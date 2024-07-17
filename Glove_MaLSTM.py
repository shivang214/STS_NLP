# -*- coding: utf-8 -*-

from datasets import load_dataset

dataset = load_dataset("sick")

import pandas as pd
train_df = pd.DataFrame(dataset['train'])
val_df = pd.DataFrame(dataset['validation'])
test_df = pd.DataFrame(dataset['test'])

train_df = train_df.drop(columns=['entailment_AB', 'entailment_BA', 'sentence_A_dataset', 'sentence_B_dataset'])
test_df = test_df.drop(columns=['entailment_AB', 'entailment_BA', 'sentence_A_dataset', 'sentence_B_dataset'])
val_df = val_df.drop(columns=['entailment_AB', 'entailment_BA', 'sentence_A_dataset', 'sentence_B_dataset'])

import gensim.downloader as api
import torch
import numpy as np
import pickle

word_vectors = torch.load("skip-gram-word-vectors.pt")
with open("word_vectors.pkl", "wb") as f:
    pickle.dump(word_vectors, f)

import pickle

with open("word_vectors.pkl", "rb") as f:
    word_vectors = pickle.load(f)

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

sentences = train_df['sentence_A'].tolist()
sentences = sentences + train_df['sentence_B'].tolist()

tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

from collections import defaultdict

word_count = defaultdict(int)

for sentence in tokenized_sentences:
    for word in sentence:
        if word.lower() in word_vectors.key_to_index:
            word_count[(word.lower())] += 1


vocab = set(['<UNK>', '<PAD>'])
counter_threshold = 0

for char, cnt in word_count.items():
    if cnt > counter_threshold:
        vocab.add(char)

print(f'Vocab Length: {len(vocab)}')

word_to_idx = {char: i for i, char in enumerate(vocab)}
idx_to_word = {i: char for char, i in word_to_idx.items()}

import torch
import numpy as np

embedding_dim = 200
embeddings = 1 * np.random.randn(len(vocab), embedding_dim)

for word, index in word_to_idx.items():
    if word in word_vectors.key_to_index:
        embeddings[index] = torch.tensor(word_vectors[(word)], dtype=torch.float32)

embeddings = torch.tensor(embeddings, dtype=torch.float32)

from typing import List
import string
import re

class SickDataSet:
    def __init__(self, data):
        self.data = data
        self.unk_id = word_to_idx.get('<UNK>')
        self.pad_id = word_to_idx.get('<PAD>')

    def __getitem__(self, idx: int) -> dict:
        processed_text_a = self.data['sentence_A'].iloc[idx].lower().translate(
            str.maketrans('', '', string.punctuation))
        tokenized_sentence_a = [
            word_to_idx.get(word, self.unk_id) for word in word_tokenize(processed_text_a)
        ]
        processed_text_b = self.data['sentence_B'].iloc[idx].lower().translate(
            str.maketrans('', '', string.punctuation))
        tokenized_sentence_b = [
            word_to_idx.get(word, self.unk_id) for word in word_tokenize(processed_text_b)
        ]

        train_sample = {
            "textA": tokenized_sentence_a,
            "textB": tokenized_sentence_b,
            "label": self.data['relatedness_score'].iloc[idx]
        }

        return train_sample

    def __len__(self) -> int:
        return len(self.data)

def collate_fn_with_padding(
    input_batch: List[dict], pad_id=word_to_idx.get('<PAD>'), max_len=256, device=torch.device('cpu')) -> dict:
    # Get the maximum sequence length in the batch
    max_seq_len = min(max(len(x['textA']) for x in input_batch), max_len)

    padded_sequences_a = []
    padded_sequences_b = []
    labels = []

    # Pad sequences and collect labels
    for sequence in input_batch:
        # Pad sequence A
        padded_sequence_a = sequence['textA']
        padded_sequences_a.append(padded_sequence_a)

        # Pad sequence B
        padded_sequence_b = sequence['textB']
        padded_sequences_b.append(padded_sequence_b)

        # Collect labels
        labels.append(sequence['label'])

    # Convert padded sequences and labels to tensors
    input_ids_a = torch.LongTensor(padded_sequences_a).to(device)
    input_ids_b = torch.LongTensor(padded_sequences_b).to(device)
    labels = torch.FloatTensor(labels).to(device)

    new_batch = {
        'input_ids_a': input_ids_a,
        'input_ids_b': input_ids_b,
        'label': labels
    }

    return new_batch

train_dataset = SickDataSet(train_df)
test_dataset = SickDataSet(test_df)

from torch.utils.data import DataLoader

batch_size = 1
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=collate_fn_with_padding, batch_size=batch_size)

test_dataloader = DataLoader(
    test_dataset, shuffle=False, collate_fn=collate_fn_with_padding, batch_size=batch_size)

import torch.nn as nn
import torch.nn.functional as F

class SiameseLSTM(nn.Module):
    def __init__(self, embedding_dim, embeddings_matrix, hidden_dim, num_layers=1):
        super(SiameseLSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings_matrix, freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        # Linear layer for concatenation
        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenating h_a and h_b, so input size is hidden_dim * 2
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )  # Concatenating h_a and h_b, so input size is hidden_dim * 2

    def forward(self, input_a, input_b):
        embedded_a = self.embedding(input_a)
        embedded_b = self.embedding(input_b)

        _, (h_a, _) = self.lstm(embedded_a)
        _, (h_b, _) = self.lstm(embedded_b)

        # Get last hidden state
        h_a_last = h_a[-1]
        h_b_last = h_b[-1]

        # Concatenate last hidden states
        concatenated_hidden = torch.cat((h_a_last, h_b_last), dim=1)

        # Pass through sequential layer
        linear_output = self.linear(concatenated_hidden)

        # Apply sigmoid activation
        output = torch.sigmoid(linear_output) * 4 + 1

        return output.squeeze()

from scipy.stats import pearsonr
from scipy.stats import spearmanr


def get_performance(model, dataloader):
    array1 = []
    array2 = []
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids_a = batch['input_ids_a'].to(device)
            input_ids_b = batch['input_ids_b'].to(device)
            labels = batch['label'].to(device)

            input_ids_a = input_ids_a.long()
            input_ids_b = input_ids_b.long()
            labels = labels.float()

            # Forward pass
            output = model(input_ids_a, input_ids_b)
            array1.append(output.cpu())
            array2.append(labels[0].cpu())

        correlation_coefficient_p, p_value_p = pearsonr(array1, array2)

        print("Pearson correlation coefficient:", correlation_coefficient_p)


        correlation_coefficient, p_value = spearmanr(array1, array2)

        print("Spearman rank correlation coefficient:", correlation_coefficient)

    return correlation_coefficient_p, correlation_coefficient

import torch.optim as optim
from torch.utils.data import DataLoader

# Assuming train_dataloader contains your training data
# Define your train_dataloader and other necessary components

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 200
hidden_dim = 256
num_layers = 2
num_epochs = 10

model = SiameseLSTM(embedding_dim, embeddings, hidden_dim, num_layers).to(device)  # Move model to GPU if available


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

data = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(train_dataloader):
        input_ids_a = batch['input_ids_a'].to(device)  # Move input tensor to GPU if available
        input_ids_b = batch['input_ids_b'].to(device)  # Move input tensor to GPU if available
        labels = batch['label'].to(device)  # Move input tensor to GPU if available


        input_ids_a = input_ids_a.long()
        input_ids_b = input_ids_b.long()
        labels = labels.float()

        optimizer.zero_grad()

        output = model(input_ids_a, input_ids_b)

        loss = criterion(output, labels)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}')
    p , sp = get_performance(model, test_dataloader)
    data.append([epoch, p, sp])

import matplotlib.pyplot as plt

epochs = [entry[0] for entry in data]
pearson_correlation = [entry[1] for entry in data]
spearman_correlation = [entry[2] for entry in data]

# Plotting
plt.figure(figsize=(10, 6))

# Plot Pearson correlation
plt.plot(epochs, pearson_correlation, label='Pearson Correlation', marker='o')

# Plot Spearman correlation
plt.plot(epochs, spearman_correlation, label='Spearman Correlation', marker='o')

# Add labels and legend
plt.xlabel('Epoch')
plt.ylabel('Correlation')
plt.title('Pearson and Spearman Correlation vs Epoch')
plt.legend()

# Show plot
plt.grid(True)
plt.show()
