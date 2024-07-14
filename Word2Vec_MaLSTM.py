# -*- coding: utf-8 -*-

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil

CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'sts-ds:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F4942392%2F8320590%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240508%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240508T145746Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D2938dee41d4dffd45e15e0b55410fcab2b10b5133cc59e5a45efedd94e6f6984a86637bbdc9a24f3330165fcba5dfbacde8f9abe1b9bbb018b06646ac5fd76d93b535e91b3f61b251d1815d54f5a84fc5fb07cc57c94f2037bd8a9ec293251e8cd9ac7752c5bdadeb746fb2ddab870f0c2aeb8ec06df68ec875ff9505eca46027a09ea5d2f0fcd91302aecb34c2c2b42e6acb68afd0652aad058556a23a0f840f565aa18da9ddcfccfd73e5ca7cacd4704e901f58d6cc3b356fe0f2bdb9a1b86768d4e66176e63c86ad5194414a08713c9977c2396d46199c580af784be3849a28b1e67209cdbe9148ffc340af3b018498b868de5e471c700118de34cc7ac3f0'

KAGGLE_INPUT_PATH='/kaggle/input'
KAGGLE_WORKING_PATH='/kaggle/working'
KAGGLE_SYMLINK='kaggle'

!umount /kaggle/input/ 2> /dev/null
shutil.rmtree('/kaggle/input', ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
  os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", 'input'), target_is_directory=True)
except FileExistsError:
  pass
try:
  os.symlink(KAGGLE_WORKING_PATH, os.path.join("..", 'working'), target_is_directory=True)
except FileExistsError:
  pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers['content-length']
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded")
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith('.zip'):
              with ZipFile(tfile) as zfile:
                zfile.extractall(destination_path)
            else:
              with tarfile.open(tfile.name) as tarfile:
                tarfile.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
        continue
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')
        continue

print('Data source import complete.')

from datasets import load_dataset

dataset = load_dataset("sentence-transformers/all-nli")

import pandas as pd

train_df = pd.read_csv('/kaggle/input/sts-ds/sts-train-cleaned.csv')
test_df = pd.read_csv('/kaggle/input/sts-ds/sts-test-cleaned.csv')
val_df = pd.read_csv('/kaggle/input/sts-ds/sts-dev-cleaned.csv')

train_df.head()

train_df = train_df.drop(columns=['source', 'dataset', 'timestamp'])
test_df = test_df.drop(columns=['source', 'dataset', 'timestamp'])
val_df = val_df.drop(columns=['source', 'dataset', 'timestamp'])

import gensim.downloader as api
import numpy as np

word_vectors = api.load("word2vec-google-news-300")

with open("word_vectors.pkl", "wb") as f:
    pickle.dump(word_vectors, f)

!du word_vectors.pkl

import pickle

with open("word_vectors.pkl", "rb") as f:
    word_vectors = pickle.load(f)

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

sentences = train_df['sentenceA'].tolist()
sentences = sentences + train_df['sentenceB'].tolist()

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

embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocab), embedding_dim)

for word, index in word_to_idx.items():
    if word in word_vectors.key_to_index:
        embeddings[index] = torch.tensor(word_vectors[(word)], dtype=torch.float32)

embeddings = torch.tensor(embeddings, dtype=torch.float32)

embeddings.shape

import torch
from torch.nn.utils.rnn import pad_sequence
from typing import List
import string
import re

class SickDataSet:
    def __init__(self, data):
        self.data = data
        self.unk_id = word_to_idx.get('<UNK>')
        self.pad_id = word_to_idx.get('<PAD>')

    def __getitem__(self, idx: int) -> dict:
        processed_text_a = self.data['sentenceA'].iloc[idx].lower().translate(
            str.maketrans('', '', string.punctuation))
        tokenized_sentence_a = [
            word_to_idx.get(word, self.unk_id) for word in word_tokenize(processed_text_a)
        ]
        processed_text_b = self.data['sentenceB'].iloc[idx].lower().translate(
            str.maketrans('', '', string.punctuation))
        tokenized_sentence_b = [
            word_to_idx.get(word, self.unk_id) for word in word_tokenize(processed_text_b)
        ]

        train_sample = {
            "textA": tokenized_sentence_a,
            "textB": tokenized_sentence_b,
            "label": self.data['score'].iloc[idx] / 5
        }

        return train_sample

    def __len__(self) -> int:
        return len(self.data)

def collate_fn_with_padding(
    input_batch: List[dict], pad_id=word_to_idx.get('<PAD>'), max_len=128, device=torch.device('cpu')) -> dict:
    # Pad sequences and collect labels
    padded_sequences_a = [torch.LongTensor(sequence['textA']) for sequence in input_batch]
    padded_sequences_b = [torch.LongTensor(sequence['textB']) for sequence in input_batch]
    labels = [sequence['label'] for sequence in input_batch]

    # Pad sequences
    padded_sequences_a = pad_sequence(padded_sequences_a, batch_first=True, padding_value=pad_id)
    padded_sequences_b = pad_sequence(padded_sequences_b, batch_first=True, padding_value=pad_id)

    # Truncate sequences if they exceed max_len
    padded_sequences_a = padded_sequences_a[:, :max_len]
    padded_sequences_b = padded_sequences_b[:, :max_len]

    # Convert labels to tensor
    labels = torch.FloatTensor(labels).to(device)

    new_batch = {
        'input_ids_a': padded_sequences_a.to(device),
        'input_ids_b': padded_sequences_b.to(device),
        'label': labels
    }

    return new_batch

train_dataset = SickDataSet(train_df)
val_dataset = SickDataSet(val_df)
test_dataset = SickDataSet(test_df)

from torch.utils.data import DataLoader

batch_size = 32
train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=collate_fn_with_padding, batch_size=batch_size)
val_dataloader = DataLoader(
    val_dataset, shuffle=False, collate_fn=collate_fn_with_padding, batch_size=batch_size)

test_dataloader = DataLoader(
    test_dataset, shuffle=False, collate_fn=collate_fn_with_padding, batch_size=batch_size)

for i in train_dataloader:
    print(i)
    break

embeddings[0].dtype

import torch.nn as nn
import torch.nn.functional as F

class SiameseLSTM(nn.Module):
    def __init__(self, embedding_dim, embeddings_matrix, hidden_dim, num_layers=1):
        super(SiameseLSTM, self).__init__()

        self.embedding = nn.Embedding.from_pretrained(embeddings_matrix, freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)


        self.linear = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, input_a, input_b):
        embedded_a = self.embedding(input_a)
        embedded_b = self.embedding(input_b)

        _, (h_a, _) = self.lstm(embedded_a)
        _, (h_b, _) = self.lstm(embedded_b)

        h_a_last = h_a[-1]
        h_b_last = h_b[-1]

        concatenated_hidden = torch.cat((h_a_last, h_b_last), dim=1)

        linear_output = self.linear(concatenated_hidden)

        output = torch.sigmoid(linear_output) * 5

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
            array2.append(labels.cpu())

        correlation_coefficient_p, p_value_p = pearsonr(array1, array2)

        print("Pearson correlation coefficient:", correlation_coefficient_p)


        correlation_coefficient, p_value = spearmanr(array1, array2)

        print("Spearman rank correlation coefficient:", correlation_coefficient)

    return correlation_coefficient_p, correlation_coefficient

import torch.optim as optim
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 300
hidden_dim = 128
num_layers = 1
num_epochs = 50

model = SiameseLSTM(embedding_dim, embeddings, hidden_dim, num_layers).to(device)  # Move model to GPU if available


criterion = nn.MSELoss()
optimizer = optim.NAdam(model.parameters(), lr=0.001)

data = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(train_dataloader):
        input_ids_a = batch['input_ids_a'].to(device)
        input_ids_b = batch['input_ids_b'].to(device)
        labels = batch['label'].to(device)


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
    p , sp = get_performance(model, val_dataloader)
    data.append([epoch, p, sp])

import matplotlib.pyplot as plt

epochs = [entry[0] for entry in data]
pearson_correlation = [entry[1] for entry in data]
spearman_correlation = [entry[2] for entry in data]

plt.figure(figsize=(10, 6))

plt.plot(epochs, pearson_correlation, label='Pearson Correlation', marker='o')

plt.plot(epochs, spearman_correlation, label='Spearman Correlation', marker='o')

plt.xlabel('Epoch')
plt.ylabel('Correlation')
plt.title('Pearson and Spearman Correlation vs Epoch')
plt.legend()

plt.grid(True)
plt.show()