# Semantic Textual Similarity

Semantic textual similarity (STS) refers to a task in which we compare the
similarity between one text and another.

## Process

1. First tried to
create the sentence representation in a vector space with different embeddings like **SVD using co occurence matrix , Skipgram using negative sampling , Elmo and BERT** which are built from scratch.

2. After doing that, we used different models like **MaLSTM and SemanticBert** (our own proposed model) for the evaluation on the STS and SICK datasets.

3. For semantic analysis, we need to know more than just tokens or morphemes. Two sentences could have many similar structures, but might have total opposite
meanings.

4. The models are then pre-trained to the **Siamese Network and Triplet Loss function** for improving the model.

## Architecture

1. ### Manhatten LSTM (MaLSTM):
        The MaLSTM architecture consists of two LSTMs which shares the same weights and it embeds the sentences. Once we have the sentence embeddings, we can apply cosine similarity and get the resultant value for the Semantic Score.

2. ### Semantic BERT:
        In this we levarage the BERT Base model and fine tune it for our needs. We use a siamese network of bert to embed the sentences and use the CosineSimilarityLoss on top of it.


## Observations

The static and non-contextual embeddings are not suitable for this task and it lacks really good understanding of semantics of a sentence. After utilizing the pre-trained bert model it outperformed the word2vec MaLSTM model due to it’s high understanding of the language. The **state of the Art Model MT DNN has **92.9** Pearson Score and XLNet has around 92.5** . Both the models uses some kind of ensemble learnings. To improve our models further we can also make use of ensemble learning like boosting and bagging.






### Note:
The own built embeddings of svd , skipgram , elmo classification code is also provided which was directly used using their .pt saved model in the main code.