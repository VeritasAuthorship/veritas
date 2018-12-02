import numpy as np
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

from utils import WordEmbeddings


def transform_dataset(dataset, authors):
    texts = [ex.passage for ex in dataset]
    labels = [authors.index_of(ex.author) for ex in dataset]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    max_seq_len = len(max(sequences, key=len))
    padded_sequences = pad_sequences(sequences, max_seq_len)

    lb = LabelBinarizer()
    labels_onehot = lb.fit(labels)

    return pad_sequences, labels_onehot, tokenizer.word_index


def construct_embeddings_matrix(word_index, word_embeddings):
    embeddings_matrix = np.zeros(shape=(len(word_index), word_embeddings.embedding_dim))

    for i, word in word_index:
        embeddings_matrix[i] = word_embeddings.get_embedding(word)

    return embeddings_matrix


def train_keras_model(embeddings: WordEmbeddings, train_data, test_data, authors):
    # texts = [ex.passage for ex in train_data]
    # labels = [ex.author for ex in train_data]
    #
    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    #
    # sequences = tokenizer.texts_to_sequences(texts)
    # max_seq_len = len(max(sequences, key=len))
    #
    # padded_sequences = pad_sequences(sequences, max_seq_len)
    #
    # lb = LabelBinarizer()
    # labels_onehot = lb.fit_transform(labels)
    #
    # embeddings_matrix = np.zeros(shape=(len(tokenizer.word_index), embeddings.embedding_dim))

    for i, word in tokenizer.word_index:
        embeddings_matrix[i] = embeddings.get_embedding(word)

    embedding_layer = Embedding(
        len(tokenizer.word_index),
        embeddings_matrix.shape[1],
        weights=embeddings_matrix,
        input_length=max_seq_len,
        trainable=False
    )
