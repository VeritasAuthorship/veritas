import numpy as np
from keras import Input, Sequential
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Embedding, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from models.attention_decoder import AttentionLayer
from nltk import pos_tag, word_tokenize

from utils import WordEmbeddings

def pos(passage):
    words = word_tokenize(passage)
    postags = pos_tag(words)

    final = []
    for i in range(len(postags) - 1):
        final.append(postags[i] + postags[i+1])


    return " ".join([tag for word, tag in postags])

def transform_dataset(dataset, authors, max_length=None):
    texts = [ex.passage for ex in dataset]
    print(set([ex.author for ex in dataset]))
    print([authors.get_object(i) for i in range(len(authors))])
    labels = [authors.index_of(ex.author) for ex in dataset]

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)

    sequences = tokenizer.texts_to_sequences(texts)
    # max_seq_len = len(max(sequences, key=len))
    max_len = max_length if max_length else len(max(sequences, key=len))
    padded_sequences = pad_sequences(sequences, max_length)

    return padded_sequences, labels, tokenizer.word_index


def construct_embeddings_matrix(word_index, word_embeddings):
    embeddings_matrix = np.zeros(shape=(len(word_index) + 1, word_embeddings.embedding_dim))

    for word in word_index:
        print("embedding for word: ", word)
        embeddings_matrix[word_index.get(word)] = word_embeddings.get_embedding(word)

    return embeddings_matrix


def train_keras_model(word_embeddings, train_data, test_data, authors):
    all_data = train_data + test_data
    label_binarizer = LabelBinarizer()
    all_seq, all_labels, all_word_index = transform_dataset(all_data, authors)
    print(set(all_labels))

    label_binarizer.fit(all_labels)

    max_seq_length = all_seq.shape[1]
    print("MAX_LENGTH", max_seq_length)

    train_seq, train_labels, _ = transform_dataset(train_data, authors, max_length=max_seq_length)
    train_labels = label_binarizer.transform(train_labels)

    train_embeddings_matrix = construct_embeddings_matrix(all_word_index, word_embeddings)

    embedding_layer = Embedding(
        len(all_word_index) + 1,
        train_embeddings_matrix.shape[1],
        weights=[train_embeddings_matrix],
        input_length=max_seq_length,
        trainable=False
    )

    # sequence_input = Input(shape=(train_seq.shape[1],), dtype='int32')
    # embedded_sequences = embedding_layer(sequence_input)

    model = Sequential()
    # model.add(Embedding(len(all_word_index) + 1, 200, input_length=max_seq_length, trainable=True))
    model.add(embedding_layer)
    model.add(LSTM(200))
    # model.add(AttentionLayer())
    model.add(Dense(len(authors), activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["acc"])



    validation_split = int(0.9 * train_seq.shape[0])

    train_x = train_seq[:validation_split]
    train_y = train_labels[:validation_split]

    val_x = train_seq[validation_split:]
    val_y = train_labels[validation_split:]

    model.fit(train_x, train_y, validation_split=0.2, batch_size=10, epochs=5, shuffle=True)

    print("Evaluating model...")
    # Evaluate model
    test_Seq, test_labels, _ = transform_dataset(test_data, authors, max_length=max_seq_length)
    test_labels = label_binarizer.transform(test_labels)
    scores = model.evaluate(test_Seq, test_labels, batch_size=100)

    # scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))
