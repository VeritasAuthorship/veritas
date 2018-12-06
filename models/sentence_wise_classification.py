import itertools

from nltk import sent_tokenize

from utils import Example


def expand(example):
    return [Example(sentence, example.author) for sentence in sent_tokenize(example.passage)]


def sentencewise(data):
    train_data, test_data, authors = data

    train_data = list(itertools.chain(*list(map(expand, train_data))))
    test_data = list(map(expand, test_data))

    return train_data, test_data, authors


def evaluate_sentences(test_data, transform):
    output = []

    for sentences in test_data:
        predictions = [transform(sentence) for sentence in sentences]

        prediction = max(set(predictions), key=predictions.count)

        output.append(prediction)

    return output