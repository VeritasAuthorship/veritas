from nltk import word_tokenize, pos_tag
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.pipeline import Pipeline

from utils import AuthorshipModel

clfs = [
    (LogisticRegression(), "logistic regression: "),
    (MultinomialNB(), "multinomial naive bayes: ")
]


def pos(passage):
    words = word_tokenize(passage)
    postags = pos_tag(words)

    final = []
    for i in range(len(postags) - 2):
        final.append(postags[i][1] + postags[i + 1][1])

    return " ".join(final)


# Gutenberg dataset:
#  1-3 grams
#    Logistic Regression: 0.413
#    Multinomial NB   : 0.56
#
# Spooky dataset:
# 1-3 grams
#    Logistic Regression: 0.80
#    Multinomial NB: 0.83

def sklearn_train(train_exs, test_exs, authors, args, pos_tags=False):
    train_texts = [ex.passage if pos_tags else ex.passage for ex in train_exs]
    test_texts = [ex.passage if pos_tags else ex.passage for ex in test_exs]
    train_labels = [ex.author for ex in train_exs]
    test_labels = [ex.author for ex in test_exs]

    for clf, msg in clfs:
        pipeline = Pipeline([
            ("features", CountVectorizer(ngram_range=(1, 3))),
            ("classifier", clf)
        ])
        pipeline.fit(train_texts, train_labels)
        predict = pipeline.predict(test_texts)

        print(msg, accuracy_score(test_labels, predict))
