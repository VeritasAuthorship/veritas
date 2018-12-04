from nltk import word_tokenize, pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np

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


#
# Gutenberg dataset:
#  1-3 grams
#    Logistic Regression: 0.413
#    Multinomial NB   : 0.56
#
# Spooky dataset:
# 1-3 grams
#    Logistic Regression: 0.80
#    Multinomial NB: 0.83

def sklearn_train(train_exs, test_exs, authors, pos_tags=False):
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    pos_vectorizer = CountVectorizer(ngram_range=(1, 3))

    train_texts = [ex.passage if pos_tags else ex.passage for ex in train_exs]
    test_texts = [ex.passage if pos_tags else ex.passage for ex in test_exs]
    vectorizer.fit(train_texts + test_texts)

    pos_train_texts = [pos(ex.passage) if pos_tags else ex.passage for ex in train_exs]
    pos_test_texts = [pos(ex.passage) if pos_tags else ex.passage for ex in test_exs]
    pos_vectorizer.fit(pos_train_texts + pos_test_texts)

    # x_train = np.concatenate((vectorizer.transform(train_texts).toarray(), pos_vectorizer.transform(pos_train_texts)), axis=1)
    # x_train = np.concatenate((vectorizer.transform(train_texts).toarray(),vectorizer.transform(pos_train_texts).toarray()), axis=1)
    # x_test = np.concatenate((vectorizer.transform(test_texts).toarray(),vectorizer.transform(pos_test_texts).toarray()), axis=1)

    train_labels = [ex.author for ex in train_exs]

    test_labels = [ex.author for ex in test_exs]

    # vectorizer.fit(train_texts + test_texts)

    X = vectorizer.transform(train_texts)
    y = train_labels

    for clf, msg in clfs:
        clf.fit(X, y)
        predict = clf.predict(vectorizer.transform(test_texts))
        # predict = clf.predict(x_test)

        print(msg, accuracy_score(test_labels, predict))
