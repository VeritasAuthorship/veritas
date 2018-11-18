from itertools import repeat

import os
import random
import string
from nltk import sent_tokenize, word_tokenize


def _gutenberg_filename(author, title):
    return "{}__{}.txt".format(author, title)


def _gutenberg_author_title(filename):
    return tuple(filename.split("__")[:2])


class GutenbergBook:

    def __init__(self, author, title, text):
        self.author = author
        self.title = title
        self.text = text

    def select_passages(self, n=5, length=1, method="paragraph"):
        """
        Select n passages at random from a book, either by paragraph
        or by sentence.
        """
        assert method in ["paragraph", "sentence"]
        sequence = []

        if method == "paragraph":
            sequence = self.text.split("\n\n")

        elif method == "sentence":
            sequence = sent_tokenize(self.text)

        assert length < len(sequence)

        def _single_passage(seq, length):
            start = random.randrange(0, len(seq) - length)
            return " ".join(seq[start:start + length])

        return [_single_passage(sequence, length) for _ in range(n)]


class GutenbergData:
    def __init__(self):
        self.books = []

    def load_from(self, path):
        for filename in os.listdir(path):
            if filename.endswith(".txt"):
                author, title = _gutenberg_author_title(filename)
                with open(path + "/" + filename, "r") as f:
                    text = f.read()

                self.books.append(GutenbergBook(author, title, text))

        return self

    def create_dataset(self, passages_per_book=10, passage_length=3, passage_type="paragraph"):
        """
        Create a dataset from a set of books. Can specify # passages, passage length, passage
        type (sentence / paragraph). Output is a tuple (X, y).
        
        X: list of passages (text)
        y: list of authors
        """
        passages = []
        labels = []

        for book in self.books:
            selected = book.select_passages(n=passages_per_book, length=passage_length, method=passage_type)
            author = repeat(book.author, len(selected))

            passages.extend(selected)
            labels.extend(author)

        return passages, labels


if __name__ == '__main__':
    gd = GutenbergData().load_from("data/american")
    data = gd.create_dataset()
    print("finished")
