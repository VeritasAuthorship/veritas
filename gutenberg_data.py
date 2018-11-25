import os
import random
from nltk import sent_tokenize

from utils import Indexer


def _gutenberg_filename(author, title):
    return "{}__{}.txt".format(author, title)


def _gutenberg_author_title(filename):
    return tuple(filename.split("__")[:2])


class GutenbergBook:

    def __init__(self, author, title, text):
        self.author = author
        self.title = title
        self.text = text

    def select_passages(self, n, length, method, min_char_length=500):
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
            passage = ""
            while len(passage) < min_char_length:
                start = random.randrange(0, len(seq) - length)
                passage = " ".join(seq[start:start + length])

            assert len(passage) >= min_char_length
            return passage

        return [_single_passage(sequence, length) for _ in range(n)]


class Example:
    def __init__(self, passage, author):
        self.passage = passage
        self.author = author

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

    def create_dataset(self, passages_per_book=33, passage_length=3, passage_type="paragraph", test_ex=3):
        """
        Create a dataset from a set of books. Can specify # passages, passage length, passage
        type (sentence / paragraph). Output is a list of Examples.

        X: list of passages (text)
        y: list of authors
        """
        examples = []
        test = []

        authors = Indexer()

        for book in self.books:
            selected = book.select_passages(n=passages_per_book, length=passage_length, method=passage_type)
            authors.get_index(book.author)

            for i in range(len(selected)):
                passage = selected[i]
                if i < passages_per_book - test_ex:
                    examples.append(Example(passage, book.author))
                else:
                    test.append(Example(passage, book.author))

        return examples, test, authors


def gutenberg_dataset(train_path):
    gd = GutenbergData().load_from(train_path)
    train_data, test_data, num_authors = gd.create_dataset()
    print("Finished loading data")

    # TODO: test data splitting
    return train_data, test_data, num_authors

if __name__ == '__main__':
    gutenberg_dataset("data/combined")