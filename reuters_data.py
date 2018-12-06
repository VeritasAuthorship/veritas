from utils import *
import os
import random

TRAIN_PATH = "data/C50/C50train"
TEST_PATH = "data/C50/C50test"

AUTHORS = os.listdir(TRAIN_PATH)


def read_file(prefix):
    def _read(filename):
        with open(prefix + filename, "r") as f:
            text = f.read()

        return text

    return _read


def create_reuters_data(args, n_authors=3, articles_per_author=50, test_split=0.3):
    authors = random.sample(AUTHORS, n_authors)

    dataset = []

    for author in authors:
        articles = os.listdir(TRAIN_PATH + "/" + author)
        selected_articles = random.sample(articles, articles_per_author)
        article_bodies = map(read_file(TRAIN_PATH + "/" + author + "/"), selected_articles)

        dataset.extend(map(lambda a: Example(a, author), article_bodies))

    split = int((1 - test_split) * len(dataset))

    authors_index = Indexer()
    for author in authors:
        authors_index.get_index(author, True)

    return dataset[:split], dataset[split:], authors_index


if __name__ == '__main__':
    create_reuters_data()
