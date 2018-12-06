from utils import *
import os
import random

TRAIN_PATH = "data/C50train"
TEST_PATH = "data/C50test"

AUTHORS = os.listdir(TRAIN_PATH)


def read_file(prefix):
    def _read(filename):
        with open(prefix + filename, "r") as f:
            text = f.read()

        return text

    return _read


def create_reuters_data(args, n_authors=3, articles_per_author=40, test_split=0.3):
    authors = random.sample(AUTHORS, n_authors)

    train_dataset = []
    test_dataset = []

    test_number = int((1 / (1 - test_split)) * articles_per_author)

    for author in authors:
        # Train Dataset
        articles = os.listdir(TRAIN_PATH + "/" + author)
        selected_articles = random.sample(articles, len(articles))
        article_bodies = map(read_file(TRAIN_PATH + "/" + author + "/"), selected_articles)
        train_dataset.extend(map(lambda a: Example(a, author), article_bodies))

        # Test Dataset
        articles = os.listdir(TEST_PATH + "/" + author)
        selected_articles = random.sample(articles, len(articles))
        article_bodies = map(read_file(TEST_PATH + "/" + author + "/"), selected_articles)

        if args.train_options == "POS":
            article_bodies = map(pos, article_bodies)

        test_dataset.extend(map(lambda a: Example(a, author), article_bodies))


    # random.shuffle(dataset)
    # split = int((1 - test_split) * len(dataset))

    authors_index = Indexer()
    for author in authors:
        authors_index.get_index(author, True)

    # return dataset[:split], dataset[split:], authors_index
    return train_dataset, test_dataset, authors_index

if __name__ == '__main__':
    create_reuters_data()
