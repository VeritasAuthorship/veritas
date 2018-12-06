import pandas as pd
from utils import Example, Indexer, pos
import random


def spooky_authorship_data(args, test_split=0.7, max_char_length=500, postags=False):
    with open("data/spooky-authorship/train.csv") as f:
        train_df = pd.read_csv(f)

    train_df.applymap(lambda s: s[1:-1] if s.startswith("\"") else s)

    print("Spooky Authorship Dataset:")
    print("   ", "authors: ", set(train_df["author"]))

    if postags:
        examples = [Example(pos(passage), author) for passage, author in zip(train_df["text"], train_df["author"]) if len(passage) <= max_char_length][:100]

    else:
        examples = [Example(passage, author) for passage, author in zip(train_df["text"], train_df["author"]) if len(passage) <= max_char_length][:100]

    random.shuffle(examples)
    test_idx = int(test_split * len(examples))

    train_exs = examples[:test_idx]
    test_exs = examples[test_idx:]

    authors = Indexer()

    for author in set(train_df["author"]):
        authors.get_index(author)

    return train_exs, test_exs, authors


if __name__ == '__main__':
    spooky_authorship_data()