# utils.py This file may be used for all utility functions
import random
from nltk import sent_tokenize, word_tokenize, pos_tag, RegexpTokenizer
import numpy as np
import datetime
import string
import pickle

'''
 Create a bijection betweeen int and object. May be used for reverse indexing
'''


class Indexer(object):
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.ints_to_objs)

    def get_object(self, i):
        return self.ints_to_objs[i]

    def contains(self, obj):
        return self.index_of(obj) != -1

    def index_of(self, obj):
        if obj in self.objs_to_ints:
            return self.objs_to_ints[obj]
        return -1

    # Get the index of the object, if add_object, add object to dict if not present
    def get_index(self, obj, add_object=True):
        if not add_object or obj in self.objs_to_ints:
            return self.index_of(obj)
        new_idx = len(self.ints_to_objs)
        self.objs_to_ints[obj] = new_idx
        self.ints_to_objs[new_idx] = obj
        return new_idx


# Add features from feats to feature indexer
# If add_to_indexer is true, that feature is indexed and added even if it is new
# If add_to_indexer is false, unseen features will be discarded
def add_dataset_features(feats, feature_indexer):
    for i in range(len(feats)):
        l = word_tokenize(feats[i].passage)
        for word in l:
            feature_indexer.get_index(word)


# Below code taken from CS 388 provided code (written by Greg Durrett <email>)

# Wraps an Indexer and a list of 1-D numpy arrays where each position in the list is the vector for the corresponding
# word in the indexer. The 0 vector is returned if an unknown word is queried.
class WordEmbeddings:
    def __init__(self, word_indexer, vectors):
        self.word_indexer = word_indexer
        self.vectors = vectors

        # assert self.vectors.shape[0] == len(self.word_indexer)

    @property
    def embedding_dim(self):
        return len(self.vectors[0])

    def get_embedding(self, word):
        word_idx = self.word_indexer.index_of(word)
        if word_idx == len(self.word_indexer):
            print("HALT")
            exit()
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of(UNK_SYMBOL)]

    def word2embedding_idx(self, word):
        word_idx = self.word_indexer.index_of(word)
        if word_idx != -1:
            return word_idx
        else:
            return self.word_indexer.index_of(UNK_SYMBOL)

    def get_embedding_idx(self, word_idx):
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.index_of(UNK_SYMBOL)]

    def get_average_score(self, word_idx):
        vec = self.get_embedding_idx(word_idx)
        return np.average(vec)


PAD_SYMBOL = "<PAD>"
UNK_SYMBOL = "<UNK>"


# Loads the given embeddings (ASCII-formatted) into a WordEmbeddings object. Augments this with an UNK embedding
# that is the 0 vector. Reads in all embeddings with no filtering -- you should only use this for relativized
# word embedding files.
def read_word_embeddings(embeddings_file):
    f = open(embeddings_file)
    word_indexer = Indexer()
    vectors = []
    for line in f:
        if line.strip() != "":
            space_idx = line.find(' ')
            word = line[:space_idx]
            numbers = line[space_idx + 1:]
            float_numbers = [float(number_str) for number_str in numbers.split()]
            # print repr(float_numbers)
            vector = np.array(float_numbers)
            word_indexer.get_index(word)
            vectors.append(vector)
            # print repr(word) + " : " + repr(vector)
    # Add PAD token
    word_indexer.get_index(PAD_SYMBOL)
    vectors.append(np.zeros(vectors[0].shape[0]))
    # Add an UNK token
    word_indexer.get_index(UNK_SYMBOL)
    vectors.append(np.zeros(vectors[0].shape[0]))

    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


# Relativization = restrict the embeddings to only have words we actually need in order to save memory
# (but this requires looking at the data in advance).

# Relativize the word vectors to the training set
def relativize(file, outfile, indexer):
    f = open(file, encoding='utf-8')
    o = open(outfile, 'w', encoding='utf-8')
    voc = []

    for line in f:
        word = line[:line.find(' ')]
        if indexer.contains(word):
            # print("Keeping word vector for " + word)
            voc.append(word)
            o.write(line)
    # for word in indexer.objs_to_ints.keys():
    # if word not in voc:
    #    print("Missing " + word)
    f.close()
    o.close()


# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len):
    result = []
    for ex in exs:
        passage = word_tokenize(ex.passage)
        result.append([input_indexer.index_of(PAD_SYMBOL) if i >= len(passage) else input_indexer.index_of(
            UNK_SYMBOL) if input_indexer.index_of(passage[i]) == -1 else input_indexer.index_of(passage[i])
                       for i in range(0, max_len)])
    return np.array(result)


def make_output_one_hot_tensor(exs, output_indexer):
    result = []
    for ex in exs:
        result.append([int(i == output_indexer.index_of(ex.author)) for i in range(len(output_indexer))])

    return np.array(result)


pos_fancy = """CC
DT
EX
FW
IN
MD
PDT
RB


WDT
WP
WP$
WRB""".split("\n")
##RBR
#RBS
##
def pos(passage, n=2, fancy=True):
    # tokenize = RegexpTokenizer(r'\w+')
    # words = tokenize.tokenize(passage)
    words = word_tokenize(passage)
    postags = pos_tag(words)

    postags_ = [("", word.lower()) if pos_tag[1] in pos_fancy else pos_tag for word, pos_tag in zip(words, postags)]

    final = []
    for i in range(len(postags) - n):
        n_gram = "".join([postags_[i + _i][1] for _i in range(n)])
        final.append(n_gram)


    return " ".join(final)


class Example:
    def __init__(self, passage, author, id=None):
        self.passage = passage
        self.author = author
        self.id = id


class AuthorshipModel:
    def __init__(self):
        self.history = None

    def _predictions(self, test_data, args):
        pass

    def _sentencewise_prediction(self, test_data, args):
        predictions = self._predictions(test_data, args)
        prediction = max(set(predictions), key=predictions.count)
        return prediction

    def evaluate(self, test_data, args):
        if args.sentencewise:
            predictions = [self._sentencewise_prediction(sentences, args) for sentences in test_data]
            labels = [sentences[0].author for sentences in test_data]
        else:
            predictions = self._predictions(test_data, args)
            labels = [example.author for example in test_data]

        correct = sum([pred == true for pred, true in zip(predictions, labels)])

        if not args.sentencewise:
            for i in range(len(test_data)):
                if labels[i]==predictions[i]:
                    print("CORRECT", labels[i])
                    print(test_data[i].passage)
                else:
                    print("INCORRECT", predictions[i], labels[i])
                    print(test_data[i].passage)

        print("Correctness: " + str(correct) + "/" + str(len(test_data)), "->", correct / len(test_data))

        if args.plot:
        
            filename = args.model + "_" + args.train_type + "_" + args.train_options + "_" + str(datetime.datetime.now()) + ".pdf"
            with open(filename, "wb") as f:
                pickle.dump((self.history, correct, len(test_data)), f)

        return correct, len(test_data)

