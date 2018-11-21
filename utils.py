# utils.py This file may be used for all utility functions
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

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

    def get_embedding(self, word):
        word_idx = self.word_indexer.get_index(word)
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.get_index(UNK_SYMBOL)]

    def get_embedding_idx(self, word_idx):
        if word_idx != -1:
            return self.vectors[word_idx]
        else:
            return self.vectors[self.word_indexer.get_index(UNK_SYMBOL)]

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
    # Add PAD token
    word_indexer.get_index(PAD_SYMBOL)
    # Add an UNK token
    word_indexer.get_index(UNK_SYMBOL)
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
    f.close()
    print("Read in " + repr(len(word_indexer)) + " vectors of size " + repr(vectors[0].shape[0]))
    vectors.append(np.zeros(vectors[0].shape[0]))
    # Turn vectors into a 2-D numpy array
    return WordEmbeddings(word_indexer, np.array(vectors))


#################
# You probably don't need to interact with this code unles you want to relativize other sets of embeddings
# to this data. Relativization = restrict the embeddings to only have words we actually need in order to save memory
# (but this requires looking at the data in advance).

# Relativize the word vectors to the training set
def relativize(file, outfile, indexer):
    f = open(file)
    o = open(outfile, 'w')
    voc = []

    for line in f:
        word = line[:line.find(' ')]
        if indexer.contains(word):
            print("Keeping word vector for " + word)
            voc.append(word)
            o.write(line)
    for word in indexer.objs_to_ints.keys():
        if word not in voc:
            print("Missing " + word)
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
