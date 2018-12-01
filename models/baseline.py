from utils import *
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
import random

'''
Baseline approach: For each author maintain a list of most used words, that are not present in stopwords
Construct tensor from this list of words
'''

# Do not add stopwords
stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", ',', '.', "''", '``', 'I', ';', 'The', '!', '--', 'old', 'But', "'s", 'And', 'said', '?', "'", ':', '-', "'ll", ')', '(', 'A', 'He', 'he', 'in', 'In', "n't", 'It', '=', 'Mr.', 'O', 'An', 'us'}

class Signature:
    ''' 
    Define the signature of an author
    '''
    def __init__(self, author, n=1000):
        self.words = []
        self.author = author
        self.counter = {}
        self.n = n

    def get_final_words(self):
        '''
        Return highest n ranked words from counter
        '''
        # print(sorted(self.counter.values())) # To see counter values
        self.words = sorted(self.counter, key=self.counter.get, reverse=True)[:self.n]
        return self.words


def train_baseline(train_data):
    word_indexer = Indexer()
    random.shuffle(train_data)
    # {author: Signature}
    authors = {}
    author_results = {}

    # Do an initial read of the entire vocabulary to add words to the indexer
    add_dataset_features(train_data, word_indexer)

    for i in range(len(train_data)):
        author = train_data[i].author
        passage = word_tokenize(train_data[i].passage)
        if author not in authors:
            authors[author] = Signature(author)
        increment_counter(passage, authors[author].counter)

    # Print data
    for author in authors:
        author_results[author] = authors[author].get_final_words()
        #print("--------------------------------------------------")
        #print("Author: " + author)
        #print(author_results[author])
    return author_results

def increment_counter(passage, counter):
    for word in passage:
        if word not in stopwords:
            if word in counter:
                counter[word] += 1
            else:
                counter[word] = 1

# American authors
# Average accuracy: .789 with 10 passages/book/author, 10 runs, 4 authors
# Average accuracy: .8519 with 30 passages/book/author, 10 runs, 4 authors
# Average accuracy: .862 with 50 passages/book/author, 10 runs, 4 authors

# British authors
# Average accuracy: .705 with 10 passages/book/author, 10 runs, 5 authors
# Average accuracy: .7498 with 30 passages/book/author, 10 runs, 5 authors
# Average accuracy: .8277 with 50 passages/book/author, 10 runs, 5 authors

# Combined authors
# Average accuracy: .6274 with 10 passages/book/author, 10 runs, 9 authors
# Average accuracy: .7607 with 30 passages/book/author, 10 runs, 9 authors
# Average accuracy: .7961 with 50 passages/book/author, 10 runs, 9 authors

# -----------------------------------------------------------------------

# British Authors
# Average accuracy: 564/2400 = .235 with 200 sentences/book/author, 5 authors (new test)

def evaluate_baseline(test_data, authors):
    random.shuffle(test_data)
    total_examples = len(test_data)
    correct = 0

    for i in range(len(test_data)):
        passage = word_tokenize(test_data[i].passage)
        max_words = 0
        max_author = ''
        for author in authors.keys():
            count = 0
            for word in passage:
                if word in authors[author]:
                    count += 1
            if count > max_words:
                max_words = count
                max_author = author
        
        if max_author == test_data[i].author:
            correct += 1

    print ("Correctness: " + str(correct) + "/" + str(total_examples), "->",correct/total_examples)