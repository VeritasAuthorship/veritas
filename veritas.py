'''search.py
 This file may be used for kicking off the model, reading in data and preprocessing
 '''

import argparse
import sys
#import models.keras_lstm as k

from models.attention import train_enc_dec_model
from utils import *
from gutenberg_data import *
from spooky_authorship import spooky_authorship_data

# sys.path.append("./models")
from models.baseline import *
#from models.sklearn_baselines import sklearn_train
from models.LSTM import *


# Read in command line arguments to the system
def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BASELINE', help="Model to run")
    parser.add_argument('--train_type', type=str, default="GUTENBERG", help="Data type - Gutenberg or custom")
    parser.add_argument('--train_path', type=str, default='data/british/', help='Path to the training set')
    parser.add_argument('--test_path', type=str, default='data/gut-test/british', help='Path to the test set')
    parser.add_argument('--train_options', type=str, default='', help="Extra train options, eg pos tags embeddings")

    # Seq-2-Seq args
    parser.add_argument('--reverse_input', type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for encoder training")
    parser.add_argument('--emb_dropout', type=float, default=0.2, help="dropout for embedding layer")
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help="dropout for RNN")
    parser.add_argument('--bidirectional', type=bool, default=True, help="lstm birectional or not")
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')
    parser.add_argument('--word_vecs_path_input', type=str, default='data/glove.6B.300d.txt',
                        help='path to word vectors file')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt',
                        help='path to word vectors file')
    parser.add_argument('--embedding_size', type=int, default=300, help='Embedding size')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parse()
    print(args)

    if args.model == 'BASELINE':
        # Get books from train path and call baselineb model train function
        if (args.train_type == 'GUTENBERG'):
            train_data, test_data, num_authors = gutenberg_dataset(args.train_path, args.test_path)
            print("training")
            authors = train_baseline(train_data)
            print("testing")
            evaluate_baseline(test_data, authors)

        elif args.train_type == "SPOOKY":
            train_data, test_data, num_authors = spooky_authorship_data()
            print("training")
            authors = train_baseline(train_data)
            print("testing")
            evaluate_baseline(test_data, authors)

    elif args.model == 'LSTM':
        if args.train_type == 'GUTENBERG':
            
            if args.train_options == 'POS':
                pretrained = False
                train_data, test_data, authors = gutenberg_dataset(args.train_path, args.test_path, postags=True)

                word_indexer = Indexer()
                add_dataset_features(train_data, word_indexer)
                add_dataset_features(test_data, word_indexer)
                word_indexer.get_index(PAD_SYMBOL)
                word_indexer.get_index(UNK_SYMBOL)

                word_vectors = WordEmbeddings(word_indexer, None)
    
            else:
                pretrained = True
                train_data, test_data, authors = gutenberg_dataset(args.train_path, args.test_path)
                word_indexer = Indexer()
                add_dataset_features(train_data, word_indexer)
                add_dataset_features(test_data, word_indexer)

                relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
                word_vectors = read_word_embeddings(args.word_vecs_path)
    
            print("Finished extracting embeddings")
            print("training")

            trained_model = train_lstm_model(train_data, test_data, authors, word_vectors, args, pretrained=pretrained)

            print("testing")
            trained_model.evaluate(test_data)

        elif args.train_type == 'SPOOKY':
            train_data, test_data, authors = spooky_authorship_data()
            word_indexer = Indexer()
            add_dataset_features(train_data, word_indexer)
            add_dataset_features(test_data, word_indexer)

            relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
            word_vectors = read_word_embeddings(args.word_vecs_path)

            print("Finished extracting embeddings")
            print("training")
            trained_model = train_lstm_model(train_data, test_data, authors, word_vectors, args)

            print("testing")
            trained_model.evaluate(test_data)


    elif args.model == "LSTM_ATTN":
        if args.train_type == 'GUTENBERG':
            train_data, test_data, authors = gutenberg_dataset(args.train_path, args.test_path)
            word_indexer = Indexer()
            add_dataset_features(train_data, word_indexer)
            add_dataset_features(test_data, word_indexer)

            relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
            word_vectors = read_word_embeddings(args.word_vecs_path)

            print("Finished extracting embeddings")
            print("training")
            trained_model = train_enc_dec_model(train_data, test_data, authors, word_vectors, args)

            print("testing")
            trained_model.evaluate(test_data)

        elif args.train_type == "SPOOKY":
            train_data, test_data, authors = spooky_authorship_data()
            word_indexer = Indexer()
            add_dataset_features(train_data, word_indexer)
            add_dataset_features(test_data, word_indexer)

            relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
            word_vectors = read_word_embeddings(args.word_vecs_path)

            print("Finished extracting embeddings")
            print("training")
            trained_model = train_enc_dec_model(train_data, test_data, authors, word_vectors, args)

            print("testing")
            trained_model.evaluate(test_data)


    elif args.model == "KERAS":
        if args.train_type == "GUTENBERG":
            train_data, test_data, authors = gutenberg_dataset(args.train_path, args.test_path)
            embeddings = read_word_embeddings(args.word_vecs_path)

            k.train_keras_model(embeddings, train_data, test_data, authors)
        elif args.train_type == "SPOOKY":
            train_data, test_data, authors = spooky_authorship_data()
            embeddings = read_word_embeddings(args.word_vecs_path)

            k.train_keras_model(embeddings, train_data, test_data, authors)

    elif args.model == "SKLEARN":
        if args.train_type == "GUTENBERG":
            train_data, test_data, authors = gutenberg_dataset(args.train_path, args.test_path)

            sklearn_train(train_data, test_data, authors)
        elif args.train_type == "SPOOKY":
            train_data, test_data, authors = spooky_authorship_data()

            sklearn_train(train_data, test_data, authors)


    elif args.model == 'VAE':
        pass

    else:
        raise Exception("Please select appropriate model")
