'''search.py
 This file may be used for kicking off the model, reading in data and preprocessing
 '''

import argparse
import sys
#import models.keras_lstm as k

from models.attention import train_lstm_attention_model
from models.gru_attention import train_gru_attention_model
from models.du_attn_classifier import train_du_attention_model
from models.du_attn_classifier_lstm import train_du_attention_model_lstm
from utils import *
from gutenberg_data import *
from spooky_authorship import spooky_authorship_data

# sys.path.append("./models")
from models.baseline import *
# from models.sklearn_baselines import sklearn_train
from models.LSTM import *
# from models.attention import train_lstm_attention_model
# sys.path.append("./models")
from models.baseline import *
from models.sentence_wise_classification import *
from models.sklearn_baselines import sklearn_train
from reuters_data import create_reuters_data
from spooky_authorship import spooky_authorship_data
from models.vae import *


# Read in command line arguments to the system
def arg_parse():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BASELINE', help="Model to run")
    parser.add_argument('--train_type', type=str, default="GUTENBERG", help="Data type - Gutenberg or custom")
    parser.add_argument('--train_path', type=str, default='data/british/', help='Path to the training set')
    parser.add_argument('--test_path', type=str, default='data/gut-test/british', help='Path to the test set')
    parser.add_argument('--train_options', type=str, default='', help="Extra train options, eg pos tags embeddings")
    parser.add_argument('--sentencewise', type=bool, default=False, help="")
    parser.add_argument('--kaggle', type=bool, default=False, help="")
    parser.add_argument('--plot', type=bool, default=False, help="")

    # Encoder-Decoder, VAE-RNN args
    parser.add_argument('--reverse_input', type=bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for encoder training")
    parser.add_argument('--emb_dropout', type=float, default=0.2, help="dropout for embedding layer")
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help="dropout for RNN")
    parser.add_argument('--bidirectional', type=bool, default=True, help="lstm birectional or not")
    parser.add_argument('--hidden_size', type=int, default=300, help='hidden state dimensionality')
    parser.add_argument('--word_vecs_path_input', type=str, default='data/glove.6B.300d.txt',
                        help='path to word vectors file')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt',
                        help='path to word vectors file')
    parser.add_argument('--embedding_size', type=int, default=300, help='Embedding size')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4 * 5, help='Learning rate')
    parser.add_argument('--z_dim', type=int, default=50, help='Size of latent space')

    args = parser.parse_args()
    return args


def get_data(args):
    assert args.train_type in ["GUTENBERG", "SPOOKY", "REUTERS"]
    if args.train_type == 'GUTENBERG':
        data = gutenberg_dataset(args.train_path, args.test_path, args=args)
    elif args.train_type == "SPOOKY":
        data = spooky_authorship_data(args=args)
    elif args.train_type == "REUTERS":
        data = create_reuters_data(args=args)

    if args.sentencewise:
        data = make_sentencewise_data(data)

    return data
"""
\dhruv{
We compared these results with those we get by replacing the LSTM cells with GRU cells in this instead of LSTM recurrent cells. It seems that the results do not differ too much, but the LSTM model does better across datasets. The GRU model achieves 76.6\% accuracy on the Spooky dataset, 30.8\% on the Gutenberg dataset, and 94.6\% on the Reuters dataset.
}

"""

if __name__ == "__main__":
    args = arg_parse()
    print(args)

    if args.model == 'BASELINE':
        # Get books from train path and call baselineb model train function
        train_data, test_data, authors = get_data(args)

        print("training baseline model")
        baseline_model = train_baseline(train_data)
        print("testing baseline")
        baseline_model.evaluate(test_data, args)

    elif args.model == 'LSTM':
        data = get_data(args)
        train_data, test_data, authors = data
        flattened_test_data = list(itertools.chain(*test_data)) if args.sentencewise else test_data


        word_indexer = Indexer()
        add_dataset_features(train_data, word_indexer)
        add_dataset_features(flattened_test_data, word_indexer)

        if args.train_options == 'POS':
            pretrained = False
            word_indexer.get_index(PAD_SYMBOL)
            word_indexer.get_index(UNK_SYMBOL)
            word_vectors = WordEmbeddings(word_indexer, None)
        else:
            pretrained = True
            relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
            word_vectors = read_word_embeddings(args.word_vecs_path)

        print("Finished extracting embeddings")
        print("training")

        trained_model: AuthorshipModel = train_lstm_model(train_data, flattened_test_data, authors, word_vectors, args,
                                                          pretrained=pretrained)
        trained_model.evaluate(test_data, args)

    elif args.model == "LSTM_ATTN":
        data = get_data(args)
        train_data, test_data, authors = data

        flattened_test_data = list(itertools.chain(*test_data)) if args.sentencewise else test_data

        word_indexer = Indexer()
        add_dataset_features(train_data, word_indexer)
        add_dataset_features(flattened_test_data, word_indexer)

        if args.train_options == 'POS':
            pretrained = False
            word_indexer.get_index(PAD_SYMBOL)
            word_indexer.get_index(UNK_SYMBOL)
            word_vectors = WordEmbeddings(word_indexer, None)
        else:
            pretrained = True
            relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
            word_vectors = read_word_embeddings(args.word_vecs_path)

        print("Finished extracting embeddings")
        print("training")

        trained_model = train_lstm_attention_model(train_data, flattened_test_data, authors, word_vectors, args, pretrained=pretrained)

        print("testing")
        trained_model.evaluate(test_data, args)

    elif args.model == "GRU_ATTN":
        data = get_data(args)
        train_data, test_data, authors = data

        flattened_test_data = list(itertools.chain(*test_data)) if args.sentencewise else test_data

        word_indexer = Indexer()
        add_dataset_features(train_data, word_indexer)
        add_dataset_features(flattened_test_data, word_indexer)

        if args.train_options == 'POS':
            pretrained = False
            word_indexer.get_index(PAD_SYMBOL)
            word_indexer.get_index(UNK_SYMBOL)
            word_vectors = WordEmbeddings(word_indexer, None)
        else:
            pretrained = True
            relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
            word_vectors = read_word_embeddings(args.word_vecs_path)

        print("Finished extracting embeddings")
        print("training")

        trained_model = train_gru_attention_model(train_data, flattened_test_data, authors, word_vectors, args, pretrained=pretrained)

        print("testing")
        trained_model.evaluate(test_data, args)

    elif args.model == "DU_ATTN":
        data = get_data(args)
        train_data, test_data, authors = data

        flattened_test_data = list(itertools.chain(*test_data)) if args.sentencewise else test_data

        word_indexer = Indexer()
        add_dataset_features(train_data, word_indexer)
        add_dataset_features(flattened_test_data, word_indexer)

        if args.train_options == 'POS':
            print("USING POS EMBEDDINGS")
            pretrained = False
            word_indexer.get_index(PAD_SYMBOL)
            word_indexer.get_index(UNK_SYMBOL)
            word_vectors = WordEmbeddings(word_indexer, None)
        else:
            pretrained = True
            relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
            word_vectors = read_word_embeddings(args.word_vecs_path)

        print("Finished extracting embeddings")
        print("training")

        trained_model = train_du_attention_model(train_data, flattened_test_data, authors, word_vectors, args, pretrained=pretrained)

        print("testing")
        trained_model.evaluate(test_data, args)

    elif args.model == "DU_ATTN_LSTM":
        data = get_data(args)
        train_data, test_data, authors = data

        flattened_test_data = list(itertools.chain(*test_data)) if args.sentencewise else test_data

        word_indexer = Indexer()
        add_dataset_features(train_data, word_indexer)
        add_dataset_features(flattened_test_data, word_indexer)

        if args.train_options == 'POS':
            print("USING POS EMBEDDINGS")
            pretrained = False
            word_indexer.get_index(PAD_SYMBOL)
            word_indexer.get_index(UNK_SYMBOL)
            word_vectors = WordEmbeddings(word_indexer, None)
        else:
            pretrained = True
            relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
            word_vectors = read_word_embeddings(args.word_vecs_path)

        print("Finished extracting embeddings")
        print("training")

        trained_model = train_du_attention_model_lstm(train_data, flattened_test_data, authors, word_vectors, args, pretrained=pretrained)

        print("testing")
        trained_model.evaluate(test_data, args)

    elif args.model == "KERAS":
        train_data, test_data, authors = get_data(args)
        embeddings = read_word_embeddings(args.word_vecs_path)
        k.train_keras_model(embeddings, train_data, test_data, authors)

    elif args.model == "SKLEARN":
        train_data, test_data, authors = data = get_data(args)
        sklearn_train(train_data, test_data, authors, args)

    elif args.model == 'VAE':
        train_data, test_data, authors = spooky_authorship_data(args)
        word_indexer = Indexer()
        add_dataset_features(train_data, word_indexer)
        add_dataset_features(test_data, word_indexer)

        pretrained = True
        relativize(args.word_vecs_path_input, args.word_vecs_path, word_indexer)
        word_vectors = read_word_embeddings(args.word_vecs_path)

        print("Finished extracting embeddings")
        print("training")
        trained_model = train_vae(train_data, test_data, authors, word_vectors, args, pretrained=pretrained)

        print("testing")
        trained_model.evaluate(test_data, args)



    else:
        raise Exception("Please select appropriate model")
