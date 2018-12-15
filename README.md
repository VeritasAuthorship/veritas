# veritas
Authorship Recognition using Attention-based Recurrent Neural Nets

Requirements:
```
Python3
Pytorch
nltk
numpy
```

To get all necessary files and datasets for the project, run 

```
get_set_go.sh
```
This file downloads 300d pretrained GloVe word embeddings, as well as the Reuters dataset. Furthermore, it calls the setup.py file which downloads further packages required for POS tag conversion.
Spooky Dataset may be downloaded from Kaggle (https://www.kaggle.com/c/spooky-author-identification/data)


The project is customizable and includes several initialization options, including baseline (ngrams model), LSTM (LSTM classifier), LSTM_ATTN (encoder decoder model with Bahdanau Attention Mechanism (https://arxiv.org/pdf/1409.0473.pdf), VAE (Variational AutoEncoder - RNN classifier), and others. 

Customizations possible:
```
--model, type=str, default='BASELINE', help="Model to run"
--train_type, type=str, default="GUTENBERG", help="Data type - Gutenberg or custom"
--train_path, type=str, default='data/american/', help='Path to the training set'
--reverse_input, type=bool, default=False
--batch_size, type=int, default=1, help="batch size for encoder training"
--emb_dropout, type=float, default=0.2, help="dropout for embedding layer"
--rnn_dropout, type=float, default=0.2, help="dropout for RNN"
--bidirectional, type=bool, default=True, help="lstm birectional or not"
--hidden_size, type=int, default=200, help='hidden state dimensionality'
--word_vecs_path_input, type=str, default='data/glove.6B.300d.txt', help='path to word vectors file'
--word_vecs_path, type=str, default='data/glove.6B.300d-relativized.txt', help='path to word vectors file'
--embedding_size, type=int, default=300, help='Embedding size'
