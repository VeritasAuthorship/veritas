# veritas
Authorship Recognition using Attention-based Recurrent Neural Nets. The paper which details results as well as techniques can be found here: https://apps.cs.utexas.edu/apps/tech-reports/171205

Our Encoder-Decoder model with Attention is able to get **96%** accuracy
Requirements:
```
Python3
Pytorch
nltk
numpy
keras (optional, remove keras_lstm.py file if not needed)
sklearn (optional, remove sklearn_baselines.py if not needed)
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
--test_path, type=str, default='data/gut-test/british', help='Path to the test set'
 --train_options', type=str, default='', help="Extra train options, eg pos tags embeddings (POS)"
 --sentencewise', type=bool, default=False, help="Train on datasets sentence-wise or passage-wise"

--reverse_input, type=bool, default=False
--batch_size, type=int, default=1, help="batch size for encoder training"
--emb_dropout, type=float, default=0.2, help="dropout for embedding layer"
--rnn_dropout, type=float, default=0.2, help="dropout for RNN"
--bidirectional, type=bool, default=True, help="lstm birectional or not"
--hidden_size, type=int, default=200, help='hidden state dimensionality'
--word_vecs_path_input, type=str, default='data/glove.6B.300d.txt', help='path to word vectors file'
--word_vecs_path, type=str, default='data/glove.6B.300d-relativized.txt', help='path to word vectors file'
--embedding_size, type=int, default=300, help='Embedding size'
--epochs, type=int, default=8, help='Number of epochs'
--lr, type=float, default=1e-4 * 5, help='Learning rate'
--z_dim', type=int, default=50, help='Size of latent representation; Only useful for VAE model'
