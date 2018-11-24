import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import *
import numpy as np


# Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
# Works for both non-batched and batched inputs
class EmbeddingLayer(nn.Module):
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, word_vectors, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding.from_pretrained(torch.from_numpy(word_vectors.vectors).float(), False)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input):
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


# One-layer RNN encoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class RNNEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, output_size, word_vectors, dropout, bidirect=False):
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                           dropout=dropout, bidirectional=self.bidirect)

        self.hiddenToLabel = nn.Linear(hidden_size, self.output_size)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)
        nn.init.xavier_uniform_(self.hiddenToLabel.weight)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray(
            [[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)

        print(h_t[0].shape)
        labels = self.hidden2Label(h_t[0])
        probs = F.log_softmax(labels[0], dim=0)
        return (probs, h_t)


class LSTMTrainedModel(object):
    def __init__(self, model, model_emb, indexer, args):
        # Add any args you need here
        self.model = model
        self.model_emb = model_emb
        self.indexer = indexer
        self.args = args


def train_lstm_model(train_data, test_data, authors, word_vectors, args):
    train_data.sort(key=lambda ex: len(word_tokenize(ex.passage)), reverse=True)
    word_indexer = word_vectors.word_indexer

    print(word_indexer.index_of(PAD_SYMBOL))

    # Create indexed input
    print("creating indexed input")
    input_lens = torch.LongTensor(np.asarray([len(word_tokenize(ex.passage)) for ex in train_data])).unsqueeze(0)
    input_max_len = torch.max(input_lens, dim=1)[0].item()
    # input_max_len = np.max(np.asarray([len(word_tokenize(ex.passage)) for ex in train_data]))
    print("train input")
    all_train_input_data = torch.LongTensor(make_padded_input_tensor(train_data, word_indexer, input_max_len)).unsqueeze(0)
    print("train output")
    all_train_output_data = torch.LongTensor(np.asarray([authors.index_of(ex.author) for ex in train_data])).unsqueeze(0)

    print(input_lens.shape)
    print(all_train_input_data.shape)
    print(all_train_output_data.shape)

    train_batch_loader = DataLoader(
        TensorDataset(
            all_train_input_data,
            all_train_output_data,
            input_lens
        ), batch_size=args.batch_size, shuffle=True
    )

    input_size = input_max_len
    hidden_size = 25
    output_size = len(authors)

    encoder = RNNEncoder(input_size, hidden_size, output_size, word_vectors, args.rnn_dropout)
    model_emb = EmbeddingLayer(word_vectors, args.emb_dropout)

    enc_optimizer = Adam(encoder.parameters())

    loss_function = nn.NLLLoss()
    num_epochs = 5

    for epoch in range(num_epochs):

        epoch_loss = 0

        for X_batch, y_batch, input_lens_batch in train_batch_loader:
            enc_optimizer.zero_grad()
            encoder.init_weight()

            print(X_batch.shape, y_batch.shape, input_lens_batch.shape)

            embedded_words = word_embeddings(X_batch)
            print("EMBEDDINGS")
            print(embedded_words.shape)
            probs, hidden = encoder.forward(embedded_words, input_lens_batch)
            print("PROBS")
            print(probs.shape)
            loss = loss_function(torch.unsqueeze(probs[-1], 0), y_batch)

        # for ex_idx in ex_indices:
        #     X = torch.from_numpy(all_train_input_data[ex_idx])
        #     y = torch.LongTensor([all_train_output_data[ex_idx]])
        #
        #     enc_optimizer.zero_grad()
        #     encoder.init_weight()
        #
        #     probs = encoder.forward(X, input_lens)
        #     category = torch.LongTensor([y])
        #
        #     loss = loss_function(torch.unsqueeze(probs[-1], 0), category)
        #
        #     loss.backward()
        #     epoch_loss += loss
        #     enc_optimizer.step()

    raise Exception('sdf')
