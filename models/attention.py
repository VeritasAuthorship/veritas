import torch
import numpy as np
import torch.nn as nn
import torch.functional as F
from torch.optim import Adam

from models.LSTM import RNNEncoder, EmbeddingLayer, LSTMTrainedModel




class AttentionRNNDecoder(nn.Module):

    def __init__(self, hidden_size, embedding_dim, output_size, max_length, args):
        super(AttentionRNNDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.output_size = output_size

        # Bahdanau Attention Mechanism
        self.attention = nn.Linear(self.hidden_size + self.embedding_dim, max_length)
        self.attention_combine = nn.Linear((2 if args.bidirectional else 1) * self.hidden_size + self.embedding_dim,
                                           hidden_size)

        # Neural Model
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, start_token, hidden, enc_outputs):
        seq_len = enc_outputs.shape[0]
        concat = torch.cat((start_token[0], hidden[0][0]), 1)
        attention_weights = F.softmax(self.attention(concat)[:, :seq_len], dim=1)

        attention_applied = torch.bmm(attention_weights.unsqueeze(0), enc_outputs.squeeze(1).unsqueeze(0))

        output = torch.cat((start_token[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)
        output = F.tanh(output)

        output, hidden = self.lstm(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden



def _run_encoder(input_tensor, model_input_emb)


def example(input_tensor, output_tensor, input_lens_tensor,
            encoder, decoder, model_input_emb, model_output_emb,
            optimizers, loss_function, input_indexer, output_indexer):

    # Run encoder

    # Run decoder, get loss

    # loss backward

    # optimizer step


def train_lstm_model(train_data, test_data, authors, word_vectors, args):
    train_data.sort(key=lambda ex: len(word_tokenize(ex.passage)), reverse=True)
    word_indexer = word_vectors.word_indexer

    # Create indexed input
    print("creating indexed input")
    input_lens = torch.LongTensor(np.asarray([len(word_tokenize(ex.passage)) for ex in train_data]))
    input_max_len = torch.max(input_lens, dim=0)[0].item()
    # input_max_len = np.max(np.asarray([len(word_tokenize(ex.passage)) for ex in train_data]))
    print("train input")
    all_train_input_data = torch.LongTensor(make_padded_input_tensor(train_data, word_indexer, input_max_len))
    print("train output")
    all_train_output_data = torch.LongTensor(np.asarray([authors.index_of(ex.author) for ex in train_data]))

    # DataLoader constructs each batch from the given data
    '''
    train_batch_loader = DataLoader(
        TensorDataset(
            all_train_input_data,
            all_train_output_data,
            input_lens
        ), batch_size=args.batch_size, shuffle=True
    )
    '''
    input_size = args.embedding_size
    output_size = len(authors)

    model_emb = EmbeddingLayer(word_vectors, args.emb_dropout).to(device)
    encoder = RNNEncoder(input_size, args.hidden_size, output_size, word_vectors, args.rnn_dropout).to(device)

    # Construct optimizer. Using Adam optimizer
    params = list(encoder.parameters()) + list(model_emb.parameters())
    lr = 1e-3
    optimizer = Adam(params, lr=lr)

    loss_function = nn.NLLLoss()
    num_epochs = 10

    for epoch in range(num_epochs):

        epoch_loss = 0

        #for X_batch, y_batch, input_lens_batch in train_batch_loader:
        for idx, X_batch in enumerate(all_train_input_data):
            if idx % 100 == 0:
                print("Example", idx, "out of", len(all_train_input_data))
            y_batch = all_train_output_data[idx].unsqueeze(0)
            input_lens_batch = input_lens[idx].unsqueeze(0).to(device)

            # Initialize optimizer
            optimizer.zero_grad()

            # Get word embeddings
            embedded_words = model_emb.forward(X_batch.unsqueeze(0).to(device)).to(device)

            # Get probability and hidden state
            probs, hidden = encoder.forward(embedded_words, input_lens_batch)
            #print(probs)
            #print("Predicted", torch.argmax(probs,0), "|| Actual" ,y_batch)
            loss = loss_function(probs.unsqueeze(0), y_batch)
            epoch_loss += loss

            # Run backward
            loss.backward()
            optimizer.step()

        print("Epoch Loss:", epoch_loss)

    return LSTMTrainedModel(encoder, model_emb, word_indexer, authors, args)