import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from models.attention import RawEmbeddingLayer, PretrainedEmbeddingLayer
from nltk import word_tokenize
from utils import *

# Run on gpu is present
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class VAEncoder(nn.Module):

    # Input_size = embedding size, z_dim is the z space size, hidden_dim = number of hidden memory units
    def __init__(self, input_size, z_dim, hidden_size, dropout, bidirect=True):
        super(VAEncoder, self).__init__()
        # setup the three linear transformations used
        self.input_size = input_size
        self.bidirect = bidirect
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(self.input_size, hidden_size, num_layers=2, batch_first=True,
                           dropout=dropout, bidirectional=self.bidirect)
        self.fc21 = nn.Linear(hidden_size, z_dim)
        self.fc22 = nn.Linear(hidden_size, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

        self.init_weight()

    def init_weight(self):
        nn.init.xavier_normal_(self.fc22.weight)
        nn.init.xavier_normal_(self.fc21.weight)
        nn.init.xavier_normal_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_normal_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    # Input is the embeddings for a sentence
    def forward(self, x):
        # Encode the embeddings using an LSTM 
        output, hn = self.rnn(x)
        
        h, c = hn[0], hn[1]
        # Grab the representations from forward and backward LSTMs
        h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
        # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
        # as the hidden size in the encoder
        new_h = self.reduce_h_W(h_)
        new_c = self.reduce_c_W(c_)
        h_t = (new_h, new_c)
    
        hidden = self.softplus(h_t[0])

        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale

# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class VAEDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, output_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = torch.sigmoid(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = torch.sigmoid(self.fc21(hidden))
        return loc_img


# define the PyTorch module that takes in p(z) and return probs
class Decoder(nn.Module):
    
    def __init__(self, z_dim, hidden_size, output_size, dropout, bidirect=True):
        super(Decoder, self).__init__()

        self.bidirect = bidirect
        self.input_size = z_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(self.input_size, hidden_size, num_layers=1, batch_first=True,
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

    # Takes in a sample in latent space and based on that assigns author probabilities
    def forward(self, z):
        # Run z through LSTM decoder and then run through softmax to get probabilities
        output, hn = self.rnn(z)
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
        labels = self.hiddenToLabel(h_t[0])
        probs = F.log_softmax(labels[0], dim=0)
        return (probs, h_t)


# define a PyTorch module for the VAE
class VAE(nn.Module):

    def __init__(self,input_size, z_dim, hidden_size, dropout):
        super(VAE, self).__init__()
        self.decoder = VAEDecoder(z_dim, hidden_size,3)
        
        self.input_size = input_size
        self.z_dim = z_dim
        self.hidden_size = hidden_size
        # create the encoder and decoder networks
        self.encoder = VAEncoder(input_size, z_dim, hidden_size, dropout)

    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))


    # NOT USED FOR CLASSIFICATION MODEL
    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = x.new_zeros(torch.Size((x.shape[0], self.z_dim)))
            z_scale = x.new_ones(torch.Size((x.shape[0], self.z_dim)))
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images
            #pyro.sample("obs", dist.Bernoulli(loc_img).to_event(1), obs=x.reshape(-1, 784))
            # return the loc so we can visualize it later
            return loc_img
            #return 0

    # Retrieve the forward distibution for the incoming sentence.
    def forward(self, x):
        # encode the sentence
        z_loc, z_scale = self.encoder.forward(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        return z

class VAERNNTrainedModel(AuthorshipModel):
    def __init__(self, vae, input_emb, decoder, input_indexer, output_indexer, args, max_len):
        # Add any args you need here
        self.vae_encoder = vae
        self.decoder = decoder
        self.input_emb = input_emb
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.args = args
        self.max_len = max_len

    def _predictions(self, test_data, args):
        test_data.sort(key=lambda ex: len(word_tokenize(ex.passage)))
        all_test_input_data = torch.LongTensor(make_padded_input_tensor(test_data, self.input_indexer, self.max_len))
        all_test_output_data = torch.LongTensor(np.asarray([self.output_indexer.index_of(ex.author) for ex in test_data]))

        correct = 0
        total = len(all_test_input_data)
        predictions = []
        for idx, X_batch in enumerate(all_test_input_data):
            X_batch = X_batch.to(device)
            y_batch = all_test_output_data[idx].unsqueeze(0).to(device)

            embedded_words = self.input_emb.forward(X_batch.unsqueeze(0).to(device)).to(device)

            z = self.vae_encoder.forward(embedded_words).unsqueeze(0).to(device)
            # Get probability and hidden state
            probs, hidden = self.decoder.forward(z)

            predictions.append(self.output_indexer.get_object(torch.argmax(probs).item()))

        return predictions

    def myevaluate(self, test_data, args):
        test_data.sort(key=lambda ex: len(word_tokenize(ex.passage)))
        all_test_input_data = torch.LongTensor(make_padded_input_tensor(test_data, self.input_indexer, self.max_len))
        all_test_output_data = torch.LongTensor(np.asarray([self.output_indexer.index_of(ex.author) for ex in test_data]))

        correct = 0
        total = len(all_test_input_data)
        for idx, X_batch in enumerate(all_test_input_data):
            X_batch = X_batch.to(device)
            y_batch = all_test_output_data[idx].unsqueeze(0).to(device)

            embedded_words = self.input_emb.forward(X_batch.unsqueeze(0).to(device)).to(device)
            
            z = self.vae_encoder.forward(embedded_words).unsqueeze(0).to(device)
            # Get probability and hidden state
            probs, hidden = self.decoder.forward(z)
               
            print(probs, max(probs))
            if torch.argmax(probs).item() == y_batch[0].item():
                correct += 1

        print("Correctness", str(correct) + "/" + str(total) + ": " + str(round(correct / total, 5)))



def train_vae(train_data, test_data, authors, word_vectors, args, pretrained=True):
    # clear param store
    pyro.clear_param_store()

    word_indexer = word_vectors.word_indexer
    input_size = args.embedding_size
    input_lens = torch.LongTensor(np.asarray([len(word_tokenize(ex.passage)) for ex in train_data]))
    test_input_lens = torch.LongTensor(np.asarray([len(word_tokenize(ex.passage)) for ex in test_data])).to(device)
    train_max_len = torch.max(input_lens, dim=0)[0].item()
    test_max_len = torch.max(test_input_lens, dim=0)[0].item()
    input_max_len = max(train_max_len, test_max_len)

    all_train_input_data = torch.LongTensor(make_padded_input_tensor(train_data, word_indexer, input_max_len))
    all_train_output_data = torch.LongTensor(np.asarray([authors.index_of(ex.author) for ex in train_data]))

    # if pretrained, use Glove embeddings, else use a raw embedding layer
    if pretrained:
        model_emb = PretrainedEmbeddingLayer(word_vectors, args.emb_dropout).to(device)
    else:
        model_emb = RawEmbeddingLayer(args.embedding_size, len(word_indexer), args.emb_dropout).to(device)
    # Set up variational auto encoder
    vae = VAE(input_size, args.z_dim, args.hidden_size, args.rnn_dropout).to(device)
    decoder = Decoder(args.z_dim, args.hidden_size, len(authors), args.rnn_dropout).to(device)

    vae.train()
    decoder.train()

    # setup the vae optimizer
    adam_args = {"lr": args.lr}
    optimizer = Adam(adam_args)

    # Setup the decoder optimizer
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    loss_function = nn.NLLLoss()
    num_epochs = args.epochs

    # setup the inference algorithm
    #elbo = JitTrace_ELBO() if args.jit else Trace_ELBO()
    elbo = Trace_ELBO()
    svi = SVI(vae.guide, vae.model, optimizer, loss=elbo)

    train_elbo = []
    # training loop. We train VAE and decoder separately
    for epoch in range(num_epochs):
        # initialize loss accumulator
        epoch_loss = 0

        for idx, X_batch in enumerate(all_train_input_data):
            if idx % 100 == 0:
                print("Example", idx, "out of", len(all_train_input_data))
            # Get word embeddings
            embedded_words = model_emb.forward(X_batch.unsqueeze(0).to(device)).to(device)

            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(embedded_words)

        # report training diagnostics
        train_elbo.append(epoch_loss)
        print("[epoch %03d]  average VAE training loss: %.4f" % (epoch, epoch_loss))

        for idx, X_batch in enumerate(all_train_input_data):
            if idx % 100 == 0:
                print("Example", idx, "out of", len(all_train_input_data))
            y_batch = all_train_output_data[idx].unsqueeze(0).to(device)
            # Get word embeddings
            embedded_words = model_emb.forward(X_batch.unsqueeze(0).to(device)).to(device)

            z = vae.forward(embedded_words).unsqueeze(0).to(device)

            # Initialize optimizer
            dec_optimizer.zero_grad()

            # Get probability and hidden state
            probs, hidden = decoder.forward(z)
            #print(probs)
            #print("Predicted", torch.argmax(probs,0), "|| Actual" ,y_batch)
            loss = loss_function(probs.unsqueeze(0).to(device), y_batch)
            epoch_loss += loss

            # Run backward
            loss.backward()
            dec_optimizer.step()
        
        print("Epoch " + str(epoch) + " Loss for Decoder:", epoch_loss)

    return VAERNNTrainedModel(vae,model_emb, decoder, word_indexer, authors, args, input_max_len)
