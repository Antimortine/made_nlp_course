import torch
import torch.nn as nn
from torchnlp.nn.attention import Attention

import random


class GruEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src, hidden=None):        
        # src = [src sent len, batch size]        
        embedded = self.embedding(src) # [src sent len, batch size, emb dim]
        embedded = self.dropout(embedded)
        
        if hidden is None:
            output, hidden = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded, hidden)
        
        # output = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        return hidden


class AttentionGruDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(
            num_embeddings=output_dim,
            embedding_dim=emb_dim
        )
        
        self.attention = Attention(dimensions=n_layers*hid_dim, attention_type='general')
        
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        self.out = nn.Linear(
            in_features=n_layers*hid_dim,
            out_features=output_dim
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, encoder_outputs):        
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # encoder_outputs = [batch size, src sent len, dimensions = n layers * hid dim]
        
        batch_size = input.shape[0]        
        input = input.unsqueeze(0) # [1, batch size]
        
        embedded = self.dropout(self.embedding(input)) # [1, batch size, emb dim]
        
        output, hidden = self.rnn(embedded, hidden)
        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        # [batch size, output length = 1, dimensions = n layers * n directions * hid dim]
        query = hidden.permute(1, 0, 2).reshape(batch_size, 1, self.n_layers * self.hid_dim)
        
        # [batch size, output length = 1, dimensions = n layers * hid dim]
        attention_output, _ = self.attention(query, encoder_outputs)
        
        # [batch size, dimensions = n layers * hid dim]
        attention_output = attention_output.squeeze(1)
        
        # [batch size, output dim]
        prediction = self.out(attention_output)
        
        return prediction, hidden


class AttentionGruSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
    
    def apply_encoder(self, src):
        # src = [src sent len, batch size]
        
        batch_size = src.shape[1]
        encoder_state_size = self.encoder.n_layers * self.encoder.hid_dim
        
        # [src sent len, batch size, dimensions = n layers * hid dim]
        encoder_states = torch.zeros(src.shape[0], batch_size, encoder_state_size).to(self.device)
        
        first_encoder_input = src[0].unsqueeze(0) # [1, batch size]
        hidden = self.encoder(first_encoder_input) # [n layers * n directions, batch size, hid dim]
        # [batch size, n layers * n directions * hid dim]
        encoder_states[0] = hidden.permute(1,0,2).reshape(batch_size, encoder_state_size)
        
        for t in range(1, src.shape[0]):
            hidden = self.encoder(src[t].unsqueeze(0), hidden)
            encoder_states[t] = hidden.permute(1,0,2).reshape(batch_size, encoder_state_size)
        
        # [batch size, src sent len, dimensions = n layers * hid dim]
        encoder_states = encoder_states.permute(1, 0, 2)
        return encoder_states, hidden
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # encoder_states = [batch size, src sent len, dimensions = n layers * hid dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        encoder_states, hidden = self.apply_encoder(src)
        decoder_outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            # output = [batch size, output dim]
            # hidden = [n layers * n directions, batch size, hid dim]
            output, hidden = self.decoder(input, hidden, encoder_states)
            decoder_outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return decoder_outputs
