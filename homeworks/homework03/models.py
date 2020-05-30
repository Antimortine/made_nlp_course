import torch
import torch.nn as nn
from torchnlp.nn.attention import Attention

import random


class GruEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, bidirectional, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.n_directions = 2 if bidirectional else 1
        
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):        
        # src = [src sent len, batch size]        
        embedded = self.embedding(src) # [src sent len, batch size, emb dim]
        embedded = self.dropout(embedded)
        
        output, hidden = self.rnn(embedded)
        
        # output = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        
        return output, hidden


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
        
    def forward(self, input, hidden, context):        
        # input = [batch size]
        # hidden = [n layers, batch size, hid dim]
        # context = [batch size, src sent len, n layers * hid dim]
        
        batch_size = input.shape[0]        
        input = input.unsqueeze(0) # [1, batch size]
        
        embedded = self.dropout(self.embedding(input)) # [1, batch size, emb dim]
        
        output, hidden = self.rnn(embedded, hidden)
        # output = [sent len, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        
        # [batch size, output length = 1, dimensions = n layers * hid dim]
        query = hidden.permute(1, 0, 2).reshape(batch_size, 1, self.n_layers * self.hid_dim)
        
        # [batch size, output length = 1, dimensions = n layers * hid dim]
        attention_output, _ = self.attention(query, context)
        
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
        
        self.state_mapper = nn.Linear(
            in_features=encoder.n_layers * encoder.n_directions * encoder.hid_dim,
            out_features=decoder.n_layers * decoder.hid_dim
        )
        
        self.context_mapper = nn.Linear(
            in_features=encoder.n_directions * encoder.hid_dim,
            out_features=decoder.n_layers * decoder.hid_dim
        )
    
    def apply_encoder(self, src):
        # src = [src sent len, batch size]
        
        src_len, batch_size = src.shape
        
        # output = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        output, hidden = self.encoder(src)
        
        context = output.view(src_len * batch_size, self.encoder.hid_dim * self.encoder.n_directions)
        context = self.context_mapper(context) # [src sent len * batch size, dec n layers * dec hid dim]
        context = (context
                   .reshape(src_len, batch_size, self.decoder.n_layers * self.decoder.hid_dim)
                   .permute(1 ,0, 2)
                   .contiguous()) # [batch size, src sent len, dec n layers * dec hid dim]
        
        # [batch size, n layers * n directions * hid dim]
        hidden = hidden.permute(1, 0, 2).reshape(batch_size,
                                                 self.encoder.n_layers * self.encoder.n_directions * self.encoder.hid_dim)
        hidden = self.state_mapper(hidden) # [batch size, dec n layers * dec hid dim]
        
        # [dec n layers, batch size, dec hid dim]
        hidden = hidden.reshape(batch_size, self.decoder.n_layers, self.decoder.hid_dim).permute(1, 0, 2).contiguous()
        
        # context = [batch size, src sent len, dec n layers * dec hid dim]
        # hidden = [dec n layers, batch size, dec hid dim]
        return context, hidden
    
    def forward(self, src, trg, teacher_forcing_ratio=0.5):        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        # context = [batch size, src sent len, dec n layers * dec hid dim]
        # hidden = [dec n layers, batch size, dec hid dim]
        context, hidden = self.apply_encoder(src)
        decoder_outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            # output = [batch size, output dim]
            # hidden = [n layers, batch size, hid dim]
            output, hidden = self.decoder(input, hidden, context)
            decoder_outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return decoder_outputs
