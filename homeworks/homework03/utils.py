import glob
import os
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def beam_search_for_sample(model, src, trg, eos, k=2):
    # src = [src sent len, batch size = 1]
    # trg = [trg sent len, batch size = 1]
    assert src.shape[1] == 1, "Batch size must be equal to 1"
    
    model.eval()
    max_len = trg.shape[0]
    trg_vocab_size = model.decoder.output_dim
    
    with torch.no_grad():
        # encoder_states = [batch size = 1, src sent len, dimensions = n layers * hid dim]
        # hidden = [n layers * n directions, batch size = 1, hid dim]
        encoder_states, hidden = model.apply_encoder(src)

        # first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        # output = [batch size = 1, output dim]
        # hidden = [n layers * n directions, batch size = 1, hid dim]
        output, hidden = model.decoder(input, hidden, encoder_states)
        output = log_softmax(output, dim=1) # output = [batch size = 1, output dim]
        top_pred = torch.topk(output, k, dim=1)
        
        encoder_states = torch.cat([encoder_states] * k, dim=0)
        
        # topk_log_probas = [batch size = 1, k]
        # top_k_outputs = [batch size = 1, k]
        topk_log_probas, top_k_outputs = top_pred.values, top_pred.indices
        
        tokens = torch.full((max_len, k*(k+1)), 0, dtype=torch.long, device=model.device)
        hidden = torch.cat([hidden] * k, dim=1) # [n layers * n directions, k, hid dim]
        scores = torch.zeros(k*(k+1), device=model.device)
              
        tokens[1,:k] = top_k_outputs[0]
        scores[:k] = topk_log_probas[0]
        
        for i in range(k):
            tokens[1, (i+1)*k:(i+2)*k] = tokens[1, i]

        for t in range(2, max_len):
            inputs = tokens[t-1,:k] # [k]
            
            # output = [k, output dim]
            # hidden = [n layers * n directions, k, hid dim]
            output, hidden = model.decoder(inputs, hidden, encoder_states)
            output = log_softmax(output, dim=1) # [k, output dim]
            top_pred = torch.topk(output, k, dim=1)
            
            # topk_log_probas = [batch size = k, top = k]
            # top_k_outputs = [batch size = k, top = k]
            topk_log_probas, top_k_outputs = top_pred.values, top_pred.indices
            tokens[t, k:] = top_k_outputs.view(-1)
            
            new_scores = (scores[:k].unsqueeze(1) * (t-1) + topk_log_probas) / t
            has_eos = (tokens[:t, k:] == eos).any(axis=0)
            scores[k:] = torch.where(has_eos, scores[k:], new_scores.view(-1))
            
            topk_indices = torch.topk(scores[k:], k).indices + k
            tokens[:,:k] = tokens[:,topk_indices]
            hidden = hidden.permute(1,0,2)
            hidden = hidden[(topk_indices - k) // k]
            hidden = hidden.permute(1,0,2).contiguous()
            scores[:k] = scores[topk_indices]

            for i in range(k):
                tokens[:, (i+1)*k:(i+2)*k] = tokens[:, i].unsqueeze(1)
        top1_index = scores[:k].max(0)[1]
        return tokens[:, top1_index]

    
def beam_search(model, src, trg, target_vocab, k=2):
    # src = [src sent len, batch size]
    # trg = [trg sent len, batch size]
    eos = target_vocab.stoi['<eos>']
    pred_tokens = [beam_search_for_sample(model, src[:,i,None], trg[:,i,None], eos, k).unsqueeze(1)
            for i in range(src.shape[1])]
    return torch.cat(pred_tokens, dim=1) # [trg sent len, batch size]


def generate_translation(src, trg, model, TRG_vocab, beam_widths=None):
    model.eval()
    beam_widths = beam_widths or []
    
    tabs = 28
    original = get_text(list(trg[:,0].cpu().numpy()), TRG_vocab)
    print('Original:\t{}'.format(' '.join(original)).expandtabs(tabs))
    
    with torch.no_grad():
        output = model(src, trg, 0) # turn off teacher forcing
        output = output.argmax(dim=-1).cpu().numpy()
        
        generated = get_text(list(output[1:, 0]), TRG_vocab)
        print('Generated (Greedy):\t{}'.format(' '.join(generated)).expandtabs(tabs))
        
        for beam_width in beam_widths:
            output = beam_search(model, src, trg, TRG_vocab, beam_width)
            generated = get_text(list(output[1:, 0]), TRG_vocab)
            print('Generated (BeamSearch@{}):\t{}'.format(beam_width, ' '.join(generated)).expandtabs(tabs))
    print()


def parse_tensorboard_logs(dir_path):
    rows = []
    event_paths = glob.glob(os.path.join(dir_path, "event*"))
    for event_path in event_paths:
        for e in tf.train.summary_iterator(event_path):
            for v in e.summary.value:
                row = {'epoch': e.step, 'metric': v.tag, 'value': v.simple_value}
                rows.append(row)
    return (pd.DataFrame.from_records(rows)
          .pivot(index='epoch', columns='metric', values='value')
          .sort_index())


def plot_metrics(logs, model_name, axes=None):
    if axes is None:
        fig, axes = plt.subplots(ncols=2, figsize=(15, 7))

    loss_columns = list(logs.filter(like='loss').columns)
    bleu_columns = list(logs.filter(like='BLEU').columns)

    for loss_column in loss_columns:
        axes[0].plot(logs.index, logs[loss_column], label=loss_column)
    axes[0].set_xlabel('Epoch', fontsize=16)
    axes[0].set_ylabel('Cross-entropy loss', fontsize=16)
    axes[0].set_title(f'Lossess for {model_name}', fontsize=16)
    axes[0].grid()
    axes[0].legend(fontsize=16)

    for bleu_column in bleu_columns:
        axes[1].plot(logs.index, logs[bleu_column], label=bleu_column)
    axes[1].set_xlabel('Epoch', fontsize=16)
    axes[1].set_ylabel('BLEU', fontsize=16)
    axes[1].set_title(f'BLEU for {model_name}', fontsize=16)
    axes[1].grid()
    axes[1].legend(fontsize=16)
    plt.tight_layout()


def _len_sort_key(x):
    return len(x.src)


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
