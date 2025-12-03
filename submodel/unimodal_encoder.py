import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.2,
                 bidirectional=False, batch_first=True, is_pad=True):
        super().__init__()

        self.is_pad = is_pad
        if num_layers == 1:
            dropout = 0.
        else:
            dropout = dropout
        self.bi_lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, x):
        if self.is_pad:
            length_list = torch.tensor([len(d) for d in x])
            data = pad_sequence(x, batch_first=True)
        else:
            x = pad_sequence(x, batch_first=True)
            length_list = [(d.sum(dim=-1)!=0).sum() for d in x]
            data = x
        data = pack_padded_sequence(data, length_list, batch_first=True, enforce_sorted=False)
        out, _ = self.bi_lstm(data)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out


class UnimodalEncoder(nn.Module):
    def __init__(self, dim_t, dim_a, dim_v, dim_m, lstm_dropout):
        super().__init__()
        self.bi_lstm_t = BiLSTM(input_size=dim_t, hidden_size=dim_m // 2, num_layers=1,
                                bidirectional=True, batch_first=True, is_pad=True, dropout=lstm_dropout)
        self.bi_lstm_a = BiLSTM(input_size=dim_a, hidden_size=dim_m // 2, num_layers=1,
                                bidirectional=True, batch_first=True, is_pad=True, dropout=lstm_dropout)
        self.bi_lstm_v = BiLSTM(input_size=dim_v, hidden_size=dim_m // 2, num_layers=1,
                                bidirectional=True, batch_first=True, is_pad=True, dropout=lstm_dropout)

    def forward(self, u_t, u_a, u_v):
        x_t = self.bi_lstm_t(u_t)
        x_a = self.bi_lstm_a(u_a)
        x_v = self.bi_lstm_v(u_v)
        return x_t, x_a, x_v