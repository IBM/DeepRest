import torch.nn as nn
import torch
import numpy as np


class QuantileRNN(nn.Module):
    def __init__(self, input_size, num_metrics, hidden_layer_size=128, num_layers=1, bidirectional=True,
                 quantiles=(.05, .50, .95), dropout=0.50):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.bidirectional = bidirectional
        self.quantiles = quantiles
        self.num_metrics = num_metrics
        self.dropout = nn.Dropout(p=dropout)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.mask_init = nn.Parameter(torch.full((1,), fill_value=1.), requires_grad=False)
        self.experts = nn.ModuleList([nn.ModuleList(
            [nn.Linear(1, hidden_layer_size),
             nn.Linear(hidden_layer_size, input_size),
             nn.GRU(input_size, hidden_layer_size, num_layers=num_layers, bidirectional=bidirectional),
             nn.Linear(hidden_layer_size * (2 if bidirectional else 1) * 2, len(quantiles))]
        ) for _ in range(num_metrics)])

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        predictions = []
        rnn_outs = []
        for mask_linear1, mask_linear2, rnn, _ in self.experts:
            mask = self.softmax(mask_linear2(self.relu(mask_linear1(self.mask_init))))

            _input_seq = input_seq * mask[None, None, :]
            _input_seq = _input_seq.permute(1, 0, 2)

            hidden_cell = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size,
                                      self.hidden_layer_size)
            rnn_out, hidden_cell = rnn(_input_seq, hidden_cell)
            rnn_out = rnn_out.permute(1, 0, 2)
            rnn_out = self.dropout(rnn_out)
            rnn_outs.append(rnn_out)

        for i in range(self.num_metrics):
            m = []
            for j in range(self.num_metrics):
                if i == j:
                    continue
                m.append(rnn_outs[j])
            m = torch.mean(torch.stack(m), dim=0)
            m = torch.cat([m, rnn_outs[i]], dim=-1)
            predictions.append(self.experts[i][-1](m))
        predictions = torch.stack(predictions).permute(1, 2, 0, 3)
        return predictions

    def quantile_loss(self, outputs, labels):
        losses = []
        for idx in range(self.num_metrics):
            losses_ = []
            for i, q in enumerate(self.quantiles):
                errors = labels[:, :, [idx]] - outputs[:, :, [idx], [i]]
                losses_.append(torch.max((q - 1) * errors, q * errors))
            losses.append(torch.mean(torch.sum(torch.cat(losses_, dim=2), dim=2)))
        loss = torch.mean(torch.stack(losses))
        return loss

    @staticmethod
    def normalization_minmax(M, split):
        min_val = np.min(M[:split])
        max_val = np.max(M[:split])
        if (max_val - min_val) != 0.0:
            M = (M - min_val) / (max_val - min_val)
        return M, min_val, max_val
