from torch.utils.data import TensorDataset
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import torch
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class QuantileRNN(nn.Module):
    window_seconds = 300
    sampling_interval = 5
    step_size = window_seconds // sampling_interval
    num_layers = 1
    hidden_layer_size = 128
    num_epochs = 50
    quantiles = (.05, .50, .95)
    batch_size = 32
    learning_rate = 0.001
    output_size = 3

    def __init__(self, widths, split, warmup, offset_estimation, dropout=0.50, bidirectional=True):
        super().__init__()
        self.split = split
        self.offset_estimation = offset_estimation
        self.widths = widths
        self.warmup = warmup
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(sum(widths), self.hidden_layer_size, num_layers=self.num_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Linear(self.hidden_layer_size * (2 if bidirectional else 1), self.output_size)

        self.dummy = nn.Parameter(torch.full((1,), fill_value=1.), requires_grad=False)
        self.cumsum_linear_1 = nn.Linear(1, self.hidden_layer_size)
        self.cumsum_linear_2 = nn.Linear(self.hidden_layer_size, 2)

        self.prob_linears = nn.ModuleList([nn.ModuleList([nn.Linear(1, self.hidden_layer_size),
                                                          nn.Linear(self.hidden_layer_size, width)]) for width in widths])

        self.optimizer = torch.optim.Adam(self.parameters(), lr=QuantileRNN.learning_rate)
        self.to(device)

    def forward(self, input_seq):
        batch_size = input_seq.shape[0]

        prob_cumsum = self.softmax(self.cumsum_linear_2(self.relu(self.cumsum_linear_1(self.dummy))))
        prob_cumsum = torch.cat([prob_cumsum[[0]].repeat(self.widths[0] + self.widths[1]),
                                 prob_cumsum[[1]].repeat(self.widths[0] + self.widths[1])])

        prob = torch.cat([self.softmax(l2(self.relu(l1(self.dummy)))) for l1, l2 in self.prob_linears])

        input_seq = input_seq * prob[None, None, :] * prob_cumsum[None, None, :]
        input_seq = input_seq.permute(1, 0, 2)

        hidden_cell = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), batch_size,
                                  self.hidden_layer_size).to(device)
        rnn_out, hidden_cell = self.rnn(input_seq, hidden_cell)
        rnn_out = rnn_out.permute(1, 0, 2)
        rnn_out = self.dropout(rnn_out)

        predictions = self.linear(rnn_out)
        return predictions

    @staticmethod
    def quantile_loss(outputs, labels):
        losses = []
        for i, q in enumerate(QuantileRNN.quantiles):
            errors = labels - outputs[:, :, [i]]
            losses.append(torch.max((q - 1) * errors, q * errors))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=2), dim=2))
        return loss

    @staticmethod
    def normalization_minmax(M, split):
        min_val = np.min(M[:split])
        max_val = np.max(M[:split])
        if (max_val - min_val) != 0.0:
            M = (M - min_val) / (max_val - min_val)
        return M, min_val, max_val

    def fit(self, X, y, X_prefix, num_epochs=None, pretrained_path=None):
        if num_epochs is None:
            num_epochs = self.num_epochs
        X_cumsum = np.cumsum(X, axis=1)
        X_prefix_cumsum = np.cumsum(X_prefix, axis=1)
        X, _, _ = QuantileRNN.normalization_minmax(X, split=self.split)
        X_cumsum, _, _ = QuantileRNN.normalization_minmax(X_cumsum, split=self.split)
        X_prefix, _, _ = QuantileRNN.normalization_minmax(X_prefix, split=self.split)
        X_prefix_cumsum, _, _ = QuantileRNN.normalization_minmax(X_prefix_cumsum, split=self.split)
        Xs = [X, X_prefix, X_cumsum, X_prefix_cumsum]
        X = np.concatenate(Xs, axis=2)
        y_offset = np.zeros(shape=(y.shape[0], 1, 1))
        if self.offset_estimation:
            y_offset = y[:, [0], :]
            y = y - y_offset
        y, scale_min_y, scale_max_y = QuantileRNN.normalization_minmax(y, split=self.split)
        scale_range_y = scale_max_y - scale_min_y
        X_train, y_train = X[:self.split], y[:self.split]
        if self.warmup:
            X_train = X_train[len(X_train)//2:]
            y_train = y_train[len(y_train)//2:]
        X_test, y_test = X[self.split:], y[self.split:]
        y_test_offset = y_offset[self.split:]

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=QuantileRNN.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

        if os.path.exists(pretrained_path):
            self.load_state_dict(torch.load(pretrained_path))
            self.eval()
            outs = []
            with torch.no_grad():
                num_cycles = 0
                for iv, (inputs, labels) in enumerate(test_loader):
                    if iv % QuantileRNN.step_size != 0 or num_cycles >= 9:
                        continue
                    num_cycles += 1
                    inputs = inputs.to(device)
                    outputs = self(inputs)
                    outputs = outputs.detach().cpu().numpy().squeeze()
                    outputs = outputs * scale_range_y + scale_min_y
                    if self.offset_estimation:
                        outputs = outputs + y_test_offset[iv].squeeze()
                    outputs = np.maximum(outputs, 1e-6)
                    outs.append(outputs)
            outs = np.concatenate(outs, axis=0)
        else:
            best_model = None
            for _ in tqdm(range(num_epochs)):
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # Forward pass
                    outputs = self(inputs)
                    loss = QuantileRNN.quantile_loss(outputs, labels)
                    # Backward and optimize
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.eval()
                with torch.no_grad():
                    loss_buffer = []
                    num_cycles = 0
                    outs = []
                    for iv, (inputs, labels) in enumerate(test_loader):
                        if iv % QuantileRNN.step_size != 0 or num_cycles >= 9:
                            continue
                        num_cycles += 1
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = self(inputs)
                        loss = QuantileRNN.quantile_loss(outputs, labels)
                        loss_buffer.append(loss.item())
                        outputs = outputs.detach().cpu().numpy().squeeze()
                        outputs = outputs * scale_range_y + scale_min_y
                        if self.offset_estimation:
                            outputs = outputs + y_test_offset[iv].squeeze()
                        outputs = np.maximum(outputs, 1e-6)
                        outs.append(outputs)
                    outs = np.concatenate(outs, axis=0)
                    if best_model is None or np.mean(loss_buffer) < best_model[0]:
                        best_model = (np.mean(loss_buffer), outs)
                self.train()
            outs = best_model[1]
        return outs
