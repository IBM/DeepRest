from torch.utils.data import TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
import json
import os
plt.style.use('ggplot')


class Baseline_Resource_Only(nn.Module):
    def __init__(self, offset_estimation, split, offset, input_size, output_size, hidden_layer_size=128):
        super().__init__()
        self.offset = offset
        self.output_size = output_size
        self.offset_estimation = offset_estimation
        self.split = split

        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.relu(out)
        out = self.linear2(out)
        return out

    @staticmethod
    def normalization_minmax(M, split):
        min_val = np.min(M[:split])
        max_val = np.max(M[:split])
        if (max_val - min_val) != 0.0:
            M = (M - min_val) / (max_val - min_val)
        return M, min_val, max_val

    def fit_and_estimate(self, X, y, verbose=1):
        if verbose != 0:
            print('Running baseline: resource-only...')
        X = np.reshape(X, newshape=(X.shape[0], -1, X.shape[1] // self.output_size))
        X, scale_min_X, scale_max_X = Baseline_Resource_Only.normalization_minmax(X, split=self.split)

        y_offset = np.zeros(shape=(y.shape[0], 1, 1))
        if self.offset_estimation:
            y_offset = y[:, [0], :]
            y = y - y_offset

        y, scale_min_y, scale_max_y = Baseline_Resource_Only.normalization_minmax(y, split=self.split)
        scale_range_y = scale_max_y - scale_min_y

        X_, y_, y_offset_ = [], [], []
        for i in range(self.offset, len(X)):
            X_.append(y[i - self.offset, :, 0])
            y_.append(y[i, :, 0])
            y_offset_.append(y_offset[i])
        X = np.asarray(X_)
        y = np.asarray(y_)
        y_offset = np.asarray(y_offset_)

        split = self.split - self.offset
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        y_test_offset = y_offset[split:]

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_func = nn.MSELoss()

        for _ in tqdm(range(100)):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs = self(inputs)
                loss = loss_func(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        inputs = torch.FloatTensor(X[[split-self.offset]]).to(self.device)
        outputs = self(inputs)
        outputs = outputs.detach().cpu().numpy().squeeze()
        outputs = outputs * scale_range_y + scale_min_y
        outputs_ = []
        for i in range(len(y_test)):
            if self.offset_estimation:
                outputs_.append(np.maximum(outputs + y_test_offset[i].squeeze(), 1e-6))
            else:
                outputs_.append(np.maximum(outputs, 1e-6))
        outputs = np.asarray(outputs_)[:, :, None]
        return outputs


class Baseline_RequestAndResource(object):
    def __init__(self, dataset, component, invocation, metric, offset_estimation, output_size, split):
        self.output_size = output_size
        self.dataset = dataset.replace('_%ds' % (output_size * 5), '')
        self.component = component
        self.metric = metric
        self.offset_estimation = offset_estimation
        self.split = split
        self.invocation = invocation[component] if component in invocation else invocation['general']


    @staticmethod
    def baseline_scaling(x, w1, w2, w3, w4):
        return (x - w1) * w2 / w3 + w4 if sum(x) > 0 else x

    def fit_and_estimate(self, X, y, verbose=1):
        if verbose != 0:
            print('Running baseline: request and resource...')
        if self.offset_estimation:
            return self._fit_and_estimate_offset(X, y, self.output_size)
        else:
            return self._fit_and_estimate_absolute(X, y, self.output_size)

    def _fit_and_estimate_offset(self, X, y, output_steps):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))
        ts_delta = [0] + [ts[i] - ts[i-1] for i in range(1, len(ts))]

        ratio = np.mean([ts_delta[i] / self.invocation[i] for i in range(len(ts_delta)) if self.invocation[i] != 0.0])

        invocation = np.asarray([self.invocation[i - output_steps:i] for i in range(output_steps, len(ts) + 1)])
        invocation_test = invocation[self.split:]
        ts_test = np.asarray([ts[i - output_steps:i] for i in range(output_steps, len(ts) + 1)])
        y_baseline = np.asarray(
            [ts_test[self.split + i - 1][0] + np.cumsum(invocation_test[i] * ratio) for i in range(len(invocation_test))]
        )[:, :, None]
        return y_baseline

    def _fit_and_estimate_absolute(self, X, y, output_steps):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))

        split = self.split + output_steps - 1

        inv_train = np.asarray(self.invocation[:split])
        metric_train = np.asarray(ts[:split])

        w1 = np.min(inv_train)
        w2 = (np.max(metric_train) - np.min(metric_train))
        w3 = (np.max(inv_train) - np.min(inv_train))
        w4 = np.min(metric_train)
        ts_hat = np.maximum(self.baseline_scaling(self.invocation, w1, w2, w3, w4), 1e-6)
        ts_hat = np.asarray([ts_hat[i - output_steps:i] for i in range(output_steps, len(ts) + 1)])
        y_baseline = ts_hat[self.split:][:, :, None]
        return y_baseline


class Baseline_RequestAndResource_RE(object):
    def __init__(self, component, invocation, offset_estimation, split, points_per_day):
        self.offset_estimation = offset_estimation
        self.points_per_day = points_per_day
        self.split = split
        self.invocation = invocation[component] if component in invocation else invocation['general']

    @staticmethod
    def baseline_scaling(x, w1, w2, w3, w4):
        return (x - w1) * w2 / w3 + w4 if sum(x) > 0 else x

    def fit_and_estimate(self, X, y):
        if self.offset_estimation:
            return self._fit_and_estimate_offset(X, y)
        else:
            return self._fit_and_estimate_absolute(X, y)

    def _fit_and_estimate_offset(self, X, y):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))
        ts_delta = [0] + [ts[i] - ts[i-1] for i in range(1, len(ts))]

        ratio = np.mean([ts_delta[i] / self.invocation[i] for i in range(self.split - int(4.5 * self.points_per_day), self.split) if self.invocation[i] != 0.0])

        out = []
        for start_idx in range(self.split, len(y), self.points_per_day):
            invocation = self.invocation[start_idx:start_idx + self.points_per_day]
            invocation_cumsum = np.cumsum(invocation) * ratio + ts[start_idx - 1]
            out += list(invocation_cumsum)
        return out

    def _fit_and_estimate_absolute(self, X, y):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))
        inv_train = np.asarray(self.invocation[self.split - int(4.5 * self.points_per_day):self.split])
        metric_train = np.asarray(ts[self.split - int(4.5 * self.points_per_day):self.split])
        w1 = np.min(inv_train)
        w2 = (np.max(metric_train) - np.min(metric_train))
        w3 = (np.max(inv_train) - np.min(inv_train))
        w4 = np.min(metric_train)
        ts_hat = np.maximum(self.baseline_scaling(self.invocation, w1, w2, w3, w4), 1e-6)
        return ts_hat[self.split:]


class Baseline_NaiveRequestAndResource_RE(object):
    def __init__(self, calls, metric, offset_estimation, split, points_per_day):
        self.metric = metric
        self.points_per_day = points_per_day
        self.offset_estimation = offset_estimation
        self.split = split
        self.calls = calls['all']

    @staticmethod
    def baseline_scaling(x, w1, w2, w3, w4):
        return (x - w1) * w2 / w3 + w4 if sum(x) > 0 else x

    def fit_and_estimate(self, X, y):
        if self.offset_estimation:
            return self._fit_and_estimate_offset(X, y)
        else:
            return self._fit_and_estimate_absolute(X, y)

    def _fit_and_estimate_offset(self, X, y):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))
        ts_delta = [0] + [ts[i] - ts[i-1] for i in range(1, len(ts))]
        ratio = np.mean([ts_delta[i] / self.calls[i] for i in range(self.split - int(4.5 * self.points_per_day), self.split) if self.calls[i] != 0.0])

        out = []
        for start_idx in range(self.split, len(y), self.points_per_day):
            invocation = self.calls[start_idx:start_idx+self.points_per_day]
            invocation_cumsum = np.cumsum(invocation) * ratio + ts[start_idx-1]
            out += list(invocation_cumsum)
        return out

    def _fit_and_estimate_absolute(self, X, y):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))

        inv_train = np.asarray(self.calls[self.split - int(4.5 * self.points_per_day):self.split])
        metric_train = np.asarray(ts[self.split - int(4.5 * self.points_per_day):self.split])

        w1 = np.min(inv_train)
        w2 = (np.max(metric_train) - np.min(metric_train))
        w3 = (np.max(inv_train) - np.min(inv_train))
        w4 = np.min(metric_train)
        ts_hat = np.maximum(self.baseline_scaling(self.calls, w1, w2, w3, w4), 1e-6)
        return ts_hat[self.split:]


class Baseline_Resource_Only_RE(nn.Module):
    def __init__(self, offset_estimation, split, offset, input_size, output_size, hidden_layer_size=128):
        super().__init__()
        self.offset = offset
        self.output_size = output_size
        self.offset_estimation = offset_estimation
        self.split = split

        self.linear1 = nn.Linear(input_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)
        self.relu = nn.ReLU()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, inputs):
        out = self.linear1(inputs)
        out = self.relu(out)
        out = self.linear2(out)
        return out

    @staticmethod
    def normalization_minmax(M, split):
        min_val = np.min(M[:split])
        max_val = np.max(M[:split])
        if (max_val - min_val) != 0.0:
            M = (M - min_val) / (max_val - min_val)
        return M, min_val, max_val

    def fit_and_estimate(self, X, y, num_test_cycles=9):
        X = np.reshape(X, newshape=(X.shape[0], -1, X.shape[1] // self.output_size))
        X, scale_min_X, scale_max_X = Baseline_Resource_Only.normalization_minmax(X, split=self.split)

        y_offset = np.zeros(shape=(y.shape[0], 1, 1))
        if self.offset_estimation:
            y_offset = y[:, [0], :]
            y = y - y_offset

        y, scale_min_y, scale_max_y = Baseline_Resource_Only.normalization_minmax(y, split=self.split)
        scale_range_y = scale_max_y - scale_min_y

        X_, y_, y_offset_ = [], [], []
        for i in range(self.offset, len(X)):
            X_.append(y[i - self.offset, :, 0])
            y_.append(y[i, :, 0])
            y_offset_.append(y_offset[i])
        X = np.asarray(X_)
        y = np.asarray(y_)
        y_offset = np.asarray(y_offset_)

        split = self.split - self.offset
        X_train, y_train = X[:split], y[:split]
        y_test_offset = y_offset[split:]

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_func = nn.MSELoss()

        for _ in range(50):
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                # Forward pass
                outputs = self(inputs)
                loss = loss_func(outputs, labels)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        inputs = torch.FloatTensor(X[[split-self.offset]]).to(self.device)
        outputs = self(inputs)
        outputs = outputs.detach().cpu().numpy().squeeze()
        outputs = outputs * scale_range_y + scale_min_y
        outputs = np.concatenate([np.maximum((outputs + y_test_offset[i*self.output_size].squeeze()) if self.offset_estimation else outputs, 1e-6) for i in range(num_test_cycles)])
        return outputs


class Baseline_RequestAndResource_RE_(object):
    def __init__(self, component, invocation, calls, offset_estimation, split, points_per_day):
        self.offset_estimation = offset_estimation
        self.points_per_day = points_per_day
        self.split = split
        self.invocation = invocation[component] if component in invocation else invocation['general']
        self.calls = calls['all']

    @staticmethod
    def baseline_scaling(x, w1, w2, w3, w4):
        return (x - w1) * w2 / w3 + w4 if sum(x) > 0 else x

    def fit_and_estimate(self, X, y):
        if self.offset_estimation:
            return self._fit_and_estimate_offset(X, y)
        else:
            return self._fit_and_estimate_absolute(X, y)

    def _fit_and_estimate_offset(self, X, y):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))
        ts_delta = [0] + [ts[i] - ts[i-1] for i in range(1, len(ts))]

        ratio = np.mean([ts_delta[i] / self.invocation[i] for i in range(self.split - 7 * self.points_per_day, self.split) if self.invocation[i] != 0.0])

        out = []
        for start_idx in range(self.split, len(y), self.points_per_day):
            invocation = self.invocation[start_idx:start_idx + self.points_per_day]
            invocation_cumsum = np.cumsum(invocation) * ratio + ts[start_idx - 1]
            out += list(invocation_cumsum)
        return out

    def _fit_and_estimate_absolute(self, X, y):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))

        peak_calls_ind = np.argmax(self.calls[:self.split])
        peak_calls = self.invocation[:self.split][peak_calls_ind]
        peak_util = ts[:self.split][peak_calls_ind]
        scaling_factors = self.invocation[self.split:] / peak_calls
        estimation = scaling_factors * peak_util
        return estimation


class Baseline_NaiveRequestAndResource_RE_(object):
    def __init__(self, calls, metric, offset_estimation, split, points_per_day):
        self.metric = metric
        self.points_per_day = points_per_day
        self.offset_estimation = offset_estimation
        self.split = split
        self.calls = calls['all']

    def fit_and_estimate(self, X, y):
        if self.offset_estimation:
            return self._fit_and_estimate_offset(X, y)
        else:
            return self._fit_and_estimate_absolute(X, y)

    def _fit_and_estimate_offset(self, X, y):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))
        ts_delta = [0] + [ts[i] - ts[i-1] for i in range(1, len(ts))]
        ratio = np.mean([ts_delta[i] / self.calls[i] for i in range(self.split - 7 * self.points_per_day, self.split) if self.calls[i] != 0.0])

        out = []
        for start_idx in range(self.split, len(y), self.points_per_day):
            invocation = self.calls[start_idx:start_idx+self.points_per_day]
            invocation_cumsum = np.cumsum(invocation) * ratio + ts[start_idx-1]
            out += list(invocation_cumsum)
        return out

    def _fit_and_estimate_absolute(self, X, y):
        ts = np.asarray([v[0] for v in y[:, :, 0][:-1]] + list(y[:, :, 0][-1]))

        peak_calls_ind = np.argmax(self.calls[:self.split])
        peak_calls = self.calls[:self.split][peak_calls_ind]
        peak_util = ts[:self.split][peak_calls_ind]
        scaling_factors = self.calls[self.split:] / peak_calls
        estimation = scaling_factors * peak_util
        return estimation