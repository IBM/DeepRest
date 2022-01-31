from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import torch


class ResourceAware(nn.Module):
    def __init__(self, split, offset, input_size, output_size, hidden_layer_size=128):
        super().__init__()
        self.offset = offset
        self.output_size = output_size
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

    def fit_and_estimate(self, X, y):
        X = np.reshape(X, newshape=(X.shape[0], -1, X.shape[1] // self.output_size))
        X, scale_min_X, scale_max_X = ResourceAware.normalization_minmax(X, split=self.split)
        y, scale_min_y, scale_max_y = ResourceAware.normalization_minmax(y, split=self.split)
        scale_range_y = scale_max_y - scale_min_y

        X_, y_ = [], []
        for i in range(self.offset, len(X)):
            X_.append(y[i - self.offset, :, 0])
            y_.append(y[i, :, 0])
        X = np.asarray(X_)
        y = np.asarray(y_)

        split = self.split - self.offset
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]

        train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        loss_func = nn.MSELoss()

        for _ in range(100):
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
            outputs_.append(np.maximum(outputs, 1e-6))
        outputs = np.asarray(outputs_)[:, :, None]
        return outputs


class ComponentAware(object):
    def __init__(self, component, invocation, metric, output_size, split):
        self.output_size = output_size
        self.component = component
        self.metric = metric
        self.split = split
        self.invocation = invocation[component] if component in invocation else invocation['general']

    @staticmethod
    def baseline_scaling(x, w1, w2, w3, w4):
        return (x - w1) * w2 / w3 + w4 if sum(x) > 0 else x

    def fit_and_estimate(self, X, y):
        return self._fit_and_estimate(X, y, self.output_size)

    def _fit_and_estimate(self, X, y, output_steps):
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
