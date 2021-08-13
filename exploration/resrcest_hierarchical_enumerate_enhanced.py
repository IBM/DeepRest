from torch.utils.data import TensorDataset
import torch.nn as nn
import numpy as np
import pickle
import torch
import copy
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


window_seconds = 300
sampling_interval = 5
step_size = window_seconds // sampling_interval
num_layers = 1
hidden_layer_size = 128
num_epochs = 20
quantiles = (.05, .50, .95)
batch_size = 32
learning_rate = 0.001


########################################################################################################################
def load_data(dataset, component, metric):
    with open('./data/%s.pkl' % dataset, 'rb') as f:
        X, y, y_names, meta = pickle.load(f)
    for i, y_name in enumerate(y_names):
        for k, v in (('cpu', 1e3),          # cores -> millicores
                     ('memory', 1e-6),      # bytes -> megabytes
                     ('tp', (512 * 1e-3)),  # blocks -> kilobytes
                     ('usage', 1e3)):       # gigabytes -> megabytes
            if k in y_name:
                y[:, [i], :] *= v
    y = y[:, [y_names.index(i + '_%s' % metric) for i in [component]], :]
    return X, y.swapaxes(1, 2), meta


def get_metric_with_unit(metric):
    metric_with_unit = metric
    unit = ''
    if metric == 'cpu':
        metric_with_unit = 'CPU (millicores)'
        unit = '(millicores)'
    elif metric == 'memory':
        metric_with_unit = 'Working Set Size (MB)'
        unit = '(MB)'
    elif metric == 'write-iops':
        metric_with_unit = 'Write IOps'
        unit = ''
    elif metric == 'write-tp':
        metric_with_unit = 'Write Throughput (KB)'
        unit = '(KB)'
    elif metric == 'usage':
        metric_with_unit = 'Disk Usage (MB)'
        unit = '(MB)'
    return metric_with_unit, unit


class QuantileRNN(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers, widths, dropout=0.50, bidirectional=True):
        super().__init__()
        self.num_layers = num_layers
        self.widths = widths
        self.hidden_layer_size = hidden_layer_size
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size, hidden_layer_size, num_layers=num_layers, bidirectional=bidirectional)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.linear = nn.Linear(hidden_layer_size * (2 if bidirectional else 1), output_size)

        self.dummy = nn.Parameter(torch.full((1,), fill_value=1.), requires_grad=False)
        self.cumsum_linear_1 = nn.Linear(1, hidden_layer_size)
        self.cumsum_linear_2 = nn.Linear(hidden_layer_size, 2)

        self.prob_linears = nn.ModuleList([nn.ModuleList([nn.Linear(1, hidden_layer_size),
                                                          nn.Linear(hidden_layer_size, width)]) for width in widths])

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
        for i, q in enumerate(quantiles):
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

whitelist = {
    'compose-post-redis': [('compose-post-redis', 'cpu'), ('compose-post-redis', 'memory')],
    'compose-post-service': [('compose-post-service', 'cpu'), ('compose-post-service', 'memory')],
    'home-timeline-redis': [('home-timeline-redis', 'cpu'), ('home-timeline-redis', 'memory')],
    'home-timeline-service': [('home-timeline-service', 'cpu'), ('home-timeline-service', 'memory')],
    'media-frontend': [('media-frontend', 'cpu'), ('media-frontend', 'memory')],
    'media-mongodb': [('media-mongodb', 'cpu'), ('media-mongodb', 'memory'),
                      ('media-mongodb-pvc', 'write-iops'), ('media-mongodb-pvc', 'write-tp'),
                      ('media-mongodb-pvc', 'usage')],
    'media-service': [('media-service', 'cpu'), ('media-service', 'memory')],
    'nginx-thrift': [('nginx-thrift', 'cpu'), ('nginx-thrift', 'memory')],
    'post-storage-mongodb': [('post-storage-mongodb', 'cpu'), ('post-storage-mongodb', 'memory'),
                             ('post-storage-mongodb-pvc', 'write-iops'), ('post-storage-mongodb-pvc', 'write-tp'),
                             ('post-storage-mongodb-pvc', 'usage')],
    'post-storage-service': [('post-storage-service', 'cpu'), ('post-storage-service', 'memory')],
    'social-graph-redis': [('social-graph-redis', 'cpu'), ('social-graph-redis', 'memory')],
    'social-graph-service': [('social-graph-service', 'cpu'), ('social-graph-service', 'memory')],
    'text-service': [('text-service', 'cpu'), ('text-service', 'memory')],
    'unique-id-service': [('unique-id-service', 'cpu'), ('unique-id-service', 'memory')],
    'url-shorten-service': [('url-shorten-service', 'cpu'), ('url-shorten-service', 'memory')],
    'url-shorten-mongodb': [('url-shorten-mongodb', 'cpu'), ('url-shorten-mongodb', 'memory'),
                              ('url-shorten-mongodb-pvc', 'write-iops'),
                              ('url-shorten-mongodb-pvc', 'write-tp'), ('url-shorten-mongodb-pvc', 'usage')],
    'user-mention-service': [('user-mention-service', 'cpu'), ('user-mention-service', 'memory')],
    'user-mongodb': [('user-mongodb', 'cpu'), ('user-mongodb', 'memory'),
                              ('user-mongodb-pvc', 'write-iops'),
                              ('user-mongodb-pvc', 'write-tp'), ('user-mongodb-pvc', 'usage')],
    'user-service': [('user-service', 'cpu'), ('user-service', 'memory')],
    'write-home-timeline-service': [('write-home-timeline-service', 'cpu'), ('write-home-timeline-service', 'memory')],
    'user-timeline-redis': [('user-timeline-redis', 'cpu'), ('user-timeline-redis', 'memory')],
    'user-timeline-service': [('user-timeline-service', 'cpu'), ('user-timeline-service', 'memory')],
    'user-timeline-mongodb': [('user-timeline-mongodb', 'cpu'), ('user-timeline-mongodb', 'memory'),
                              ('user-timeline-mongodb-pvc', 'write-iops'),
                              ('user-timeline-mongodb-pvc', 'write-tp'), ('user-timeline-mongodb-pvc', 'usage')],
}

datasets = [
    'composePost_uploadMedia_readUserTimeline-steps_waves-seen_compositions-1x',
    'composePost_uploadMedia_readUserTimeline-waves_steps-seen_compositions-1x',
    'composePost_uploadMedia_readUserTimeline-waves_waves-seen_compositions-1x',
    'composePost_uploadMedia_readUserTimeline-waves_waves-seen_compositions-2x',
    'composePost_uploadMedia_readUserTimeline-waves_waves-seen_compositions-3x',
    'composePost_uploadMedia_readUserTimeline-waves_waves-unseen_compositions-1x',
    'composePost_uploadMedia_readUserTimeline-waves_waves-unseen_compositions-2x',
    'composePost_uploadMedia_readUserTimeline-waves_waves-unseen_compositions-3x'
]

for _did, dataset in enumerate(datasets):
    for _kid, k in enumerate(whitelist):
        for component, metric in whitelist[k]:
            print('Dataset: %d/%d | Component: %d/%d' % (_did+1, len(datasets), _kid+1, len(whitelist)))
            load_path = './%s_%s.pth' % (dataset, metric)
            offset_estimation = metric in ['memory', 'usage']

            X, y, meta = load_data(dataset, component, metric)
            split = meta['split']

            X_cumsum = np.cumsum(X, axis=1)

            X_prefix = meta['X_prefix']
            X_prefix_cumsum = np.cumsum(X_prefix, axis=1)

            X, scale_min_X, scale_max_X = QuantileRNN.normalization_minmax(X, split=split)
            scale_range_X = scale_max_X - scale_min_X
            X_cumsum, _, _ = QuantileRNN.normalization_minmax(X_cumsum, split=split)

            X_prefix, _, _ = QuantileRNN.normalization_minmax(X_prefix, split=split)
            X_prefix_cumsum, _, _ = QuantileRNN.normalization_minmax(X_prefix_cumsum, split=split)

            Xs = [X, X_prefix, X_cumsum, X_prefix_cumsum]
            X = np.concatenate(Xs, axis=2)
            widths = [x.shape[-1] for x in Xs]

            y_offset = np.zeros(shape=(y.shape[0], 1, 1))
            if offset_estimation:
                y_offset = y[:, [0], :]
                y = y - y_offset

            y, scale_min_y, scale_max_y = QuantileRNN.normalization_minmax(y, split=split)
            scale_range_y = scale_max_y - scale_min_y
            X_train, y_train = X[:split], y[:split]
            if metric == 'memory':
                X_train = X_train[len(X_train)//2:]
                y_train = y_train[len(y_train)//2:]
            X_test, y_test = X[split:], y[split:]
            y_test_offset = y_offset[split:]

            train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
            test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
            train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

            model = QuantileRNN(input_size=X.shape[-1], hidden_layer_size=hidden_layer_size,
                                output_size=len(quantiles), num_layers=num_layers, widths=widths).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            best_model_l2 = None
            for epoch in range(num_epochs):
                # Stage 1: Training
                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # Forward pass
                    outputs = model(inputs)
                    loss = QuantileRNN.quantile_loss(outputs, labels)
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    loss_buffer = []
                    num_cycles = 0
                    for iv, (inputs, labels) in enumerate(test_loader):
                        if iv % step_size != 0 or num_cycles >= 9:
                            continue
                        num_cycles += 1
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                        outputs = model(inputs)
                        loss = QuantileRNN.quantile_loss(outputs, labels)
                        loss_buffer.append(loss.item())

                        labels = labels.detach().cpu().numpy().squeeze()
                        outputs = outputs.detach().cpu().numpy().squeeze()

                        labels = labels * scale_range_y + scale_min_y
                        outputs = outputs * scale_range_y + scale_min_y

                        if offset_estimation:
                            labels = labels + y_test_offset[iv].squeeze()
                            outputs = outputs + y_test_offset[iv].squeeze()
                        outputs = np.maximum(outputs, 1e-6)

                    if best_model_l2 is None or np.mean(loss_buffer) < best_model_l2[0]:
                        best_model_l2 = (np.mean(loss_buffer), copy.deepcopy(model.state_dict()))
                model.train()

            if not os.path.exists('./models_all/%s' % dataset):
                os.makedirs('./models_all/%s' % dataset)
            torch.save(best_model_l2[1], './models_all/%s/%s_%s.pth' % (dataset, component, metric))
