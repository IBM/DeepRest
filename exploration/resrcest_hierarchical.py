from baselines import Baseline_Resource_Only, Baseline_RequestAndResource
from torch.utils.data import TensorDataset
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import pickle
import torch
import copy
plt.style.use('ggplot')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


window_seconds = 300
sampling_interval = 5
step_size = window_seconds // sampling_interval
dataset = 'composePost_uploadMedia_readUserTimeline-waves_waves-unseen_compositions-3x'
num_layers = 1
hidden_layer_size = 128
num_epochs = 50
quantiles = (.05, .50, .95)
batch_size = 32
learning_rate = 0.001
retrain = True

# ================================================== compute ==================================================
# media-frontend        | media-mongodb         | nginx-thrift
# compose-post-service  |
# user-timeline-service | user-timeline-mongodb |
# post-storage-service  | post-storage-mongodb  |
# =============================================================================================================
# available metrics: cpu | memory
# =================================== storage ====================================
# media-mongodb-pvc       | post-storage-mongodb-pvc | user-timeline-mongodb-pvc |
# ================================================================================
# available metrics: write-iops | write-tp | usage
component = 'compose-post-service'
metric = 'cpu'


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


########################################################################################################################
load_path = './%s_%s.pth' % (dataset, metric)
offset_estimation = metric in ['memory', 'usage']
metric_with_unit, unit = get_metric_with_unit(metric)
timeline = [t * sampling_interval for t in range(0, step_size)]

########################################################################################################################
print('Loading dataset...')
print('   > path     : %s' % dataset)
print('   > component: %s' % component)
print('   > metric   : %s' % metric)
X, y, meta = load_data(dataset, component, metric)
split = meta['split']

baseline_resrc = Baseline_Resource_Only(offset_estimation=offset_estimation, split=split,
                                        offset=step_size-1, input_size=step_size, output_size=step_size)
y_test_baseline_resrc = baseline_resrc.fit_and_estimate(X, y)
baseline_reqresrc = Baseline_RequestAndResource(dataset=dataset, component=component, invocation=meta['invocation'],
                                                metric=metric, offset_estimation=offset_estimation,
                                                split=split, output_size=step_size)
y_test_baseline_reqresrc = baseline_reqresrc.fit_and_estimate(X, y)


print('Max-min normalization on inputs (requests)...')
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
print('   > X.shape: %s' % str(X.shape))
print('   > X.min  : %.4f' % scale_min_X)
print('   > X.max  : %.4f' % scale_max_X)


y_offset = np.zeros(shape=(y.shape[0], 1, 1))
if offset_estimation:
    print('Enabling offset estimation for metric = %s...' % metric)
    y_offset = y[:, [0], :]
    y = y - y_offset

print('Max-min normalization on outputs (resources)...')
y, scale_min_y, scale_max_y = QuantileRNN.normalization_minmax(y, split=split)
scale_range_y = scale_max_y - scale_min_y
print('   > y.shape: %s' % str(y.shape))
print('   > y.min  : %.4f' % scale_min_y)
print('   > y.max  : %.4f' % scale_max_y)

print('Splitting train-test datasets...')
X_train, y_train = X[split//2:split], y[split//2:split]
X_test, y_test = X[split:], y[split:]
y_test_offset = y_offset[split:]
print('   > [Train             ] X: %s | y: %s' % (str(X_train.shape), str(y_train.shape)))
print('   > [Test              ] X: %s | y: %s' % (str(X_test.shape), str(y_test.shape)))
print('   > [Test-BL (Resrc)   ]                    y: %s' % str(y_test_baseline_resrc.shape))
print('   > [Test-BL (ReqResrc)]                    y: %s' % str(y_test_baseline_reqresrc.shape))
assert y_test.shape == y_test_baseline_resrc.shape == y_test_baseline_reqresrc.shape, 'y_test\'s sizes do not match.'

train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


model = QuantileRNN(input_size=X.shape[-1], hidden_layer_size=hidden_layer_size,
                    output_size=len(quantiles), num_layers=num_layers, widths=widths).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if retrain:
    xs_train, xs_test = [], []
    ys_train, ys_test = [], []
    best_model = None
    for epoch in range(num_epochs):
        # Stage 1: Training
        loss_buffer = []
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
            # Logging
            loss_buffer.append(loss.item())
        xs_train.append(epoch)
        ys_train.append(np.mean(loss_buffer))

        # Stage 3: Testing (optional)
        yerr_ours, yerr_baseline_resrc, yerr_baseline_reqresrc = {'abs': [], 'cdf': []}, \
                                                                 {'abs': [], 'cdf': []}, \
                                                                 {'abs': [], 'cdf': []}
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
                outputs_baseline_resrc = y_test_baseline_resrc[iv].squeeze()
                outputs_baseline_reqresrc = y_test_baseline_reqresrc[iv].squeeze()

                labels = labels * scale_range_y + scale_min_y
                outputs = outputs * scale_range_y + scale_min_y

                if offset_estimation:
                    labels = labels + y_test_offset[iv].squeeze()
                    outputs = outputs + y_test_offset[iv].squeeze()
                outputs = np.maximum(outputs, 1e-6)

                yerr_ours['cdf'] += list(np.abs(np.sort(outputs[:, 1]) - np.sort(labels)))
                yerr_baseline_resrc['cdf'] += list(np.abs(np.sort(outputs_baseline_resrc) - np.sort(labels)))
                yerr_baseline_reqresrc['cdf'] += list(np.abs(np.sort(outputs_baseline_reqresrc) - np.sort(labels)))

                yerr_ours['abs'] += list(np.abs(outputs[:, 1] - labels))
                yerr_baseline_resrc['abs'] += list(np.abs(outputs_baseline_resrc - labels))
                yerr_baseline_reqresrc['abs'] += list(np.abs(outputs_baseline_reqresrc - labels))

            xs_test.append(epoch)
            ys_test.append(np.mean(loss_buffer))
            if best_model is None or ys_test[-1] < best_model[1]:
                best_model = (epoch, ys_test[-1], copy.deepcopy(model.state_dict()))
        print('X           : [' + ', '.join(['%.2f' % v for v in list(torch.cat(
            [model.softmax(l2(model.relu(l1(model.dummy)))) for l1, l2 in
             [model.prob_linears[0]]]).detach().cpu().numpy())]) + ']')
        print('X_p         : [' + ', '.join(['%.2f' % v for v in list(torch.cat(
            [model.softmax(l2(model.relu(l1(model.dummy)))) for l1, l2 in
             [model.prob_linears[1]]]).detach().cpu().numpy())]) + ']')
        print('X - cumsum  : [' + ', '.join(['%.2f' % v for v in list(torch.cat(
            [model.softmax(l2(model.relu(l1(model.dummy)))) for l1, l2 in
             [model.prob_linears[2]]]).detach().cpu().numpy())]) + ']')
        print('X_p - cumsum: [' + ', '.join(['%.2f' % v for v in list(torch.cat(
            [model.softmax(l2(model.relu(l1(model.dummy)))) for l1, l2 in
             [model.prob_linears[3]]]).detach().cpu().numpy())]) + ']')
        print('count v.s. cumsum: [' + ', '.join(['%.2f' % v for v in model.softmax(
            model.cumsum_linear_2(model.relu(model.cumsum_linear_1(model.dummy)))).detach().cpu().numpy()]) + ']')
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {ys_train[-1]:.6f}, '
              f'Test Loss: {ys_test[-1]:.6f}')
        model.train()
        print('   [Absolute] Resrc     => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
            np.median(yerr_baseline_resrc['abs']), np.percentile(yerr_baseline_resrc['abs'], q=90), np.percentile(yerr_baseline_resrc['abs'], q=95),
            np.percentile(yerr_baseline_resrc['abs'], q=99), np.max(yerr_baseline_resrc['abs'])))
        print('   [Absolute] Req+Resrc => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
            np.median(yerr_baseline_reqresrc['abs']), np.percentile(yerr_baseline_reqresrc['abs'], q=90), np.percentile(yerr_baseline_reqresrc['abs'], q=95),
            np.percentile(yerr_baseline_reqresrc['abs'], q=99), np.max(yerr_baseline_reqresrc['abs'])))
        print('   [Absolute] Ours      => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
            np.median(yerr_ours['abs']), np.percentile(yerr_ours['abs'], q=90), np.percentile(yerr_ours['abs'], q=95),
            np.percentile(yerr_ours['abs'], q=99), np.max(yerr_ours['abs'])))

        print('   [CDF     ] Resrc     => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
            np.median(yerr_baseline_resrc['cdf']), np.percentile(yerr_baseline_resrc['cdf'], q=90), np.percentile(yerr_baseline_resrc['cdf'], q=95),
            np.percentile(yerr_baseline_resrc['cdf'], q=99), np.max(yerr_baseline_resrc['cdf'])))
        print('   [CDF     ] Req+Resrc => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
            np.median(yerr_baseline_reqresrc['cdf']), np.percentile(yerr_baseline_reqresrc['cdf'], q=90), np.percentile(yerr_baseline_reqresrc['cdf'], q=95),
            np.percentile(yerr_baseline_reqresrc['cdf'], q=99), np.max(yerr_baseline_reqresrc['cdf'])))
        print('   [CDF     ] Ours      => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
            np.median(yerr_ours['cdf']), np.percentile(yerr_ours['cdf'], q=90), np.percentile(yerr_ours['cdf'], q=95), np.percentile(yerr_ours['cdf'], q=99),
            np.max(yerr_ours['cdf'])))

    print('Reloading the best model at Epoch %d' % (best_model[0] + 1))
    model.load_state_dict(best_model[2])
    plt.clf()
    plt.title('Learning Curves')
    plt.plot(xs_train, ys_train, label='Train', marker='x')
    plt.plot(xs_test, ys_test, label='Test', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([0, max(xs_train)])
    plt.legend()
    plt.tight_layout()
    plt.show()
    print('Saving the trained model to %s...' % load_path)
    torch.save(model.state_dict(), load_path)
else:
    print('Loading the pre-trained model from %s...' % load_path)
    model.load_state_dict(torch.load(load_path))
model.eval()

print('Testing the model...')
yerr_ours, yerr_baseline_resrc, yerr_baseline_reqresrc = {'abs': [], 'cdf': []}, \
                                                         {'abs': [], 'cdf': []}, \
                                                         {'abs': [], 'cdf': []}
with torch.no_grad():
    num_cycles = 0
    for iv, (inputs, labels) in enumerate(test_loader):
        if iv % step_size != 0 or num_cycles >= 9:
            continue
        num_cycles += 1
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)

        labels = labels.detach().cpu().numpy().squeeze()
        outputs = outputs.detach().cpu().numpy().squeeze()
        outputs_baseline_resrc = y_test_baseline_resrc[iv].squeeze()
        outputs_baseline_reqresrc = y_test_baseline_reqresrc[iv].squeeze()

        labels = labels * scale_range_y + scale_min_y
        outputs = outputs * scale_range_y + scale_min_y

        if offset_estimation:
            labels = labels + y_test_offset[iv].squeeze()
            outputs = outputs + y_test_offset[iv].squeeze()
        outputs = np.maximum(outputs, 1e-6)

        yerr_ours['cdf'] += list(np.abs(np.sort(outputs[:, 1]) - np.sort(labels)))
        yerr_baseline_resrc['cdf'] += list(np.abs(np.sort(outputs_baseline_resrc) - np.sort(labels)))
        yerr_baseline_reqresrc['cdf'] += list(np.abs(np.sort(outputs_baseline_reqresrc) - np.sort(labels)))

        yerr_ours['abs'] += list(np.abs(outputs[:, 1] - labels))
        yerr_baseline_resrc['abs'] += list(np.abs(outputs_baseline_resrc - labels))
        yerr_baseline_reqresrc['abs'] += list(np.abs(outputs_baseline_reqresrc - labels))

        plt.clf()
        plt.figure(figsize=(12, 12))
        metric_limit = [
            max(min(min(labels), min(outputs_baseline_resrc), min(outputs_baseline_reqresrc), np.min(outputs)) // 10 * 10 - 10.5, 0.0),
            max(max(labels), max(outputs_baseline_resrc), max(outputs_baseline_reqresrc), np.max(outputs)) // 10 * 10 + 10.5]
        plt.subplot(3, 2, 1)
        plt.title(component + '_' + metric)
        plt.plot(timeline, labels, label='Ground Truth', linestyle='--', color='red')
        plt.plot(timeline, outputs_baseline_resrc, label='Baseline: Resrc-aware ANN', color='green')
        plt.plot(timeline, outputs_baseline_reqresrc, label='Baseline: Req-aware LinearRegr', color='orange')
        plt.plot(timeline, outputs[:, 1], label='Ours: API-aware QRNN', color='mediumblue')
        plt.ylim(metric_limit)
        plt.xlabel('Timeline')
        plt.xlim([min(timeline), max(timeline)])
        plt.ylabel(metric_with_unit)
        plt.legend()
        plt.subplot(3, 2, 2)
        plt.plot(np.sort(labels), np.arange(step_size) / step_size, color='red', linestyle='--', label='Ground Truth')
        plt.plot(np.sort(outputs_baseline_resrc), np.arange(step_size) / step_size, color='green', label='Baseline: Resrc-aware ANN')
        plt.plot(np.sort(outputs_baseline_reqresrc), np.arange(step_size) / step_size, color='orange', label='Baseline: Req-aware LinearRegr')
        plt.plot(np.sort(outputs[:, 1]), np.arange(step_size) / step_size, color='mediumblue', label='Ours: API-aware QRNN')
        plt.xlim(metric_limit)
        plt.xlabel(metric_with_unit)
        plt.ylabel('CDF')
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(timeline, labels, label='Ground Truth', color='red', linestyle='--')
        plt.plot(timeline, outputs[:, 0], 'k-')
        plt.plot(timeline, outputs[:, 1], label='Ours: API-aware QRNN', color='mediumblue')
        plt.plot(timeline, outputs[:, 2], 'k-')
        xx = np.atleast_2d(np.linspace(0 * 5, step_size * 5, step_size)).T
        plt.fill(np.concatenate([xx, xx[::-1]]), np.concatenate([outputs[:, 2], outputs[:, 0][::-1]]),
                 alpha=.3, fc='b', ec='None',
                 label='%d%% Prediction Interval' % (int(quantiles[-1] * 100) - int(quantiles[0] * 100)))
        plt.xlabel('Timeline')
        plt.xlim([min(timeline), max(timeline)])
        plt.ylabel(metric_with_unit)
        plt.ylim(metric_limit)
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(np.sort(labels), np.arange(step_size) / step_size, color='red', label='Ground Truth', linestyle='--')
        plt.plot(np.sort(outputs[:, 1]), np.arange(step_size) / step_size, color='mediumblue', label='Ours: Req-aware QRNN')
        outputs_lower_sorted = np.sort(outputs[:, 0])
        outputs_upper_sorted = np.sort(outputs[:, 2])
        plt.plot(outputs_lower_sorted, np.arange(step_size) / step_size, 'k-')
        plt.plot(outputs_upper_sorted, np.arange(step_size) / step_size, 'k-')
        xx = np.atleast_2d(np.linspace(0, 1, step_size)).T
        plt.fill(np.concatenate([outputs_upper_sorted, outputs_lower_sorted[::-1]]), np.concatenate([xx, xx[::-1]]),
                 alpha=.3, fc='b', ec='None',
                 label='%d%% Prediction Interval' % (int(quantiles[-1] * 100) - int(quantiles[0] * 100)))
        plt.xlim(metric_limit)
        plt.xlabel(metric_with_unit)
        plt.ylabel('CDF')
        plt.legend()

        plt.subplot(3, 2, 5)
        plt.plot(timeline, np.sum(X_test[iv, :len(timeline), :X_cumsum.shape[-1]], axis=1) * scale_range_X + scale_min_X, label='ALL APIs')

        matstyles = {
            'composePost': ('--', 'goldenrod'),
            'readUserTimeline': (':', 'blue'),
            'uploadMedia': ('-.', 'slategrey')
        }
        for api, indices in meta['api2indices'].items():
            plt.plot(timeline, np.sum(X_test[iv, :len(timeline)][:, indices], axis=1) * scale_range_X + scale_min_X, label='/%s' % api,
                     linestyle=matstyles[api][0], color=matstyles[api][1])

        plt.xlabel('Timeline')
        plt.xlim([min(timeline), max(timeline)])
        plt.ylabel('Number of Traces')
        plt.ylim([0, (max(np.sum(X_test[iv, :len(timeline), :X_cumsum.shape[-1]], axis=1) * scale_range_X + scale_min_X) * 1.3)])
        plt.legend(ncol=2)
        plt.tight_layout()
        plt.show()

print('Mean CDF L2 Error - %s_%s' % (component, metric))
print('   [Absolute] Resrc     => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
    np.median(yerr_baseline_resrc['abs']), np.percentile(yerr_baseline_resrc['abs'], q=90),
    np.percentile(yerr_baseline_resrc['abs'], q=95), np.percentile(yerr_baseline_resrc['abs'], q=99),
    np.max(yerr_baseline_resrc['abs'])))
print('   [Absolute] Req+Resrc => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
    np.median(yerr_baseline_reqresrc['abs']), np.percentile(yerr_baseline_reqresrc['abs'], q=90),
    np.percentile(yerr_baseline_reqresrc['abs'], q=95), np.percentile(yerr_baseline_reqresrc['abs'], q=99),
    np.max(yerr_baseline_reqresrc['abs'])))
print('   [Absolute] Ours      => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
    np.median(yerr_ours['abs']), np.percentile(yerr_ours['abs'], q=90),
    np.percentile(yerr_ours['abs'], q=95), np.percentile(yerr_ours['abs'], q=99),
    np.max(yerr_ours['abs'])))
print('   [CDF     ] Resrc     => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
    np.median(yerr_baseline_resrc['cdf']), np.percentile(yerr_baseline_resrc['cdf'], q=90),
    np.percentile(yerr_baseline_resrc['cdf'], q=95), np.percentile(yerr_baseline_resrc['cdf'], q=99),
    np.max(yerr_baseline_resrc['cdf'])))
print('   [CDF     ] Req+Resrc => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
    np.median(yerr_baseline_reqresrc['cdf']), np.percentile(yerr_baseline_reqresrc['cdf'], q=90),
    np.percentile(yerr_baseline_reqresrc['cdf'], q=95), np.percentile(yerr_baseline_reqresrc['cdf'], q=99),
    np.max(yerr_baseline_reqresrc['cdf'])))
print('   [CDF     ] Ours      => Median: %.4f | 90-th: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
    np.median(yerr_ours['cdf']), np.percentile(yerr_ours['cdf'], q=90), np.percentile(yerr_ours['cdf'], q=95),
    np.percentile(yerr_ours['cdf'], q=99), np.max(yerr_ours['cdf'])))

data = []
for i in range(len(yerr_ours['cdf'])):
    data.append(('Baseline:\nResrc-aware ANN', yerr_baseline_resrc['cdf'][i], yerr_baseline_resrc['abs'][i]))
    data.append(('Baseline:\nReq-aware LinearRegr', yerr_baseline_reqresrc['cdf'][i], yerr_baseline_reqresrc['abs'][i]))
    data.append(('Ours:\nAPI-aware QRNN', yerr_ours['cdf'][i], yerr_ours['abs'][i]))
data = pd.DataFrame(data, columns=['Approach', 'CDF Error', 'Abs. Error'])

plt.clf()
plt.figure(figsize=(11, 5))
plt.subplot(1, 2, 1)
plt.title('Component: %s' % component)
sns.boxplot(x=data['Approach'], y=data['Abs. Error'])
plt.xlabel('')
plt.ylabel('Absolute Error %s' % unit)
plt.ylim([plt.gca().get_ylim()[0], max(plt.gca().get_ylim()[1], plt.gca().get_ylim()[0] + 10)])
plt.subplot(1, 2, 2)
plt.title('Metric: %s' % metric)
sns.boxplot(x=data['Approach'], y=data['CDF Error'])
plt.xlabel('')
plt.ylabel('CDF Error %s' % unit)
plt.ylim([plt.gca().get_ylim()[0], max(plt.gca().get_ylim()[1], plt.gca().get_ylim()[0] + 10)])
plt.tight_layout()
plt.show()