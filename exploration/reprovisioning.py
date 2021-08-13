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

labels = [7.7170677, 7.7170677, 7.6409035, 6.939763, 5.587872, 5.9391966, 5.9391966, 5.9391966, 4.5939174, 5.3875833, 5.483395, 5.483395, 5.456231, 5.456231, 5.456231, 4.9868593, 4.9868593, 4.5217505, 4.3412676, 6.900634, 7.1310735, 7.1310735, 8.213073, 8.213073, 9.550764, 8.671026, 8.671026, 5.818715, 7.1940103, 7.1940103, 7.1940103, 6.691777, 6.691777, 6.691777, 6.6993103, 6.6993103, 6.6993103, 7.221347, 8.489367, 8.852322, 8.852322, 9.217558, 9.217558, 10.035177, 10.035177, 10.035177, 6.5338664, 8.053828, 8.053828, 8.053828, 6.9923835, 6.0109777, 6.0109777, 5.233871, 4.7844505, 3.2525878, 4.0060816, 3.2957869, 3.8053322, 3.5275195]
outputs_lower = [5.1781006, 5.230154, 5.1515346, 5.070452, 4.9982033, 4.959194, 4.896367, 4.8148346, 4.7563963, 4.730379, 4.6904163, 4.6710587, 4.6823473, 4.7478733, 4.846086, 4.9916706, 5.23293, 5.5592318, 5.9767537, 6.423752, 6.8666945, 7.292914, 7.5596995, 7.7130356, 7.7191863, 7.6364183, 7.4376845, 7.176301, 6.8140388, 6.3774967, 5.868669, 5.37728, 4.948423, 4.575962, 4.381595, 4.332881, 4.466343, 4.7960253, 5.2823963, 5.88624, 6.555723, 7.2095523, 7.800447, 8.165113, 8.268969, 8.179232, 7.9310794, 7.5441494, 7.04303, 6.4262905, 5.7814302, 5.1361427, 4.5189466, 3.9734492, 3.5172968, 3.2479835, 3.1721983, 3.2601187, 3.4977486, 3.7219133]
outputs_middle = [7.068549, 7.1055665, 6.980012, 6.8485093, 6.7243237, 6.6371665, 6.517457, 6.371211, 6.248276, 6.1593175, 6.0484104, 5.959219, 5.9053974, 5.9181576, 5.9733744, 6.0915956, 6.333597, 6.6897984, 7.168311, 7.6973357, 8.238129, 8.775707, 9.1439495, 9.393722, 9.478757, 9.457813, 9.294476, 9.048174, 8.671028, 8.190166, 7.60408, 7.016069, 6.4798803, 5.9910355, 5.6995583, 5.5737047, 5.66521, 5.996934, 6.526773, 7.2110915, 7.9942102, 8.785484, 9.5337925, 10.049324, 10.28503, 10.306125, 10.139723, 9.801482, 9.316458, 8.678493, 7.9829874, 7.25838, 6.533418, 5.8583794, 5.2545147, 4.841668, 4.6314144, 4.595724, 4.7433205, 4.9253983]
outputs_upper = [8.480825, 8.732828, 8.57976, 8.389002, 8.2286, 8.133976, 8.012831, 7.862788, 7.7436714, 7.6650963, 7.5561466, 7.465929, 7.4195886, 7.4515996, 7.543172, 7.735947, 8.121055, 8.690794, 9.444248, 10.281185, 11.134426, 11.971893, 12.572439, 12.987562, 13.159216, 13.182461, 13.013283, 12.712795, 12.215687, 11.541616, 10.707329, 9.847506, 9.045931, 8.315438, 7.868185, 7.695274, 7.8650155, 8.404102, 9.228704, 10.268801, 11.421198, 12.558018, 13.588293, 14.274719, 14.588806, 14.619984, 14.416552, 13.98369, 13.334209, 12.445417, 11.458017, 10.404896, 9.344712, 8.356201, 7.4925604, 6.9274015, 6.671723, 6.66334, 6.7794666, 6.5805225]
outputs_bl = [4.582846651171376, 4.560712520021616, 4.560712520021616, 4.582846651171376, 4.597602738604549, 4.590224694887962, 4.590224694887962, 4.538578388871855, 4.531200345155268, 4.575468607454789, 4.582846651171376, 4.597602738604549, 4.6787612194870025, 4.789431875235804, 5.032907317883167, 5.512480159461306, 6.4421136677512365, 7.784917624170025, 9.533513985001083, 11.569854050779027, 13.525035635674515, 15.458083089420242, 16.99271618247029, 18.14369100225782, 18.697044281001826, 18.984787985948707, 18.9331416799326, 18.54948340667009, 17.649362073246508, 16.321314204260894, 14.58009588714642, 12.610158214817758, 10.714000979654964, 8.950648531390732, 7.7480274055870915, 7.305344782591886, 7.674246968421224, 8.935892443957556, 10.706622935938377, 13.01595061923003, 15.524485482869526, 17.83381316616118, 19.840641057072773, 21.190823057208146, 22.09094439063173, 22.37131005186203, 22.29752961469616, 21.744176335952154, 20.77765260907929, 19.110214729130686, 17.14027705680202, 14.72027871776157, 12.211743854122075, 9.90241617083042, 7.888210236202239, 6.486381930050757, 5.527236246894479, 5.077175580182688, 4.804187962668978, 4.66400513205383]
xs = range(len(labels))

labels = [v * 100 / 25 for v in labels]
outputs_lower = [v * 100 / 25 for v in outputs_lower]
outputs_middle = [v * 100 / 25 for v in outputs_middle]
outputs_upper = [v * 100 / 25 for v in outputs_upper]
outputs_bl = [v * 100 / 25 for v in outputs_bl]

plt.figure(figsize=(5, 3))
plt.axhline(y=max(outputs_bl), color='firebrick', linestyle=':', label='Estimated Max. Usage: Linear Scaling')
plt.axhline(y=max(outputs_middle), color='blue', linestyle='--', label='Estimated Max. Usage: Ours')
plt.plot(xs, labels, color='green', label='Actual Usage')

# plt.plot(xs, outputs_lower, color='black')
# plt.plot(xs, outputs_upper, color='black')
# xx = np.atleast_2d(np.linspace(0, len(labels), len(labels))).T
# plt.fill(np.concatenate([xx, xx[::-1]]), np.concatenate([outputs_lower, outputs_upper[::-1]]),
#          alpha=.3, fc='b', ec='None', label='90% Pred. Interval')
plt.ylabel('CPU Utilization (%)')
plt.ylim([0, 100])
plt.xlim([0, len(xs)])
plt.xticks([0, 10, 20, 30, 40, 50, 60], ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '00:00'])
plt.tight_layout()
plt.show()

exit()
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
retrain = False

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
component = 'user-timeline-mongodb'
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
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]
y_test_offset = y_offset[split:]
print('   > [Train             ] X: %s | y: %s' % (str(X_train.shape), str(y_train.shape)))
print('   > [Test              ] X: %s | y: %s' % (str(X_test.shape), str(y_test.shape)))
print('   > [Test-BL (Resrc)   ]                    y: %s' % str(y_test_baseline_resrc.shape))
print('   > [Test-BL (ReqResrc)]                    y: %s' % str(y_test_baseline_reqresrc.shape))
assert y_test.shape == y_test_baseline_resrc.shape == y_test_baseline_reqresrc.shape, 'y_test\'s sizes do not match.'
print((np.max(y_train) * scale_range_y + scale_min_y) * 3)
yy = []
xx = []
xx_ = []
xx__ = []
for iv in range(len(y)):
    if iv % step_size != 0:
        continue
    yy = yy + list(y[iv] * scale_range_y + scale_min_y)
    xx = xx + list(np.sum(X[iv][:, meta['api2indices']['readUserTimeline']], axis=1) * scale_range_X + scale_min_X)
    xx_ = xx_ + list(np.sum(X[iv][:, meta['api2indices']['composePost']], axis=1) * scale_range_X + scale_min_X)
    xx__ = xx__ + list(np.sum(X[iv][:, meta['api2indices']['uploadMedia']], axis=1) * scale_range_X + scale_min_X)

plt.subplot(2, 1, 1)
plt.plot(xx, label='/readUserTimeline', color='red')
plt.plot(xx_, label='/composePost', color='blue')
plt.plot(xx__, label='/uploadMedia', color='green')
plt.legend()
for iv in range(split, len(xx), step_size):
    plt.axvline(x=iv)
for iv in range(split, 0, -step_size):
    plt.axvline(x=iv)
plt.subplot(2, 1, 2)
plt.plot(yy)
for iv in range(split, len(yy), step_size):
    plt.axvline(x=iv)
for iv in range(split, 0, -step_size):
    plt.axvline(x=iv)
plt.xlabel('Timeline')
plt.ylabel('CPU Utilization (millicores)')
plt.tight_layout()
plt.show()


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
        if num_cycles != 6:
            continue
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

        print(list(labels))
        print(list(outputs[:, 0]))
        print(list(outputs[:, 1]))
        print(list(outputs[:, 2]))
        print(list(outputs_baseline_reqresrc))

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
        print('Cycle #%d' % num_cycles)
        for api, indices in meta['api2indices'].items():
            plt.plot(timeline, np.sum(X_test[iv, :len(timeline)][:, indices], axis=1) * scale_range_X + scale_min_X, label='/%s' % api,
                     linestyle=matstyles[api][0], color=matstyles[api][1])
            print('%s: %d' % (api, int(np.max(np.ceil(np.sum(X_test[iv, :len(timeline)][:, indices], axis=1) * scale_range_X + scale_min_X)))))
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
# plt.show()