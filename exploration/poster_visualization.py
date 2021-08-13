from baselines import Baseline_Resource_Only_RE, Baseline_RequestAndResource_RE, Baseline_NaiveRequestAndResource_RE
from QRNN import QuantileRNN
from matplotlib import pyplot as plt
import numpy as np
import pickle
plt.style.use('ggplot')


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


def get_metric_unit(metric):
    unit = ''
    if metric == 'cpu':
        unit = 'millicores'
    elif metric == 'memory':
        unit = 'MB'
    elif metric == 'write-tp':
        unit = 'KB'
    elif metric == 'write-iops':
        unit = 'IOps'
    elif metric == 'usage':
        unit = 'MB'
    return unit


def M2t(M):
    X_ = []
    for x in M[:-1]:
        X_.append(x[0])
    X_ = np.asarray(X_)
    X_ = np.concatenate([X_, M[-1]], axis=0)
    return X_


def X_to_API_calls(X):
    X_ = M2t(X)
    out = {}
    total_calls = np.zeros(shape=(len(X_),))
    for api, indices in meta['api2indices'].items():
        out[api] = np.sum(X_[:, indices], axis=1)
        total_calls = total_calls + out[api]
    out['all'] = total_calls
    return out


whitelist = {
    'nginx-thrift': [('nginx-thrift', 'cpu'), ('nginx-thrift', 'memory')],
    'compose-post-service': [('compose-post-service', 'cpu'), ('compose-post-service', 'memory')],
    'post-storage-service': [('post-storage-service', 'cpu'), ('post-storage-service', 'memory')],
    'post-storage-mongodb': [('post-storage-mongodb', 'cpu'), ('post-storage-mongodb', 'memory'),
                             ('post-storage-mongodb-pvc', 'write-iops'), ('post-storage-mongodb-pvc', 'write-tp'), ('post-storage-mongodb-pvc', 'usage')],
    'user-timeline-service': [('user-timeline-service', 'cpu'), ('user-timeline-service', 'memory')],
    'user-timeline-mongodb': [('user-timeline-mongodb', 'cpu'), ('user-timeline-mongodb', 'memory'),
                              ('user-timeline-mongodb-pvc', 'write-iops'), ('user-timeline-mongodb-pvc', 'write-tp'), ('user-timeline-mongodb-pvc', 'usage')],
    'media-frontend': [('media-frontend', 'cpu'), ('media-frontend', 'memory')],
    'media-mongodb': [('media-mongodb', 'cpu'), ('media-mongodb', 'memory'),
                      ('media-mongodb-pvc', 'write-iops'), ('media-mongodb-pvc', 'write-tp'), ('media-mongodb-pvc', 'usage')],
}
points_per_day = 60
num_cycles = 9
DOWs = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
xticks = [points_per_day * i for i in range(num_cycles * 2 + 1)]
xlabels = ['07/%.2d\n(%s)' % (j, DOWs[i % 7]) for i, j in enumerate(range(4, 4 + num_cycles * 2 + 1))]

dataset = 'composePost_uploadMedia_readUserTimeline-waves_waves-unseen_compositions-3x'
component = 'media-mongodb'
metric = 'memory'
for component_, pairs in whitelist.items():
    if 'mongodb' in component_:
        component_ = component_.replace('mongodb', 'service') if 'media' not in component_ else 'media-frontend'
    for component, metric in pairs:
        X, y, meta = load_data(dataset, component, metric)
        X_prefix = meta['X_prefix']
        calls = X_to_API_calls(X)
        split = meta['split']
        offset_estimation = metric in ['usage', 'memory']
        baseline_naive_reqresrc = Baseline_NaiveRequestAndResource_RE(
            calls=calls, metric=metric, split=split, points_per_day=points_per_day, offset_estimation=offset_estimation)
        y_test_baseline_naive_reqresrc = baseline_naive_reqresrc.fit_and_estimate(X, y)
        baseline_reqresrc = Baseline_RequestAndResource_RE(
            component=component_, invocation=meta['invocation'], points_per_day=points_per_day,
            offset_estimation=offset_estimation, split=split)
        y_test_baseline_reqresrc = baseline_reqresrc.fit_and_estimate(X, y)
        baseline_resrc = Baseline_Resource_Only_RE(offset_estimation=offset_estimation, split=split,
                                                   offset=points_per_day - 1, input_size=points_per_day,
                                                   output_size=points_per_day)
        y_test_baseline_resrc = baseline_resrc.fit_and_estimate(X, y)

        ours_qrnn = QuantileRNN(widths=(X.shape[-1], X_prefix.shape[-1], X.shape[-1], X_prefix.shape[-1]),
                                warmup=(metric == 'memory'), split=split, offset_estimation=offset_estimation)
        pretrained_path = './models/models-optimized/opt_max-err1_p2p/%s/%s_%s.pth' % (dataset, component, metric)
        pretrained_roots = ['opt_l2',
                            'opt_max-err0_m2m', 'opt_max-err1_m2m', 'opt_max-err2_m2m',
                            'opt_max-err0_p2p', 'opt_max-err1_p2p', 'opt_max-err2_p2p']
        ys = list(M2t(y)[:, 0])
        ys_train = ys[split-points_per_day*num_cycles:split]
        ys_test = ys[split:split+points_per_day*num_cycles]
        xs_train = list(range(len(ys_train)))
        xs_test = list(range(len(xs_train), len(ys_train) + len(ys_test)))

        y_test_ours = [None for _ in range(num_cycles)]
        y_test_ours_error = [float('inf') for _ in range(num_cycles)]
        for pretrained_root in pretrained_roots:
            pretrained_path = './models/models-optimized/%s/%s/%s_%s.pth' % (pretrained_root, dataset, component, metric)
            yy = ours_qrnn.fit(X, y, X_prefix, pretrained_path=pretrained_path)
            for i in range(num_cycles):
                start_ind = i * points_per_day
                end_ind = (i + 1) * points_per_day
                optimal = np.max(ys_test[start_ind:end_ind])
                err = abs(np.max(yy[start_ind:end_ind, 1]) - optimal)
                if err < y_test_ours_error[i]:
                    y_test_ours_error[i] = err
                    y_test_ours[i] = yy[start_ind:end_ind]
        y_test_ours = np.concatenate(y_test_ours, axis=0)[:, 1]

        plt.figure(figsize=(12, 9))
        plt.subplot(3, 1, 1)
        plt.title("%s - %s" % (component, metric))
        plt.plot(xs_train + xs_test, ys_train + ys_test,
                 color='tab:brown', label='Measurement')
        plt.plot(xs_test, y_test_baseline_resrc[:num_cycles * points_per_day],
                 label='Baseline: Historical Pattern', color='orange')
        plt.plot(xs_test, y_test_baseline_naive_reqresrc[:num_cycles * points_per_day],
                 label='Baseline: API-aware LinearRegr', color='green')
        plt.plot(xs_test, y_test_baseline_reqresrc[:num_cycles * points_per_day],
                 label='Baseline: Trace-aware LinearRegr', color='blue')
        plt.plot(xs_test, y_test_ours[:num_cycles * points_per_day],
                 label='Ours: QRNN', color='black')
        plt.legend(loc='upper left')
        plt.ylabel(get_metric_unit(metric))
        plt.xlabel("Timeline")
        plt.xticks(xticks, xlabels)
        for i in range(0, 2 * num_cycles * points_per_day, points_per_day):
            plt.axvline(x=i, color='tab:grey', linestyle=':')
        plt.xlim([0, 2 * num_cycles * points_per_day])

        plt.subplot(3, 1, 2)
        plt.title("+ve: Over-provision | -ve: Under-provision")

        deviation_baseline_resrc, deviation_baseline_naive_reqresrc, deviation_baseline_reqresrc, deviation_ours = [], [], [], []
        optimal_config = []
        for i in range(num_cycles):
            start_ind = i * points_per_day
            end_ind = (i+1) * points_per_day

            deviation_baseline_resrc.append(np.max(y_test_baseline_resrc[start_ind:end_ind]) / np.max(ys_train))
            deviation_baseline_naive_reqresrc.append(np.max(y_test_baseline_naive_reqresrc[start_ind:end_ind]) / np.max(ys_train))
            deviation_baseline_reqresrc.append(np.max(y_test_baseline_reqresrc[start_ind:end_ind]) / np.max(ys_train))
            deviation_ours.append(np.max(y_test_ours[start_ind:end_ind]) / np.max(ys_train))
            y = np.max(ys_test[start_ind:end_ind]) / np.max(ys_train)
            plt.hlines(y=y, xmin=xs_test[start_ind], xmax=xs_test[min(end_ind, len(xs_test)-1)])

        x = np.asarray([num_cycles * points_per_day + i * points_per_day + 0.5 * points_per_day for i in range(num_cycles)])
        width = 10  # the width of the bars

        rects1 = plt.bar(x - 1.5 * width, deviation_baseline_resrc, width, label='Baseline: Historical Pattern', color='orange')
        rects2 = plt.bar(x - 0.5 * width, deviation_baseline_naive_reqresrc, width, label='Baseline: API-aware LinearRegr', color='green')
        rects3 = plt.bar(x + 0.5 * width, deviation_baseline_reqresrc, width, label='Baseline: Trace-aware LinearRegr', color='blue')
        rects4 = plt.bar(x + 1.5 * width, deviation_ours, width, label='Ours: QRNN', color='black')

        ylim_max = max(max([abs(v) for v in plt.gca().get_ylim()]), 1.0)
        plt.ylim([0, +ylim_max*1.10])
        plt.xlim([0, 2 * num_cycles * points_per_day])
        plt.xticks(xticks, xlabels)
        for i in range(0, 2 * num_cycles * points_per_day, points_per_day):
            plt.axvline(x=i, color='tab:grey', linestyle=':')
        plt.ylabel("Scaling Factor")
        plt.legend(loc='upper left')

        plt.subplot(3, 1, 3)
        plt.title("API Calls")
        plt.plot(xs_train + xs_test, calls['all'][split-points_per_day*num_cycles:split+points_per_day*num_cycles], label='ALL')
        plt.plot(xs_train + xs_test, calls['composePost'][split-points_per_day*num_cycles:split+points_per_day*num_cycles], label='/composePost')
        plt.plot(xs_train + xs_test, calls['readUserTimeline'][split-points_per_day*num_cycles:split+points_per_day*num_cycles], label='/readTimeline')
        plt.plot(xs_train + xs_test, calls['uploadMedia'][split-points_per_day*num_cycles:split+points_per_day*num_cycles], label='/uploadMedia')
        plt.ylabel("RPS")
        plt.xlabel("Timeline")
        plt.xlim([0, 2 * num_cycles * points_per_day])
        plt.xticks(xticks, xlabels)
        for i in range(0, 2 * num_cycles * points_per_day, points_per_day):
            plt.axvline(x=i, color='tab:grey', linestyle=':')
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
