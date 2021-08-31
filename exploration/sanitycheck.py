from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
plt.style.use('ggplot')


def figure_1():
    plt.figure(figsize=(12, 3))
    plt.title("Post Storage MongoDB - CPU Utilization")
    plt.plot(xs_train + xs_test, ys_train + ys_test, color='tab:brown')
    plt.ylabel("Utilization (%)")
    plt.xlabel("Timeline")
    plt.xticks(xticks, xlabels)
    for i in range(0, 2 * num_cycles * points_per_day, points_per_day):
        plt.axvline(x=i, color='tab:grey', linestyle=':')
    plt.xlim([0, 2 * num_cycles * points_per_day])
    plt.ylim([0, 100])
    plt.tight_layout()
    return plt


def figure_2():
    plt.figure(figsize=(12, 3))
    plt.title("Post Storage MongoDB - CPU Utilization")
    plt.plot(xs_train, ys_train, label="Application Learning")
    plt.plot(xs_test, ys_test, label='Sanity Check')
    plt.ylabel("Utilization (%)")
    plt.xlabel("Timeline")
    plt.xticks(xticks, xlabels)
    for i in range(0, 2 * num_cycles * points_per_day, points_per_day):
        plt.axvline(x=i, color='tab:grey', linestyle=':')
    plt.xlim([0, 2 * num_cycles * points_per_day])
    plt.ylim([0, 100])
    plt.legend(loc='upper left')
    plt.tight_layout()
    return plt


def figure_3():
    outputs = np.load('./visualization/sanitycheck/mongodb_outputs.npy')
    outputs = np.reshape(outputs, newshape=(outputs.shape[0] * outputs.shape[1], outputs.shape[2])) * 100 / 30
    outputs_ub = outputs[:, 0]
    outputs_lb = outputs[:, 2]

    plt.figure(figsize=(12, 3))
    plt.title("Post Storage MongoDB - CPU Utilization")
    plt.plot(xs_train, ys_train, label="Application Learning")
    plt.plot(xs_test, ys_test, label='Sanity Check')
    plt.ylabel("Utilization (%)")
    plt.xlabel("Timeline")
    plt.xticks(xticks, xlabels)
    for i in range(0, 2 * num_cycles * points_per_day, points_per_day):
        plt.axvline(x=i, color='tab:grey', linestyle=':')
    plt.xlim([0, 2 * num_cycles * points_per_day])
    plt.ylim([0, 100])
    plt.tight_layout()
    xx = np.atleast_2d(np.linspace(min(xs_test), max(xs_test), len(xs_test))).T
    plt.fill(np.concatenate([xx, xx[::-1]]), np.concatenate([outputs_ub, outputs_lb[::-1]]),
             alpha=.6, fc='orange', ec='None',
             label='Expected Utilization Interval')
    plt.legend(loc='upper left', ncol=2)
    return plt


def figure_4():
    outputs = np.load('./visualization/sanitycheck/mongodb_outputs.npy')
    outputs = np.reshape(outputs, newshape=(outputs.shape[0] * outputs.shape[1], outputs.shape[2])) * 100 / 30
    outputs_ub = outputs[:, 0]
    outputs_lb = outputs[:, 2]

    plt.figure(figsize=(8, 4))
    plt.title("Post Storage MongoDB - CPU Utilization")
    plt.plot(xs_test, ys_test, label='Measurement', color='tab:blue')
    plt.ylabel("Utilization (%)")
    plt.xlabel("Timeline")
    plt.xticks(xticks, ['\n\n\n%s' % v for v in xlabels])
    for i in range(0, 2 * num_cycles * points_per_day, points_per_day):
        plt.axvline(x=i, color='tab:grey', linestyle=':')
    plt.xlim([num_cycles * points_per_day, 2 * num_cycles * points_per_day])
    plt.ylim([0, 100])
    plt.tight_layout()
    xx = np.atleast_2d(np.linspace(min(xs_test), max(xs_test), len(xs_test))).T
    plt.fill(np.concatenate([xx, xx[::-1]]), np.concatenate([outputs_ub, outputs_lb[::-1]]),
             alpha=.6, fc='orange', ec='None',
             label='Expected Utilization Interval')
    plt.legend(loc='upper left', ncol=2)
    return plt


def figure_5():
    outputs = np.load('./visualization/sanitycheck/mongodb_outputs.npy')
    outputs = np.reshape(outputs, newshape=(outputs.shape[0] * outputs.shape[1], outputs.shape[2])) * 100 / 30
    outputs_ub = outputs[:, 0]
    outputs_lb = outputs[:, 2]

    plt.figure(figsize=(8, 4))
    anomaly_score = np.zeros(shape=(len(ys_test),))
    for i in range(len(ys_test)):
        if ys_test[i] > outputs_ub[i]:
            anomaly_score[i] = (ys_test[i] - outputs_ub[i]) / outputs_ub[i]
        elif ys_test[i] < outputs_lb[i]:
            anomaly_score[i] = (outputs_lb[i] - ys_test[i]) / outputs_lb[i]
    anomaly_score = gaussian_filter1d(anomaly_score, sigma=4)
    plt.imshow([anomaly_score for _ in range(50)], cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    return plt


dataset = 'composePost_uploadMedia_readUserTimeline-ransomware'
points_per_day = 60
num_cycles = 9
DOWs = ['SUN', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT']
xticks = [points_per_day * i for i in range(num_cycles * 2 + 1)]
xlabels = ['07/%.2d\n(%s)' % (j, DOWs[i % 7]) for i, j in enumerate(range(4, 4 + num_cycles * 2 + 1))]

with open('./data/%s.pkl' % dataset, 'rb') as f:
    X, y, y_names, meta = pickle.load(f)
split = meta['split']

ys = [v * 1e3 * 100 / 30 for v in meta['microservices']['post-storage-mongodb']['cpu']]

ys_train = ys[split-points_per_day*num_cycles:split]
ys_test = ys[split:split+points_per_day*num_cycles]
xs_train = list(range(len(ys_train)))
xs_test = list(range(len(xs_train), len(ys_train) + len(ys_test)))


with open('./visualization/sanitycheck/mongodb_outputs_all.pkl', 'rb') as f:
    results = pickle.load(f)

print('{' + ', '.join(['"%s": ""' % component for component in results.keys()]) + '}')
component_names = {"media-frontend": "Media Frontend",
                   "media-mongodb": "Media MongoDB",
                   "nginx-thrift": "NGINX Thrift",
                   "compose-post-service": "Compose Post Service",
                   "user-timeline-service": "User Timeline Service",
                   "user-timeline-mongodb": "User Timeline MongoDB",
                   "post-storage-service": "Post Storage Service",
                   "post-storage-mongodb": "Post Storage MongoDB"}
metric_names = {
    'cpu': 'CPU Utilization',
    'memory': 'Memory (Working Set Size)',
    'usage': 'Disk Usage',
    'write-iops': 'Write IOps',
    'write-tp': 'Write Throughput'
}
metric_units = {
    'cpu': 'CPU (millicores)',
    'memory': 'Memory (MB)',
    'usage': 'Usage (MB)',
    'write-iops': 'IOps',
    'write-tp': 'Throughput (KB)'
}

whitelist = {'post-storage-mongodb': ['cpu', 'write-iops', 'write-tp'],
             'media-frontend': ['cpu'],
             'user-timeline-service': ['cpu']}

for component in whitelist:
    for metric in whitelist[component]:
        predictability, measurements, outputs = results[component][metric]

        plt.figure(figsize=(5, 4))
        if component == 'post-storage-mongodb' and metric == 'cpu':
            measurements = [v * 100 / 30 for v in measurements]
            outputs = outputs * 100 / 30
            plt.ylim([0, 100])
            plt.ylabel("Utilization (%)")
        if component == 'post-storage-mongodb' and metric == 'write-iops':
            plt.ylim([0, 25])
            plt.ylabel("IOps")
        if component == 'post-storage-mongodb' and metric == 'write-tp':
            plt.ylim([0, 275])
            plt.ylabel("Throughput (KB)")
        if component == 'media-frontend' and metric == 'cpu':
            measurements = [v * 100 / 170 for v in measurements]
            outputs = outputs * 100 / 170
            plt.ylim([0, 100])
            plt.ylabel("Utilization (%)")
        if component == 'user-timeline-service' and metric == 'cpu':
            measurements = [v * 100 / 70 for v in measurements]
            outputs = outputs * 100 / 70
            plt.ylim([0, 100])
            plt.ylabel("Utilization (%)")
        outputs_lb = outputs[:, 0]
        outputs_ub = outputs[:, 2]
        plt.title("%s - %s" % (component_names[component], metric_names[metric]))
        plt.plot(xs_test, measurements, label='Measurement', color='tab:blue')
        plt.xlabel("Timeline")
        plt.xticks(xticks, ['\n\n\n%s' % v for v in xlabels])
        for i in range(0, 2 * num_cycles * points_per_day, points_per_day):
            plt.axvline(x=i, color='tab:grey', linestyle=':')
        plt.xlim([num_cycles * points_per_day, 2 * num_cycles * points_per_day])

        plt.tight_layout()
        xx = np.atleast_2d(np.linspace(min(xs_test), max(xs_test), len(xs_test))).T
        plt.fill(np.concatenate([xx, xx[::-1]]), np.concatenate([outputs_ub, outputs_lb[::-1]]),
                 alpha=.6, fc='orange', ec='None',
                 label='Expected Utilization Interval')
        plt.legend(loc='upper left', ncol=1)
        plt.show()

        plt.figure(figsize=(8, 4))
        anomaly_score = np.zeros(shape=(len(measurements),))
        for i in range(len(measurements)):
            if measurements[i] > outputs_ub[i]:
                anomaly_score[i] = (measurements[i] - outputs_ub[i]) / outputs_ub[i]
            elif measurements[i] < outputs_lb[i]:
                anomaly_score[i] = (outputs_lb[i] - measurements[i]) / outputs_lb[i]
        anomaly_score = np.asarray(anomaly_score)
        # anomaly_score = (anomaly_score - anomaly_score.min()) / (anomaly_score.max() - anomaly_score.min())
        anomaly_score = gaussian_filter1d(anomaly_score, sigma=4)
        plt.imshow([anomaly_score for _ in range(50)], cmap='jet', vmin=0, vmax=2)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
