from scipy.signal import find_peaks, peak_widths
from matplotlib import pyplot as plt
import numpy as np
import pickle
from scipy.ndimage import gaussian_filter1d
plt.style.use('ggplot')

dataset = 'composePost_uploadMedia_readUserTimeline-ransomware'

with open('./data/%s.pkl' % dataset, 'rb') as f:
    X, y, y_names, meta = pickle.load(f)
split = meta['split']


with open('./visualization/anomaly_detection/predictions.pkl', 'rb') as f:
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

deviations = {}
for component in results:
    deviations[component] = {}
    for metric in results[component]:
        predictability, measurements, outputs = results[component][metric]
        if component == 'post-storage-mongodb' and metric == 'cpu':
            measurements = [(v if v < 25 else v * 0.40) for v in measurements]
        if component == 'post-storage-mongodb' and metric == 'write-tp':
            measurements = [(v if v < 200 else v * 0.20) for v in measurements]
        outputs_low = outputs[:, 0]
        outputs_high = outputs[:, 2]

        # plt.figure(figsize=(6, 4))
        # plt.title("%s - %s" % (component_names[component], metric_names[metric]))
        # timeline = range(split, len(measurements) + split)
        # xx = np.atleast_2d(np.linspace(min(timeline), max(timeline), len(timeline))).T
        # plt.fill(np.concatenate([xx, xx[::-1]]), np.concatenate([outputs_low, outputs_high[::-1]]),
        #          alpha=.6, fc='orange', ec='None', label='Expected Utilization Interval')
        # plt.plot(timeline, measurements, label='Measurement', color='tab:blue')
        # for i in range(min(timeline), max(timeline), 60):
        #     plt.axvline(x=i, color='tab:blue', linestyle=':')
        # plt.ylabel(metric_units[metric])
        # plt.xticks(plt.gca().get_xticks(), ['' for _ in plt.gca().get_xticks()])
        # plt.xlabel("\n\nTimeline")
        # plt.legend(loc='upper left')
        # plt.tight_layout()
        # plt.xlim([min(timeline), max(timeline)])
        # plt.savefig('./visualization/anomaly_detection/ensemble/%s_%s.png' % (component, metric))
        # plt.show()

        deviations[component][metric] = np.zeros(shape=(len(measurements),))
        for i in range(len(measurements)):
            if measurements[i] > outputs_high[i]:
                deviations[component][metric][i] = (measurements[i] - outputs_high[i]) / outputs_high[i]
            elif measurements[i] < outputs_low[i]:
                deviations[component][metric][i] = (measurements[i] - outputs_low[i]) / outputs_low[i]
        # plt.style.use('default')
        # plt.figure(figsize=(6, 4))
        # heatmap_1d = np.abs(deviations[component][metric])
        # heatmap_1d = heatmap_1d / np.max(heatmap_1d)
        # plt.imshow([heatmap_1d for _ in range(50)], cmap='jet')
        # plt.tight_layout()
        # plt.xticks([], [])
        # plt.yticks([], [])
        # plt.savefig('./visualization/anomaly_detection/ensemble/%s_%s-heatmap.png' % (component, metric))
        # # plt.show()
        # plt.style.use('ggplot')

heatmap = np.asarray([np.abs(deviations[component][metric]) for component in results for metric in results[component]])
# Weighted by the predictability of each component-metric pair
weights = np.asarray([results[component][metric][0] for component in results for metric in results[component]])[:, None]
heatmap = heatmap * weights
heatmap = np.sum(heatmap, axis=0)
# heatmap[315:335] = heatmap[315:335] * 0.3
heatmap = gaussian_filter1d(heatmap, 4)

# plt.style.use('default')
# plt.figure(figsize=(6, 4))
# plt.imshow([heatmap for _ in range(50)], cmap='jet')
# plt.tight_layout()
# plt.xticks([], [])
# plt.yticks([], [])
# plt.savefig('./visualization/anomaly_detection/ensemble/heatmap.png')
# plt.show()
# plt.style.use('ggplot')

peaks, properties = find_peaks(heatmap, prominence=max(heatmap) * 0.20, distance=12)
peak_width_results = peak_widths(heatmap, peaks, rel_height=0.90)
start_indices = peak_width_results[2]
end_indices = peak_width_results[3]
recommendations = []
for j in range(len(start_indices)):
    start_ind = int(np.floor(start_indices[j]))
    end_ind = int(np.ceil(end_indices[j]))
    plt.axvspan(start_ind, end_ind, color='red', alpha=0.2)
    found = False
    recommendations.append([range(start_ind, end_ind+1), np.sum(heatmap[start_ind:end_ind+1])])


recommendations = list(sorted(recommendations, key=lambda recommendation: -recommendation[1]))
for recommendation in recommendations:
    likelihood = float(np.sum([np.abs(deviations[component][metric][list(recommendation[0])]) for component in deviations for metric in deviations[component]]))
    print('Anomalous Event(s) ')
    print('   > Index     : [%d, %d]' % (min(recommendation[0]), max(recommendation[0])))
    print('   > Likelihood: %.4f' % likelihood)
    print('   > Rationale :')
    rationale = []
    for component in deviations:
        percentage = [np.abs(deviations[component][metric][list(recommendation[0])]) for metric in deviations[component]]
        rationale_details = []
        for metric in deviations[component]:
            heatmap_subset = deviations[component][metric][list(recommendation[0])]
            count_high = np.sum(heatmap_subset > 0)
            count_low = np.sum(heatmap_subset < 0)
            if count_high > count_low:
                rationale_details.append((metric, np.max(heatmap_subset)))
            elif count_high < count_low:
                rationale_details.append((metric, np.min(heatmap_subset)))
        rationale_details = list(sorted(rationale_details, key=lambda tup: -abs(tup[1])))
        if len(rationale_details) > 0:
            rationale.append((component, np.sum(percentage) * 100 / likelihood, rationale_details))
    rationale = list(sorted(rationale, key=lambda r: -r[1]))
    for component, percentage, details in rationale:
        print('       => [%.1f%%] %s' % (percentage, component))
        for metric, factor in details:
            print('                  %s: %.2f%% %s than expectation' % (metric, abs(factor) * 100, 'higher' if factor > 0 else 'lower'))
