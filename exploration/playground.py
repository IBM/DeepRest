import numpy as np
import pickle
import plotly.graph_objects as go


with open('story_results.pkl', 'rb') as f:
    results = pickle.load(f)

results = results['composePost_uploadMedia_readUserTimeline-story']

components = ['nginx-thrift', 'compose-post-service', 'post-storage-service', 'post-storage-mongodb', 'user-timeline-service', 'user-timeline-mongodb', 'media-frontend', 'media-mongodb']

print(components)
z = []
for component in components:
    metrics = ['cpu', 'memory', 'write-iops', 'write-tp', 'usage']
    print(component)
    row = []
    for metric in metrics:
        if metric in results[component]:
            measurements = results[component][metric]['measurement'][-60:]
            ours = results[component][metric]['prediction_ours'][-60:]
            error = np.sum(abs(np.asarray(measurements) - np.asarray(ours)))
            row.append(error)
        elif component + '-pvc' in results and metric in results[component + '-pvc']:
            measurements = results[component + '-pvc'][metric]['measurement'][-60:]
            ours = results[component + '-pvc'][metric]['prediction_ours'][-60:]
            error = np.sum(abs(np.asarray(measurements) - np.asarray(ours)))
            row.append(error)
        else:
            row.append(None)
    z.append(row)


fig = go.Figure(data=go.Heatmap(z=z, x=metrics, y=components, hoverongaps = False, colorscale = 'brwnyl'))
fig.show()

