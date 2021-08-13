from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
from mpl_toolkits import mplot3d
import numpy as np
import torch
import os
plt.style.use('ggplot')

model_root = './models/'

component2models = {}

for dataset in os.listdir(model_root):
    if 'compose' in dataset or 'mongodb' in dataset:

        for fname in os.listdir(os.path.join(model_root, dataset)):
            fpath = os.path.join(model_root, dataset, fname)
            M = torch.load(fpath)
            print(M.keys())
            fname = fname.replace('composePost_uploadMedia_readUserTimeline-ransomware-', '')

            component, metric = fname.replace('.pth', '').split('_')
            if component.replace('-pvc', '') not in component2models:
                component2models[component.replace('-pvc', '')] = []
            component2models[component.replace('-pvc', '')].append(np.concatenate([v.numpy().flatten() for k, v in M.items() if 'rnn' in k]))


component2models = {component: np.asarray(models) for component, models in component2models.items()}

mnames = list(component2models.keys())
mparam = list([np.mean(component2models[component], axis=0) for component in mnames])
M = np.asarray(mparam)
M_2d = PCA(n_components=2).fit_transform(M)
plt.scatter(M_2d[:, 0], M_2d[:, 1])
for i, txt in enumerate(mnames):
    plt.gca().annotate(txt, (M_2d[i, 0], M_2d[i, 1]))
plt.tight_layout()
plt.show()

