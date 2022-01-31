from baselines import ResourceAware, ComponentAware
from torch.utils.data import TensorDataset
from utils import sliding_window
from qrnn import QuantileRNN
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
plt.style.use('ggplot')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########################################################################################################################
path_to_input = './input.pkl'     # The formatted data generated by featurize.py
num_epochs    = 50                # Number of training epochs
batch_size    = 32                # Batch size
learning_rate = 0.001             # Learning rate of the optimizer
split         = 0.40              # The first split*100% of the time-series will be used for application learning
step_size     = 60                # The window size for the Quantile RNN and visualization
########################################################################################################################

if __name__ == '__main__':
    with open(path_to_input, 'rb') as f:
        traffic, resources, invocations = pickle.load(f)

    names = list(resources.keys())
    X = sliding_window(traffic, step_size)
    y = sliding_window(np.concatenate([resources[name] for name in names], axis=-1), step_size)
    split = int(len(X) * split)

    # Baseline methods to estimate resources
    y_test_resrc, y_test_comp = [], []
    for idx, name in enumerate(names):
        component, metric = name.split('_')
        y_test_resrc.append(ResourceAware(split=split, output_size=step_size, offset=step_size-1,
                                          input_size=step_size).fit_and_estimate(X, y[:, :, [idx]]))
        y_test_comp.append(ComponentAware(split=split, output_size=step_size, component=component, metric=metric,
                                          invocation=invocations).fit_and_estimate(X, y[:, :, [idx]]))
    y_test_resrc = np.concatenate(y_test_resrc, axis=-1)
    y_test_comp = np.concatenate(y_test_comp, axis=-1)

    # Data normalization
    X, _, _ = QuantileRNN.normalization_minmax(X, split=split)
    scales = []
    for idx in range(len(names)):
        y_, min_, max_ = QuantileRNN.normalization_minmax(y[:, :, [idx]], split=split)
        y[:, :, [idx]] = y_
        scales.append((max_ - min_, min_))

    # Prepare training and test sets
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    assert y_test.shape == y_test_resrc.shape == y_test_comp.shape, 'y_test\'s sizes do not match.'

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # Initialize Quantile RNN and its optimizer
    model = QuantileRNN(input_size=X.shape[-1], num_metrics=len(names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Begin model training
    xs_train, ys_train, xs_test, ys_test = [], [], [], []
    for epoch in range(num_epochs):
        loss_buffer = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = model.quantile_loss(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_buffer.append(loss.item())
        xs_train.append(epoch)
        ys_train.append(np.mean(loss_buffer))

        # Stage 3: Testing
        yerr_deeprest, yerr_resrc, yerr_comp = [[] for _ in names], [[] for _ in names], [[] for _ in names]
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
                loss = model.quantile_loss(outputs, labels)
                loss_buffer.append(loss.item())

                labels = labels.detach().cpu().numpy()[0]
                outputs_deeprest = np.maximum(outputs.detach().cpu().numpy(), 1e-6)[0]
                outputs_resrc = y_test_resrc[iv]
                outputs_comp = y_test_comp[iv]

                for idx in range(len(names)):
                    labels_ = labels[:, idx] * scales[idx][0] + scales[idx][1]
                    outputs_deeprest_ = outputs_deeprest[:, idx, 1] * scales[idx][0] + scales[idx][1]
                    outputs_resrc_ = outputs_resrc[:, idx]
                    outputs_comp_ = outputs_comp[:, idx]
                    yerr_deeprest[idx] += list(np.abs(outputs_deeprest_ - labels_))
                    yerr_resrc[idx] += list(np.abs(outputs_resrc_ - labels_))
                    yerr_comp[idx] += list(np.abs(outputs_comp_ - labels_))

            xs_test.append(epoch)
            ys_test.append(np.mean(loss_buffer))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {ys_train[-1]:.6f}, Test Loss: {ys_test[-1]:.6f}')
        for idx, name in enumerate(names):
            print('===== %s =====' % name)
            print('   RESRC => Median: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
                float(np.median(yerr_resrc[idx])), np.percentile(yerr_resrc[idx], q=95),
                np.percentile(yerr_resrc[idx], q=99), np.max(yerr_resrc[idx])))
            print('   COMP  => Median: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
                float(np.median(yerr_comp[idx])), np.percentile(yerr_comp[idx], q=95),
                np.percentile(yerr_comp[idx], q=99), np.max(yerr_comp[idx])))
            print('   DEEPR => Median: %.4f | 95-th: %.4f | 99-th: %.4f | Max: %.4f' % (
                float(np.median(yerr_deeprest[idx])), np.percentile(yerr_deeprest[idx], q=95),
                np.percentile(yerr_deeprest[idx], q=99), np.max(yerr_deeprest[idx])))
            model.train()

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

    # Begin model testing with visualization
    model.eval()
    with torch.no_grad():
        num_cycles = 0
        for iv, (inputs, labels) in enumerate(test_loader):
            if iv % step_size != 0 or num_cycles >= 9:
                continue
            num_cycles += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

            labels = labels.detach().cpu().numpy()[0]
            outputs_deeprest = outputs.detach().cpu().numpy()[0]
            outputs_resrc = y_test_resrc[iv]
            outputs_comp = y_test_comp[iv]

            plt.clf()
            plt.figure(figsize=(len(names) * 6, 6))
            for idx in range(len(names)):
                plt.subplot(1, len(names), idx+1)
                labels_ = labels[:, idx] * scales[idx][0] + scales[idx][1]
                outputs_deeprest_ = outputs_deeprest[:, idx, 1] * scales[idx][0] + scales[idx][1]
                outputs_resrc_ = outputs_resrc[:, idx]
                outputs_comp_ = outputs_comp[:, idx]

                plt.title(names[idx])
                plt.plot(labels_, label='Ground Truth', linestyle='--', color='red')
                plt.plot(outputs_resrc_, label='Baseline: Resrc-aware ANN', color='green')
                plt.plot(outputs_comp_, label='Baseline: Req-aware LinearRegr', color='orange')
                plt.plot(outputs_deeprest_, label='DeepRest', color='mediumblue')
                plt.tight_layout()
            plt.show()
            plt.close()
