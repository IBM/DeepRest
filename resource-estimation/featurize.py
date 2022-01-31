import numpy as np
import pickle
import copy

########################################################################################################################
path_to_raw_data = './raw_data.pkl'  # The raw data following the format specified in the README.md
path_to_save     = './input.pkl'     # The location to save the formatted data to be used by estimate.py
########################################################################################################################


def traverse_construct(M, node, prefix):
    prefix = copy.deepcopy(prefix)
    prefix.append(node['component'] + '_' + node['operation'])
    if str(prefix) not in M:
        M[str(prefix)] = len(M)
    for child in node['children']:
        M = traverse_construct(M, child, prefix)
    return M


def construct_feature_space(M, T):
    for trace in T:
        M = traverse_construct(M, trace, [])
    return M


def traverse_extract(x, node, prefix, M):
    prefix = copy.deepcopy(prefix)
    prefix.append(node['component'] + '_' + node['operation'])
    x[M[str(prefix)]] += 1
    for child in node['children']:
        x = traverse_extract(x, child, prefix, M)
    return x


def extract_feature(M, T):
    x = [0 for _ in range(len(M))]
    for trace in T:
        x = traverse_extract(x, trace, [], M)
    return x


def traverse_count(c, node):
    if node['component'] not in c:
        c[node['component']] = 0
    c[node['component']] += 1
    for child in node['children']:
        c = traverse_count(c, child)
    return c


def count_invocations(T):
    c = {'general': 0}
    for trace in T:
        c['general'] += 1
        c = traverse_count(c, trace)
    return c


if __name__ == '__main__':
    ####################################################################################################################
    # 0. Load raw data from disk
    with open(path_to_raw_data, 'rb') as f:
        raw_data = pickle.load(f)

    ####################################################################################################################
    # 1. Format resources (the outputs)
    resources = {}
    for bucket in raw_data:
        for metric in bucket['metrics']:
            identifier = '%s_%s' % (metric['component'], metric['resource'])
            if identifier not in resources:
                resources[identifier] = []
            resources[identifier].append(metric['value'])
    resources = {k: np.asarray(v) for k, v in resources.items()}

    ####################################################################################################################
    # 2. Format distributed traces (the inputs)
    # Construct the feature space
    M = {}
    for bucket in raw_data:
        M = construct_feature_space(M, bucket['traces'])
    # Extract features
    traffic = np.asarray([extract_feature(M, bucket['traces']) for bucket in raw_data])

    ####################################################################################################################
    # 3. Format component-based invocation counts (for the baseline)
    # Find all components
    components = set()
    for m in M:
        for identifier in eval(m):
            component, _ = identifier.split('_')
            components.add(component)

    # Count invocations
    invocations = {component: [] for component in components.union({'general'})}
    for bucket in raw_data:
        c = count_invocations(bucket['traces'])
        for component in invocations:
            invocations[component].append(c[component] if component in c else 0)
    invocations = {k: np.asarray(v) for k, v in invocations.items()}

    ####################################################################################################################
    # 4. Save the formatted data for estimate.py
    with open(path_to_save, 'wb') as f:
        pickle.dump([traffic, resources, invocations], f)
