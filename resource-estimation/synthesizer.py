import numpy as np
import pickle
import copy

########################################################################################################################
path_to_raw_data = './raw_data.pkl'  # The raw data following the format specified in the README.md
########################################################################################################################


class TraceSynthesizer:
    def __init__(self):
        self.M = None
        self.api2dist = None

    def fit(self, data):
        # Construct the feature space
        M = {}
        for bucket in data:
            M = TraceSynthesizer.construct_feature_space(M, bucket['traces'])
        api2dist = {}
        for m in M:
            _m = eval(m)
            if len(_m) == 1:
                _m = _m[0]
                api2dist[_m] = {}

        # Learn probability
        for bucket in data:
            for trace in bucket['traces']:
                api_endpoint = '%s_%s' % (trace['component'], trace['operation'])
                feature_vector = TraceSynthesizer.extract_feature(M, [trace])
                if str(feature_vector) not in api2dist[api_endpoint]:
                    api2dist[api_endpoint][str(feature_vector)] = 0
                api2dist[api_endpoint][str(feature_vector)] += 1
        for api, dist in api2dist.items():
            candidates = list(dist.keys())
            weights = [dist[candidate] for candidate in candidates]
            api2dist[api] = (candidates, weights)
        self.M = M
        self.api2dist = api2dist
        return self

    def synthesize(self, expected_api_calls):
        for api in expected_api_calls:
            assert api in self.api2dist, 'API endpoint `%s` does not exist.' % api
        x = np.zeros(shape=(len(self.M),), dtype=np.int)
        for api, count in expected_api_calls.items():
            candidates, weights = self.api2dist[api]
            for fv in np.random.choice(candidates, size=count, replace=True):
                fv = np.asarray(eval(fv))
                x = x + fv
        return x

    @staticmethod
    def traverse_construct(M, node, prefix):
        prefix = copy.deepcopy(prefix)
        prefix.append(node['component'] + '_' + node['operation'])
        if str(prefix) not in M:
            M[str(prefix)] = len(M)
        for child in node['children']:
            M = TraceSynthesizer.traverse_construct(M, child, prefix)
        return M

    @staticmethod
    def construct_feature_space(M, T):
        for trace in T:
            M = TraceSynthesizer.traverse_construct(M, trace, [])
        return M

    @staticmethod
    def traverse_extract(x, node, prefix, M):
        prefix = copy.deepcopy(prefix)
        prefix.append(node['component'] + '_' + node['operation'])
        x[M[str(prefix)]] += 1
        for child in node['children']:
            x = TraceSynthesizer.traverse_extract(x, child, prefix, M)
        return x

    @staticmethod
    def extract_feature(M, T):
        x = [0 for _ in range(len(M))]
        for trace in T:
            x = TraceSynthesizer.traverse_extract(x, trace, [], M)
        return x


if __name__ == '__main__':
    # 1. Load raw data from disk
    with open(path_to_raw_data, 'rb') as f:
        raw_data = pickle.load(f)

    # 2. Train the trace synthesizer
    synthesizer = TraceSynthesizer().fit(raw_data)

    # 3. Synthesize example
    # Print all API endpoints found in the data
    print('%d API endpoints are found:' % len(synthesizer.api2dist))
    for api in synthesizer.api2dist:
        print('    > %s' % api)

    # To use the synthesizer, prepare a list of dictionary specifying the number of expected API calls. The t-th entry
    # in the list refers to the expected API calls at the t-th time step.
    # Example:
    # expected_api_traffic = [
    #     {'/compose': 3, '/read': 2},
    #     {'/compose': 0, '/read': 3},
    #     {'/compose': 2, '/read': 7}
    # ]
    # At the 1st time step, we expect 3 `/compose` API calls and 2 `/read` API calls.
    # At the 2nd time step, we expect no `/compose` API calls and 3 `/read` API calls.
    # At the 3rd time step, we expect 2 `/compose` API calls and 7 `/read` API calls.
    expected_api_traffic = [
        {'API_ENDPOINT_1': 3, 'API_ENDPOINT_2': 2},
        {'API_ENDPOINT_1': 0, 'API_ENDPOINT_2': 3},
        {'API_ENDPOINT_1': 2, 'API_ENDPOINT_2': 7}
    ]
    for t, calls in enumerate(expected_api_traffic):
        print('Expected API calls at time %d: %s' % (t, calls))
        print('               Feature vector: %s' % str(synthesizer.synthesize(calls)))
