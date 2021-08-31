try:
    from math import gcd
except ImportError:
    from fractions import gcd


# python2 workaround for python3 nonlocal keyword
class Store:
    __slots__ = ('index', 'weight')

    def __init__(self, index, weight):
        self.index = index
        self.weight = weight


def weighted(dataset):
    current = Store(index=-1, weight=0)

    dataset_length = len(dataset)
    dataset_max_weight = 0
    dataset_gcd_weight = 0

    for _, weight in dataset:
        if dataset_max_weight < weight:
            dataset_max_weight = weight
        dataset_gcd_weight = gcd(dataset_gcd_weight, weight)

    def get_next():
        while True:
            current.index = (current.index + 1) % dataset_length
            if current.index == 0:
                current.weight = current.weight - dataset_gcd_weight
                if current.weight <= 0:
                    current.weight = dataset_max_weight
                    if current.weight == 0:
                        return None
            if dataset[current.index][1] >= current.weight:
                return dataset[current.index][0]

    return get_next
