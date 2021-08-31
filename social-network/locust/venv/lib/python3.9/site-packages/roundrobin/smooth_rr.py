def smooth(dataset):
    dataset_length = len(dataset)
    dataset_extra_weights = [ItemWeight(*x) for x in dataset]

    def get_next():
        if dataset_length == 0:
            return None
        if dataset_length == 1:
            return dataset[0][0]

        total_weight = 0
        result = None
        for extra in dataset_extra_weights:
            extra.current_weight += extra.effective_weight
            total_weight += extra.effective_weight
            if extra.effective_weight < extra.weight:
                extra.effective_weight += 1
            if not result or result.current_weight < extra.current_weight:
                result = extra
        if not result:  # this should be unreachable, but check anyway
            raise RuntimeError
        result.current_weight -= total_weight
        return result.key

    return get_next


class ItemWeight:

    __slots__ = ('key', 'weight', 'current_weight', 'effective_weight')

    def __init__(self, key, weight):
        self.key = key
        self.weight = weight
        self.current_weight = 0
        self.effective_weight = weight
