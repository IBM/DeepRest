from itertools import cycle


def basic(dataset):
    iterator = cycle(dataset)

    def get_next():
        return next(iterator)

    return get_next
