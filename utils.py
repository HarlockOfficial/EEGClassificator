import numpy as np


def to_categorical(labels: list[str]|str, np_2d_array: bool = False):
    labels_to_numbers = {
        'feet': 0,
        'left_hand': 1,
        'right_hand': 2,
        'rest': 3
    }
    if np_2d_array:
        if isinstance(labels, str):
            index = labels_to_numbers[labels]
            np.array([0 if i != index else 1 for i in range(4)])
        return np.array([[0 if i != labels_to_numbers[label] else 1 for i in range(4)] for label in labels])

    if isinstance(labels, str):
        return labels_to_numbers[labels]
    return np.array([labels_to_numbers[label] for label in labels])

def from_categorical(labels: list[int]|int):
    numbers_to_labels = {
        0: 'feet',
        1: 'left_hand',
        2: 'right_hand',
        3: 'rest'
    }
    if isinstance(labels, int):
        return numbers_to_labels[labels]
    return [numbers_to_labels[label] for label in labels]
