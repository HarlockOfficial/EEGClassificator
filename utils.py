import numpy as np


def to_categorical(labels: list[str]|str):
    labels_to_numbers = {
        'feet': 0,
        'left_hand': 1,
        'right_hand': 2,
    }
    if isinstance(labels, str):
        return labels_to_numbers[labels]
    return np.array([labels_to_numbers[label] for label in labels])

def from_categorical(labels: list[int]|int):
    numbers_to_labels = {
        0: 'feet',
        1: 'left_hand',
        2: 'right_hand',
    }
    if isinstance(labels, int):
        return numbers_to_labels[labels]
    return [numbers_to_labels[label] for label in labels]
