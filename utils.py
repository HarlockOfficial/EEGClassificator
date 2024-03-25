import numpy as np


def to_categorical(labels: list[str]):
    labels_to_numbers = {
        'feet': 0,
        'left_hand': 1,
        'right_hand': 2,
    }
    return np.array([labels_to_numbers[label] for label in labels])

def from_categorical(labels: list[int]):
    numbers_to_labels = {
        0: 'feet',
        1: 'left_hand',
        2: 'right_hand',
    }
    return [numbers_to_labels[label] for label in labels]