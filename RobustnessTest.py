import sys

import numpy as np
import torch
from moabb.datasets import PhysionetMI

import DatasetAugmentation.utils
import EEGClassificator.ClassificationAccuracyTesterGAN
import EEGClassificator.ClassificationAccuracyTester
import EEGClassificator.utils

def create_confusion_matrix_gan(network, gans, sample_count=1000, noise_level=None):
    if noise_level is None:
        noise_level = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    keys_count = len(gans.keys())
    confusion_matrix = np.zeros(shape=(keys_count, keys_count), dtype=int)
    for label, gan in gans.items():
        true_label = int(label.split('_')[0])
        noise = torch.normal(mean=0, std=noise_level, size=(sample_count, 58, 65)).to(device).to(torch.float32).unsqueeze(1)
        sample, _ = gan(noise)
        sample = sample.detach().cpu().numpy().squeeze(1)
        predictions = network.predict(sample)
        for prediction in predictions:
            if type(prediction) == np.ndarray:
                prediction = prediction.argmax()
            confusion_matrix[true_label][prediction] += 1

    percentage_confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)
    return confusion_matrix, percentage_confusion_matrix

def create_confusion_matrix(network, dataset_by_label, noise_level=None):
    keys_count = len(dataset_by_label.keys())
    confusion_matrix = np.zeros(shape=(keys_count, keys_count), dtype=int)
    for label, dataset in dataset_by_label.items():
        label = EEGClassificator.utils.to_categorical(label)
        noise = np.random.normal(0, noise_level, (len(dataset), 58, 65)).astype(np.float32)
        dataset += noise
        predictions = network.predict(dataset)
        for prediction in predictions:
            if type(prediction) == np.ndarray:
                prediction = prediction.argmax()
            confusion_matrix[label][prediction] += 1
    percentage_confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)
    return confusion_matrix, percentage_confusion_matrix


def main(path_to_gan, path_to_networks):
    gans = EEGClassificator.ClassificationAccuracyTesterGAN.load_gans(path_to_gan)
    x, y = DatasetAugmentation.utils.load_dataset(PhysionetMI, events=['left_hand', 'right_hand', 'feet', 'rest'])
    dataset_by_label = DatasetAugmentation.utils.split_dataset_by_label(x, y)
    del x, y
    noise_levels = np.arange(0, 10, 0.1)
    for noise_level in noise_levels:
        print(f"Running for noise level {noise_level}")
        print("GANs data")
        tmp_gan = lambda n, g: create_confusion_matrix_gan(n, g, noise_level=noise_level)
        EEGClassificator.ClassificationAccuracyTesterGAN.compute_confusion_matrix(gans, path_to_networks, local_create_confusion_matrix=tmp_gan)
        print("Dataset data")
        tmp_dataset = lambda n, d: create_confusion_matrix(n, d, noise_level=noise_level)
        EEGClassificator.ClassificationAccuracyTester.compute_confusion_matrix(dataset_by_label, path_to_networks, local_create_confusion_matrix=tmp_dataset)

if __name__ == '__main__':
    path_to_gan = sys.argv[1]
    path_to_networks = sys.argv[2:]
    main(path_to_gan, path_to_networks)