"""
This will load the dataset and the network and create a confusion matrix
"""
import os
import re
import sys

import numpy as np
from moabb.datasets import PhysionetMI

import DatasetAugmentation.utils
import EEGClassificator.utils
import VirtualController.main

def create_confusion_matrix(network, dataset_by_label):
    keys_count = len(dataset_by_label.keys())
    confusion_matrix = np.zeros(shape=(keys_count, keys_count), dtype=int)
    for label, dataset in dataset_by_label.items():
        label = EEGClassificator.utils.to_categorical(label)
        for sample in dataset:
            sample = np.expand_dims(sample, axis=0)
            assert sample.shape == (1, 58, 65)
            prediction = network.predict(sample)[0]
            if type(prediction) == np.ndarray:
                prediction = prediction.argmax()
            confusion_matrix[label][prediction] += 1
    percentage_confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)
    return confusion_matrix, percentage_confusion_matrix

def compute_confusion_matrix_single_subject(dataset_list_by_label: list[dict], path_to_network):
    if isinstance(path_to_network, list):
        for file in path_to_network:
            print(f"Recursively running compute confusion matrix for {file}")
            compute_confusion_matrix_single_subject(dataset_list_by_label, file)
        return
    elif path_to_network.endswith('/*'):
        # enumerate all files in folder and run main for each file
        for file in os.listdir(path_to_network[:-1]):
            print(f"Recursively running compute confusion matrix for {file}")
            compute_confusion_matrix_single_subject(dataset_list_by_label, path_to_network[:-1] + file)
        return
    if not path_to_network.endswith('.pkl'):
        print(f"Invalid network file {path_to_network}")
        return
    network = VirtualController.load_classificator(path_to_network)
    net_name = os.path.basename(path_to_network)
    if re.match(r'[a-zA-Z0-9\+]*_[0-9]+_[0-9]+_[0-9]+\.[0-9]+\.pkl', net_name):
        subject_index = int(re.search(r'_\d+_(\d+)_\d+\.\d+\.pkl', net_name).group(1))
        confusion_matrix, percentage_confusion_matrix = create_confusion_matrix(network, dataset_list_by_label[subject_index])
    else:
        all_subjects_dict = dict()
        for subject in dataset_list_by_label:
            for label, dataset in subject.items():
                if label not in all_subjects_dict.keys():
                    all_subjects_dict[label] = []
                all_subjects_dict[label].extend(dataset)
        confusion_matrix, percentage_confusion_matrix = create_confusion_matrix(network, all_subjects_dict)
    print(path_to_network, '\n' , confusion_matrix, '\n', percentage_confusion_matrix, flush=True)

def compute_confusion_matrix(dataset_by_label: dict, path_to_network, local_create_confusion_matrix=create_confusion_matrix):
    if isinstance(path_to_network, list):
        for file in path_to_network:
            print(f"Recursively running compute confusion matrix for {file}")
            compute_confusion_matrix(dataset_by_label, file, local_create_confusion_matrix)
        return
    elif path_to_network.endswith('/*'):
        # enumerate all files in folder and run main for each file
        for file in os.listdir(path_to_network[:-1]):
            print(f"Recursively running compute confusion matrix for {file}")
            compute_confusion_matrix(dataset_by_label, path_to_network[:-1] + file, local_create_confusion_matrix)
        return
    if not path_to_network.endswith('.pkl'):
        print(f"Invalid network file {path_to_network}")
        return
    network = VirtualController.load_classificator(path_to_network)
    confusion_matrix, percentage_confusion_matrix = local_create_confusion_matrix(network, dataset_by_label)
    print(path_to_network, '\n' , confusion_matrix, '\n', percentage_confusion_matrix, flush=True)


def main(path_to_network):
    """
    dataset_list_by_label = []
    for i in range(1, 110):
        x, y = DatasetAugmentation.utils.load_dataset(PhysionetMI, subject_id=[i], events=['left_hand', 'right_hand', 'feet', 'rest'])
        dataset_by_label = DatasetAugmentation.utils.split_dataset_by_label(x, y)
        dataset_list_by_label.append(dataset_by_label)
        compute_confusion_matrix_single_subject(dataset_list_by_label, path_to_network)
    """
    x, y = DatasetAugmentation.utils.load_dataset(PhysionetMI, events=['left_hand', 'right_hand', 'feet', 'rest'])
    dataset_by_label = DatasetAugmentation.utils.split_dataset_by_label(x, y)
    compute_confusion_matrix(dataset_by_label, path_to_network)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python Tester.py <path to network>")
        sys.exit(1)
    if len(sys.argv) > 2:
        net_path = sys.argv[1:]
    else:
        net_path = sys.argv[1]
    main(net_path)