"""
This will load the dataset and the network and create a confusion matrix
"""
import os
import sys

import numpy as np
import torch

import VirtualController.main


def create_confusion_matrix(network, gans, sample_count=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    keys_count = len(gans.keys())
    confusion_matrix = np.zeros(shape=(keys_count, keys_count), dtype=int)
    for label, gan in gans.items():
        true_label = int(label.split('_')[0])
        noise = torch.rand((sample_count, 58, 65)).to(device).to(torch.float32).unsqueeze(1)
        sample, _ = gan(noise)
        sample = sample.detach().cpu().numpy().squeeze(1)
        predictions = network.predict(sample)
        for prediction in predictions:
            if type(prediction) == np.ndarray:
                prediction = prediction.argmax()
            confusion_matrix[true_label][prediction] += 1
    percentage_confusion_matrix = confusion_matrix / np.sum(confusion_matrix, axis=1)
    return confusion_matrix, percentage_confusion_matrix

def compute_confusion_matrix(gans: dict, path_to_network, local_create_confusion_matrix=create_confusion_matrix):
    if isinstance(path_to_network, list):
        for file in path_to_network:
            print(f"Recursively running compute confusion matrix for {file}")
            compute_confusion_matrix(gans, file, local_create_confusion_matrix)
        return
    elif path_to_network.endswith('/*'):
        # enumerate all files in folder and run main for each file
        for file in os.listdir(path_to_network[:-1]):
            print(f"Recursively running compute confusion matrix for {file}")
            compute_confusion_matrix(gans, path_to_network[:-1] + file, local_create_confusion_matrix)
        return
    if not path_to_network.endswith('.pkl'):
        print(f"Invalid network file {path_to_network}")
        return
    network = VirtualController.load_classificator(path_to_network)
    confusion_matrix, percentage_confusion_matrix = local_create_confusion_matrix(network, gans)
    print(path_to_network, '\n' , confusion_matrix, '\n', percentage_confusion_matrix, flush=True)


def load_gans(gans_folder_path: str):
    gans = dict()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for file in os.listdir(gans_folder_path):
        if not file.endswith('.pkl') or not 'generator' in file:
            continue
        label = file.split('/')[-1].split('pkl')[0]
        gans[label] = VirtualController.load_classificator(gans_folder_path + '/' + file)
        gans[label].eval()
        gans[label].to(device)
    return gans


def main(gans_folder, path_to_network):
    gans = load_gans(gans_folder)
    compute_confusion_matrix(gans, path_to_network)



if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python Tester.py <path to gans folder> <path to network>")
        sys.exit(1)
    gans_folder = sys.argv[1]
    if len(sys.argv) > 3:
        net_path = sys.argv[2:]
    else:
        net_path = sys.argv[2]
    main(gans_folder, net_path)