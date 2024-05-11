"""
This will load the dataset and the network and create a confusion matrix
"""
import os
import re
import sys
import time
import pickle

import numpy as np
import pandas as pd
from moabb.datasets import PhysionetMI
from torch.profiler import profile, ProfilerActivity, record_function

import DatasetAugmentation.utils
import EEGClassificator.utils
import VirtualController.main

def classification_time(network, data_points):
    list_dict_time = []
    for sample in data_points:
        sample = np.expand_dims(sample, axis=0)
        assert sample.shape == (1, 58, 65)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, with_flops=True) as prof:
            network.predict(sample)
        start_time = time.time()
        network.predict(sample)
        end_time = time.time()
        list_dict_time.append({
            'time': (end_time-start_time)*1000, # in ms
            'flops': prof.key_averages().total_average().flops,
            'cpu_time': prof.key_averages().total_average().cpu_time_str,
            'cuda_time': prof.key_averages().total_average().cuda_time_str,
            'cpu_memory': prof.key_averages().total_average().cpu_memory_usage, # in bytes
            'cuda_memory': prof.key_averages().total_average().cuda_memory_usage, # in bytes
            'cpu_time_total': prof.key_averages().total_average().cpu_time_total_str,
            'cuda_time_total': prof.key_averages().total_average().cuda_time_total_str,
            'self_cuda_memory_usage': prof.key_averages().total_average().self_cuda_memory_usage, # in bytes
            'self_cpu_memory_usage': prof.key_averages().total_average().self_cpu_memory_usage, # in bytes
            'self_cpu_time_total': prof.key_averages().total_average().self_cpu_time_total_str,
            'self_cuda_time_total': prof.key_averages().total_average().self_cuda_time_total_str
        })
    df = pd.DataFrame(list_dict_time)
    return list_dict_time, df


def compute_classification_time(data_points: list, path_to_network):
    if isinstance(path_to_network, list):
        for file in path_to_network:
            print(f"Recursively running compute confusion matrix for {file}")
            compute_classification_time(data_points, file)
        return
    elif path_to_network.endswith('/*'):
        # enumerate all files in folder and run main for each file
        for file in os.listdir(path_to_network[:-1]):
            print(f"Recursively running compute confusion matrix for {file}")
            compute_classification_time(data_points, path_to_network[:-1] + file)
        return
    if not path_to_network.endswith('.pkl'):
        print(f"Invalid network file {path_to_network}")
        return
    network = VirtualController.load_classificator(path_to_network)
    list_dict_time, data_frame = classification_time(network, data_points)
    average = data_frame.mean()
    min = data_frame.min()
    max = data_frame.max()
    base_path = os.path.dirname(path_to_network)
    with open(f'{base_path}/classification_time.pkl', 'wb') as f:
        pickle.dump(data_frame, f)
    with open(f'{base_path}/classification_time_min.pkl', 'wb') as f:
        pickle.dump(min, f)
    with open(f'{base_path}/classification_time_max.pkl', 'wb') as f:
        pickle.dump(max, f)
    with open(f'{base_path}/classification_time_avg.pkl', 'wb') as f:
        pickle.dump(average, f)
    print(path_to_network)
    print("Classification Time:")
    print(data_frame)
    print("Average:")
    print(average)
    print("Min:")
    print(min)
    print("Max:")
    print(max)


def main(path_to_network):
    # Not interested in overall accuracy or inter-subject classification accuracy,
    # ``ClassificationTester.py'' computes it
    x, _ = DatasetAugmentation.utils.load_dataset(PhysionetMI)
    compute_classification_time(x, path_to_network)



if __name__ == '__main__':
    """
    if len(sys.argv) < 2:
        print("Usage: python Tester.py <path to network>")
        sys.exit(1)
    if len(sys.argv) > 2:
        net_path = sys.argv[1:]
    else:
        net_path = sys.argv[1]
    """
    net_path = '../models/2024_03_25_23_21_59/LSTMNet_0.5943600867678959.pkl'
    main(net_path)