import os
import sys

import numpy as np

from PrettyPrintConfusionMatrix.pretty_confusion_matrix import pp_matrix
from PrettyPrintConfusionMatrix.utils import get_confusion_matrix_df


def main(base_path, filename):
    network_names = filter(lambda x: not x.endswith('.log'), os.listdir(base_path))
    network_names = list(set(map(lambda x: x.split('_')[0]+'_', network_names)))

    path = os.path.join(base_path, filename)
    print("Working with", path, flush=True)

    if not os.path.exists('./figures'):
        os.makedirs('./figures')

    with open(path, 'r') as f:
        lines = f.readlines()

    for network_name in network_names:
        net_name = os.path.join(base_path, network_name)
        net_list = [(lines[idx+1:idx+1+3], lines[idx+4:idx+4+3]) for idx, line in enumerate(lines) if line.startswith(net_name)]
        print(net_name, len(net_list))

        flat_results, percentage_results = zip(*list(map(lambda x: (np.array(np.matrix(';'.join(x[0]))), np.array(np.matrix(';'.join(x[1])))), net_list)))
        flat_results, percentage_results = list(flat_results), list(percentage_results)
        avg_flat_results = np.mean(flat_results, axis=0)
        avg_percentage_results = np.mean(percentage_results, axis=0)
        print(net_name)
        print(avg_flat_results)
        print(avg_percentage_results)

        df = get_confusion_matrix_df(avg_flat_results)
        fig, _ = pp_matrix(df, pred_val_axis='x', show_null_values=2, annot=True, fz=10, lw=0.5, cbar=True, cmap='Blues',
                        title=f'Confusion Matrix {network_name[:-1]}')
        fig.savefig(f'./figures/confusion_matrix_{network_name[:-1]}_flat.png')
        fig.cla()
        fig.clf()
        fig.close()

        df = get_confusion_matrix_df(avg_percentage_results)
        fig, _ = pp_matrix(df, pred_val_axis='x', show_null_values=2, fz=10, lw=0.5, cbar=True, cmap='Blues',
                        title=f'Confusion Matrix {network_name[:-1]}')
        fig.savefig(f'./figures/confusion_matrix_{network_name[:-1]}_percentage.png')
        fig.cla()
        fig.clf()
        fig.close()

if __name__ == '__main__':
    base_path = sys.argv[1]
    filename = sys.argv[2]
    main(base_path, filename)