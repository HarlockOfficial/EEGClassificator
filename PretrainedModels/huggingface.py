# https://huggingface.co/PierreGtch/EEGNetv4
import pickle
from collections import OrderedDict
from datetime import datetime

import torch
from huggingface_hub import hf_hub_download
from moabb.evaluations import WithinSessionEvaluation
from moabb.paradigms import MotorImagery
from requests import ReadTimeout
from sklearn.base import TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from skorch import NeuralNet
from skorch.utils import to_numpy
from torch import nn


def remove_clf_layers(model: nn.Sequential):
    """
    Remove the classification layers from braindecode models.
    Tested on EEGNetv4, Deep4Net (i.e. DeepConvNet), and EEGResNet.
    """
    new_layers = []
    for name, layer in model.named_children():
        if 'classif' in name:
            continue
        if 'softmax' in name:
            continue
        new_layers.append((name, layer))
    return nn.Sequential(OrderedDict(new_layers))


def freeze_model(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model

class FrozenNeuralNetTransformer(NeuralNet, TransformerMixin):
    def __init__(
            self,
            *args,
            criterion=nn.MSELoss,  # should be unused
            unique_name=None,  # needed for a unique digest in MOABB
            **kwargs
    ):
        super().__init__(
            *args,
            criterion=criterion,
            **kwargs
        )
        self.initialize()
        self.unique_name = unique_name

    def fit(self, X, y=None, **fit_params):
        return self  # do nothing

    def transform(self, X):
        X = self.infer(X)
        return to_numpy(X)

    def __repr__(self):
        return super().__repr__() + str(self.unique_name)

def flatten_batched(X):
    return X.reshape(X.shape[0], -1)

paradigm = MotorImagery(
    channels=['C3', 'Cz', 'C4'],
    events=['left_hand', 'right_hand', 'feet'],
    n_classes=3,
    fmin=0.5,
    fmax=40,
    tmin=0,
    tmax=3,
    resample=128,
)

path_kwargs = hf_hub_download(
    repo_id='PierreGtch/EEGNetv4',
    filename='EEGNetv4_PhysionetMI/kwargs.pkl',
)
path_params = hf_hub_download(
    repo_id='PierreGtch/EEGNetv4',
    filename='EEGNetv4_PhysionetMI/model-params.pkl',
)
with open(path_kwargs, 'rb') as f:
    kwargs = pickle.load(f)

module_cls = kwargs['module_cls']
module_kwargs = kwargs['module_kwargs']
module_kwargs['n_chans'] = module_kwargs['in_chans']
module_kwargs['n_outputs'] = module_kwargs['n_classes']
module_kwargs['n_times'] = module_kwargs['input_window_samples']
del module_kwargs['in_chans']
del module_kwargs['n_classes']
del module_kwargs['input_window_samples']

# load the model with pre-trained weights:
torch_module = module_cls(**module_kwargs)
torch_module.load_state_dict(torch.load(path_params, map_location='cuda'))

embedding = freeze_model(remove_clf_layers(torch_module)).double()

sklearn_pipeline = make_pipeline(
    FrozenNeuralNetTransformer(embedding),
    FunctionTransformer(flatten_batched),
    LogisticRegression()
)

datasets = []
for dataset in paradigm.datasets:
    if paradigm.is_valid(dataset):
        try:
            dataset.download()
            datasets.append(dataset)
        except ReadTimeout:
            continue

curr_datetime_to_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
evaluation = WithinSessionEvaluation(
    paradigm=paradigm,
    datasets=datasets,
    overwrite=True,
    suffix='demo_' + curr_datetime_to_string,
    random_state=42,
    hdf5_path='./models/'+ curr_datetime_to_string + '/',
)

results = evaluation.process(pipelines=dict(demo_pipeline=sklearn_pipeline))
scores = results['score']
with open('results_'+curr_datetime_to_string+'.pkl', 'wb') as f:
    pickle.dump(results, f)
print("Results",results)
print("Mean",scores.mean())