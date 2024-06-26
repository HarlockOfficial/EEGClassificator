import copy
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import torch
from braindecode.models import EEGNetv4
from mne.decoding import CSP
from moabb.datasets import PhysionetMI
from moabb.paradigms import MotorImagery
from pyriemann.classification import MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVC

import DatasetAugmentation.utils
import EEGClassificator.utils
from EEGClassificator.NetworkArchitectures import TransformerClassifier, LSTMBasedArchitecture, MLPArchitecture


class NeuralNetTransformer(TransformerMixin, BaseEstimator):
    def __init__(self, model, optimizer=None, loss_fn=None, train_step=1000, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if str(type(model)) == "<class 'docstring_inheritance.NumpyDocstringInheritanceInitMeta'>":
            self.model = model(**kwargs)
            if optimizer is None:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
            self.optimizer = optimizer
            if loss_fn is None:
                loss_fn = torch.nn.CrossEntropyLoss()
            self.loss_fn = loss_fn
        else:
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn
        self.train_step = train_step
        self.model.to(self.device)
        self.__getattr__ = self.getattr__
    def fit(self, X, y=None):
        self.model.train(True)
        t = torch.FloatTensor if self.device == torch.device("cpu") else torch.cuda.FloatTensor
        X = torch.from_numpy(X).type(t)

        y = torch.as_tensor(y, device=self.device, dtype=torch.float)

        for step in range(self.train_step):
            print(f"Step {step+1} of {self.train_step}", flush=True)
            start_time = datetime.now()
            out = self.model(X)
            loss = self.loss_fn(out, y)
            accuracy = (out.argmax(dim=1) == y.argmax(dim=1)).float().mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            print(f"Loss: {loss} Train Accuracy: {accuracy} Run Time: {datetime.now() - start_time}", flush=((step+1) % 50 == 0))
        return self

    def transform(self, X):
        return self.predict(X)

    def score(self, X, y):
        self.model.eval()
        t = torch.FloatTensor if self.device == torch.device("cpu") else torch.cuda.FloatTensor
        X = torch.from_numpy(X).type(t)
        y = torch.as_tensor(y, device=self.device, dtype=torch.float)
        out = self.model(X)
        loss = self.loss_fn(out, y)
        accuracy = (out.argmax(dim=1) == y.argmax(dim=1)).float().mean()
        print(f"Validation Loss: {loss} Validation Accuracy: {accuracy}", flush=True)
        return accuracy

    def predict(self, X):
        self.model.to(self.device)
        self.model.eval()
        t = torch.FloatTensor if self.device == torch.device("cpu") else torch.cuda.FloatTensor
        X = torch.from_numpy(X).type(t)
        out = self.model(X)
        return out.detach().cpu().numpy()
    
    def getattr__(self, item):
        return getattr(self.model, item)


def flatten_batched(X):
    return X.reshape(X.shape[0], -1)


OUTPUT_CLASSES = 4
SECOND_DURATION = 0.5  # seconds


def main(path_to_models=None):
    # The following code was used to determine the suitable channels, that have been hardcoded in the ALL_EEG_CHANNELS list
    # The single dataset channel list was extracted during debugging,
    # and the intersection of both lists was used to determine the suitable channels
    """
    PhysionetChannels = {'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5',
                         'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
                         'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7', 'T8', 'T9', 'T10', 'TP7',
                         'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                         'O1', 'Oz', 'O2', 'Iz'}

    WeiboChannels = {'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                     'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
                     'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2',
                     'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2',
                     'VEO', 'HEO', 'STIM014'}

    # Intersection of All with Physionet and Weibo
    ALL_EEG_CHANNELS = list(PhysionetChannels.intersection(WeiboChannels))
    """
    # print("Using the ", DatasetAugmentation.utils.INPUT_CHANNELS, "Channels:", DatasetAugmentation.utils.ALL_EEG_CHANNELS)

    paradigm = MotorImagery(channels=DatasetAugmentation.utils.ALL_EEG_CHANNELS, events=['left_hand', 'right_hand', 'feet', 'rest'],
        n_classes=OUTPUT_CLASSES, fmin=0.5, fmax=40, tmin=0, tmax=SECOND_DURATION, resample=DatasetAugmentation.utils.SAMPLE_RATE)
    if path_to_models is None:
        eegnetv4model_for_pipe = NeuralNetTransformer(EEGNetv4, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS, n_outputs=OUTPUT_CLASSES,
                                             n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION))
        eegnetv4model = NeuralNetTransformer(EEGNetv4, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS, n_outputs=OUTPUT_CLASSES,
                                                n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION))
        pureEegnetv4model = NeuralNetTransformer(EEGNetv4, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS,
                                             n_outputs=OUTPUT_CLASSES,
                                             n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION))
        lstmnetmodel_for_pipe = NeuralNetTransformer(LSTMBasedArchitecture, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS, n_outputs=OUTPUT_CLASSES,
                                            n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION))
        lstmnetmodel = NeuralNetTransformer(LSTMBasedArchitecture, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS,
                                            n_outputs=OUTPUT_CLASSES,
                                            n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION))
        purelstmnetmodel = NeuralNetTransformer(LSTMBasedArchitecture, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS,
                                            n_outputs=OUTPUT_CLASSES,
                                            n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION))
        transformermodel_for_pipe = NeuralNetTransformer(TransformerClassifier, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS,
                                                n_outputs=OUTPUT_CLASSES, n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION), num_layers=4)
        transformermodel = NeuralNetTransformer(TransformerClassifier, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS,
                                                n_outputs=OUTPUT_CLASSES, n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION), num_layers=4)
        puretransformermodel = NeuralNetTransformer(TransformerClassifier, n_chans=DatasetAugmentation.utils.INPUT_CHANNELS,
                                                n_outputs=OUTPUT_CLASSES, n_times=int(DatasetAugmentation.utils.SAMPLE_RATE * SECOND_DURATION), num_layers=4)
        # print("Warning: Overriding Num Layers in TransformerClassifier to 2", flush=True)

        # mlpnetmodel_for_pipe = NeuralNetTransformer(MLPArchitecture, n_chans=INPUT_CHANNELS, n_outputs=OUTPUT_CLASSES,
        #                                       n_times=int(SAMPLE_RATE * SECOND_DURATION))
        # mlpnetmodel = NeuralNetTransformer(MLPArchitecture, n_chans=INPUT_CHANNELS, n_outputs=OUTPUT_CLASSES,
        #                                         n_times=int(SAMPLE_RATE * SECOND_DURATION))

        pipelines = dict()
        pipelines["csp+lda"] = make_pipeline(CSP(n_components=8), LinearDiscriminantAnalysis())
        pipelines["tgsp+svm"] = make_pipeline(Covariances("oas"), TangentSpace(metric="riemann"), SVC(kernel="linear"))
        pipelines["MDM"] = make_pipeline(Covariances("oas"), MDM(metric="riemann"))
        # pipelines["MLPNetPipe"] = Pipeline([('net', mlpnetmodel_for_pipe), ('flatten', FunctionTransformer(flatten_batched)),
        #     ('logistic_regression', LogisticRegression())])
        # pipelines["MLPNet"] = Pipeline([('net', mlpnetmodel), ('logistic_regression', LogisticRegression())])
        pipelines["EEGNetV4Pipe"] = Pipeline([('net', eegnetv4model_for_pipe), ('flatten', FunctionTransformer(flatten_batched)),
            ('logistic_regression', LogisticRegression())])
        pipelines["EEGNetV4"] = Pipeline([('net', eegnetv4model), ('logistic_regression', LogisticRegression())])
        pipelines["LSTMNetPipe"] = Pipeline([('net', lstmnetmodel_for_pipe), ('flatten', FunctionTransformer(flatten_batched)),
            ('logistic_regression', LogisticRegression())])
        pipelines["LSTMNet"] = Pipeline([('net', lstmnetmodel), ('logistic_regression', LogisticRegression())])
        pipelines["TransformerNetPipe"] = Pipeline([('net', transformermodel_for_pipe), ('flatten', FunctionTransformer(flatten_batched)),
                ('logistic_regression', LogisticRegression())])
        pipelines["TransformerNet"] = Pipeline([('net', transformermodel), ('logistic_regression', LogisticRegression())])
        pipelines["PureEEGNetV4"] = pureEegnetv4model
        pipelines["PureLSTMNet"] = purelstmnetmodel
        pipelines["PureTransformerNet"] = puretransformermodel
    else:
        print("Loading Models from:", path_to_models)
        if not os.path.exists(path_to_models):
            raise FileNotFoundError("Path to Models does not exist")
        if path_to_models.endswith('.pkl'):
            with open(path_to_models, 'rb') as f:
                pipelines = pickle.load(f)
            if not isinstance(pipelines, dict):
                pipe = pipelines
                pipelines = dict()
                pipe_name = path_to_models.split('/')[-1].split('_')[0]
                pipelines[pipe_name] = pipe
        else:
            pipelines = dict()
            for file in os.listdir(path_to_models):
                with open(path_to_models + '/' + file, 'rb') as f:
                    pipe_name = file.split('_')[0]
                    pipelines[pipe_name] = pickle.load(f)
        print("Loaded Models:", pipelines.keys(), flush=True)
    # This commented code was used to determine the suitable dataset, that have been hardcoded in the datasets list
    """
    from requests import ReadTimeout
    datasets = []
    for dataset in paradigm.datasets:
        if paradigm.is_valid(dataset):
            try:
                dataset.download()
                x, y, _ = paradigm.get_data(dataset)
                print(type(dataset), "Shapes:", x.shape, y.shape)
                datasets.append(dataset)
            except ReadTimeout as e:
                print(dataset, e)
                # Cannot download dataset, website down or something else, skipping
                continue
            except ValueError as e:
                print(dataset, e)
                # Dataset channels are not enough, skipping
                continue
    print("Selected Datasets:", datasets)
    """
    datasets = PhysionetMI()#, Weibo2014()]

    # Removed fit and score, evaluation will internally call both,
    # must check whether the pipeline is being fitted and scored correctly
    """
    for dataset in datasets:
        print("Fitting Pipeline on:", dataset)
        x, y, _ = paradigm.get_data(dataset)
        pipeline = pipeline.fit(x, y)

    for dataset in datasets:
        x, y, _ = paradigm.get_data(dataset)
        print("Scoring Pipeline on:", dataset, "result:", pipeline.score(x, y))
    """

    # To fix a library error multi threading was removed in all evaluations,
    # fix was suggested in https://mne.discourse.group/t/valueerror-data-copying-was-not-requested-by-copy-none-but-it-was-required-to-get-to-double-floating-point-precision-cross-val-score/7036/5
    # and seems to solve the problem
    curr_datetime_to_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + '_networks'
    """
    evaluation = WithinSessionEvaluation(paradigm=paradigm, datasets=datasets, overwrite=False,
        suffix='within_session_network_' + curr_datetime_to_string, random_state=42,
        hdf5_path='./models/' + curr_datetime_to_string + '/', )
    """
    # cant use it, at a certain point, an internal library function is called with one less parameter,
    # most probably is an error in my code, but don't know the origin, nor the source, therefore I cant fix it
    """
    from moabb.evaluations import CrossSubjectEvaluation
    evaluation = CrossSubjectEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        overwrite=True,
        suffix='cross_subject_network_' + curr_datetime_to_string,
        random_state=42,
        hdf5_path='./models/'+ curr_datetime_to_string + '/',
    )
    """

    # cant use it, usable datasets are not valid for this evaluation method,
    # as both of them have been recorded in 1 session (or are marked as such in the library)
    """
    from moabb.evaluations import CrossSessionEvaluation
    evaluation = CrossSessionEvaluation(
        paradigm=paradigm,
        datasets=datasets,
        overwrite=True,
        suffix='cross_session_network_' + curr_datetime_to_string,
        random_state=42,
        hdf5_path='./models/'+ curr_datetime_to_string + '/',
    )
    """
    """
    # hold-out cross-validation
    x, y, tmp = paradigm.get_data(datasets)
    y = EEGClassificator.utils.to_categorical(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=False)
    """
    """
    # k-fold cross-validation
    subjects_number = 109
    dataset_by_subject = []
    for i in range(1, subjects_number+1):
        x, y, tmp = paradigm.get_data(datasets, subjects=[i])
        dataset_by_subject.append((x, y))
    kfold = KFold(n_splits=subjects_number)
    for key, value in pipelines.items():
        pipelines[key] = [copy.deepcopy(value) for _ in range(subjects_number)]
    """
    x, y, tmp = paradigm.get_data(datasets)
    train_dataset_by_label, test_dataset_by_label = DatasetAugmentation.utils.split_dataset_by_label(x, y, test_size=0.3)
    downsampled_train_dataset_by_label = DatasetAugmentation.utils.downsample_dataset_by_label(train_dataset_by_label, min([len(dataset) for dataset in train_dataset_by_label.values()]))
    x_train, y_train = DatasetAugmentation.utils.merge_dataset_by_label(downsampled_train_dataset_by_label)
    y_train = EEGClassificator.utils.to_categorical(y_train, np_2d_array=True)
    x_test, y_test = DatasetAugmentation.utils.merge_dataset_by_label(test_dataset_by_label)
    y_test = EEGClassificator.utils.to_categorical(y_test, np_2d_array=True)
    del train_dataset_by_label
    del test_dataset_by_label
    del downsampled_train_dataset_by_label
    del paradigm
    del datasets
    del x
    del y
    del tmp
    while len(pipelines) > 0:
        key, pipe = pipelines.popitem()
        print("Fitting Pipeline:", key, flush=True)
        pipe.fit(x_train, y_train)
        score = pipe.score(x_test, y_test)
        if hasattr(score, 'item'):
            score = score.item()

        print("Computing Confusion Matrix for Pipeline:", key)
        result = pipe.predict(x_test)
        confusion_matrix = np.zeros((OUTPUT_CLASSES, OUTPUT_CLASSES))
        for i in range(y_test.shape[0]):
            confusion_matrix[y_test[i].argmax(), result[i].argmax()] += 1
        print("Scoring Pipeline:", key, "result:", score, "confusion matrix:", ''.join(str(confusion_matrix).splitlines()))

        if not os.path.exists('models/' + curr_datetime_to_string):
            os.makedirs('models/' + curr_datetime_to_string)
        with open('models/' + curr_datetime_to_string + '/' + key + '_' + str(score) + '_confusion_matrix.pkl',
                  'wb') as f:
            pickle.dump(confusion_matrix, f)
        with open('models/' + curr_datetime_to_string + '/' + key + '_' + str(score) + '.pkl', 'wb') as f:
            pickle.dump(pipe, f)
        if hasattr(pipe, 'named_steps') and 'net' in pipe.named_steps:
            pipe['net'].model.cpu()
            del pipe['net'].model
        del pipe
    """
    for key, pipe_lst in pipelines.items():
        print("Fitting Pipeline:", key, flush=True) 
        pipe_index = 0
        for train_index, test_index in kfold.split(dataset_by_subject):
            assert len(test_index.shape) == 1 and test_index.shape[0] == 1
            train = [dataset_by_subject[idx] for idx in train_index]
            # add further processing to get x_train, y_train
            x_train, y_train = train[0]
            for i in range(1, len(train)):
                x_train = np.vstack((x_train, train[i][0]))
                y_train = np.hstack((y_train, train[i][1]))
            y_train = EEGClassificator.utils.to_categorical(y_train)
            x_test, y_test = dataset_by_subject[test_index[0]]
            y_test = EEGClassificator.utils.to_categorical(y_test)
            pipe = pipe_lst[0]
            pipe.fit(x_train, y_train)
            score = pipe.score(x_test, y_test)
            print("Scoring Pipeline:", key, "result:", score, "run", pipe_index, "test", test_index[0], flush=True)
            if not os.path.exists('models/' + curr_datetime_to_string):
                os.makedirs('models/' + curr_datetime_to_string)
            with open('models/' + curr_datetime_to_string + '/' + key + '_' + str(test_index[0]) + '_' + str(pipe_index) + '_' + str(score) + '.pkl', 'wb') as f:
                pickle.dump(pipe, f)
            pipe_index += 1
            pipe['lstm_net'].model.cpu()
            del pipe['lstm_net'].model
            del pipe
            del pipe_lst[0]
        del pipelines[key]
        assert pipe_index == subjects_number
        """
    """
    results = evaluation.process(pipelines=pipelines)
    curr_datetime_to_string = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results.to_csv('within_session_network_results' + curr_datetime_to_string + '.csv')
    print(results)

    fig, _ = score_plot(results)
    fig.savefig('within_session_network_results' + curr_datetime_to_string + '.png')
    """

if __name__ == '__main__':
    path_to_models = sys.argv[1] if len(sys.argv) > 1 else None
    main(path_to_models)
