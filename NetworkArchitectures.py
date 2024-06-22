# Copying networks from:
# - https://www.nature.com/articles/s41598-023-41653-w#Sec3
import torch
import torch.nn as nn
import torch.nn.functional as F
from braindecode.models import EEGModuleMixin
from braindecode.models.base import deprecated_args
from einops.layers.torch import Rearrange
from torchviz import make_dot


def lambda_fn(x):
    return x[:, -1, :]

class LambdaLayer(nn.Module):
    def __init__(self, lambda_fn):
        super(LambdaLayer, self).__init__()
        self.lambda_fn = lambda_fn

    def forward(self, x):
        return self.lambda_fn(x)

class LSTMBasedArchitecture(EEGModuleMixin, nn.Sequential):
    """
    DOI: 10.1038/s41598-023-41653-w
    """

    def __init__(self, n_chans=None, n_outputs=None, n_times=None, chs_info=None, input_window_seconds=None, sfreq=None,
                 in_chans=None, n_classes=None, input_window_samples=None):
        n_chans, n_outputs, n_times = deprecated_args(self, ("in_chans", "n_chans", in_chans, n_chans),
                                                      ("n_classes", "n_outputs", n_classes, n_outputs), (
                                                      "input_window_samples", "n_times", input_window_samples,
                                                      n_times), )
        super().__init__(n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                         input_window_seconds=input_window_seconds, sfreq=sfreq, )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples
        """
        This architecture uses a fully connected layer with input sequence $X \in R^{T_s \times N_e}$, 
        where $N_e$ is the number of electrodes used for collection of MI-EEG signals and $T_s$ ($T_s < N_d$)
        is the time sequence, as the input layer. 
        The total length of time series data is denoted by $N_d$. 
        The details regarding the dataset preparation are provided later in the Experiments section. 
        These neurons are passed to an LSTM Network with 100 hidden units. 
        The LSTM is used for processing the time-series data and extracting temporal features. 
        A typical diagram of LSTM architecture29 is depicted in Fig. 1. 
        It consists of different memory blocks called cells, where 'e' represents the input signal, 
        'b' denotes the bias value, $f_t$ refers to the activation vector of forget gate, 
        'w' is the weight matrix associated with the signal, $v_{t-1}$ is the output of the prior cell, 
        's' is the cell state vector, $\widetilde{s_t}$ and $i_t$ denote the activation vectors of cell state 
        and input, respectively, $v_t$ is the output from the output gate.
        he LSTM-100 (LSTM with 100 hidden units) has a dropout of 0.2 to regularize the learning and 
        prevent over-training. The output of this dropout layer is fed to another LSTM with 50 hidden 
        units (LSTM-50). The ReLU activation function is used for each node. LSTM-50 also has a dropout 
        of 0.2. This layer is connected with another neural network layer with $O_N_c$ neurons for each class. 
        To obtain the probability vector for each class, the output of these neurons is passed to the Softmax 
        layer. The proposed architecture is shown in Fig. 2.
        """
        # Input format should be (batch, self.n_times, self.n_chans)
        # https://stackoverflow.com/questions/58587057/multi-dimensional-inputs-in-pytorch-linear-method
        self.add_module('reorder', Rearrange("batch channels times -> batch times channels"))
        self.add_module('input', nn.Linear(self.n_chans, 100))
        self.add_module('lstm1', nn.LSTM(100, 100, batch_first=True, dropout=0.2))
        self.add_module('lstm2', nn.LSTM(100, 50, batch_first=True, dropout=0.2))
        self.add_module('lambda_fn', LambdaLayer(lambda_fn))
        self.add_module('output', nn.Linear(50, self.n_outputs))
        self.add_module('softmax', nn.Softmax(dim=1))

    def forward(self, x):
        x = self.reorder(x)
        x = self.input(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.lambda_fn(x)
        x = self.output(x)
        x = self.softmax(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=6000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.2):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.2):
        super(TransformerEncoder, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src):
        return self.encoder(src)


class TransformerClassifier(EEGModuleMixin, nn.Module):
    def __init__(self, n_chans=None, n_outputs=None, n_times=None, chs_info=None, input_window_seconds=None, sfreq=None,
                 in_chans=None, n_classes=None, input_window_samples=None, num_layers=4, d_model=512, nhead=8,
                 dim_feedforward=2048, dropout=0.2):
        n_chans, n_outputs, n_times = deprecated_args(self, ("in_chans", "n_chans", in_chans, n_chans),
                                                      ("n_classes", "n_outputs", n_classes, n_outputs), (
                                                      "input_window_samples", "n_times", input_window_samples,
                                                      n_times), )
        super().__init__(n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                         input_window_seconds=input_window_seconds, sfreq=sfreq, )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples
        self.reorder_dims = Rearrange("batch channels times -> batch times channels")
        self.embedding = nn.Linear(self.n_chans, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.transformer_encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        self.classifier = nn.Linear(d_model, self.n_outputs)
        self.invert_reorder = Rearrange("batch times channels -> batch channels times")

    def forward(self, x):
        x = self.reorder_dims(x)
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = self.classifier(x)  # Average pooling over time steps
        x = F.softmax(x, dim=-1)
        # x = self.invert_reorder(x)
        x = x[:, -1, :]
        return x

    def plot(self, input_shape=(1, 65, 58)):
        x = torch.randn(input_shape)
        y = self.forward(x)
        dot = make_dot(y, params=dict(self.named_parameters()))
        dot.format = 'png'
        dot.render("transformer_classifier", cleanup=True)


class MLPArchitecture(EEGModuleMixin, nn.Sequential):
    def __init__(self, n_chans=None, n_outputs=None, n_times=None, chs_info=None, input_window_seconds=None, sfreq=None,
                 in_chans=None, n_classes=None, input_window_samples=None, num_layers=10):
        n_chans, n_outputs, n_times = deprecated_args(self, ("in_chans", "n_chans", in_chans, n_chans),
                                                      ("n_classes", "n_outputs", n_classes, n_outputs), (
                                                          "input_window_samples", "n_times", input_window_samples,
                                                          n_times), )
        super().__init__(n_outputs=n_outputs, n_chans=n_chans, chs_info=chs_info, n_times=n_times,
                         input_window_seconds=input_window_seconds, sfreq=sfreq, )
        del n_outputs, n_chans, chs_info, n_times, input_window_seconds, sfreq
        del in_chans, n_classes, input_window_samples

        self.add_module('reorder', Rearrange("batch channels times -> batch times channels"))
        self.add_module('input', nn.Linear(self.n_chans, 100))
        self.add_module('relu', nn.ReLU())
        for i in range(num_layers):
            self.add_module(f'hidden_{i}', nn.Linear(100, 100))
            self.add_module(f'relu_{i}', nn.ReLU())
        self.add_module('lambda_fn', LambdaLayer(lambda_fn))
        self.add_module('output', nn.Linear(100, self.n_outputs))
        self.add_module('softmax', nn.Softmax(dim=1))
