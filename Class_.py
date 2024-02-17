from create_dataset import series_to_supervised
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_boston
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.io as scio
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
from torchvision import transforms as tfs

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)

INPUT_SIZE = 7  # rnn input size
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=18,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            bidirectional=True
        )
        self.out = nn.Linear(36, 2)
    def forward(self, x):
        r_out, (hidden_state1, hidden_state2) = self.lstm(x, None) 
        outs = []  
        for time_step in range(r_out.size(1)):  
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1)
    
class Dataset(Dataset): 
    def __init__(self, x, y, z, q, w):
        self.x = torch.from_numpy(x).to(torch.double)
        self.y = torch.from_numpy(y).to(torch.double)
        self.z = torch.from_numpy(z).to(torch.double)
        self.q = torch.from_numpy(q).to(torch.double)
        self.w = torch.from_numpy(w).to(torch.double)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):  
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx], self.w[idx]
    

class CombinedAE(nn.Module):
    def __init__(self, input_size, encode2_input_size, output_size, activation_fn, use_dx_in_forward):
        super(CombinedAE, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        self.fc2 = nn.Linear(encode2_input_size, 1)
        self.fc3 = nn.Linear(output_size, output_size)
        self.activation_fn = activation_fn
        self.use_dx_in_forward = use_dx_in_forward

    def encode(self, x):
        return self.fc1(x)

    def encode2(self, x):
        return torch.sigmoid(self.fc2(x))

    def decode(self, z):
        return self.activation_fn(self.fc3(z))

    def forward(self, x, dx, q):
        z = self.encode(x) + self.encode2(q) + dx
        re = self.decode(z)
        return re, z

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat, y):
        return torch.sqrt(self.mse(yhat, y))
