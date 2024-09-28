import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick
import os
import warnings
import matplotlib
from Function_ import *
from Class_ import *
import math
import math
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
import scipy.stats as stats
import seaborn as sns
import pickle

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
print(torch.cuda.device_count())

warnings.filterwarnings('ignore')

#----------------------------------------data loading------------------------------
VIN_data = pd.read_excel('./data/Name_list.xls')  # Changed path and removed Chinese characters
vin = VIN_data.iloc[1, 0]
print(vin)

lstm = torch.load('./models/lstm.pth').to(device)  # Changed path
with open('./data/test/' + vin + '/vin_1.pkl', 'rb') as file:  # Changed path
    test_X = pickle.load(file)
# test


#----------------------------------------Hyper Parameter Setting---------------------
TIME_STEP = 1  # rnn time step
INPUT_SIZE = 7  # rnn input size
LR = 1e-4  # learning rate
lr_decay_freq = 25
EPOCH = 100
BATCH_SIZE = 100

#----------------------------------------Training for overall state prediction--------
train_X, train_y = prepare_training_data(test_X, INPUT_SIZE, TIME_STEP, device)

train_dataset = MyDataset(train_X, train_y)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

lstm = LSTM().to(device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()

i = 0
hidden_state = None
loss_train_100 = []
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):  # gives batch data
        lstm = lstm.double()
        output = lstm(b_x)  # rnn output
        loss = loss_func(b_y, output)  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients
        if step % 50 == 0:
            loss_train_100.append(loss.cpu().detach().numpy())

with open('./data/test/' + vin + '/vin_2.pkl', 'rb') as file:  # Changed path
    combined_tensor = pickle.load(file)
with open('./data/test/' + vin + '/vin_3.pkl', 'rb') as file:  # Changed path
    combined_tensorx = pickle.load(file)


#----------------------------------------Training for MC-AE--------------------------
dim_x = 2
dim_y = 110
dim_z = 110
dim_q = 3

x_recovered = combined_tensor[:, :dim_x]
y_recovered = combined_tensor[:, dim_x:dim_x + dim_y]
z_recovered = combined_tensor[:, dim_x + dim_y: dim_x + dim_y + dim_z]
q_recovered = combined_tensor[:, dim_x + dim_y + dim_z:]

dim_x2 = 2
dim_y2 = 110
dim_z2 = 110
dim_q2= 4

x_recovered2 = combined_tensorx[:, :dim_x2]
y_recovered2 = combined_tensorx[:, dim_x2:dim_x2 + dim_y2]
z_recovered2 = combined_tensorx[:, dim_x2 + dim_y2: dim_x2 + dim_y2 + dim_z2]
q_recovered2 = combined_tensorx[:, dim_x2 + dim_y2 + dim_z2:]

EPOCH = 300
LR = 5e-4
BATCHSIZE = 100
class Dataset(Dataset):
    def __init__(self, x, y, z, q):
        self.x = x.to(torch.double)
        self.y = y.to(torch.double)
        self.z = z.to(torch.double)
        self.q = q.to(torch.double)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx], self.q[idx]
train_loader_u = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=BATCHSIZE,
                      shuffle=False)

# Instantiate the networks
net = CombinedAE(input_size=2, encode2_input_size=3, output_size=110, activation_fn=custom_activation, use_dx_in_forward=True).to(device)
netx = CombinedAE(input_size=2, encode2_input_size=4, output_size=110, activation_fn=torch.sigmoid, use_dx_in_forward=True).to(device)

optimizer = torch.optim.Adam(net.parameters(), lr=LR)
l1_lambda = 0.01
loss_f = nn.MSELoss()
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    for iteration, (x, y, z, q) in enumerate(train_loader_u):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        net = net.double()
        recon_im , recon_p = net(x,z,q)
        loss_u = loss_f(y,recon_im)
        total_loss += loss_u.item()
        num_batches += 1
        optimizer.zero_grad()
        loss_u.backward()
        optimizer.step()
    avg_loss = total_loss / num_batches
    print('Epoch: {:2d} | Average Loss: {:.4f}'.format(epoch, avg_loss))

train_loader2 = DataLoader(Dataset(x_recovered, y_recovered, z_recovered, q_recovered), batch_size=len(x_recovered),
                        shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loader2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    net = net.double()
    recon_imtest, recon = net(x, z, q)
AA = recon_imtest.cpu().detach().numpy()
yTrainU = y_recovered.cpu().detach().numpy()
ERRORU = AA - yTrainU

train_loader_soc = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=BATCHSIZE, shuffle=False)
optimizer = torch.optim.Adam(netx.parameters(), lr=LR)
loss_f = nn.MSELoss()
avg_loss_list_x = []
for epoch in range(EPOCH):
    total_loss = 0
    num_batches = 0
    for iteration, (x, y, z, q) in enumerate(train_loader_soc):
        x = x.to(device)
        y = y.to(device)
        z = z.to(device)
        q = q.to(device)
        netx = netx.double()
        recon_im , z  = netx(x,z,q)
        loss_x = loss_f(y,recon_im)
        total_loss += loss_x.item()
        num_batches += 1
        optimizer.zero_grad()
        loss_x.backward()
        optimizer.step()
    avg_loss = total_loss / num_batches
    avg_loss_list_x.append(avg_loss)
    print('Epoch: {:2d} | Average Loss: {:.4f}'.format(epoch, avg_loss))

train_loaderx2 = DataLoader(Dataset(x_recovered2, y_recovered2, z_recovered2, q_recovered2), batch_size=len(x_recovered2), shuffle=False)
for iteration, (x, y, z, q) in enumerate(train_loaderx2):
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    q = q.to(device)
    netx = netx.double()
    recon_imtestx, z = netx(x, z, q)

BB = recon_imtestx.cpu().detach().numpy()
yTrainX = y_recovered2.cpu().detach().numpy()
ERRORX = BB - yTrainX

df_data = DiagnosisFeature(ERRORU,ERRORX)

v_I, v, v_ratio, p_k, data_mean, data_std, T_95_limit, T_99_limit, SPE_95_limit, SPE_99_limit, P, k, P_t, X, data_nor = PCA(df_data,0.95,0.95)
