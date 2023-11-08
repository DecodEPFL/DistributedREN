from models import RNNModel
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from os.path import dirname, join as pjoin
import torch
from torch import nn

dtype = torch.float
device = torch.device("cpu")

plt.close('all')
# Import Data
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'dataset_sysID_3tanks.mat')
data = scipy.io.loadmat(filepath)

dExp, yExp, dExp_val, yExp_val, Ts = data['dExp'], data['yExp'], \
    data['dExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

t = np.arange(0, np.size(dExp[0, 0], 1) * Ts-Ts, Ts)

t_end = t.size

u = torch.zeros(nExp, t_end, 1)
y = torch.zeros(nExp, t_end, 3)
inputnumberD =1

for j in range(nExp):
    inputActive = (torch.from_numpy(dExp[0, j])).T
    u[j, :, :] = torch.unsqueeze(inputActive[:,inputnumberD], 1)
    y[j, :, :] = (torch.from_numpy(yExp[0, j])).T

seed = 1
torch.manual_seed(seed)

idd = 1
hdd = 20
ldd = 5
odd = yExp[0, 0].shape[0]

RNN = RNNModel(idd, hdd, ldd, odd)
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-2
optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].shape[1]

epochs = 500
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-4
        optimizer = torch.optim.Adam(RNN.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0

    yRNN = RNN(u)
    yRNN = torch.squeeze(yRNN)
    loss = MSE(yRNN, y)
    loss.backward()
    optimizer.step()
    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss

t_end = yExp_val[0, 0].shape[1]

nExp = yExp_val.size

uval = torch.zeros(nExp, t_end, 1)
yval = torch.zeros(nExp, t_end, 3)

for j in range(nExp):
    inputActive = (torch.from_numpy(dExp_val[0, j])).T
    uval[j, :, :] = torch.unsqueeze(inputActive[:,inputnumberD], 1)
    yval[j, :, :] = (torch.from_numpy(yExp_val[0, j])).T

yRNN_val = RNN(uval)
yRNN_val = torch.squeeze(yRNN_val)

loss_val = MSE(yRNN_val, yval)

plt.figure('8')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

plt.figure('9')
plt.plot(yRNN[0, :, 0].detach().numpy(), label='REN')
plt.plot(y[0, :, 0].detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

plt.figure('10')
plt.plot(yRNN_val[:, 0].detach().numpy(), label='REN val')
plt.plot(yval[0, :, 0].detach().numpy(), label='y val')
plt.title("output 1 val single RNN")
plt.legend()
plt.show()

plt.figure('11')
plt.plot(yRNN[0, :, 1].detach().numpy(), label='REN')
plt.plot(y[0, :, 1].detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

plt.figure('12')
plt.plot(yRNN_val[:, 1].detach().numpy(), label='REN val')
plt.plot(yval[0, :, 1].detach().numpy(), label='y val')
plt.title("output 1 val single REN")
plt.legend()
plt.show()

plt.figure('13')
plt.plot(yRNN[0, :, 2].detach().numpy(), label='REN')
plt.plot(y[0, :, 2].detach().numpy(), label='y train')
plt.title("output 1 train single RNN")
plt.legend()
plt.show()

plt.figure('14')
plt.plot(yRNN_val[:, 2].detach().numpy(), label='REN val')
plt.plot(yval[0, :, 2].detach().numpy(), label='y val')
plt.title("output 1 val single RNN")
plt.legend()
plt.show()

# plt.figure('15')
# plt.plot(d[inputnumberD, :].detach().numpy(), label='input train')
# plt.plot(dval[inputnumberD, :].detach().numpy(), label='input val')
# plt.title("input single REN")
# plt.legend()
# plt.show()

print(f"Loss Validation single RNN: {loss_val}")
