from models import LSTModel
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
filepath = pjoin(folderpath, 'dataset_sysID_3tanks_final.mat')
data = scipy.io.loadmat(filepath)

dExp, yExp, dExp_val, yExp_val, Ts = data['dExp'], data['yExp'], \
    data['dExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

t = np.arange(0, np.size(dExp[0, 0], 1) * Ts, Ts)

# plt.plot(t, yExp[0,-1])
# plt.show()

seed = 1
torch.manual_seed(seed)

n = 1  # input dimensions
inputnumberD = 1
p = 3  # output dimensions

n_xi = 9
# nel paper n1, numero di stati
l = 9  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE

RENsys = torch.nn.Sequential(LSTModel(n, n_xi, l, p))

# Define the system

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-1
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].shape[1] - 1

epochs = 1
LOSS = np.zeros(epochs)

for epoch in range(epochs):
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-2
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        y = torch.from_numpy(yExp[0, exp]).float().to(device)
        y = y.squeeze()
        yRENm = torch.randn(p, t_end + 1, device=device, dtype=dtype)
        yRENm[:,0] = y[:,0]
        xi = torch.randn(n_xi)
        d = torch.from_numpy(dExp[0, exp]).float().to(device)
        for t in range(1, t_end):
            u = torch.tensor([d[inputnumberD, t]])
            yRENm[:, t] = RENsys(u)
        loss = loss + MSE(yRENm[:, 0:yRENm.size(1)], y[:, 0:t_end + 1])
        # ignorare da loss effetto condizione iniziale

    loss = loss / nExp
    #loss.backward()
    loss.backward(retain_graph=True)

    optimizer.step()
    RENsys.set_model_param()

    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    LOSS[epoch] = loss


t_end = yExp_val[0, 0].shape[1] - 1

yval = torch.from_numpy(yExp_val[0, 0]).float().to(device)
yval = yval.squeeze()

yRENm_val = torch.zeros(p, t_end + 1, device=device, dtype=dtype)
yRENm_val[:,0] = yval[:,0]
dval = torch.from_numpy(dExp_val[0, 0]).float().to(device)
loss_val = 0
for t in range(1, t_end):
    u = torch.tensor([dval[inputnumberD, t]])
    yRENm_val[:, t]= RENsys(u)
    loss_val = loss_val + MSE(yRENm_val[:, 0:yRENm_val.size(1)], yval[:, 0:t_end + 1])

plt.figure('8')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

plt.figure('9')
plt.plot(yRENm[0, :].detach().numpy(), label='REN')
plt.plot(y[0, :].detach().numpy(), label='y train')
plt.title("output 1 train single REN")
plt.legend()
plt.show()

plt.figure('10')
plt.plot(yRENm_val[0, 0:t_end].detach().numpy(), label='REN val')
plt.plot(yval[0, 0:t_end].detach().numpy(), label='y val')
plt.title("output 1 val single REN")
plt.legend()
plt.show()

plt.figure('11')
plt.plot(yRENm[1, :].detach().numpy(), label='REN')
plt.plot(y[1, :].detach().numpy(), label='y train')
plt.title("output 2 train single REN")
plt.legend()
plt.show()

plt.figure('12')
plt.plot(yRENm_val[1, 0:t_end].detach().numpy(), label='REN val')
plt.plot(yval[1, 0:t_end].detach().numpy(), label='y val')
plt.title("output 2 val single REN")
plt.legend()
plt.show()

plt.figure('13')
plt.plot(yRENm[2, :].detach().numpy(), label='REN')
plt.plot(y[2, :].detach().numpy(), label='y train')
plt.title("output 3 train single REN")
plt.legend()
plt.show()

plt.figure('14')
plt.plot(yRENm_val[2, 0:t_end].detach().numpy(), label='REN val')
plt.plot(yval[2, 0:t_end].detach().numpy(), label='y val')
plt.title("output 3 val single REN")
plt.legend()
plt.show()


plt.figure('15')
plt.plot(d[inputnumberD, :].detach().numpy(), label='input train')
plt.plot(dval[inputnumberD, :].detach().numpy(), label='input val')
plt.title("input single REN")
plt.legend()
plt.show()

print(f"Loss Validation single REN: {loss_val}")

scipy.io.savemat('data_singleREN.mat', dict(yRENm_val=yRENm_val.detach().numpy(), yval=yval.detach().numpy()))
