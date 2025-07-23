import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# RECURRENT NEURAL NETWORK
class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, nonlinearity='relu', batch_first=True)

        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state randomly
        h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
        # One time step
        out, hn = self.rnn(x, h0)
        out = self.fc(out)
        out = out.squeeze()
        return out

# Robust REN implementation in the acyclic version
class REN(nn.Module):
    # ## Implementation of REN model, modified from "Recurrent Equilibrium Networks: Flexible Dynamic Models with
    # Guaranteed Stability and Robustness" by Max Revay et al.
    def __init__(self, m, p, n, l, bias=False, mode="l2stable", gamma=0.3, gammaTrain=False, Q=None, R=None, S=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.m = m  # input dimension
        self.n = n  # state dimension
        self.l = l  # dimension of v(t) and w(t)
        self.p = p  # output dimension
        self.mode = mode
        self.device = device
        self.gamma = gamma
        self.gammaTrain = gammaTrain
        if gammaTrain:
            self.sg = nn.Parameter(torch.tensor(gamma))
        # # # # # # # # # IQC specification # # # # # # # # #
        self.Q = Q
        self.R = R
        self.S = S
        # # # # # # # # # Training parameters # # # # # # # # #
        std = 0.01
        # Sparse training matrix parameters
        self.x0 = nn.Parameter((torch.randn(1, n, device=device) * std))
        self.X = nn.Parameter((torch.randn(2 * n + l, 2 * n + l, device=device) * std))
        self.Y = nn.Parameter((torch.randn(n, n, device=device) * std))
        self.Z3 = nn.Parameter(torch.randn(abs(p - m), min(p, m), device=device) * std)
        self.X3 = nn.Parameter(torch.randn(min(p, m), min(p, m), device=device) * std)
        self.Y3 = nn.Parameter(torch.randn(min(p, m), min(p, m), device=device) * std)
        self.D12 = nn.Parameter(torch.randn(l, m, device=device))
        #self.D21 = nn.Parameter((torch.randn(p, l, device=device) * std)) #set to zero later to enable
        # computable Networked RENs
        self.B2 = nn.Parameter((torch.randn(n, m, device=device) * std))
        self.C2 = nn.Parameter((torch.randn(p, n, device=device) * std))

        if bias:
            self.bx = nn.Parameter(torch.randn(n, device=device) * std)
            self.bv = nn.Parameter(torch.randn(l, device=device) * std)
            self.bu = nn.Parameter(torch.randn(p, device=device) * std)
        else:
            self.bx = torch.zeros(n, device=device)
            self.bv = torch.zeros(l, device=device)
            self.bu = torch.zeros(p, device=device)
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements

        self.x = torch.zeros(1, n, device=device)
        self.epsilon = 0.001
        self.F = torch.zeros(n, n, device=device)
        self.B1 = torch.zeros(n, l, device=device)
        self.E = torch.zeros(n, n, device=device)
        self.Lambda = torch.ones(l, device=device)
        self.C1 = torch.zeros(l, n, device=device)
        self.D11 = torch.zeros(l, l, device=device)
        self.D22 = torch.zeros(p, m, device=device)
        self.P = torch.zeros(n, n, device=device)
        self.P_cal = torch.zeros(n, n, device=device)
        self.D21 = torch.zeros(p, l, device=device)
        self.set_param(gamma)

    def set_param(self, gamma=0.3):
        if self.gammaTrain:
            gamma = self.sg**2
        self.gamma = gamma
        n, l, m, p = self.n, self.l, self.m, self.p
        # Updating of Q,S,R with variable gamma if needed
        self.Q, self.R, self.S = self._set_mode(self.mode, gamma, self.Q, self.R, self.S)
        M = F.linear(self.X3.T, self.X3.T) + self.Y3 - self.Y3.T + F.linear(self.Z3.T,
                                                                            self.Z3.T) + self.epsilon * torch.eye(
            min(m, p), device=self.device)
        if p >= m:
            N = torch.vstack((F.linear(torch.eye(m, device=self.device) - M,
                                       torch.inverse(torch.eye(m, device=self.device) + M).T),
                              -2 * F.linear(self.Z3, torch.inverse(torch.eye(m, device=self.device) + M).T)))
        else:
            N = torch.hstack((F.linear(torch.inverse(torch.eye(p, device=self.device) + M),
                                       (torch.eye(p, device=self.device) - M).T),
                              -2 * F.linear(torch.inverse(torch.eye(p, device=self.device) + M), self.Z3)))

        Lq = torch.linalg.cholesky(-self.Q).T
        Lr = torch.linalg.cholesky(self.R - torch.matmul(self.S, torch.matmul(torch.inverse(self.Q), self.S.T))).T
        self.D22 = -torch.matmul(torch.inverse(self.Q), self.S.T) + torch.matmul(torch.inverse(Lq),
                                                                                 torch.matmul(N, Lr))
        # Calculate psi_r:
        R_cal = self.R + torch.matmul(self.S, self.D22) + torch.matmul(self.S, self.D22).T + torch.matmul(self.D22.T,
                                                                                                          torch.matmul(
                                                                                                              self.Q,
                                                                                                              self.D22))
        R_cal_inv = torch.linalg.inv(R_cal)
        C2_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.C2).T
        D21_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.D21).T - self.D12
        vec_r = torch.cat((C2_cal, D21_cal, self.B2), dim=0)
        psi_r = torch.matmul(vec_r, torch.matmul(R_cal_inv, vec_r.T))
        # Calculate psi_q:
        vec_q = torch.cat((self.C2.T, self.D21.T, torch.zeros(self.n, self.p, device=self.device)), dim=0)
        psi_q = torch.matmul(vec_q, torch.matmul(self.Q, vec_q.T))
        # Create H matrix:
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n + l, device=self.device) + psi_r - psi_q
        h1, h2, h3 = torch.split(H, [n, l, n], dim=0)
        H11, H12, H13 = torch.split(h1, [n, l, n], dim=1)
        H21, H22, _ = torch.split(h2, [n, l, n], dim=1)
        H31, H32, H33 = torch.split(h3, [n, l, n], dim=1)
        self.P_cal = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + self.P_cal + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21
        # Matrix P
        self.P = torch.matmul(self.E.T, torch.matmul(torch.inverse(self.P_cal), self.E))

    def forward(self, u, x, t):
        decay_rate = 0.95
        vec = torch.zeros(self.l, device=self.device)
        epsilon = torch.zeros(self.l, device=self.device)
        if self.l > 0:
            vec[0] = 1
            v = F.linear(x, self.C1[0, :]) + F.linear(u, self.D12[0, :]) + (decay_rate ** t) * self.bv[0]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l, device=self.device)
            vec[i] = 1
            v = F.linear(x, self.C1[i, :]) + F.linear(epsilon, self.D11[i, :]) + F.linear(u, self.D12[i, :]) + (
                    decay_rate ** t) * self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
        E_x_ = F.linear(x, self.F) + F.linear(epsilon, self.B1) + F.linear(u, self.B2) + (decay_rate ** t) * self.bx

        x_ = F.linear(E_x_, self.E.inverse())

        y = F.linear(x, self.C2) + F.linear(epsilon, self.D21) + F.linear(u, self.D22) + (decay_rate ** t) * self.bu

        return y, x_

    def _set_mode(self, mode, gamma, Q, R, S, eps=1e-4):
        # We set Q to be negative definite. If Q is nsd we set: Q - \epsilon I.
        # I.e. The Q we define here is denoted as \matcal{Q} in REN paper.
        if mode == "l2stable":
            Q = -(1. / gamma) * torch.eye(self.p, device=self.device)
            R = gamma * torch.eye(self.m, device=self.device)
            S = torch.zeros(self.m, self.p, device=self.device)
        elif mode == "input_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing input passivity.")
            Q = torch.zeros(self.p, self.p, device=self.device) - eps * torch.eye(self.p, device=self.device)
            R = -2. * gamma * torch.eye(self.m, device=self.device)
            S = torch.eye(self.p, device=self.device)
        elif mode == "output_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing output passivity.")
            Q = -2. * gamma * torch.eye(self.p, device=self.device)
            R = torch.zeros(self.m, self.m, device=self.device)
            S = torch.eye(self.m, device=self.device)
        else:
            print("Using matrices R,Q,S given by user.")
            # Check dimensions:
            if not (len(R.shape) == 2 and R.shape[0] == R.shape[1] and R.shape[0] == self.m):
                raise NameError("The matrix R is not valid. It must be a square matrix of %ix%i." % (self.m, self.m))
            if not (len(Q.shape) == 2 and Q.shape[0] == Q.shape[1] and Q.shape[0] == self.p):
                raise NameError("The matrix Q is not valid. It must be a square matrix of %ix%i." % (self.p, self.p))
            if not (len(S.shape) == 2 and S.shape[0] == self.m and S.shape[1] == self.p):
                raise NameError("The matrix S is not valid. It must be a matrix of %ix%i." % (self.m, self.p))
            # Check R=R':
            if not (R == R.T).prod():
                raise NameError("The matrix R is not valid. It must be symmetric.")
            # Check Q is nsd:
            eigs, _ = torch.linalg.eig(Q)
            if not (eigs.real <= 0).prod():
                print('oh!')
                raise NameError("The matrix Q is not valid. It must be negative semidefinite.")
            if not (eigs.real < 0).prod():
                # We make Q negative definite: (\mathcal{Q} in the REN paper)
                Q = Q - eps * torch.eye(self.p, device=self.device)
        return Q, R, S


# Stable networked operator made by RENs with fully trainable l2 gains and interconnection matrices
class NetworkedRENs(nn.Module):
    def __init__(self, N, Muy, Mud, Mey, m, p, n, l, top=True,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.device = device
        self.top = top  # If set to True, the topology of M is preserved, otherwise it trains a
        # potentially full matrix Q
        self.p = p  # output dimension for each REN
        self.m = m  # input dimension for each REN
        self.n = n  # state dimension for each REN
        self.l = l  # number of nonlinear layers for each REN
        self.Muy = Muy
        self.Mud = Mud
        self.Mey = Mey
        self.diag_params = nn.Parameter(torch.randn(sum(p)))  # For trainable Mey matrix
        self.N = N
        self.r = nn.ModuleList([REN(self.m[j], self.p[j], self.n[j], self.l[j]) for j in range(N)])
        self.s = nn.Parameter(torch.randn(N, device=device))
        self.gammaw = torch.nn.Parameter(4 * torch.randn(1, device=device))
        if top:
            # Create a mask where M is non-zero
            self.mask = Muy.ge(0.1)
            # Count the number of non-zero elements in M
            num_params = self.mask.sum().item()
            # Initialize the trainable parameters
            self.params = nn.Parameter(0.03 * torch.randn(num_params))
            # Create a clone of M to create Q (the trainable version of M)
            self.Q = Muy.clone()
        else:
            self.Q = nn.Parameter(0.01 * torch.randn((sum(m), sum(p))))

    def forward(self, t, d, x, checkLMI=False):
        # checkLMI if set to True, checks if the dissipativity LMI is satisfied at every step
        Q = self.Q
        if self.top:
            params = self.params
            # Assign the parameters to the corresponding positions in Q
            masked_values = torch.zeros_like(Q, device=self.device)
            masked_values[self.mask] = params
            Q = masked_values

        gammaw = self.gammaw
        #Mey = self.Mey
        tMey = torch.diag(self.diag_params)
        H = torch.matmul(tMey.T, tMey)
        sp = torch.abs(self.s)
        gamma_list = []
        C2s = []
        D22s = []
        row_sum = torch.sum(self.Mud, 1)
        A1t = torch.nonzero(row_sum == 1, as_tuple=False).squeeze(dim=1)
        A0t = torch.nonzero(row_sum == 0, as_tuple=False).squeeze(dim=1)
        uindex = []
        yindex = []
        xindex = []
        startu = 0
        starty = 0
        startx = 0
        pesi = torch.zeros(self.N)
        for j, l in enumerate(self.r):
            # Free parametrization of individual l2 gains ensuring stability of networked REN
            xi = torch.arange(startx, startx + l.n)
            ui = torch.arange(startu, startu + l.m)
            yi = torch.arange(starty, starty + l.p)
            setu = set(ui.numpy())
            A1 = torch.tensor(list(setu.intersection(set(A1t.numpy()))), device=self.device)
            A0 = torch.tensor(list(setu.intersection(set(A0t.numpy()))), device=self.device)
            a = H[j, j] + torch.max(torch.stack([torch.sum(torch.abs(Q[:, j])) for j in yi])) + sp[j]
            #a = H[j, j] + torch.max(torch.stack([torch.sum(torch.abs(Q[:, j])) for j in yi]))
            pesi[j] = a
            if A0.numel() != 0:
                if A1.numel() != 0:
                    gamma = torch.sqrt(
                        1 / a * torch.minimum(gammaw ** 2 / (torch.max(torch.stack([torch.sum(torch.abs(Q[j, :]))
                                                                                    for j in
                                                                                    A1])) * gammaw ** 2 + 1),
                                              1 / (torch.max(torch.stack([torch.sum(torch.abs(Q[j, :]))
                                                                          for j in
                                                                          A0])))))
                else:
                    gamma = torch.sqrt(1 / (a * torch.max(torch.stack([torch.sum(torch.abs(Q[j, :]))
                                                                       for j in
                                                                       A0]))))
            else:
                gamma = torch.sqrt(1 / a * (gammaw ** 2 / (torch.max(torch.stack([torch.sum(torch.abs(Q[j, :]))
                                                                                  for j in
                                                                                  A1])) * gammaw ** 2 + 1)))

            l.set_param(gamma)
            gamma_list.append(gamma)
            C2s.append(l.C2)
            D22s.append(l.D22)
            startu += l.m
            starty += l.p
            startx += l.n
            uindex.append(ui)
            yindex.append(yi)
            xindex.append(xi)
        C2 = torch.block_diag(*C2s)
        D22 = torch.block_diag(*D22s)
        # compute the stacked input for each REN
        u = torch.matmul(torch.inverse(torch.eye(self.Muy.size(0)) - torch.matmul(Q, D22)),
                         (torch.matmul(torch.matmul(Q, C2), x)) + torch.matmul(self.Mud, d))
        y_list = []
        x_list = []
        # update REN dynamics
        for j, l in enumerate(self.r):
            yt, xtemp = l(u[uindex[j]], x[xindex[j]], t)
            y_list.append(yt)
            x_list.append(xtemp)

        y = torch.cat(y_list)
        x_ = torch.cat(x_list)
        e = torch.matmul(tMey, y)
        gammawout = gammaw ** 2

        # check Dissipativity LMI
        if checkLMI:
            with torch.no_grad():
                Nu = torch.block_diag(*[pesi[j] * gamma_list[j] ** 2 * torch.eye(self.m[j]) for j in range(self.N)])
                Ny = torch.block_diag(*[pesi[j] * torch.eye(self.p[j]) for j in range(self.N)])
                Xi = torch.block_diag(Nu, -Ny)
                S = torch.block_diag(gammawout * torch.eye(sum(self.m)), -torch.eye(sum(self.p)))
                XiS = torch.block_diag(Xi, -S)

                M1 = torch.hstack((Q.data, self.Mud))
                M2 = torch.hstack((torch.eye(sum(self.p)), torch.zeros((sum(self.p), sum(self.m)))))
                M3 = torch.hstack((torch.zeros((sum(self.m), sum(self.p))), torch.eye(sum(self.m))))
                M4 = torch.hstack((tMey, torch.zeros((sum(self.p), sum(self.m)))))
                M = torch.vstack((M1, M2, M3, M4))
                lmi = M.T @ XiS @ M
                lmip = torch.linalg.eigvals(lmi)

        return e, x_