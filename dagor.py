# coding = utf-8
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

from castle.common import BaseLearner, Tensor
from castle.common import consts
from castle.common.validator import check_args_value

from castle.algorithms.gradient.dag_gnn.torch.utils import functions as func
from models.modules import Encoder, Decoder


def set_seed(seed):
    """
    Referred from:
    - https://stackoverflow.com/questions/38469632/tensorflow-non-repeatable-results
    """

    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        os.environ['PYTHONHASHSEED'] = str(seed)
    except:
        pass


def H_A(A, m):
    s = torch.tensor(1.0).to('cuda')
    Id = torch.eye(m).to('cuda')
    M = s * Id - A * A
    h_A_2 = - torch.linalg.slogdet(M)[1] + m * torch.log(s)

    '''
    
    # *****************synthetic linear data ER2 TOPO sorts***********
    [6, 8, 9, 4, 2, 3, 0, 5, 1, 7]
    [2, 5, 7, 12, 4, 9, 14, 16, 6, 15, 3, 0, 13, 17, 1, 8, 18, 10, 11, 19]
    [2, 5, 6, 10, 13, 23, 26, 30, 31, 39, 0, 19, 32, 24, 34, 9, 14, 4, 18, 1, 29, 8, 20, 21, 28, 3, 17, 27, 36, 12, 15, 22, 25, 11, 38, 35, 33, 37, 16, 7]
    [4, 12, 16, 24, 26, 33, 35, 40, 42, 43, 50, 51, 54, 55, 0, 56, 38, 41, 8, 5, 21, 1, 14, 15, 22, 25, 47, 9, 31, 57, 3, 28, 36, 59, 2, 48, 6, 30, 53, 37, 44, 27, 23, 19, 32, 45, 10, 7, 20, 17, 39, 34, 46, 49, 52, 29, 11, 58, 18, 13]
     
    # *****************synthetic nonlinear data SF2 TOPO sorts***********
    [2, 8, 9, 6, 4, 3, 5, 7, 0, 1]
    [2, 4, 5, 16, 3, 7, 9, 12, 15, 18, 6, 13, 0, 1, 11, 17, 8, 10, 14, 19]
    [5, 6, 10, 13, 23, 24, 27, 31, 39, 4, 30, 34, 35, 0, 25, 33, 11, 1, 9, 18, 2, 19, 26, 36, 37, 22, 8, 32, 3, 21, 29, 12, 17, 7, 28, 38, 15, 16, 14, 20]
    [5, 8, 9, 25, 40, 43, 50, 51, 29, 1, 24, 26, 35, 4, 12, 13, 32, 33, 47, 38, 54, 21, 28, 42, 37, 15, 20, 34, 2, 16, 22, 31, 41, 59, 0, 39, 36, 23, 55, 3, 11, 14, 44, 46, 52, 30, 45, 48, 18, 17, 58, 6, 10, 19, 27, 56, 57, 7, 53, 49]
    
    # *****************benchmark data TOPO sorts***********
    [1, 2, 10, 4, 5, 0, 8, 6, 9, 7, 3]
    [18, 6, 2, 13, 17, 25, 9, 26, 12, 7, 16, 0, 23, 8, 3, 20, 1, 14, 19, 21, 11, 4, 15,	10, 24, 5, 22]
    [19, 6, 10, 26, 27, 24, 34, 32,	29,	31,	33,	1, 36, 9, 25, 30, 5, 0, 28,	4, 11, 2, 21, 16, 15, 18, 35, 22, 20, 14, 12, 17, 7, 13, 23, 3, 8]
    [18, 33, 39, 20, 47, 50, 48, 27, 6, 36,	2, 28, 46, 1, 4, 49, 37, 13, 23, 32, 15, 10, 29, 22, 34, 42, 21, 12, 14, 30, 35, 24, 41, 25, 16, 54, 43, 44, 52, 31, 8, 0, 40, 51, 38, 7, 45, 9, 26, 11, 3, 5, 53, 55, 19, 17]
    
    '''

    p = [18, 33, 39, 20, 47, 50, 48, 27, 6, 36,	2, 28, 46, 1, 4, 49, 37, 13, 23, 32, 15, 10, 29, 22, 34, 42, 21, 12, 14, 30, 35, 24, 41, 25, 16, 54, 43, 44, 52, 31, 8, 0, 40, 51, 38, 7, 45, 9, 26, 11, 3, 5, 53, 55, 19, 17]

    P = np.zeros((m, m))
    for i in range(m):
        P[i][p[i]] = 1
    PT = np.transpose(P)
    # Pp = torch.tensor(P).to("cuda")
    # PTp = torch.tensor(PT).to("cuda")
    W = A.detach().cpu().numpy()
    Wp = P @ W @ PT

    Wp = torch.tensor(Wp, requires_grad=True).to("cuda")

    # Q, R = sl.qr(Wp)
    Q, R = torch.linalg.qr(Wp)
    # Q_ = torch.linalg.inv(Q)
    # S = Wp - torch.matmul(Q_, Wp)
    # S_ = Wp @ R_
    # Id = torch.eye(m).to('cuda')
    # S = Q - Id
    # h_A_1 = torch.tensor(0.5 / m * (S ** 2).sum())
    h_A_1 = torch.linalg.slogdet(Q)[1]

    h_A = h_A_1
    return h_A


class DAGOR(BaseLearner):
    """
    Parameters
    ----------
    encoder_type: str, default: 'mlp'
        choose an encoder, 'mlp' or 'sem'.
    decoder_type: str, detault: 'mlp'
        choose a decoder, 'mlp' or 'sem'.
    encoder_hidden: int, default: 64
        MLP encoder hidden layer dimension, just one hidden layer.
    latent_dim: int, default equal to input dimension
        encoder output dimension
    decoder_hidden: int, default: 64
        MLP decoder hidden layer dimension, just one hidden layer.
    encoder_dropout: float, default: 0.0
        Dropout rate (1 - keep probability).
    decoder_dropout: float, default: 0.0
        Dropout rate (1 - keep probability).
    epochs: int, default: 300
        train epochs
    k_max_iter: int, default: 1e2
        the max iteration number for searching lambda and c.
    batch_size: int, default: 100
        Sample size of each training batch
    lr: float, default: 3e-3
        learning rate
    lr_decay: int, default: 200
        Period of learning rate decay.
    gamma: float, default: 1.0
        Multiplicative factor of learning rate decay.
    lambda_a: float, default: 0.0
        coefficient for DAG constraint h(A).
    c_a: float, default: 1.0
        coefficient for absolute value h(A).
    c_a_thresh: float, default: 1e20
        control loop by c_a
    eta: int, default: 10
        use for update c_a, greater equal than 1.
    multiply_h: float, default: 0.25
        use for judge whether update c_a.
    tau_a: float, default: 0.0
        coefficient for L-1 norm of A.
    h_tolerance: float, default: 1e-8
        the tolerance of error of h(A) to zero.
    use_a_connect_loss: bool, default: False
        flag to use A connect loss
    use_a_positiver_loss: bool, default: False
        flag to enforce A must have positive values
    graph_threshold: float, default: 0.3
        threshold for learned adjacency matrix binarization.
        greater equal to graph_threshold denotes has causal relationship.
    optimizer: str, default: 'Adam'
        choose optimizer, 'Adam' or 'SGD'
    seed: int, default: 42
        random seed
    device_type: str, default: cpu
        ``cpu`` or ``gpu``
    device_ids: int or str, default None
        CUDA devices, it's effective when ``use_gpu`` is True.
        For single-device modules, ``device_ids`` can be int or str, e.g. 0 or '0',
        For multi-device modules, ``device_ids`` must be str, format like '0, 1'.

    Examples
    --------
    # >>> from castle.algorithms.gradient.dag_gnn.torch import DAG_GNN
    # >>> from castle.datasets import load_dataset
    # >>> from castle.common import GraphDAG
    # >>> from castle.metrics import MetricsDAG
    # >>> X, true_dag, _ = load_dataset('IID_Test')
    # >>> m = DAG_GNN()
    # >>> m.learn(X)
    # >>> GraphDAG(m.causal_matrix, true_dag)
    # >>> met = MetricsDAG(m.causal_matrix, true_dag)
    # >>> print(met.metrics)
    """

    # lr = 3e-3
    # batch_size=100

    @check_args_value(consts.GNN_VALID_PARAMS)
    def __init__(self, encoder_type='mlp', decoder_type='mlp',
                 encoder_hidden=64, latent_dim=None, decoder_hidden=64,
                 encoder_dropout=0.0, decoder_dropout=0.0, epochs=300, k_max_iter=10, tau_a=0.0,
                 batch_size=100, lr=3e-3, lr_decay=200, gamma=1.0, init_lambda_a=0.0, init_c_a=1.0,
                 c_a_thresh=1e20, eta=10, multiply_h=0.25, h_tolerance=1e-8,
                 use_a_connect_loss=False, use_a_positiver_loss=False, graph_threshold=0.3,
                 optimizer='adam', seed=42, device_type='gpu', device_ids='0'):
        # asia:-8, sachs:-6, batch_size=256, lr:3e-5
        # def __init__(self, encoder_type='mlp', decoder_type='mlp',
        #              encoder_hidden=64, latent_dim=None, decoder_hidden=64,
        #              encoder_dropout=0.0, decoder_dropout=0.0, epochs=300, k_max_iter=1e2, tau_a=0.0,
        #              batch_size=100, lr=3e-3, lr_decay=200, gamma=1.0, init_lambda_a=0.0, init_c_a=1.0,
        #              c_a_thresh=1e20, eta=10, multiply_h=0.25, h_tolerance=1e-8,
        #              use_a_connect_loss=False, use_a_positiver_loss=False, graph_threshold=0.3,
        #              optimizer='adam', seed=42, device_type='gpu', device_ids='0'):
        super(DAGOR, self).__init__()
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.encoder_hidden = encoder_hidden
        self.latent_dim = latent_dim
        self.decoder_hidden = decoder_hidden
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.epochs = epochs
        self.k_max_iter = int(k_max_iter)
        self.batch_size = batch_size
        self.lr = lr
        self.lr_decay = lr_decay
        self.gamma = gamma
        self.init_lambda_a = init_lambda_a
        self.init_c_a = init_c_a
        self.c_a_thresh = c_a_thresh
        self.eta = eta
        self.multiply_h = multiply_h
        self.tau_a = tau_a
        self.h_tolerance = h_tolerance
        self.use_a_connect_loss = use_a_connect_loss
        self.use_a_positiver_loss = use_a_positiver_loss
        self.graph_threshold = graph_threshold
        self.optimizer = optimizer
        self.seed = seed
        self.device_type = device_type
        self.device_ids = device_ids

        if torch.cuda.is_available():
            logging.info('GPU is available.')
        else:
            logging.info('GPU is unavailable.')
            if self.device_type == 'gpu':
                raise ValueError("GPU is unavailable, "
                                 "please set device_type = 'cpu'.")
        if self.device_type == 'gpu':
            if self.device_ids:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(self.device_ids)
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        self.device = device

        self.input_dim = None

    def learn(self, data, columns=None, **kwargs):

        set_seed(self.seed)

        if data.ndim == 2:
            data = np.expand_dims(data, axis=2)
        self.n_samples, self.n_nodes, self.input_dim = data.shape

        if self.latent_dim is None:
            self.latent_dim = self.input_dim
        train_loader = func.get_dataloader(data, batch_size=self.batch_size, device=self.device)

        # =====initialize encoder and decoder=====
        # p = [5, 1, 0, 3, 4, 2, 7, 6]
        # p = [3, 17, 9, 5, 19, 26, 15, 33, 23, 8, 31, 6, 7, 36, 11, 0, 16, 27, 28, 18, 35, 25, 24, 32, 10, 21, 2, 4, 1,
        #      20, 29, 30, 12, 14, 34, 13, 22]
        # p = [9, 0, 6, 8, 7, 1, 5, 3, 2, 4, 10]
        # P = np.zeros((len(p), len(p)))
        # for i in range(len(p)-1):
        #     for j in range(i+1, len(p)):
        #         P[p[i]][p[j]] = 0.5
        # # print(P)
        # adj_A = torch.tensor(P, requires_grad=True, device=self.device)
        # print(adj_A)
        adj_A = torch.zeros((self.n_nodes, self.n_nodes), requires_grad=True, device=self.device)
        # structure = np.load('./dataset/Alarm/Structure3.5.npy')
        # adj_A = torch.tensor(0.5 * structure.T, requires_grad=True, device=self.device)
        self.encoder = Encoder(input_dim=self.input_dim,
                               hidden_dim=self.encoder_hidden,
                               output_dim=self.latent_dim,
                               adj_A=adj_A,
                               device=self.device,
                               encoder_type=self.encoder_type.lower()
                               )
        self.decoder = Decoder(input_dim=self.latent_dim,
                               hidden_dim=self.decoder_hidden,
                               output_dim=self.input_dim,
                               device=self.device,
                               decoder_type=self.decoder_type.lower()
                               )
        # =====initialize optimizer=====
        if self.optimizer.lower() == 'adam':
            optimizer = optim.Adam([{'params': self.encoder.parameters()},
                                    {'params': self.decoder.parameters()}],
                                   lr=self.lr)
        elif self.optimizer.lower() == 'sgd':
            optimizer = optim.SGD([{'params': self.encoder.parameters()},
                                   {'params': self.decoder.parameters()}],
                                  lr=self.lr)
        else:
            raise
        self.scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay, gamma=self.gamma)

        ################################
        # main training
        ################################
        c_a = self.init_c_a
        lambda_a = self.init_lambda_a
        h_a_new = torch.tensor(1.0)
        h_a_old = np.inf
        elbo_loss = np.inf
        best_elbo_loss = np.inf
        origin_a = adj_A

        epoch = 0
        for step_k in range(self.k_max_iter):
            while c_a < self.c_a_thresh:
                for epoch in range(self.epochs):
                    elbo_loss, origin_a = self._train(train_loader=train_loader,
                                                      optimizer=optimizer,
                                                      lambda_a=lambda_a,
                                                      c_a=c_a)
                    if elbo_loss < best_elbo_loss:
                        best_elbo_loss = elbo_loss
                if elbo_loss > 2 * best_elbo_loss:
                    break
                # update parameters
                a_new = origin_a.detach().clone()

                h_a_new = H_A(a_new, self.n_nodes)

                # expm_A = func.matrix_poly(abs(a_new), self.n_nodes)
                # h_a_new = torch.trace(expm_A) - self.n_nodes

                if h_a_new.item() > self.multiply_h * h_a_old:
                    c_a *= self.eta  # eta
                else:
                    break
                # break
            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
            h_a_old = h_a_new.item()

            logging.info(f"Iter: {step_k}, epoch: {epoch}, h_new: {h_a_old}")
            lambda_a += c_a * h_a_new.item()
            if h_a_old <= self.h_tolerance:
                break

        origin_a = origin_a.detach().cpu().numpy()

        origin_a[np.abs(origin_a) < self.graph_threshold] = 0
        origin_a[np.abs(origin_a) >= self.graph_threshold] = 1

        self.causal_matrix = Tensor(origin_a, index=columns, columns=columns)

    def _train(self, train_loader, optimizer, lambda_a, c_a):

        self.encoder.train()
        self.decoder.train()
        # structure = np.load('./dataset/Insurance/Structure_alpha.npy')
        # adj_A1 = torch.tensor(structure.T, requires_grad=True, device=self.device)
        # update optimizer
        optimizer, lr = func.update_optimizer(optimizer, self.lr, c_a)

        nll_train = []
        kl_train = []
        # str_train = []
        origin_a = None

        for batch_idx, (data, relations) in enumerate(train_loader):
            x = Variable(data).double()

            optimizer.zero_grad()

            logits, origin_a = self.encoder(x)
            z_gap = self.encoder.z
            z_positive = self.encoder.z_positive

            wa = self.encoder.wa

            x_pred = self.decoder(logits, adj_A=origin_a, wa=wa)  # X_hat

            # reconstruction accuracy loss
            loss_nll = func.nll_gaussian(x_pred, x)

            # s = 8
            # I_ = torch.eye(self.n_nodes).to('cuda')
            # h_A_loss = torch.slogdet(s * I_ - origin_a)[1] + self.n_nodes * np.log(s)

            # structure loss

            # origin_a1 = origin_a.detach().cpu().numpy()
            # origin_a1[np.abs(origin_a1) < self.graph_threshold] = 0
            # origin_a1[np.abs(origin_a1) >= self.graph_threshold] = 1
            # origin_a1 = torch.tensor(origin_a1, requires_grad=True, device=self.device)

            # loss_str = 0 * (
            #             1 - torch.mean(torch.cosine_similarity(torch.abs(origin_a), self.graph_threshold * adj_A1,
            #                                                    dim=1)))

            # loss_fn3 = torch.nn.MSELoss(reduction='mean')
            # loss_str = loss_fn3(origin_a, adj_A1)

            # KL loss
            loss_kl = func.kl_gaussian_sem(logits)

            # ELBO loss:
            # loss = loss_kl + loss_nll + loss_str
            loss = loss_kl + loss_nll
            # add A loss
            one_adj_a = origin_a  # torch.mean(adj_A_tilt_decoder, dim =0)
            sparse_loss = self.tau_a * torch.sum(torch.abs(one_adj_a))

            # other loss term
            if self.use_a_connect_loss:
                connect_gap = func.a_connect_loss(one_adj_a, self.graph_threshold, z_gap)
                loss += lambda_a * connect_gap + 0.5 * c_a * connect_gap * connect_gap

            if self.use_a_positiver_loss:
                positive_gap = func.a_positive_loss(one_adj_a, z_positive)
                loss += .1 * (lambda_a * positive_gap
                              + 0.5 * c_a * positive_gap * positive_gap)

            # compute h(A)

            # expm_A = matrix_poly(A*A, m)
            # h_A = torch.trace(expm_A) - m
            # return h_A

            h_A = H_A(origin_a, self.n_nodes)

            loss += (lambda_a * h_A
                     + 0.5 * c_a * h_A * h_A
                     + 100. * torch.trace(origin_a * origin_a)
                     + sparse_loss)  # +  0.01 * torch.sum(variance * variance)
            # loss += (lambda_a * torch.abs(h_A)
            #          + 0.5 * c_a * torch.abs(h_A)
            #          + 100. * torch.trace(torch.abs(origin_a))
            #          + sparse_loss)
            if np.isnan(loss.detach().cpu().numpy()):
                raise ValueError(f"The loss value is Nan, "
                                 f"suggest to set optimizer='adam' to solve it. "
                                 f"If you already set, please check your code whether has other problems.")
            loss.backward()
            optimizer.step()
            self.scheduler.step()

            nll_train.append(loss_nll.item())
            kl_train.append(loss_kl.item())
            # str_train.append(loss_str.item())

        # return np.mean(np.mean(kl_train) + np.mean(nll_train) + np.mean(str_train)), origin_a
        return np.mean(np.mean(kl_train) + np.mean(nll_train)), origin_a
