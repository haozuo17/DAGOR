# %%
from Topo_utils import threshold_W, create_Z, find_idx_set, create_new_topo, create_new_topo_greedy, gradient_l1
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.special import expit as sigmoid
import scipy.linalg as slin
from copy import copy
import pandas as pd
from BDeu_score import reward_BDeu
from BIC_score import reward_BIC
from BICscore import score as bic_score1
import torch
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation

class TOPO_linear:
    def __init__(self, score, regress):
        super().__init__()
        self.score = score
        self.regress = regress

    def _init_W_slice(self, idx_y, idx_x):
        y = self.X[:, idx_y]
        x = self.X[:, idx_x]
        w = self.regress(X=x, y=y)
        return w

    def _init_W(self, Z):
        W = np.zeros((self.d, self.d))
        for j in range(self.d):
            if (~Z[:, j]).any():

                # non-linear
                M = self.regress(X=X[:, ~Z[:, j]], y=X[:, j])
                M_ = list(0.0 for _ in range(len(M[0])))
                for m in M:
                    M_ += m
                M__ = [item / len(M) for item in M_]
                W[~Z[:, j], j] = M__

                # linear
                # W[~Z[:, j], j] = self.regress(X=X[:, ~Z[:, j]], y=X[:, j])

            else:
                W[:, j] = 0
        return W

    def _h(self, W):
        """Evaluate value and gradient of acyclicity constraint.
        Option 1: h(W) = Tr(I+|W|/d)^d-d
        """

        """
        h(W) = -log det(sI-W*W) + d log (s)
        nabla h(W) = 2 (sI-W*W)^{-T}*W
        """
        I = np.eye(self.d)
        s = 1
        M = s * I - np.abs(W)
        h = - np.linalg.slogdet(M)[1] + self.d * np.log(s)
        G_h = slin.inv(M).T

        return h, G_h

    def _update_topo_linear(self, W, topo, idx, opt=1):

        topo0 = copy(topo)
        W0 = np.zeros_like(W)
        i, j = idx
        i_pos, j_pos = topo.index(i), topo.index(j)

        W0[:, topo[:j_pos]] = W[:, topo[:j_pos]]
        W0[:, topo[(i_pos + 1):]] = W[:, topo[(i_pos + 1):]]
        topo0 = create_new_topo(topo=topo0, idx=idx, opt=opt)
        for k in range(j_pos, i_pos + 1):
            if len(topo0[:k]) != 0:

                # non-linear
                M = self._init_W_slice(idx_y=topo0[k], idx_x=topo0[:k])
                M_ = list(0.0 for _ in range(len(M[0])))
                for m in M:
                    M_ += m
                M__ = [item / len(M) for item in M_]
                W0[topo0[:k], topo0[k]] = M__

                # linear
                # W0[topo0[:k], topo0[k]] = self._init_W_slice(idx_y=topo0[k], idx_x=topo0[:k])

            else:
                W0[:, topo0[k]] = 0
        return W0, topo0

    def fit(self, X, topo: list, no_large_search, size_small, size_large):
        self.n, self.d = X.shape
        self.X = X
        iter_count = 0
        large_space_used = 0
        if not isinstance(topo, list):
            raise TypeError
        else:
            self.topo = topo

        Z = create_Z(self.topo)
        self.Z = Z
        self.W = self._init_W(self.Z)
        loss, G_loss = self.score(X=self.X, W=self.W)
        h, G_h = self._h(W=self.W)
        idx_set_small, idx_set_large = find_idx_set(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small,
                                                    size_large=size_large)
        idx_set = list(idx_set_small)
        while bool(idx_set):

            idx_len = len(idx_set)
            loss_collections = np.zeros(idx_len)

            for i in range(idx_len):
                W_c, topo_c = self._update_topo_linear(W=self.W, topo=self.topo, idx=idx_set[i])
                loss_c, _ = self.score(X=self.X, W=W_c)
                loss_collections[i] = loss_c

            if np.any(loss > np.min(loss_collections)):
                self.topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss)

            else:
                if large_space_used < no_large_search:
                    idx_set = idx_set_large.difference(idx_set_small)
                    idx_set = list(idx_set)
                    idx_len = len(idx_set)
                    loss_collections = np.zeros(idx_len)
                    for i in range(idx_len):
                        W_c, topo_c = self._update_topo_linear(W=self.W, topo=self.topo, idx=idx_set[i])
                        loss_c, _ = self.score(X=self.X, W=W_c)
                        loss_collections[i] = loss_c

                    if np.any(loss > loss_collections):
                        large_space_used += 1
                        self.topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss)
                    else:
                        print("Using larger search space, but we cannot find better loss")
                        break


                else:
                    print("We reach the number of chances to search large space, it is {}".format(
                        no_large_search))
                    break

            self.Z = create_Z(self.topo)
            self.W = self._init_W(self.Z)
            loss, G_loss = self.score(X=self.X, W=self.W)
            h, G_h = self._h(W=self.W)
            idx_set_small, idx_set_large = find_idx_set(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small,
                                                        size_large=size_large)
            idx_set = list(idx_set_small)

            iter_count += 1

        return self.W, self.topo, Z, loss


if __name__ == '__main__':
    import utils
    from timeit import default_timer as timer


    # rd_int = int(np.random.randint(10000, size=1))
    #
    # print(rd_int)
    #
    # utils.set_random_seed(rd_int)
    # n, d, s0 = 100, 50, 200
    # graph_type, sem_type = 'ER', 'gauss'
    #
    # B_true = utils.simulate_dag(d, s0, graph_type)
    # W_true = utils.simulate_parameter(B_true)
    # X = utils.simulate_linear_sem(W_true, n, sem_type)

    ## Linear Model
    # def regress(X, y):
    #     reg = LinearRegression(fit_intercept=False)
    #     reg.fit(X=X, y=y)
    #     return reg.coef_

    # Non-linear

    def regress(X, y, C=0.5):
        reg = LogisticRegression(multi_class='ovr', fit_intercept=False, penalty='l1', C=C, solver='liblinear')
        # reg = LogisticRegression(multi_class='ovr', fit_intercept=False, penalty='l2', C=C, solver='lbfgs')
        # reg = LinearRegression(fit_intercept=False)
        reg.fit(X=X, y=y)
        return reg.coef_


    def score(X, W):
        M = X @ W

        # linear
        # R = X - M
        # loss = 0.5 / X.shape[0] * (R ** 2).sum()
        # G_loss = - 1.0 / X.shape[0] * X.T @ R

        # non-linear
        G_loss1 = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        G_loss = G_loss1 + gradient_l1(W, G_loss1, 2)

        bic = reward_bic.compute_score(torch.tensor(W.T))
        bic_ = bic.cpu().detach().numpy()
        # # asia, sachs: loss = bedu_.sum() * (-0.0001)
        loss = bic_.sum() * (-0.0001)
        # print("看一下小数点: ", loss)
        return loss, G_loss


    '''
    ## Logistic Model
    n, d, s0 = 10000, 20, 80
    graph_type, sem_type = 'ER', 'logistic'

    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)


    def regress(X,y,C = 0.1):
        reg = LogisticRegression(multi_class='ovr', fit_intercept=False, penalty='l1', C=C,
                                 solver='liblinear')                         
        reg.fit(X = X, y = y)
        return reg.coef_

    def score(X,W,C = 0.1):
        lambda1 = 1/C
        M = X @ W
        loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum() + lambda1 * (np.abs(W)).sum()
        G_loss1 = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        G_loss = G_loss1 + gradient_l1(W, G_loss1, lambda1)
        return loss, G_loss
    '''
    # **************************Synthetic data***************
    # type = 'ER'  # or `SF`
    # h = 2  # ER2 when h=5 --> ER5
    # n_nodes = 10
    # n_edges = h * n_nodes
    # # method = 'nonlinear'
    # # sem_type = 'mlp'
    # method = 'linear'
    # sem_type = 'gauss'
    # # sem_type = 'gp'
    # if type == 'ER':
    #     weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
    #                                           weight_range=(0.5, 2.0), seed=300)
    # elif type == 'SF':
    #     weighted_random_dag = DAG.scale_free(n_nodes=n_nodes, n_edges=n_edges,
    #                                          weight_range=(0.5, 2.0), seed=300)
    # else:
    #     raise ValueError('Just supported `ER` or `SF`.')
    #
    # dataset = IIDSimulation(W=weighted_random_dag, n=1000,
    #                         method=method, sem_type=sem_type)
    # true_dag, X = dataset.B, dataset.X
    #
    # df = pd.DataFrame(X)

    # df.to_csv('sample1.csv', index=False)
    # np.save('./sample1.npy', X.to_numpy())
    # ***************************************************************
    # data = pd.read_csv('./sample.csv')
    sort_output = "sort_output.txt"

    for dataset in ['Sachs', 'Insurance', 'Alarm', 'Hailfinder']:
        data = pd.read_csv('./dataset/{}/data.csv'.format(dataset))
        v = data.columns
        X = data.values
        d = len(v)
        datanp = data.to_numpy()
        # reward_bdeu = reward_BDeu(torch.tensor(datanp), len(data.columns))
        reward_bic = reward_BIC(torch.tensor(datanp), len(data.columns))
        # scorebdeu = reward_bdeu.compute_score(torch.tensor(true_dag.T))
        # true_dag = np.load('./dataset/Sachs/True_DAG.npy')
        # scorebic = reward_bic.compute_score(torch.tensor(true_dag.T))
        # print("***************Groundtruth***************")
        # print("BIC:", scorebic.sum())
        # print("BDeu:", scorebdeu.sum())
        # print("BIC:", bic_score1(data, true_dag.T))
        bic_0 = float('-inf')
        TOPO_est_ = []
        for i in range(25):
            print("数据集{}".format(dataset))
            print("第{}轮:".format(i + 1))
            model = TOPO_linear(regress=regress, score=score)
            topo_init = list(np.random.permutation(range(d)))
            # start = timer()
            causual_DAG, TOPO_est, _, _ = model.fit(X=X, topo=topo_init, no_large_search=10, size_small=100, size_large=1000)
            print(causual_DAG)
            # scorebdeu = reward_bdeu.compute_score(torch.tensor(causual_DAG.T))
            scorebic = reward_bic.compute_score(torch.tensor(causual_DAG.T))
            print("***************Causual_Sorts***************")
            # print("BDeu:", scorebdeu.sum())
            print("BIC:", scorebic.sum())
            bic_m = scorebic.sum()
            # acc = utils.count_accuracy(true_dag, threshold_W(W=causual_DAG) != 0)
            # tpr = acc.get('tpr')
            if bic_m > bic_0:
                TOPO_est_ = TOPO_est
                bic_0 = bic_m

            print("{}当前最优bic:".format(dataset), bic_0)
            TOPO_est_v = []
            # for i in TOPO_est_:
            #     TOPO_est_v.append(v[i])
            print("{}当前最优排序TOPO_est_v:".format(dataset), TOPO_est_)
        print("{}最优排序TOPO_est_v:".format(dataset), TOPO_est_)
        with open(sort_output, 'w') as file:
            file.write("{}最优排序:".format(dataset))
            file.write("{}\n".format(TOPO_est_))

