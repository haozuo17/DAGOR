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
import pandas as pd
import os
import numpy as np
import torch

os.environ['CASTLE_BACKEND'] = 'pytorch'

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import Notears, GraNDAG, DAG_GNN, RL
from dagor import DAGOR as DR
from model.BDeu_score import reward_BDeu
from model.BIC_score import reward_BIC

type = 'SF'  # or `SF`
h = 2  # ER2 when h=5 --> ER5
# [10, 20, 40, 60]
for n_nodes in [10, 20, 40, 60]:
    n_edges = h * n_nodes
    method = 'nonlinear'
    sem_type = 'mlp'
    # method = 'linear'
    # sem_type = 'gauss'
    # sem_type = 'gp'
    if type == 'ER':
        weighted_random_dag = DAG.erdos_renyi(n_nodes=n_nodes, n_edges=n_edges,
                                              weight_range=(0.5, 2.0), seed=300)
    elif type == 'SF':
        weighted_random_dag = DAG.scale_free(n_nodes=n_nodes, n_edges=n_edges,
                                             weight_range=(0.5, 2.0), seed=300)
    else:
        raise ValueError('Just supported `ER` or `SF`.')

    dataset = IIDSimulation(W=weighted_random_dag, n=1000,
                            method=method, sem_type=sem_type)
    true_dag, X, sort = dataset.B, dataset.X, dataset.ordered_vertices

    # # #################### notears learn ###############################
    # nt = Notears(h_tol=1e-8)
    # nt.learn(X)
    #
    # # plot est_dag and true_dag
    # GraphDAG(nt.causal_matrix, true_dag)
    #
    # # calculate accuracy
    # met = MetricsDAG(nt.causal_matrix, true_dag)
    # print("************{}***NOTEARS***************".format(n_nodes))
    # print(met.metrics)
    #
    # # # ########################## GraNDAG ###############################
    # gnd = GraNDAG(input_dim=X.shape[1], device_type='GPU', iterations=3000)
    # gnd.learn(X)
    #
    # # plot predict_dag and true_dag
    # GraphDAG(gnd.causal_matrix, true_dag)
    #
    # # calculate accuracy
    # mm = MetricsDAG(gnd.causal_matrix, true_dag)
    # print("************{}***GraN-DAG***************".format(n_nodes))
    # print(mm.metrics)
    #
    # # # ########################## GNN learn #############################
    # gnn = DAG_GNN(device_type='gpu')
    # gnn.learn(X)
    # # plot est_dag and true_dag
    # GraphDAG(gnn.causal_matrix, true_dag)
    #
    # # calculate accuracy
    # met = MetricsDAG(gnn.causal_matrix, true_dag)
    # print("************{}***DAG-GNN***************".format(n_nodes))
    # print(met.metrics)

    ######################## DAGOR learn #############################
    # dr = DR(device_type='gpu')
    # dr.learn(X)
    # # plot est_dag and true_dag
    # GraphDAG(dr.causal_matrix, true_dag)
    #
    # # calculate accuracy
    # met = MetricsDAG(dr.causal_matrix, true_dag)
    # print("************{}***DAGOR***************".format(n_nodes))
    # print(met.metrics)

    # # # ########################## RL-BIC learn #############################
    # rl learn
    rl = RL(nb_epoch=100, device_type='gpu')
    rl.learn(X)

    # plot est_dag and true_dag
    GraphDAG(rl.causal_matrix, true_dag)

    # calculate accuracy
    met = MetricsDAG(rl.causal_matrix, true_dag)
    print("************{}***RL-BIC***************".format(n_nodes))
    print(met.metrics)
