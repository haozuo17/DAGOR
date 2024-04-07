import pandas as pd
import os
import numpy as np
import torch

os.environ['CASTLE_BACKEND'] = 'pytorch'

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
# from castle.adrorithms import Notears, GraNDAG, DAG_GNN, GAE, RL
from dagor import DAGOR as DR
from BICscore import score as bic_score
from BDeuscore import score as bdeu_score
from model.BDeu_score import reward_BDeu
from model.BIC_score import reward_BIC


# for dataset in ['Sachs', 'Insurance', 'Alarm', 'Hailfinder']:
for dataset in ['Hailfinder']:
    data = pd.read_csv('./dataset/{}/data.csv'.format(dataset), dtype=float)
    Y = data.values
    true_dag = np.load('./dataset/{}/True_DAG.npy'.format(dataset))

    print("************{}***Groundtruth***************".format(dataset))
    # # calculate BDeuScore
    # bdeu_sco = bdeu_score(data, true_dag.T)
    # print("BDeu:", bdeu_sco)
    # # calculate BICScore
    # bic_sco = bic_score(data, true_dag.T)
    # print("BIC:", bic_sco)
    datanp = data.to_numpy()
    reward_bdeu = reward_BDeu(torch.tensor(datanp), len(data.columns))
    reward_bic = reward_BIC(torch.tensor(datanp), len(data.columns))
    scorebdeu = reward_bdeu.compute_score(torch.tensor(true_dag.T))
    scorebic = reward_bic.compute_score(torch.tensor(true_dag.T))
    print("BDeu:", scorebdeu.sum())
    print("BIC:", scorebic.sum())

    # # #################### notears learn ###############################
    # nt = Notears()
    # nt.learn(Y)
    #
    # # plot est_dag and true_dag
    # GraphDAG(nt.causal_matrix, true_dag)
    #
    # # calculate accuracy
    # met = MetricsDAG(nt.causal_matrix, true_dag)
    # print("************{}***NOTEARS***************".format(dataset))
    # print(met.metrics)
    #
    # causual_DAG = nt.causal_matrix
    # scorebdeu = reward_bdeu.compute_score(torch.tensor(causual_DAG.T))
    # scorebic = reward_bic.compute_score(torch.tensor(causual_DAG.T))
    #
    # print("BDeu:", scorebdeu.sum())
    # print("BIC:", scorebic.sum())
    #
    # # # calculate BDeuScore
    # # bdeu_sco = bdeu_score(data, nt.causal_matrix.T)
    # # print("BDeu:", bdeu_sco)
    # # # calculate BICScore
    # # bic_sco = bic_score(data, nt.causal_matrix.T)
    # # print("BIC:", bic_sco)
    #
    # # # ########################## GraNDAG ###############################
    # gnd = GraNDAG(input_dim=Y.shape[1], device_type='GPU')
    # gnd.learn(Y)
    #
    # # plot predict_dag and true_dag
    # GraphDAG(gnd.causal_matrix, true_dag)
    #
    # # calculate accuracy
    # mm = MetricsDAG(gnd.causal_matrix, true_dag)
    # print("************{}***GraN-DAG***************".format(dataset))
    # print(mm.metrics)
    #
    # causual_DAG = gnd.causal_matrix
    # scorebdeu = reward_bdeu.compute_score(torch.tensor(causual_DAG.T))
    # scorebic = reward_bic.compute_score(torch.tensor(causual_DAG.T))
    #
    # print("BDeu:", scorebdeu.sum())
    # print("BIC:", scorebic.sum())
    # # # calculate BDeuScore
    # # bdeu_sco = bdeu_score(data, gnd.causal_matrix.T)
    # # print("BDeu:", bdeu_sco)
    # # # calculate BICScore
    # # bic_sco = bic_score(data, gnd.causal_matrix.T)
    # # print("BIC:", bic_sco)
    #
    # # ########################## GNN learn #############################
    # gnn = DAG_GNN(device_type='gpu')
    # gnn.learn(Y)
    # # plot est_dag and true_dag
    # GraphDAG(gnn.causal_matrix, true_dag)
    #
    # # calculate accuracy
    # met = MetricsDAG(gnn.causal_matrix, true_dag)
    # print("************{}***GNN-DAG***************".format(dataset))
    # print(met.metrics)
    #
    # causual_DAG = gnn.causal_matrix
    # scorebdeu = reward_bdeu.compute_score(torch.tensor(causual_DAG.T))
    # scorebic = reward_bic.compute_score(torch.tensor(causual_DAG.T))
    #
    # print("BDeu:", scorebdeu.sum())
    # print("BIC:", scorebic.sum())
    # # calculate BDeuScore
    # bdeu_sco = bdeu_score(data, gnn.causal_matrix.T)
    # print("BDeu:", bdeu_sco)
    # # calculate BICScore
    # bic_sco = bic_score(data, gnn.causal_matrix.T)
    # print("BIC:", bic_sco)

    # ########################## DAGOR learn #############################
    dr = DR(device_type='gpu')
    dr.learn(Y)
    # plot est_dag and true_dag
    GraphDAG(dr.causal_matrix, true_dag)

    # calculate accuracy
    met = MetricsDAG(dr.causal_matrix, true_dag)
    print("************{}***DAGOR***************".format(dataset))
    print(met.metrics)

    # # calculate BDeuScore
    causual_DAG = dr.causal_matrix
    scorebdeu = reward_bdeu.compute_score(torch.tensor(causual_DAG.T))
    scorebic = reward_bic.compute_score(torch.tensor(causual_DAG.T))

    print("BDeu:", scorebdeu.sum())
    print("BIC:", scorebic.sum())
    # bdeu_sco = bdeu_score(data, dr.causal_matrix.T)
    # print("BDeu:", bdeu_sco)
    # # calculate BICScore
    # bic_sco = bic_score(data, dr.causal_matrix.T)
    # print("BIC:", bic_sco)
    # # ########################## RL-BIC learn #############################
    # # rl learn
    # rl = RL(nb_epoch=100, device_type='gpu')
    # rl.learn(Y)
    #
    # # plot est_dag and true_dag
    # GraphDAG(rl.causal_matrix, true_dag)
    #
    # # calculate accuracy
    # met = MetricsDAG(rl.causal_matrix, true_dag)
    # print("************{}***RL-BIC***************".format(dataset))
    # print(met.metrics)
    # causual_DAG = rl.causal_matrix
    # scorebdeu = reward_bdeu.compute_score(torch.tensor(causual_DAG.T))
    # scorebic = reward_bic.compute_score(torch.tensor(causual_DAG.T))
    #
    # print("BDeu:", scorebdeu.sum())
    # print("BIC:", scorebic.sum())