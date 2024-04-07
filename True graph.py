import pyAgrum as gum
from pgmpy import readwrite
from pgmpy.models import BayesianNetwork
import pandas as pd
import numpy as np

for dataset in ['Survey']:
    bn = gum.loadBN("./dataset/{}/{}.bif".format(dataset, dataset))
    gum.generateCSV(bn, "./dataset/{}/data.csv".format(dataset), 1000, True)

    bifmodel = readwrite.BIF.BIFReader(path="./dataset/{}/{}.bif".format(dataset, dataset))
    datadf = pd.read_csv("./dataset/{}/data.csv".format(dataset))
    nodes = datadf.columns
    print(nodes)
    model = BayesianNetwork(bifmodel.variable_edges)
    model.name = bifmodel.network_name
    model.add_nodes_from(bifmodel.variable_names)

    # print(model.edges())
    print(len(model.edges()))
    a = len(nodes)
    w_g = np.zeros((a, a))
    for (i, j) in model.edges():
        w_g[list(nodes).index(i), list(nodes).index(j)] = 1
    print(w_g)
    np.save('./dataset/{}/True_DAG.npy'.format(dataset), w_g)
