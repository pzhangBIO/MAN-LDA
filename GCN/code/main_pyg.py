from __future__ import division
from __future__ import print_function

import argparse
import time

import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn as nn
import torch
from model_pyg import mlp,Gate_GNN
from optimizer import loss_function,ContrastiveLoss
from utils import load_data, mask_test_edges, preprocess_graph, get_roc_score,fuse,EarlyStopping
import pandas as pd
from torch_geometric.data import Data

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=128, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=128, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
#parser.add_argument('--GeneFeat', type=str, default='../Process_data/Step7_GeneAttributes.txt', help='gene feature')


args = parser.parse_args()

##input data
gtr_adj, gtr_features= load_data(args)




gtr_n_nodes, gtr_feat_dim = gtr_features.shape


gtr_adj_orig = gtr_adj.copy()


gtr_adj_orig = gtr_adj_orig - sp.dia_matrix((gtr_adj_orig.diagonal()[np.newaxis, :], [0]), shape=gtr_adj_orig.shape)

gtr_adj_orig.eliminate_zeros()

print('step 1')
gtr_adj_train, gtr_train_edges = mask_test_edges(gtr_adj)

gtr_adj = gtr_adj_train


# Some preprocessing norm 
gtr_adj_norm = preprocess_graph(gtr_adj)
gtr_adj_orig = preprocess_graph(gtr_adj_orig)


gtr_adj_label = gtr_adj_train + sp.eye(gtr_adj_train.shape[0])
gtr_adj_label = torch.FloatTensor(gtr_adj_label.toarray())

gtr_pos_weight = torch.tensor(float(gtr_adj.shape[0] * gtr_adj.shape[0] - gtr_adj.sum()) / gtr_adj.sum())

gtr_norm = gtr_adj.shape[0] * gtr_adj.shape[0] / float((gtr_adj.shape[0] * gtr_adj.shape[0] - gtr_adj.sum()) * 2)

gtr_model = Gate_GNN(gtr_n_nodes, args.hidden1, args.hidden2, args.dropout)
optimizer = optim.Adam(list(gtr_model.parameters()), lr=args.lr)

#early_stopping = EarlyStopping(patience=5, verbose=False)
print('step 2')
hidden_emb = None
for epoch in range(args.epochs):
    t = time.time()
    gtr_model.train()
    optimizer.zero_grad()

              
    gtr_data=Data(x=gtr_features, edge_index=gtr_adj_norm)
    gtr_recovered, gtr_mu=gtr_model(gtr_data.x, gtr_data.edge_index)
    gtr_loss = loss_function(preds=gtr_recovered, labels=gtr_adj_label,
                            mu=gtr_mu, logvar=gtr_mu, n_nodes=gtr_n_nodes,
                            norm=gtr_norm, pos_weight=gtr_pos_weight)

    loss=   gtr_loss              

    loss.backward()
    cur_loss = loss.item()
    optimizer.step()
    
    
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss),
            "time=", "{:.5f}".format(time.time() - t)
            )


gtr_model.eval()


with torch.no_grad():

    gtr_data=Data(x=gtr_features, edge_index=gtr_adj_orig)
 
    gtr_recovered, gtr_mu=gtr_model(gtr_data.x, gtr_data.edge_index)
    hidden_emb=gtr_mu.numpy()
    print(hidden_emb)
    pd.DataFrame(hidden_emb).to_csv('./hidden_emb.csv',index=0,header=0)
    