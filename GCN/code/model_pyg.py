import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution
from torch.nn.parameter import Parameter

from torch import Tensor
import torch_geometric as tg
from torch_geometric.typing import OptTensor
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
from torch_geometric.nn.glob import *
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import GINConv,GINEConv,APPNP,GCNConv,GraphConv,SAGEConv


class SelfAttnConv(MessagePassing):
    def __init__(self, emb_dim, attn_dim=0, num_relations=1, reverse=False):
        super(SelfAttnConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        attn_dim = attn_dim if attn_dim > 0 else emb_dim
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, attn_dim)
        else:
            self.wea = False
        self.attn_lin = nn.Linear(attn_dim, 1)

    # h_attn, edge_attr are optional
    def forward(self, h, edge_index, edge_attr=None, h_attn=None, **kwargs):
        # HACK assume x contains only message sources
        if edge_index is None:
            h_attn = h_attn if h_attn is not None else h
            attn_weights = self.attn_linear(h_attn).squeeze(-1)
            attn_weights = F.softmax(attn_weights, dim=-1)
            return torch.mm(attn_weights, h)
   
        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        return self.propagate(edge_index, h=h, edge_attr=edge_embedding, h_attn=h_attn)

    def message(self,edge_index, h_j, edge_attr, h_attn_j):

        h_attn = h_attn_j if h_attn_j is not None else h_j
        h_attn = h_attn + edge_attr if self.wea else h_attn
  
        index = edge_index[0]
      
        a_j = self.attn_lin(h_attn)
        a_j = softmax(a_j, index)
        t = h_j * a_j
  
        return t

    def update(self, aggr_out):
        return aggr_out



class GatedSumConv(MessagePassing):  # dvae needs outdim parameter
    def __init__(self, emb_dim, num_relations=1, reverse=False, mapper=None, gate=None):
        super(GatedSumConv, self).__init__(aggr='add', flow='target_to_source' if reverse else 'source_to_target')

        assert emb_dim > 0
        if num_relations > 1:
            self.wea = True
            self.edge_encoder = torch.nn.Linear(num_relations, emb_dim)
        else:
            self.wea = False
        self.mapper = nn.Linear(emb_dim, emb_dim) if mapper is None else mapper
        self.gate = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.Sigmoid()) if gate is None else gate

    def forward(self, x, edge_index, edge_attr=None, **kwargs):
        # HACK assume x contains only message sources
        if edge_index is None:
            h = self.gate(x) * self.mapper(x)
            return torch.sum(h, dim=1)
        edge_embedding = self.edge_encoder(edge_attr) if self.wea else None
        #print('edge_embedding',edge_embedding)
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding)

    def message(self, x_j, edge_attr):
        h_j = x_j + edge_attr if self.wea else x_j
        return self.gate(h_j) * self.mapper(h_j)

    def update(self, aggr_out):
        return aggr_out

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(out_channels, out_channels)
        self.linear2 = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x



class Gate_GNN(nn.Module):
    def __init__(self, feature, hidden1, hidden2,dropout=0.5):
        super(Gate_GNN, self).__init__()
        #GraphConv
        self.encoder_o1 = GraphConv(hidden1, hidden1,aggr='add',bias= True)
        self.encoder_o2 = GraphConv(hidden1, hidden1,aggr='add',bias= True)
        # self.mlp_o1 = MLP(hidden1, hidden1)
        # self.mlp_o2 = MLP(hidden1 * 2, hidden2)
        # self.encoder_o1 = GINConv(self.mlp_o1,eps=0., train_eps=True)
        # self.encoder_o2 = GINConv(self.mlp_o1,eps=0., train_eps=True)   
        # self.encoder_o1 = SAGEConv(hidden1, hidden1,aggr='mean',bias= True)
        # self.encoder_o2 = SAGEConv(hidden1, hidden1,aggr='mean',bias= True)

        self.gate=GatedSumConv(emb_dim=64, num_relations=1, reverse=False, mapper=None, gate=None)
        self.attn=SelfAttnConv(emb_dim=64, attn_dim=0, num_relations=1, reverse=False)

        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.mlp = nn.Linear(hidden1, hidden1)
        self.mlp2 = nn.Linear(hidden1, hidden1)
        self.mlp3 = nn.Linear(hidden1, 64)
        self.ln=nn.LayerNorm([feature,hidden1])


        self.dropout = nn.Dropout(p=0.5)
        

    def forward(self, x, adj):
        
        x1 = F.relu(self.encoder_o1(x, adj))
        x1 = self.dropout(x1)
        x1 = self.encoder_o1(x1, adj)
        #x1=self.gate(x1, adj,edge_attr=None)
        #x1=self.attn(x1, adj,edge_attr=None, h_attn=None)
        #print('1x1',x1.shape)
       # x1=F.relu(self.encoder_o2(x1,adj))
        #x1 = self.dropout(x1)
        #x1=self.gate(x1, adj,edge_attr=None)
        #x1=self.attn(x1, adj,edge_attr=None, h_attn=None)
       # x1 = self.dropout(x1)
       # x2=self.mlp(x1)
       # x1=self.ln(x2)
       # x1=self.mlp2(x1)
       # x1=self.mlp3(x1)
        #print('x1',x1.shape)
        return self.dc(x1),x1


        

class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(hidden_dim1, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)
        self.mlp = nn.Linear(hidden_dim2, hidden_dim2)
        self.ln=nn.LayerNorm([input_feat_dim,hidden_dim1])
    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        mu=self.mlp(mu)
        #print(mu.shape)
        logvar=self.mlp(logvar)
        mu=self.ln(mu)
        print('mu',mu)
        z = self.reparameterize(mu, logvar)
        return self.dc(mu), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.sigmoid = nn.Sigmoid()
        self.weight = Parameter(torch.FloatTensor(64, 64))
        

    def reset_parameters(self):
        #torch.nn.init.xavier_uniform_(self.weight)  
        torch.nn.init.normal_(self.weight)
    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
       # print('z',z)
        #support = torch.mm(z, self.weight)
        #print('support',support)
        output = torch.mm(z, z.t())
       # print('output',output.shape)

        adj = self.sigmoid(output)

        return adj

class mlp(nn.Module):
    def __init__(self, nhid, nclass, dropout):
        super(mlp, self).__init__()

        self.mlp1 = nn.Linear(nhid, nclass)
        self.mlp2 = nn.Linear(nclass, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp1.weight,std=0.05)
        nn.init.normal_(self.mlp2.weight,std=0.05)

    def forward(self, x):
        x=torch.from_numpy(x)
        x = self.mlp1(x)
        x = self.mlp2(x)


        return x
