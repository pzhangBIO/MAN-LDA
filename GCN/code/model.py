import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution
from torch.nn.parameter import Parameter



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
        self.weight = Parameter(torch.FloatTensor(128, 128))
        

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)  

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        print('z',z)
        support = torch.mm(z, self.weight)
        output = torch.mm(support, z.t())
        print('output',output)
        #adj = torch.mm(z, z.t())
        adj = self.sigmoid(output)
       # print('adj',torch.mm(z, z.t()))

       # print(adj)
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
