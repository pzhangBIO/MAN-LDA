import torch
import torch.nn.modules.loss
import torch.nn.functional as F


def loss_function(preds, labels, mu, logvar, n_nodes, norm, pos_weight):

    criterion = torch.nn.BCELoss(reduction='mean')
                
                
    print('preds',preds)
    print('labels',labels)
    cost = criterion(preds, labels)
    #cost = norm * F.binary_cross_entropy_with_logits(preds, labels, pos_weight=pos_weight)
    
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
  #  KLD = -0.5 / n_nodes * torch.mean(torch.sum(
   #     1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))

   # print('KLD',KLD)
    return cost #+ KLD
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +     # calmp夹断用法
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))     
 

        return loss_contrastive
