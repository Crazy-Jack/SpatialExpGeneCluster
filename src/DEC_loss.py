'''
modified from https://github.com/xiaopeng-liao/DEC_pytorch
'''
import torch
import torch.nn as nn
from torch.autograd import Variable


class DECLoss(nn.Module):
    def __init__(self):
        super(DECLoss, self).__init__()
    
    def target_distribution(self, q):
        weight = q ** 2 / q.sum(0)
        return Variable((weight.t() / weight.sum(1)).t().data, requires_grad=True)

    def KL_div(self, q, p):
        # logsumexp
        # print("q: ", q)
        # print("p: ", p)
        # print("q shape: {}; p shape: {}".format(q.shape, p.shape))

        res = torch.sum(p * torch.log(p/q))
        # print("KLD: ", res)
        return res

    def forward(self, q):
        p = self.target_distribution(q)
        return self.KL_div(q, p)
    
