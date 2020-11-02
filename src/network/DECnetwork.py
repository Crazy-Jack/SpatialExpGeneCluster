'''
modified from https://github.com/xiaopeng-liao/DEC_pytorch
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from .resnet_deconv import ResNet18, DecovResNet18

class GELU(nn.Module):
    """Applies the Gaussian Error Linear Units function:

    .. math:: \text{GELU}(x) = x * \Phi(x)

    where :math:`\Phi(x)` is the Cumulative Distribution Function for Gaussian Distribution.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input

    .. image:: ../scripts/activation_images/GELU.png

    Examples::

        >>> m = nn.GELU()
        >>> input = torch.randn(2)
        >>> output = m(input)
    """
    def forward(self, input):
        return F.gelu(input)


class DECNetwork(nn.Module):
    def __init__(self, input_channel, feature_dim, latent_class_num, alpha=1.0, decode_constraint=False):
        '''
        temporarily hard-code layer_dims to have length of 3
        '''
        super(DECNetwork, self).__init__()
        self.input_channel = input_channel
        self.latent_class_num = latent_class_num
        self.pretrainMode = True
        self.alpha = alpha
        self.clusterCenter = nn.Parameter(torch.zeros(latent_class_num, feature_dim))

        self.encoder = ResNet18(img_channel=input_channel)
        print("define encoder in channel", input_channel)
        self.encoder_shape_before_avgpool = (2048, 1, 1) # for input size (bs, 7, 32, 32); \
                                                         # will be (2048, 2, 2) for input size (bs, 7, 64, 64)
        self.fc1 = nn.Linear(2048, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 2048)
        self.relu2 = nn.ReLU()
        self.decoder = DecovResNet18(self.encoder_shape_before_avgpool, img_channel=input_channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight)

    def setPretrain(self, mode):
        """To set training mode to pretrain or not,
        so that it can control to run only the Encoder or Encoder+Decoder"""
        self.pretrainMode = mode

    def clusterCenterInitialization(self, cc):
        """
        To update the cluster center. This is a method for pre-train phase.
        When a center is being provided by kmeans, we need to update it so
        that it is available for further training
        :param cc: the cluster centers to update, size of latent_class_num x feature_dim
        """
        assert self.pretrainMode == True

        self.clusterCenter.data = torch.from_numpy(cc)
        # print("CLUSTER CENTER: ", self.clusterCenter.data)

    def getTDistribution(self, x, clusterCenter):
        """
        student t-distribution, as same as used in t-SNE algorithm.
         q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.

         :param x: input data, in this context it is encoder output
         :param clusterCenter: the cluster center from kmeans
         """
        xe = torch.unsqueeze(x, 1).cuda() - clusterCenter.cuda()
        q = 1.0 / (1.0 + (torch.sum(torch.mul(xe,xe), 2) / self.alpha))
        q = q ** (self.alpha + 1.0) / 2.0
        q = (q.t() / torch.sum(q, 1)).t() # due to divison, we need to transpose q

        return q

    def forward(self, x):
        '''
        At pretrainMode: return latent_feature, reconstructed_input
        Not At pretrainMode: return latent_feature, p(Y|A)
        '''
        # print("forward encoder x", x.shape)
        x = self.encoder(x)
        # print(x.shape)
        encoder_shape = x.shape
        feature = self.fc1(x.reshape(-1, x.shape[1])) # [bsz, latent_dim]
        if self.pretrainMode:
            rec_x = self.relu2(self.fc2(feature))
            rec_x = self.decoder(rec_x.reshape(encoder_shape))
            return feature, rec_x
        else:
            # print("center shape: ", self.clusterCenter.shape)
            return feature, self.getTDistribution(feature, self.clusterCenter)





if __name__ == '__main__':
    attr_num, feature_dim, latent_class_num = 126, 32, 21
    torch.manual_seed(0)
    model = DECNetwork(attr_num, feature_dim, latent_class_num)
    x = torch.randint(2, (3000, 126)).float()
    model.clusterCenterInitialization(0.0001 * torch.rand(latent_class_num, feature_dim).numpy())
    model.setPretrain(False)
    feature, q = model(x)
    print(feature)
    print("q", q)
