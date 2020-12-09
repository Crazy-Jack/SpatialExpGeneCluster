import torch
from torch import nn
import torch.nn.functional as F
import torchvision
'''
batch_size 8
epoch every 100 checkpoint
learning_rate 1e-4?
'''


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, downblock, num_layers, data_channel):
        super(ResNet, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(data_channel, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_downlayer(downblock, 64, num_layers[0])
        self.layer2 = self._make_downlayer(downblock, 128, num_layers[1],
                                           stride=2)
        self.layer3 = self._make_downlayer(downblock, 256, num_layers[2],
                                           stride=2)
        self.layer4 = self._make_downlayer(downblock, 512, num_layers[3],
                                           stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels * block.expansion),
            )
        layers = []
        layers.append(
            block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # self.shape_before_avgpool = x.shape[1:]
        x = self.avgpool(x).reshape(x.shape[:2])
        # print("Resnet", x.shape)
        return x

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)

        out += shortcut
        out = self.relu(out)

        return out


class DeconvResNet(nn.Module):
    def __init__(self, upblock, num_layers, out_channel, z_dim, shape_before_avgpool):
        super(DeconvResNet, self).__init__()

        self.in_channels = 2048

        ## decoder
        self.shape_before_avgpool = shape_before_avgpool
        self.fc_layer_dim = shape_before_avgpool[-3] * shape_before_avgpool[-2] * shape_before_avgpool[-1]
        self.fc = nn.Linear(z_dim, self.fc_layer_dim)
        self.relu2 = nn.ReLU()
        self.uplayer1 = self._make_up_block(
            upblock, 512,  num_layers[3], stride=2)
        self.uplayer2 = self._make_up_block(
            upblock, 256, num_layers[2], stride=2)
        self.uplayer3 = self._make_up_block(
            upblock, 128, num_layers[1], stride=2)
        self.uplayer4 = self._make_up_block(
            upblock, 64,  num_layers[0], stride=2)

        upsample = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,  # 256
                               64,
                               kernel_size=1, stride=2,
                               bias=False, output_padding=1),
            nn.BatchNorm2d(64),
        )
        self.uplayer_top = DeconvBottleneck(
            self.in_channels, 64, 1, 2, upsample)

        self.conv1_1 = nn.ConvTranspose2d(64, out_channel, kernel_size=1, stride=1,
                                          bias=False)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        # expansion = block.expansion
        if stride != 1 or self.in_channels != init_channels * 2:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels * 2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(init_channels * 2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(
            block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def forward(self, x, img_size=None):
        x = self.fc(x.reshape(-1, x.shape[1]))
        x = self.relu2(x)
        x = x.reshape(-1, self.shape_before_avgpool[0], self.shape_before_avgpool[1], self.shape_before_avgpool[2])
        x = self.uplayer1(x)
        x = self.uplayer2(x)
        x = self.uplayer3(x)
        x = self.uplayer4(x)
        x = self.uplayer_top(x)
        if img_size:
            x = self.conv1_1(x, output_size=img_size)#, output_size=image_size)
        else:
            x = self.conv1_1(x)
        return x

    # def forward(self, x):
    #     img = x
    #     tmp1 = self.encoder(x) # torch.Size([2, 2048, 7, 7])
    #     print("encoder output", tmp1.shape)
    #     tmp2 = self.decoder(tmp1, img.size())

    #     return tmp1, tmp2


def DecovResNet50(shape_before_enc_avgpool, img_channel=7):
    return DeconvResNet(DeconvBottleneck, [3, 4, 6, 3], img_channel, 2048, shape_before_enc_avgpool)

def DecovResNet18(shape_before_enc_avgpool, img_channel=7):
    return  DeconvResNet(DeconvBottleneck, [2, 2, 2, 2], img_channel, 2048, shape_before_enc_avgpool)

def ResNet50(img_channel=7, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], img_channel)

def ResNet18(img_channel=7, **kwargs):
    return ResNet(Bottleneck, [2, 2, 2, 2], img_channel)

model_dict = {
    'resnet18': [ResNet18, 2048],
    'resnet50': [ResNet50, 2048],
}


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet50', head='mlp', feat_dim=128, img_channel=1):
        super(SupConResNet, self).__init__()
        
        model_fun, dim_in = model_dict[name]
        self.encoder_out_dim = dim_in
        self.encoder = model_fun(img_channel=img_channel)
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat



if __name__ == "__main__":
    encoder = ResNet18(img_channel=3).cuda()
    input = torch.autograd.Variable(torch.randn(2, 3, 64, 64)).cuda()
    feat = encoder(input)
    # print("encoder outshape", encoder.shape_before_avgpool)
    print("latent", feat.shape)
    encoder_shape_before_avgpool = (2048, 2, 2)
    decoder = DecovResNet18(encoder_shape_before_avgpool, img_channel=3).cuda()
    out_img = decoder(feat)
    print("out_img", out_img.shape)
    # test
    # from torchsummary import summary
    # summary(decoder, (7, 32, 32))
    # summary(encoder, (7, 32, 32))

    # '''
    # load pre_trained_model
    # '''
    # pretrained_dict = torch.load("./resnet50-19c8e357.pth")
    # model_dict = model.state_dict()
    # # 1. filter out unnecessary keys
    # pretrained_dict = {k: v for k,
    #                    v in pretrained_dict.items() if k in model_dict}
    # # 2. overwrite entries in the existing state dict
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    

    # input = torch.autograd.Variable(torch.randn(2, 7, 32, 32)).cuda()
    # o = decoder(input)
    # print(o)