import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image
import os



class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding

class NewPad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return F.pad(img, get_padding(img), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)

# data_transforms = {
#     'train': transforms.Compose([
#         NewPad(),
#         transforms.Resize((224,224)),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
#     'val': transforms.Compose([
#         NewPad(),
#         transforms.Resize((224,224)),
#         transforms.Grayscale(num_output_channels=3),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

class MyTransform:
    """Class for costomize transform"""
    def __init__(self, opt):
        super(MyTransform).__init__()
        # normolize
        self.mean = (0.5,)
        self.std = (0.5,)
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.opt = opt

    def train_transform(self, ssl=True, rrc_scale=(0.8, 1.)):
        """Transform for train"""
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.opt.img_size, scale=rrc_scale),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            self.normalize,
        ])

        if ssl:
            train_transform = TwoCropTransform(train_transform)

        return train_transform

    def val_transform(self, rrc_scale=(0.8, 1.)):
        """Transform for val"""
        val_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=self.opt.img_size, scale=rrc_scale),
            transforms.ToTensor(),
            self.normalize,
        ])

        return val_transform


def block_stack(imgs):
    """stack a block of images
    to pad the images as square as possible
    params: 
        - imgs: a list of numpy arrays, e.g. img=[np.random.rand(10, 1, 32, 32) for i in range(7)]
    """
    # calculate how many blocks are should be there
    square_num = int(np.ceil(np.sqrt(len(imgs))))
    # pad the images
    pad_shape = imgs[0].shape
    padded_list = imgs + [np.zeros(pad_shape) for i in range(square_num ** 2 - len(imgs))]

    # building the blocks list
    blocked_pad_img = np.block([[padded_list[i * square_num + j] for j in range(square_num)] for i in range(square_num)])
    return blocked_pad_img




class SpatialDataset(Dataset):
    """My dataset for spatial expression data"""

    def __init__(self, folder, name, return_idx=False, transform=None):
        self.data = np.load(os.path.join(folder, name)).transpose(0, 3, 1, 2)
        self.data_list = np.array_split(self.data, self.data.shape[1], axis=1)
        self.data = block_stack(self.data_list)
        # print(self.data.shape)
        self.return_idx = return_idx
        self.transform = transform
        # print(self.data.max())
        # normalized to 256
        self.data_min, self.data_max = self.data.min(), self.data.max()
        self.data = (self.data - self.data_min) / (self.data_max - self.data_min)
        # print(self.data.min())
        
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img = self.data[idx, :, :, :].transpose(1, 2, 0)
        # print("img inside shape {}, max {}, min {}".format(img.shape, img.max(), img.min()))
        # load
        img = img.reshape(img.shape[0], img.shape[1])
        img = Image.fromarray(img, mode='L')
        # transform
        if not (self.transform is None):
            img = self.transform(img)
        # print(img.max())
        if self.return_idx:
            return img, "", idx 
        else:
            return img, ""



if __name__ == '__main__':
    transform = transforms.Compose([
            transforms.RandomResizedCrop(size=48, scale=(0.4, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
        ])
    dataset = SpatialDataset('../data/spatial_Exp/32-32/', 'data.npy', transform=transform)
    print(dataset[22][0])
    # a = [np.random.rand(2,2) for i in range(7)]
    # imgs = block_stack(a)
    # print(imgs)
    # print(imgs.shape)
    # for i in a:
    #     print(i)