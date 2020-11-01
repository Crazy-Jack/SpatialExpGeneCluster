import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
from tqdm import tqdm
import pickle
from PIL import Image
import os


class SpatialDataset(Dataset):
    """My dataset for spatial expression data"""

    def __init__(self, folder, name):
        self.data = np.load(os.path.join(folder, name)).transpose(0, 3, 1, 2)
        print(self.data.shape)
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx, :, :32, :32]), ""



