import argparse
import os, sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import ortho_group
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from utlis import set_args, set_optimizer
from utlis import save_model
from utlis import AverageMeter
from utlis import txt_logger
from network.DECnetwork import DECNetwork
from DEC_loss import DECLoss
from data_utlis import SpatialDataset


def costomize_args(args):
    
    return args


def set_dataloader(args):
    """use args.dataset decide which dataset to use and return dataloader"""
    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_dataset = torchvision.datasets.MNIST(root=args.loading_path, train=True, download=True, 
                            transform=transform)
        test_dataset = torchvision.datasets.MNIST(root=args.loading_path, train=False, download=True, 
                            transform=transform)
    elif args.dataset == 'spatial':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_dataset = SpatialDataset(args.data_root, args.data_file_name)
        test_dataset = SpatialDataset(args.data_root, args.data_file_name)

    else:
        raise NotImplemented("dataset {} is not implemented.".format(args.dataset))
    # train loader
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)
    # test loader
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

    return train_dataloader, test_dataloader


def get_model(args):
    model = DECNetwork(args.input_channel, args.feature_dim, args.latent_class_num,
                                alpha=1.0, decode_constraint=False)
    latent_class_criterion = DECLoss()

    rec_criterion = torch.nn.MSELoss()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Used devices: {}".format(torch.cuda.device_count()))
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        latent_class_criterion = latent_class_criterion.cuda()
        rec_criterion = rec_criterion.cuda()
        cudnn.benchmark = True

        if args.resume_model_path:
            # get pre ssl epoch
            ckpt = torch.load(args.resume_model_path, map_location='cpu')
            state_dict = ckpt['model']
            new_state_dict = {}
            for k, v in state_dict.items():
                if torch.cuda.device_count() > 1:
                    print(k)
                    #if k.split(".")[0] != 'head':
                    #    k = ".".join([k.split(".")[0], "module"] + k.split(".")[1:])
                else:
                    k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
            model.load_state_dict(state_dict)

            print("Model loaded from epoch {}!".format(args.pretrain_epoch))

    return model, latent_class_criterion, rec_criterion



def get_feature(train_loader, model, epoch, args, optimizer, scheduler, pretrain_criterion):
    model.eval()
    model.setPretrain(False)

    losses = AverageMeter()
    all_features = []
    Y_assignment = []
    for idx, (img, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = img.cuda()
        bsz = img.shape[0]
        # compute probrability
        with torch.no_grad():
            feature, prob = model(img)
            feature.detach()
        all_features.append(feature)

        # get Y and T assignment
        prob = prob.float()
        Y_assignment.extend(prob.argmax(dim=1).cpu().numpy())

    all_features_mat = torch.cat(all_features)
    return all_features_mat, Y_assignment

def main():
    args = set_args()
    args = costomize_args(args)

    train_loader, test_loader = set_dataloader(args)
    model, UnSup_criterion, pretrain_criterion = get_model(args)
    optimizer, scheduler = set_optimizer(args, model)


    if args.pretrain_mode == 'autoencoder':
        # pre_train
        pre_train_optimizer = optim.Adam(model.parameters(), weight_decay=5e-4)
        pre_train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pre_train_optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        epoch = 0
        all_features_mat, Y_assignment = get_feature(train_loader, model, epoch, args, pre_train_optimizer, pre_train_scheduler, pretrain_criterion)
        all_features_mat = all_features_mat.cpu().numpy()
        print(all_features_mat.shape)
        print(len(Y_assignment))
        # np.save(os.path.join(args.saving_path, "Y_assignment_{}.npy".format(args.pretrain_epoch)), Y_assignment)

    # tSNE
    converged_Y = np.load("/home/tianqinl/Code/SpatialExpGeneCluster/train_related/spatial/Y_assignment_100.npy")
    print("Begin t-SNE -----------------")
    X_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300).fit_transform(all_features_mat)
    print("X_embedding shape", X_embedded.shape)
    df_2d = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'lbl': converged_Y})
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="x", y="y",
        hue="lbl",
        palette=sns.color_palette("hls", len(np.unique(converged_Y))),
        data=df_2d,
        legend=False,
        alpha=0.3
    )
    plt.savefig(os.path.join(args.saving_path, 'tSNE_{}.png'.format(args.pretrain_epoch)))



    

        
    return 

    
if __name__ == '__main__':
    """
    python plot_tSNE.py --resume_model_path /home/tianqinl/Code/SpatialExpGeneCluster/train_related/spatial_old/ckpt_epoch_100.pth \
                         --dataset spatial --input_channel 7 --pre_train_epochs 20 --pretrain_mode autoencoder --data_root ../data/spatial_Exp/32-32/
    """
    main()





    