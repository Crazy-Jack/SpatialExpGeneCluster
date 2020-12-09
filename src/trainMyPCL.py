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

from utlis import set_args, set_optimizer
from utlis import save_model
from utlis import AverageMeter
from utlis import txt_logger
from network.resnet_deconv import SupConResNet
from losses import SupConLoss
from data_utlis import SpatialDataset
from data_utlis import MyTransform

def costomize_args(args):
    
    return args


def set_dataloader(args):
    """use args.dataset decide which dataset to use and return dataloader"""
    mytransform = MyTransform(args)
    train_transform = mytransform.train_transform(ssl=True)
    eval_transform = mytransform.val_transform()
    if args.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root=args.loading_path, train=True, download=True,
                            transform=train_transform)
        test_dataset = torchvision.datasets.MNIST(root=args.loading_path, train=False, download=True, 
                            transform=eval_transform)
    elif args.dataset == 'spatial':
        
        train_dataset = SpatialDataset(args.data_root, args.data_file_name, return_idx=True, transform=train_transform)
        test_dataset = SpatialDataset(args.data_root, args.data_file_name, return_idx=True, transform=eval_transform)

    else:
        raise NotImplemented("dataset {} is not implemented.".format(args.dataset))

    # train loader
    train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
    # test loader
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

    return train_dataloader, test_dataloader, train_dataset, test_dataset


def get_model(args, logger):
    model = SupConResNet(name=args.model)
    criterion = SupConLoss(temperature=args.temp)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("Used devices: {}".format(torch.cuda.device_count()))
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        
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

            logger.logger.info("Model loaded! Pretrained from epoch {}".format(args.pre_ssl_epoch))

    return model, criterion



def train(train_loader, model, optimizer, epoch, args, scheduler, criterion, label_store):
    """one epoch training
    params:
        - label_store: [len(train_loader),], a torch tensor, each entry represents its pseudo-class in this train epoch
    """
    # TODO: rewrite this and fill all empty lines!
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    for _, (images, _, idx) in tqdm(enumerate(train_loader), total=len(train_loader)):
        """params:
                img: [bz, C, H, W]
                labels: [bz,]
                idx: [bz,]
        """
        images = torch.cat([images[0], images[1]], dim=0)
        images = images.cuda()
        # labels = labels.cuda()
        bsz = idx.shape[0]

        # convert idx to real labels
        labels = torch.LongTensor(label_store[idx]).cuda() # [bz, ]

        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # compute loss
        loss = criterion(features, labels)
        
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.lr_scheduling == 'reduce': # reduce on pleatau
        scheduler.step(loss)

    return losses.avg


def kmean_cluster(args, model, k, train_loader, label_store):
    """perform clustering on given feture space"""
    features_list = []
    idxs = []
    model.eval()
    for _, (images, _, idx) in tqdm(enumerate(train_loader), total=len(train_loader)):
        with torch.no_grad():
            images = images.cuda()
            features = model(images)
            features_list.append(features.cpu().numpy())
            idxs.append(idx)

    features_np = np.concatenate(features_list, axis=0)
    idxs = np.concatenate(idxs, axis=0)

    print("clustering on ", features_np.shape)
    k_means = KMeans(n_clusters=k, n_init=20)
    k_means.fit(features_np)
    new_labels = k_means.labels_
    kmeans_loss = k_means.inertia_
    centers = k_means.cluster_centers_

    # print("idx max{} min {}; label_store {}".format(idxs.max(), idxs.min(), label_store.shape))
    label_store[idxs] = new_labels

    return label_store, centers, kmeans_loss


def main():
    args = set_args()
    args = costomize_args(args)
    # log
    scalar_logger = txt_logger(args.saving_path, args, 'python ' + ' '.join(sys.argv))
    # data loader
    train_loader, eval_loader, train_dataset, eval_dataset = set_dataloader(args)
    model, criterion = get_model(args, scalar_logger)
    optimizer, scheduler = set_optimizer(args, model)

    # training routine
    # resume model path
    if args.resume_model_path:
        start = args.pre_ssl_epoch
    else:
        start = 0

    kmeans_losses = []
    # k-means clustering for initialization
    print("Initialization (K-means) ---------")
    label_store = np.zeros(len(train_dataset), dtype=np.int32)
    print("---label_store size", label_store.shape)
    k_clusters = args.k_clusters
    label_store, centers, kmeans_loss = kmean_cluster(args, model, k_clusters, eval_loader, label_store)
    kmeans_losses.append(kmeans_loss) # initial kmeans loss
    # train
    losses = []
    print("Begin Training -------------------------")
    for epoch in range(start + 1, args.epochs + 1):
        # train for one epoch
        loss = train(train_loader, model, optimizer, epoch, args, scheduler, criterion, label_store)
        losses.append(loss)
        # file logger
        scalar_logger.log_value(epoch, ('loss', loss),
                                    ('learning_rate', optimizer.param_groups[0]['lr']),
                                    ('lc_len', len(np.unique(label_store))),
                                    )

        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs:
            save_file = os.path.join(
                args.saving_path, 'ckpt_epoch_{}.pth'.format(epoch+1))
            save_model(model, optimizer, args, epoch, save_file)
            # save loss
            np.save(os.path.join(args.saving_path, "train_loss.npy"), losses)
            # save latent class assignment
            save_file_lat_class_assign = os.path.join(args.saving_path, 'latent_class.npy')
            np.save(save_file_lat_class_assign, label_store)
            # log latent class
            unique_latent_class = np.unique(label_store)
            latent_class_stats = {}
            for i in unique_latent_class:
                latent_class_stats[i] = np.where(label_store == i)[0].shape[0]
            scalar_logger.log_value(epoch, ('lc_assign', latent_class_stats))

        if (epoch + 1) % args.kmeans_freq == 0:
            print("Perform (K-means) ---------")
            label_store, centers, kmeans_loss = kmean_cluster(args, model, k_clusters, eval_loader, label_store)
            kmeans_losses.append(kmeans_loss)
            np.save(os.path.join(args.saving_path, "Kmeans_loss.npy"), kmeans_losses)

    return 

    
if __name__ == '__main__':
    main()





    