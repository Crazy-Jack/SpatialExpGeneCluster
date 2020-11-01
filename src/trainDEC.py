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
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
    # test loader
    test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

    return train_dataloader, test_dataloader


def get_model(args, logger):
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

            logger.logger.info("Model loaded! Pretrained from epoch {}".format(opt.pre_ssl_epoch))

    return model, latent_class_criterion, rec_criterion



def pre_train(train_loader, model, epoch, args, optimizer, scheduler, pretrain_criterion):
    model.train()
    model.setPretrain(True)

    losses = AverageMeter()

    for idx, (img, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        img = img.cuda()
        bsz = img.shape[0]

        # compute probrability
        feature, rec_img = model(img)

        # compute loss
        loss = pretrain_criterion(rec_img, img)

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if args.use_scheduler_pretrain:
        scheduler.step(loss)

    return losses.avg


def train(train_loader, model, optimizer, epoch, args, scheduler, UnSup_criterion, rec_criterion=None):
    """one epoch training"""
    # TODO: rewrite this and fill all empty lines!
    model.train()
    model.setPretrain(False)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    Y_assignment = []
    T_assignment = []
    for idx, (img, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        """params:
                img: [bz, C, H, W]
                labels: [bz,]
        """
        img = img.cuda()
        # labels = labels.cuda()
        bsz = img.shape[0]

        # compute probrability - p(y|a) dim: [bsz, |Y|]
        features, prob = model(img)
        prob = prob.float()
        # get Y and T assignment
        Y_assignment.extend(prob.argmax(dim=1).cpu().numpy())

        # compute loss
        # DEC loss
        loss = UnSup_criterion(prob)
        
        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if args.lr_scheduling == 'reduce': # reduce on pleatau
        scheduler.step(loss)

    # # compute H(Y|T) and I(Y;T)
    # # TODO: LOW PRIORITY!! use pytorch to implement H and MI calucalation
    # Y_assignment = np.array(Y_assignment)
    # # print("Y_assign", Y_assignment[:200])
    # T_assignment = np.array(T_assignment)
    # H_Y_T = conditional_entropy(Y_assignment, T_assignment)
    # MI = mutual_information(Y_assignment, T_assignment)

    return losses.avg, Y_assignment



def main():
    args = set_args()
    args = costomize_args(args)

    train_loader, test_loader = set_dataloader(args)

    scalar_logger = txt_logger(args.saving_path, args, 'python ' + ' '.join(sys.argv))
    model, UnSup_criterion, pretrain_criterion = get_model(args, scalar_logger)
    optimizer, scheduler = set_optimizer(args, model)

    # training routine
    # resume model path
    if args.resume_model_path:
        start = opt.pre_ssl_epoch
    else:
        start = 0

    if args.pretrain_mode == 'autoencoder':
        # pre_train
        pre_train_optimizer = optim.Adam(model.parameters(), weight_decay=5e-4)
        pre_train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pre_train_optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        for epoch in range(start + 1, args.pre_train_epochs + 1):
            pre_train_loss = pre_train(train_loader, model, epoch, args, pre_train_optimizer, pre_train_scheduler, pretrain_criterion)

            scalar_logger.log_value(epoch, ('pre_train loss', pre_train_loss))

        # k-means clustering for initialization
        print("Initialization (K-means) ---------")
        features = []
        model.eval()
        for idx, (img, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            with torch.no_grad():
                img = img.cuda()
                feature, rec_img = model(img)
                features.extend(feature.cpu().numpy())

        features = np.array(features)
        features = features.reshape(features.shape[0], features.shape[1])
        print(features.shape)
        k_means = KMeans(n_clusters=args.latent_class_num, n_init=20)
        k_means.fit(features)
        model.clusterCenterInitialization(k_means.cluster_centers_)

    elif args.pretrain_mode == 'None':
        # random othogonal init
        if args.latent_class_num < args.feature_dim:
            mu_init = ortho_group.rvs(dim=args.feature_dim)[:args.latent_class_num]
        else:
            mu_init = np.random.rand(args.latent_class_num, args.feature_dim)
        model.clusterCenterInitialization(mu_init)
    else:
        raise NotImplementedError("pretrain mode {} has not been implemented.".format(args.pretrain_mode))

    # train
    print("Begin Training -------------------------")
    for epoch in range(start + 1, args.epochs + 1):
        # train for one epoch
        loss, Y_assignment = train(train_loader, model, optimizer, epoch, args, scheduler, UnSup_criterion)
        # latent_class statistics
        unique_latent_class = set(Y_assignment)

        # file logger
        scalar_logger.log_value(epoch, ('loss', loss),
                                    ('learning_rate', optimizer.param_groups[0]['lr']),
                                    ('lc_len', len(unique_latent_class)),
                                    )


        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.saving_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, args, epoch, save_file)

    # TODO: save latent class assignment
    save_file_lat_class_assign = os.path.join(args.saving_path, 'latent_class.npy')
    np.save(save_file_lat_class_assign, Y_assignment)
    # log latent class statistics
    latent_class_stats = {}
    for i in unique_latent_class:
        latent_class_stats[i] = np.where(Y_assignment == i)[0].shape[0]
    scalar_logger.log_value(epoch, ('final_lc_assign', latent_class_stats))

    # save the last model
    save_file = os.path.join(
        args.saving_path, 'last.pth')
    save_model(model, optimizer, args, args.epochs, save_file)

    return 

    
if __name__ == '__main__':
    main()





    