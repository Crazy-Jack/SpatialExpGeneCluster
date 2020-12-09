import argparse
import os, sys
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from PIL import Image
import numpy as np
import tqdm



def set_args():
    # Initial
    parser = argparse.ArgumentParser('10617 project gene clustering via deep learning')

    # I/O related
    parser.add_argument('--dataset', type=str, required=True, help="which dataset do you want")
    parser.add_argument('--data_root', type=str, default="../data/spatial_Exp/32-32")
    parser.add_argument('--save_root', type=str, default="../train_related/")
    parser.add_argument('--save_freq', type=int, default=10, help="save freq of the model")
    parser.add_argument('--data_file_name', type=str, default="data.npy", help="data file name")
    parser.add_argument('--experiment_name', type=str, default="test", help="costomized experiment name")
    
    # train parameters
    parser.add_argument('--batch_size', type=int, default=256, help="batch")
    parser.add_argument('--num_workers', type=int, default=3, help="number of workers to use \
                                                                    for data loading")
    parser.add_argument('--epochs', type=int, default=500, help="total epochs")
    parser.add_argument('--img_size', type=int, default=64, help="img size")

    # optimization parameters
    parser.add_argument('--optimizer', type=str, default='adam', help="which optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate for optimizer")
    parser.add_argument('--min_lr', type=float, default=1e-5, help='SGD min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--schedular_patients', type=int, default=10, help='reduce on pleatau shceduling')
    parser.add_argument('--schedular_verbose', action='store_true', help="if print scheduling")
    parser.add_argument('--lr_scheduling', type=str, default='', help="optimizer scheduling type")
    parser.add_argument('--warmup_percent', type=float, default=0.33,
                        help='percent of epochs that used for warmup')
    parser.add_argument('--temp', type=float, default=0.1, help="temperature for training contrastive learning")

    # model
    parser.add_argument('--model', type=str, default='resnet50', help="what model we choose")
    parser.add_argument('--input_channel', type=int, default=7, help="input channels")
    parser.add_argument('--feature_dim', type=int, default=128, help="only support 2048 for now")
    parser.add_argument('--latent_class_num', type=int, default=100)

    # pretrain
    parser.add_argument('--pretrain_mode', type=str, default="autoencoder")
    parser.add_argument('--pre_train_epochs', type=int, default=100, help="pretraining epochs")
    parser.add_argument('--use_scheduler_pretrain', type=str, default="if use scheduler for pretrain")

    # kmeans clustering
    parser.add_argument('--k_clusters', type=int, default=100, help="kmeans clustering k")
    parser.add_argument('--kmeans_freq', type=int, default=20, help="freq to perform kmeans clustering assignment")

    # resume path
    parser.add_argument('--resume_model_path', type=str, default='', help="resume model path")
    # parse args
    args = parser.parse_args()
    
    # process args
    args = Process_args(args).process()
    return args 


class Process_args:

    def __init__(self, args):
        self.args = args
        self.dataset = args.dataset
        self.data_root = args.data_root
        self.save_root = args.save_root
        self.experiment_name = args.experiment_name
    
    def process(self, costomized_name=None):
        """process all sort of situation for args"""
        self.IO_path(costomized_name=costomized_name)
        if self.args.resume_model_path:
            self.resume_epoch()
        return self.args


    def IO_path(self, costomized_name=None):
        if costomized_name:
            dataset_name = costomized_name
        else:
            dataset_name = self.dataset

        # input path
        self.args.loading_path = os.path.join(self.data_root, dataset_name)
        # output path
        filename = f"{self.experiment_name}"
        self.args.saving_path = os.path.join(self.save_root, dataset_name, filename)
        os.makedirs(self.args.saving_path, exist_ok=True)

    def resume_epoch(self):
        self.args.pretrain_epoch = int(self.args.resume_model_path.split(".")[0].split("_")[-1])


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.lr_scheduling == 'adam':
        return None
    elif args.lr_scheduling == 'cosine':
        eta_min = lr * (args.lr_decay_rate ** 3)

        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif args.lr_scheduling == 'exp_decay':
        if epoch == 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(lr, args.min_lr)
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'] * args.exp_decay_rate, args.min_lr)

    elif args.lr_scheduling == 'warmup':
        assert args.learning_rate >= args.min_lr, "learning rate should >= min lr"
        warmup_epochs = int(args.epochs * args.warmup_percent)
        up_slope = (args.learning_rate - args.min_lr) / warmup_epochs
        down_slope = (args.learning_rate - args.min_lr) / (args.epochs - warmup_epochs)
        if epoch <= warmup_epochs:
            lr = args.min_lr + up_slope * epoch
        else:
            # lr = args.learning_rate - slope * (epoch - warmup_epochs)
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)

            lr = eta_min + (args.learning_rate - eta_min) * (
                1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs))) / 2

        for param_group in optimizer.param_groups:
            param_group['lr'] = max(lr, args.min_lr)


    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(args, model):

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.learning_rate,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                            lr=args.learning_rate,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=True)
    else:
        raise NotImplemented("optimizer '{}' not implemented. ".format(args.optimizer))
    

    # scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                          mode='min',
                                                          patience=args.schedular_patients,
                                                          verbose=args.schedular_verbose)
    return optimizer, scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.00)


class txt_logger:
    def __init__(self, save_folder, args, argv):
        self.save_folder = save_folder
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        if os.path.isfile(os.path.join(save_folder, 'logfile.log')):
            os.remove(os.path.join(save_folder, 'logfile.log'))

        file_log_handler = logging.FileHandler(os.path.join(save_folder, 'logfile.log'))
        self.logger.addHandler(file_log_handler)

        stdout_log_handler = logging.StreamHandler(sys.stdout)
        self.logger.addHandler(stdout_log_handler)
        # commend line
        self.logger.info("# COMMEND LINE ===========")
        self.logger.info(argv)
        self.logger.info("# =====================")
        # meta info
        self.logger.info("# META INFO ===========")
        attrs = vars(args)
        for item in attrs.items():
            self.logger.info("%s: %s"%item)
        # self.logger.info("Saved in: {}".format(save_folder))
        self.logger.info("# =====================")

    def log_value(self, epoch, *info_pairs):
        log_str = "Epoch: {}; ".format(epoch)
        for name, value in info_pairs:
            log_str += (str(name) + ": {}; ").format(value)
        self.logger.info(log_str)

    def save_value(self, name, list_of_values):
        np.save(os.path.join(self.save_folder, name), list_of_values)


def save_model(model, optimizer, opt, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state