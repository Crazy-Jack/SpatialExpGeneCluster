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
    parser.add_argument('--data_root', type=str, default="../data/")
    parser.add_argument('--save_root', type=str, default="../train_related/")
    parser.add_argument('--save_freq', type=int, default=1, help="save freq of the model")
    parser.add_argument('--data_file_name', type=str, default="data.npy", help="data file name")
    # train parameters
    parser.add_argument('--batch_size', type=int, default=256, help="batch")
    parser.add_argument('--num_workers', type=int, default=3, help="number of workers to use \
                                                                    for data loading")
    parser.add_argument('--epochs', type=int, default=200, help="total epochs")

    # optimization parameters
    parser.add_argument('--optimizer', type=str, default='adam', help="which optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate for \
                                                                                    optimizer")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--schedular_patients', type=int, default=10, help='reduce on pleatau shceduling')
    parser.add_argument('--schedular_verbose', action='store_true', help="if print scheduling")
    parser.add_argument('--lr_scheduling', type=str, default='', help="optimizer scheduling type")

    # model
    parser.add_argument('--model', type=str, default='resnet18', help="what model we choose")
    parser.add_argument('--input_channel', type=int, default=7, help="input channels")
    parser.add_argument('--feature_dim', type=int, default=128, help="only support 2048 for now")
    parser.add_argument('--latent_class_num', type=int, default=100)

    # pretrain
    parser.add_argument('--pretrain_mode', type=str, default="autoencoder")
    parser.add_argument('--pre_train_epochs', type=int, default=100, help="pretraining epochs")
    parser.add_argument('--use_scheduler_pretrain', type=str, default="if use scheduler for pretrain")


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
        self.args.saving_path = os.path.join(self.save_root, dataset_name)
        os.makedirs(self.args.saving_path, exist_ok=True)

    def resume_epoch(self):
        self.args.pretrain_epoch = int(self.args.resume_model_path.split(".")[0].split("_")[-1])


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