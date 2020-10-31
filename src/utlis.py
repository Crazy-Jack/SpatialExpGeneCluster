import argparse
import os

import torch
import torch.nn as nn
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

    # train parameters
    parser.add_argument('--batch_size', type=int, default=256, help="batch")
    parser.add_argumnet('--num_workers', type=int, default=3, help="number of workers to use \
                                                                    for data loading")
    parser.add_argument('--epochs', type=int, default=200, help="total epochs")

    # optimization parameters
    parser.add_argument('--optimizer', type=str, default='adam', help="which optimizer to use")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate for \
                                                                                    optimizer")
    parser.add_argument('--momentum', type=float, default=0.9, help="momentum")
    parser.add_argument('--weight decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--schedular_patients', type=int, default=10, help='reduce on pleatau shceduling')
    parser.add_argument('--schedular_verbose', action='store_true', help="if print scheduling")
    parser.add_argument('--lr_scheduling', type=str, default='', help="optimizer scheduling type")

    # model
    parser.add_argument('--model', type=str, default='resnet18', help="what model we choose")


    # pretrain

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

        return self.args


    def IO_path(self, costomized_name=None):
        if costomized_name:
            dataset_name = costomized_name
        else:
            dataset_name = self.dataset

        # input path
        self.args.loading_path = os.path.join(self.data_root, dataset_name)
        # output path
        self.args.saving_path = os.path.join(self.data_root, dataset_name)


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