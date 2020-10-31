# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pandas as pd
import shutil
import getpass
import multiprocessing
import subprocess

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from tqdm import tqdm
from torchsummary import summary
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import ortho_group


from data_utils import AttributeOnlyDataset
from util import set_default_path
from util import optimization_argument
from util import AverageMeter
from util import adjust_learning_rate
from util import set_optimizer, save_model
from util import txt_logger, set_parser, parser_processing, suppress_std
from networks.DEC_network import DECNetwork
from DEC_loss import DECLoss
from loss_neural_selection import SemiSupLoss, NeuSelectLoss
from loss_attribute import WeakAttributeLoss
from stats import mutual_information, conditional_entropy

def get_arg():
    parser = argparse.ArgumentParser('Neural Selection')

    parser.add_argument('--batch_size', type=int, default=2000, help='batch_size')
    parser.add_argument('--save_freq', type=int, default=250, help='save frequency')
    parser.add_argument('--save_path', type=str, default='/projects/rsalakhugroup/tianqinl/train_related', help='where to save file')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=500, help='number of training epochs')
    parser.add_argument('--pretrain_mode', type=str, required=True, help='define how to pretrain the network: [`autoencoder`, `None`]')
    parser.add_argument('--pre_train_epochs', type=int, default=200, help='number of pre-training epochs')
    parser.add_argument('--dataset', type=str, required=True, choices=['deepfashion', 'ut-zap50k', 'ut-zap50k-sub', 'cifar10', 'cifar100', 'CUB', 'SUN', 'aPascal', 'Wider'], help='what dataset to use')
    parser.add_argument('--data_folder', type=str, default='', help='data foldr, being handled by util.set_default_path if not specify')
    parser.add_argument('--data_root_name', type=str, default='', help="dataset img folder name, only needed when dataset is organized by folders of img, , being handled by util.set_default_path if not specify")
    parser.add_argument('--meta_file_train', type=str, default='meta_data_bin_train.csv',
                        help='meta data for ssl training')
    parser.add_argument('--meta_file_test', type=str, default='meta_data_bin_test.csv',
                        help='meta data for ssl testing')
    parser.add_argument('--latent_class_num', type=int, required=True, help='number of latent class')
    parser.add_argument('--feature_dim', type=int, required=True, help='dimension of latent feature')
    parser.add_argument('--alpha', type=float, default=1, help='alpha, temperature as used in DEC paper when get the q from t-distribution')
    parser.add_argument('--rec_weight', type=float, default=0.0, help='weight of reconstruction loss')
    parser.add_argument('--layer_dims', type=int, nargs='+', default=[500, 500, 2000], help='mlp dimensions')
    parser.add_argument('--SemiSup_col', default='mask', help='the column name of mask for whether the class labels are known')
    parser.add_argument('--customized_name', type=str, default='', help='adding customized name for saving folder')
    parser.add_argument('--one_batch', action='store_true', help='use one batch dataloader')
    parser.add_argument('--attr_num', type=int, default=0, help='use how many top k attributes')
    parser.add_argument('--semi_metrics', type=str, default='normal', help='semi supervised loss metrics, default is `normal` which is just using hyper parameter to construct')
    parser.add_argument('--lambda_semi_hgt_mi_hgy', type=str, default='0.5,1,0', help="relative weight within semi supvised loss for H(Y|T)-I(Y;T)-H(T|Y)")
    parser.add_argument('--lambda_semi', type=float, default=1e-2, help='weight of overall semi supervised loss for H(T|Y)')
    parser.add_argument('--lambda_hy', type=float, default=1e-2, help='weight of H(Y)')
    parser.add_argument('--HY', action='store_true', help='whether control H(Y)')

    
    parser.add_argument('--verbose', action='store_true', help='whether print varity of values to check')


    # optimization
    optimization_argument(parser)

    # resume training
    parser.add_argument('--resume_model_path', type=str, default='',
                        help='model_path to resume')
    # for spead up testing purpose
    parser.add_argument('--test', action='store_true',
                        help='determine if in a testing mode (shuttle down iterations)')

    opt = parser.parse_args()

    # parse lambda_semi
    opt.lambda_semi_hgt, opt.lambda_semi_mi, opt.lambda_semi_hgy = [float(i) for i in opt.lambda_semi_hgt_mi_hgy.split(",")]


    if hasattr(opt, 'stage_info'):
        opt.stage_info = [(int(i.split(":")[0]), int(i.split(":")[1])) for i in opt.stage_info.split(",")]

    # set the path according to the environment
    if not opt.data_folder:
        set_default_path(opt)

    # get user name
    opt.user_name = getpass.getuser()

    opt.data_root_folder = os.path.join(opt.data_folder, opt.data_root_name)

    opt.model_path = os.path.join(opt.save_path, 'WeakSupCon_{}/{}/DEC/train_file_{}/UnSup_{}/num_latent_{}_layers_dim_{}_zdim_{}_rec_{}_semimetric_{}_semiWt_{}_{}'.format(
                                    opt.user_name, opt.dataset, opt.meta_file_train.replace("meta_data_train_", "").replace(".csv", ""), opt.SemiSup_col,
                                    opt.latent_class_num, "_".join([str(i) for i in opt.layer_dims]), opt.feature_dim, opt.rec_weight, opt.semi_metrics, opt.lambda_semi, opt.customized_name))
    opt.model_name = 'lr_{}_decay_{}_bsz_{}_scheduling_{}_trial_{}'.\
                    format(opt.learning_rate,
                    opt.weight_decay, opt.batch_size, opt.lr_scheduling, opt.trial)

    # MODIFY model name
    if opt.lr_scheduling == 'cosine':
        opt.change_init_lr = True
    else:
        opt.change_init_lr = False

    # if opt.batch_size > 256:
    #     opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.lr_scheduling == 'cosine':
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    if opt.test:
        opt.model_name = '{}_test'.format(opt.model_name)

    if opt.resume_model_path:
        opt.pre_ssl_epoch = int(opt.resume_model_path.split('/')[-1].split('.')[0].split('_')[-1])
        opt.model_name += '_resume_from_epoch_{}'.format(opt.pre_ssl_epoch)

    if opt.semi_metrics == 'normal':
        opt.model_name += '_rwhgt_{}_rwmi_{}_rwhgy_{}'.format(opt.lambda_semi_hgt, opt.lambda_semi_mi, opt.lambda_semi_hgy)

    opt.model_name += '_pretrain_{}'.format(opt.pretrain_mode)

    # END of MODIFY model name

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    opt.tb_folder = os.path.join(opt.save_folder, 'tensorboard')

    if os.path.isdir(opt.tb_folder):
        delete = input("Are you sure to delete folder {}? (Y/n)".format(opt.tb_folder))
        if delete.lower() == 'y':
            rm_command = "rm -rf " + str(opt.tb_folder)
            os.system(rm_command)
            # shutil.rmtree(opt.tb_folder)
        else:
            sys.exit("{} FOLDER is untouched.".format(opt.tb_folder))

    os.makedirs(opt.tb_folder, exist_ok=True)


    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder, exist_ok=True)

    # get current git id
    opt.git_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode()[:-1]

    return opt

class OneBatchDataLoader():
    def __init__(self, df, label_known_mask=''):
        '''
        label_known_mask: str, the column name of mask for whether the class labels are known
        '''
        self.df = df
        self.label_known_mask = label_known_mask
        self.attr_list = [i for i in self.df.columns if 'attr_val' in i]
        self.attr_num = len(self.attr_list)

        if self.label_known_mask:
            self.num_class = df.iloc[-1]['class']

        if -1 in self.df.index:
            self.df = df.drop([-1])

        # get p(t|a) for all data and store results in self.pta_df (dim: [all_data_num, class_num])
        self.labels = torch.LongTensor(self.df['class'].to_numpy())
        attr_id = self.df[self.attr_list].apply(lambda row: "-".join([str(i) for i in row]), 1)
        mapping_pd = pd.DataFrame({'attr_id': attr_id, 'class': self.labels.numpy()})
        grouped = mapping_pd.groupby('attr_id')['class']

        mapping = grouped.apply(lambda x: self.get_tga_row(torch.LongTensor(x.to_numpy()), self.num_class).numpy())
        self.pta_mapping_pd = pd.DataFrame({i:mapping[i] for i in mapping.index}) # col: attr_id, row: class, value: p(t|a)
        pta_df_dict = {}
        for class_id in range(self.num_class):
            pta_df_dict['class_{}'.format(class_id)] = attr_id.apply(lambda a_id: self.pta_mapping_pd.loc[class_id, a_id], 1)
        self.pta_df = pd.DataFrame(pta_df_dict)



    def get_tga_row(self, class_info, total_class):
        """being applied in the groupby function to get prob of each attrs_comb
        param: class_info: [instance num in the same group, ] : int, specific for certain attrs_id
               total_class: scalar: total number of class (|T|)
        return: [total_class, ], indicating p(t|a) for this attrs_id
        """
        assert torch.max(class_info) <= (total_class - 1), "max number in class_info should be less than total class number"
        class_count = torch.zeros(class_info.shape[0], total_class).scatter(1, class_info.view(-1,1), 1).sum(0)
        return class_count / class_count.sum()



    def __len__(self):
        return 1

    def __iter__(self):
        '''
        if self.label_known_mask is set:
            return attributes, class, mask
            Note: mask: 1 indicate known, 0 indicate unknown
                  class: -1 for unknown, otherwise known.
        else:
            return attributes, -1, 0
        '''
        attributes = torch.Tensor(self.df[self.attr_list].to_numpy().astype('float32'))
        attributes = attributes[:, :opt.attr_num]
        # get p(t|a):
        p_t_a = torch.FloatTensor(self.pta_df.to_numpy()) #[instance_data, class_num]

        if self.label_known_mask:
            yield attributes, self.labels, torch.LongTensor(self.df[self.label_known_mask].to_numpy()), p_t_a
        else:
            # TODO: dont expose label
            # yield attributes, torch.LongTensor([-1] * len(self.df)), torch.LongTensor([0] * len(self.df))
            yield attributes, torch.LongTensor(self.df['class'].to_numpy()), torch.LongTensor([0] * len(self.df)), p_t_a

class NeuralSelectDataset(Dataset):
    """Customized dataset for non one batch loader"""
    def __init__(self, df, label_known_mask=''):
        super(NeuralSelectDataset, self).__init__()
        self.df = df
        self.label_known_mask = label_known_mask
        self.attr_list = [i for i in self.df.columns if 'attr_val' in i]
        self.attr_num = len(self.attr_list)

        if self.label_known_mask:
            self.num_class = df.iloc[-1]['class']

        if -1 in self.df.index:
            self.df = df.drop([-1])

        # get p(t|a) for all data and store results in self.pta_df (dim: [all_data_num, class_num])
        self.labels = torch.LongTensor(self.df['class'].to_numpy())
        attr_id = self.df[self.attr_list].apply(lambda row: "-".join([str(i) for i in row]), 1)
        mapping_pd = pd.DataFrame({'attr_id': attr_id, 'class': self.labels.numpy()})
        grouped = mapping_pd.groupby('attr_id')['class']

        mapping = grouped.apply(lambda x: self.get_tga_row(torch.LongTensor(x.to_numpy()), self.num_class).numpy())
        self.pta_mapping_pd = pd.DataFrame({i:mapping[i] for i in mapping.index}) # col: attr_id, row: class, value: p(t|a)
        pta_df_dict = {}
        for class_id in range(self.num_class):
            pta_df_dict['class_{}'.format(class_id)] = attr_id.apply(lambda a_id: self.pta_mapping_pd.loc[class_id, a_id], 1)
        self.pta_df = pd.DataFrame(pta_df_dict)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Need to be complete"""
        return None



def set_loader(opt):
    train_meta_df = pd.read_csv(os.path.join(opt.data_folder, 'neural_selection', opt.meta_file_train), index_col=0)
    if opt.one_batch:
        train_loader = OneBatchDataLoader(train_meta_df, opt.SemiSup_col)
        if opt.attr_num == 0:
            opt.attr_num = train_loader.attr_num
    else:
        train_dataset = AttributeOnlyDataset(train_meta_df, opt.SemiSup_col)
        opt.attr_num = train_dataset.attr_num
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=opt.batch_size, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True)


    return train_loader

def get_model(opt, logger):
    model = DECNetwork(opt.attr_num, opt.feature_dim, opt.latent_class_num, layer_dims=opt.layer_dims, alpha=opt.alpha, decode_constraint=opt.rec_weight!=0)
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

        if opt.resume_model_path:
            # get pre ssl epoch
            ckpt = torch.load(opt.resume_model_path, map_location='cpu')
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

def pre_train(train_loader, model, epoch, opt, optimizer, scheduler, pretrain_criterion):
    model.train()
    model.setPretrain(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    for idx, (attrs, labels, masks, _) in enumerate(train_loader):
        if not opt.test:
            attrs = attrs.cuda()
            bsz = attrs.shape[0]

            # compute probrability
            feature, rec_attr = model(attrs)

            # compute loss
            loss = pretrain_criterion(rec_attr, attrs)

            # update metric
            losses.update(loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    scheduler.step(loss)

    return losses.avg

def train(train_loader, model, optimizer, epoch, opt, scheduler, UnSup_criterion, rec_criterion=None, SemiSup_criterion=None, HYA_criterion=None):
    """one epoch training"""
    # TODO: rewrite this and fill all empty lines!
    model.train()
    model.setPretrain(False)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    dec_losses = AverageMeter()
    semi_losses = AverageMeter()
    hy_losses = AverageMeter()
    losses = AverageMeter()

    Y_assignment = []
    T_assignment = []
    end = time.time()
    for idx, (attrs, labels, masks, prob_tga) in enumerate(train_loader):
        """params:
                attrs: [bz, attrs_num]
                labels: [bz,]
                masks: [bz, ]: one hot
                prob_tga: [bsz, |T|]: float
        """
        if not opt.test:
            attrs = attrs.cuda()
            labels = labels.cuda()
            masks = masks.cuda()
            prob_tga = prob_tga.cuda()
            bsz = attrs.shape[0]

            # compute probrability - p(y|a) dim: [bsz, |Y|]
            features, prob = model(attrs)
            prob = prob.float()
            # get Y and T assignment
            Y_assignment.extend(prob.argmax(dim=1).cpu().numpy())
            T_assignment.extend(labels.cpu().numpy())

            # compute loss
            # 1) DEC loss
            dec_loss = UnSup_criterion(prob)
            # 2) SemiSup Loss
            if SemiSup_criterion:
                semi_sup_loss = SemiSup_criterion(prob, prob_tga, masks)
            else:
                semi_sup_loss = torch.FloatTensor([0])
            # 3) control H(Y)
            if HYA_criterion:
                hy_loss = HYA_criterion(prob, hy_only=True)
            else:
                hy_loss = torch.FloatTensor([0])

            loss = opt.lambda_semi * semi_sup_loss + dec_loss + opt.lambda_hy * hy_loss

            # update metric
            dec_losses.update(dec_loss.item(), bsz)
            semi_losses.update(semi_sup_loss.item(), bsz)
            hy_losses.update(hy_loss.item(), bsz)

            losses.update(loss.item(), bsz)

            # SGD
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if opt.batch_scheduler:
                scheduler.batch_step(loss.item(), bsz)

        elif opt.test:
            scheduler.batch_step(0, bsz)

    # compute H(Y|T) and I(Y;T)
    # TODO: LOW PRIORITY!! use pytorch to implement H and MI calucalation
    Y_assignment = np.array(Y_assignment)
    # print("Y_assign", Y_assignment[:200])
    T_assignment = np.array(T_assignment)
    H_Y_T = conditional_entropy(Y_assignment, T_assignment)
    MI = mutual_information(Y_assignment, T_assignment)

    return losses.avg, semi_losses.avg, dec_losses.avg, hy_losses.avg, H_Y_T, MI, Y_assignment

def main(opt):
    tf_logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    scalar_logger = txt_logger(opt.save_folder, opt, 'python ' + ' '.join(sys.argv))

    train_loader = set_loader(opt)

    model, UnSup_criterion, pretrain_criterion = get_model(opt, scalar_logger)

    if opt.SemiSup_col:
        # TODO: add semisup_criterion
        SemiSup_criterion = SemiSupLoss(opt, verbose=opt.verbose)
    else:
        SemiSup_criterion = None

    if opt.HY:
        HYA_criterion = NeuSelectLoss()
    else:
        HYA_criterion = None

    scheduler, optimizer = set_optimizer(opt, model)

    # training routine
    # resume model path
    if opt.resume_model_path:
        start = opt.pre_ssl_epoch
    else:
        start = 0

    if opt.pretrain_mode == 'autoencoder':
        # pre_train
        pre_train_optimizer = optim.Adam(model.parameters(), weight_decay=5e-4)
        pre_train_scheduler = optim.lr_scheduler.ReduceLROnPlateau(pre_train_optimizer, mode='min', factor=0.5, patience=20, verbose=True)
        for epoch in range(start + 1, opt.pre_train_epochs + 1):
            pre_train_loss = pre_train(train_loader, model, epoch, opt, pre_train_optimizer, pre_train_scheduler, pretrain_criterion)

            scalar_logger.log_value(epoch, ('pre_train loss', pre_train_loss))

        # k-means clustering for initialization
        features = []
        model.eval()
        for idx, (attrs, labels, masks, _) in tqdm(enumerate(train_loader), total=len(train_loader)):
            with torch.no_grad():
                attrs = attrs.cuda()
                feature, rec_attrs = model(attrs)
                features.extend(feature.cpu().numpy())

        features = np.array(features)
        k_means = KMeans(n_clusters=opt.latent_class_num, n_init=20)
        k_means.fit(features)
        model.clusterCenterInitialization(k_means.cluster_centers_)

    elif opt.pretrain_mode == 'None':
        # random othogonal init
        if opt.latent_class_num < opt.feature_dim:
            mu_init = ortho_group.rvs(dim=opt.feature_dim)[:opt.latent_class_num]
        else:
            mu_init = np.random.rand(opt.latent_class_num, opt.feature_dim)
        model.clusterCenterInitialization(mu_init)
    else:
        raise NotImplementedError("pretrain mode {} has not been implemented.".format(opt.pretrain_mode))

    # train
    # TODO: modify all the thing below
    for epoch in range(start + 1, opt.epochs + 1):
        # adjust lr
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        start_time_epoch = time.time()
        loss, semi_loss, dec_loss, hy_loss, H_Y_T, MI, Y_assignment = train(train_loader, model, optimizer, epoch, opt, scheduler, UnSup_criterion, SemiSup_criterion=SemiSup_criterion, 
                                                                    HYA_criterion=HYA_criterion)
        end_time_epoch = time.time()
        if not opt.batch_scheduler and opt.lr_scheduling != 'warmup':
            scheduler.step(loss)

        # latent_class statistics
        unique_latent_class = set(Y_assignment)

        # tensorboard logger
        tf_logger.log_value('loss', loss, epoch)
        tf_logger.log_value('semi_loss', semi_loss, epoch)
        tf_logger.log_value('dec_loss', dec_loss, epoch)
        tf_logger.log_value('H(Y|T)', H_Y_T, epoch)
        tf_logger.log_value('MI', MI, epoch)
        tf_logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # file logger
        scalar_logger.log_value(epoch, ('loss', loss),
                                    ('semi_loss', semi_loss),
                                    ('dec_loss', dec_loss),
                                    ('H(Y)', hy_loss),
                                    ('H(Y|T)', H_Y_T),
                                    ('MI', MI),
                                    ('learning_rate', optimizer.param_groups[0]['lr']),
                                    # ('Scheduler (Bad epochs)', "{}/{}".format(scheduler.num_bad_epochs, scheduler.patience)),
                                    ('lc_len', len(unique_latent_class)),
                                    )


        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # TODO: save latent class assignment
    save_file_lat_class_assign = os.path.join(opt.save_folder, 'latent_class.npy')
    np.save(save_file_lat_class_assign, Y_assignment)
    # log latent class statistics
    latent_class_stats = {}
    for i in unique_latent_class:
        latent_class_stats[i] = np.where(Y_assignment == i)[0].shape[0]
    scalar_logger.log_value(epoch, ('final_lc_assign', latent_class_stats))

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)


    return opt.save_folder

if __name__ == '__main__':
    opt = get_arg()

    main(opt)
