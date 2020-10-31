import argparse

import torch
import torch.nn as nn
import torchvision
from PIL import Image
import numpy as np
import tqdm

from utlis import set_args, set_optimizer
from utlis import AverageMeter

def costomize_args(args):
    pass



def set_dataloader(args):
    """use args.dataset decide which dataset to use and return dataloader"""
    if args.dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(root=args.loading_path, train=True)
        test_dataset = torchvision.datasets.MNIST(root=args.loading_path, train=False)
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
    model = DECNetwork(args.attr_num, opt.feature_dim, opt.latent_class_num, 
                                    layer_dims=opt.layer_dims, alpha=opt.alpha,
                                    decode_constraint=opt.rec_weight!=0)
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



def pre_train(train_loader, model, epoch, args, optimizer, scheduler, pretrain_criterion):
    model.train()
    model.setPretrain(True)

    losses = AverageMeter()

    for idx, (img, labels, masks, _) in enumerate(train_loader):
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



def main():
    args = set_args()
    args = costomize_args(args)

    train_loader, test_loader = set_dataloader(args)
    model = set_model
    optimizer, scheduler = set_optimizer(args, model)

    ################## modify below
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

    





    