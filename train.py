import os
import math
import argparse
import socket
from datetime import datetime
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model.utils import save_ckpt
from utils.dataset import TrainSetLoader, TestSetLoader
from utils.metric import SigmoidMetric, SamplewiseSigmoidMetric, PD_FA, ROCMetric, mIoU
from utils.engine import train_one_epoch, evaluate
from utils.loss import SoftLoULoss1 as SoftLoULoss

from model.PBT_v2 import WindowPBTNet
#

from config import load_config
from load_dataset import load_dataset
from argparse import ArgumentParser
import torch.nn as nn
import os.path as ops
import time
import numpy as np

torch.cuda.manual_seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

os.environ["CUDA_VISIBLE_DEVICES"] = "1, 0, "
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

resume = False

resume_dir = ''


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of HCT model')
    #parser.add_argument('--model', type=str, default='hct_base_patch32_512', help='model_name:')
    parser.add_argument('--dataset', type=str, default='IRSTD-Air',
                        help='dataset:IRSTD-1k; NUDT-SIRST; Flir; ')
    parser.add_argument('--suffix', type=str, default='.png')
    #
    # Training parameters
    #
    parser.add_argument('--aug', type=float, default=0.)
    parser.add_argument('--workers', type=int, default=8, metavar='N', help='dataloader threads')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=5000 , help='number of epochs')
    parser.add_argument('--optimizer', type=str, default='Adagrad', help='Adam, Adagrad')
    parser.add_argument('--scheduler', default='CosineAnnealingLR', choices=['CosineAnnealingLR', 'ReduceLROnPlateau'])
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR', help='9e-3learning rate (default: 0.1)')
    parser.add_argument('--min_lr', default=1e-2, type=float, help='3e-3minimum learning rate')
    #
    # Net parameters
    #

    #
    # Dataset parameters
    #

    args = parser.parse_args()
    return args


def main(args):

    dataset = args.dataset
    cfg = load_config()
    root, split_method, size, batch, aug = cfg['dataset']['root'], cfg['dataset'][dataset]['split_method'], \
                                      cfg['dataset'][dataset]['size'], cfg['dataset'][dataset]['batch'], cfg['dataset'][dataset]['aug']
    args.img_size = size
    args.batch_size = batch
    args.aug = aug
    args.model = cfg['dataset'][dataset]['model']
    train_img_ids, val_img_ids, test_txt = load_dataset(root, dataset, split_method)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + dataset + "_" + split_method + "_" + args.model)
    tb_writer = SummaryWriter(log_dir=log_dir)
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    dataset_dir = root + '/' + dataset
    args.use_prior = True
    print('use_prior_loss: ', args.use_prior)
    trainset = TrainSetLoader(dataset_dir, img_id=train_img_ids, base_size=size, crop_size=size,
                              transform=input_transform, suffix=args.suffix, aug=args.aug, useprior=True)
    print(len(trainset))

    valset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=size, crop_size=size,
                            transform=input_transform, suffix=args.suffix)
    train_data = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.workers, drop_last=True, pin_memory=True)
    val_data = DataLoader(dataset=valset, batch_size=args.batch_size, num_workers=args.workers,
                           drop_last=False)

    model = WindowPBTNet(num_classes=1,input_channels=3, num_blocks=[2, 2, 2, 2, 2], nb_filter=[ 16, 32, 64, 128, 256, 524, 1024],
                     deep_supervision=False, depth=[1, 2, 2, 2, 2], drop=0.1, attn_drop=0.1, drop_path=0.1,mlp_ratio=4.,
                     # heads=[[1, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1], [1, 1], [1]], token_projection='linear',
                     heads=[[1, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1], [1, 1], [1]], token_projection='linear',
                     token_mlp='leff', win_size=8, img_size=size,)

    print('img size: ', size)
    print('dataset: ', dataset)#xiangniqwqqqqqq
    print('# model_restoration parameters: %.2f M' % (sum(param.numel() for param in model.parameters()) / 1e6))
    print('device_count: ', torch.cuda.device_count())
    if torch.cuda.device_count() > 1:


        print("Let's use ", torch.cuda.device_count(), " GPUs!")
        model = nn.DataParallel(model, device_ids=[0, 1, ])

    model = model.to(device)

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=.9)
    else:
        raise
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=min(args.epochs,2000), eta_min=args.min_lr)
    else:
        raise
    restart = 0
    if resume == True:
        ckpt = torch.load(resume_dir)
        print(ckpt['mean_IOU'])
        model.load_state_dict(ckpt['state_dict'], strict=True)

        restart = ckpt['epoch']

        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt["scheduler"])
        print('resuming')

    best_iou = 0
    best_nIoU = 0

    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    roc = ROCMetric(1, 10)
    pdfa = PD_FA(1, 10)
    miou = mIoU(1)

    folder_name = '%s_%s_%s' % (time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())),
                                args.dataset, args.model)

    save_folder = log_dir
    save_pkl = ops.join(save_folder, 'checkpoint')
    if not ops.exists('result'):
        os.mkdir('result')
    if not ops.exists(save_folder):
        os.mkdir(save_folder)
    if not ops.exists(save_pkl):
        os.mkdir(save_pkl)
    tb_writer.add_text(folder_name, 'Args:%s, ' % args)

    loss_func = SoftLoULoss(a=0.).to(device)
    last_name_miou = ' '
    last_name_niou = ' '
    for epoch in range(restart+1, args.epochs):
        train_loss, current_lr, loss1, loss2 = train_one_epoch(model, optimizer, train_data, device, epoch, loss_func,)
        if epoch > 500:
            val_loss, iou_, niou_, miou_, ture_positive_rate, false_positive_rate, recall, precision, pd, fa = \
                evaluate(model, val_data, device, epoch, iou_metric, niou_metric, pdfa, miou, roc, len(valset), loss_func)
            tags = ['train_loss', 'val_loss', 'IoU', 'nIoU', 'mIoU', 'PD', 'tp', 'fa', 'rc', 'pr']
            tb_writer.add_scalar('LR.a', loss_func.a, epoch)
            tb_writer.add_scalar('LR', current_lr, epoch)
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar('loss1', loss1, epoch)
            tb_writer.add_scalar('loss2', loss1, epoch)
            tb_writer.add_scalar(tags[1], val_loss, epoch)
            tb_writer.add_scalar(tags[2], iou_, epoch)
            tb_writer.add_scalar(tags[3], niou_, epoch)

            name = 'Epoch-%3d_IoU-%.4f_nIoU-%.4f.pth.tar' % (epoch, iou_, niou_)
            if resume == True or (resume == False and epoch >= 100):
                if iou_ > best_iou:
                    save_ckpt({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'loss': val_loss,
                        'mean_IOU': iou_,
                        'n_IoU': niou_,
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, save_path=save_pkl,
                        filename='Best_mIoU_' + name)
                    best_iou = iou_
                    if ops.exists(ops.join(save_pkl, 'Best_mIoU_' + last_name_miou)):
                        os.remove(ops.join(save_pkl, 'Best_mIoU_' + last_name_miou))
                    last_name_miou = name

                if niou_ > best_nIoU:
                    save_ckpt({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'loss': val_loss,
                        'mean_IOU': iou_,
                        'n_IoU': niou_,
                        'optimizer': optimizer.state_dict(),
                    }, save_path=save_pkl,
                        filename='Best_nIoU_' + name)
                    best_nIoU = niou_
                    if ops.exists(ops.join(save_pkl, 'Best_nIoU_' + last_name_niou)):
                        os.remove(ops.join(save_pkl, 'Best_nIoU_' + last_name_niou))
                    last_name_niou = name
if __name__ == '__main__':

    args = parse_args()
    main(args)

