import os
import sys
import json
import pickle
import random
from colorama import Fore
import torch
from tqdm import tqdm
from utils.loss import SoftLoULoss1
from model.utils import AverageMeter, save_Pred_GT, total_visulization_generation

import matplotlib.pyplot as plt

def train_one_epoch(model, optimizer, data_loader, device, epoch, loss):
    model.train()
    loss_function = loss
    losses = AverageMeter()
    loss1 = AverageMeter()
    loss2 = AverageMeter()
    optimizer.zero_grad()
    data_loader = tqdm(data_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.LIGHTRED_EX, Fore.RESET))
    current_lr = optimizer.state_dict()['param_groups'][0]['lr']
    for step, data in enumerate(data_loader):
        images, labels = data

        labels[labels > 0] = 1
        labels = torch.Tensor(labels).long().to(device)
        pred = model(images.to(device))

        if isinstance(pred, list):
            loss = 0
            # print('data', prior.shape)
            for p in pred:
                loss += loss_function(p, labels)
                loss1.update(loss_function.loss1, p.size(0))
                loss2.update(loss_function.loss2, p.size(0))
            loss /= len(pred)
            pred = pred[-1]
        else:
            # print('data', prior.shape)
            loss = loss_function(pred, labels)
            loss1.update(loss_function.loss1, pred.size(0))
            loss2.update(loss_function.loss2, pred.size(0))
        losses.update(loss.item(), pred.size(0))

        torch.autograd.set_detect_anomaly(False)

        loss.backward()
        optimizer.step()

        # for name, param in model.named_parameters():
        #     print('name', name)
        #     print('parm', param.shape)
        #     print('grad_require', param.requires_grad)
        #     print('grad_val', param.grad)
        #     print('-------------')

        optimizer.zero_grad()

        data_loader.desc = "[train epoch {}] loss: {:.8f}, lr: {:.8f}, 1: {:.4f}, 2: {:.4f}, a:{:.2f}".format(epoch, losses.avg, current_lr, loss1.avg, loss2.avg, loss_function.a)
    return losses.avg, current_lr, loss1.avg, loss2.avg,

@torch.no_grad()
def evaluate(model, data_loader, device, epoch, iou_metric, nIoU_metric, PD_FA, mIoU, ROC, len_val, loss):
    loss_function = loss
    model.eval()
    mIoU.reset()
    iou_metric.reset()
    nIoU_metric.reset()
    PD_FA.reset()
    ROC.reset()
    losses = AverageMeter()
    data_loader = tqdm(data_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))

    for step, data in enumerate(data_loader):
        images, labels = data
        labels[labels > 0] = 1
        labels = torch.Tensor(labels).long().to(device)
        pred = model(images.to(device))
        if isinstance(pred, list):
            loss = 0
            for p in pred:
                loss += loss_function(p, labels)
            loss /= len(pred)
            pred = pred[-1]
        else:
            loss = loss_function(pred, labels)
        losses.update(loss.item(), pred.size(0))
        pred, labels = pred.cpu(), labels.cpu()
        iou_metric.update(pred, labels)
        nIoU_metric.update(pred, labels)
        # ROC.update(pred, labels)
        mIoU.update(pred, labels)
        # PD_FA.update(pred, labels)
        FA, PD = PD_FA.get(len_val)
        ture_positive_rate, false_positive_rate, recall, precision = ROC.get()
        _, mean_IOU = mIoU.get()
        _, IoU = iou_metric.get()
        _, nIoU = nIoU_metric.get()
        data_loader.desc = "[valid epoch {}] loss: {:.6f}, mIoU: {:.6f}, nIoU: {:.6f}".format(epoch, losses.avg, IoU, nIoU)
    return losses.avg, IoU, nIoU, mean_IOU, ture_positive_rate, false_positive_rate, recall, precision, PD, FA


@torch.no_grad()
def visual(model, data_loader, device, epoch, iou_metric, nIoU_metric, PD_FA, mIoU, ROC, len_val, loss, path, ids, suffix, dataset_dir, test_txt):
    loss_function = loss
    model.eval()
    mIoU.reset()
    iou_metric.reset()
    nIoU_metric.reset()
    PD_FA.reset()
    ROC.reset()
    losses = AverageMeter()
    data_loader = tqdm(data_loader, file=sys.stdout, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))
    iou_list = []
    for step, data in enumerate(data_loader):
        images, labels = data
        labels[labels > 0] = 1
        labels = torch.Tensor(labels).long().to(device)
        pred = model(images.to(device))

        if isinstance(pred, list):
            loss = 0
            for p in pred:
                loss += loss_function(p, labels, None)
            loss /= len(pred)
            pred = pred[-1]
        else:
            loss = loss_function(pred, labels, None)
        iou_list.append(loss_function.iou.tolist())
        losses.update(loss.item(), pred.size(0))
        pred, labels = pred.cpu(), labels.cpu()
        iou_metric.update(pred, labels)
        nIoU_metric.update(pred, labels)
        ROC.update(pred, labels)
        mIoU.update(pred, labels)
        PD_FA.update(pred, labels)
        FA, PD = PD_FA.get(len_val)
        ture_positive_rate, false_positive_rate, recall, precision = ROC.get()
        _, mean_IOU = mIoU.get()
        _, IoU = iou_metric.get()
        _, nIoU = nIoU_metric.get()
        data_loader.desc = "[valid epoch {}] loss: {:.6f}, mIoU: {:.6f}, nIoU: {:.6f}".format(epoch, losses.avg, IoU, nIoU)
        save_Pred_GT(pred, labels, path, ids, step, suffix)
    total_visulization_generation(dataset_dir, test_txt, suffix, path, path, iou_list)
    return losses.avg, IoU, nIoU, mean_IOU, ture_positive_rate, false_positive_rate, recall, precision, PD, FA
