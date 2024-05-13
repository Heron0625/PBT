import os
import shutil
import sys
from collections import OrderedDict
import torch.nn.functional as F
import torch

from model.PBT_v2 import WindowPBTNet
from argparse import ArgumentParser
from datetime import datetime
from skimage import measure

from colorama import Fore
from PIL import Image, ImageOps, ImageFilter
from torch.utils.data.dataset import Dataset

from tqdm import tqdm
import cv2
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
import time
from  matplotlib import pyplot as plt

# model_dir=f'weights/Best_nIoU_IoU-0.7896_nIoU-0.7943.pth.tar'
model_dir=f'weights/Best_nIoU_IoU-0.9096_nIoU-0.9187.pth.tar'
model_dir=f'weights/Best_mIoU_Best_nIoU_IoU-0.9096_nIoU-0.9187-state_dict.pth.tar'
root_dir = model_dir.split('Best')[0]
model_name = model_dir.split('Best')[1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')
def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='dataset:IRSTD-1k; MFIRST; NUAA-SIRST; NUDT-SIRST; IRSTD-Air')
    parser.add_argument('--split_method', type=str, default='6_2_2',help='6_4')
    parser.add_argument('--size', type=int, default=256,help='')
    parser.add_argument('--root', type=str, default='../HCTNet/dataset', help='')
    parser.add_argument('--suffix', type=str, default='.png')

    #
    # Training parameters
    #
    # Net parameters
    #
    #
    # Dataset parameters
    #
    args = parser.parse_args()
    return args

class SigmoidMetric():
    def __init__(self):
        self.reset()

    def update(self, pred, labels):
        correct, labeled = self.batch_pix_accuracy(pred, labels)
        inter, union = self.batch_intersection_union(pred, labels)

        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union

    def get(self):
        """Gets the current evaluation result."""
        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0

    def batch_pix_accuracy(self, output, target):
        assert output.shape == target.shape
        output = output.detach().numpy()
        target = target.detach().numpy()

        predict = (output > 0).astype('int64') # P
        pixel_labeled = np.sum(target > 0) # T
        pixel_correct = np.sum((predict == target)*(target > 0)) # TP
        assert pixel_correct <= pixel_labeled
        return pixel_correct, pixel_labeled

    def batch_intersection_union(self, output, target):
        mini = 1
        maxi = 1 # nclass
        nbins = 1 # nclass
        predict = (output.detach().numpy() > 0).astype('int64') # P
        target = target.numpy().astype('int64') # T
        intersection = predict * (predict == target) # TP

        # areas of intersection and union
        area_inter, _ = np.histogram(intersection, bins=nbins, range=(mini, maxi))
        area_pred, _ = np.histogram(predict, bins=nbins, range=(mini, maxi))
        area_lab, _ = np.histogram(target, bins=nbins, range=(mini, maxi))
        area_union = area_pred + area_lab - area_inter
        assert (area_inter <= area_union).all()
        return area_inter, area_union



class SamplewiseSigmoidMetric():
    def __init__(self, nclass, score_thresh=0.5):
        self.nclass = nclass
        self.score_thresh = score_thresh
        self.reset()

    def update(self, preds, labels):
        """Updates the internal evaluation result."""
        inter_arr, union_arr = self.batch_intersection_union(preds, labels,
                                                             self.nclass, self.score_thresh)
        self.total_inter = np.append(self.total_inter, inter_arr)
        self.total_union = np.append(self.total_union, union_arr)

    def get(self):
        """Gets the current evaluation result."""
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return IoU, mIoU

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.total_inter = np.array([])
        self.total_union = np.array([])
        self.total_correct = np.array([])
        self.total_label = np.array([])

    def batch_intersection_union(self, output, target, nclass, score_thresh):
        """mIoU"""
        # inputs are tensor
        # the category 0 is ignored class, typically for background / boundary
        mini = 1
        maxi = 1  # nclass
        nbins = 1  # nclass

        predict = (F.sigmoid(output).detach().numpy() > score_thresh).astype('int64') # P
        target = target.detach().numpy().astype('int64') # T
        intersection = predict * (predict == target) # TP

        num_sample = intersection.shape[0]
        area_inter_arr = np.zeros(num_sample)
        area_pred_arr = np.zeros(num_sample)
        area_lab_arr = np.zeros(num_sample)
        area_union_arr = np.zeros(num_sample)

        for b in range(num_sample):
            # areas of intersection and union
            area_inter, _ = np.histogram(intersection[b], bins=nbins, range=(mini, maxi))
            area_inter_arr[b] = area_inter

            area_pred, _ = np.histogram(predict[b], bins=nbins, range=(mini, maxi))
            area_pred_arr[b] = area_pred

            area_lab, _ = np.histogram(target[b], bins=nbins, range=(mini, maxi))
            area_lab_arr[b] = area_lab

            area_union = area_pred + area_lab - area_inter
            area_union_arr[b] = area_union

            assert (area_inter <= area_union).all()

        return area_inter_arr, area_union_arr
class ROCMetric():
    """Computes pixAcc and mIoU metric scores
    """
    def __init__(self, nclass, bins):  #bin的意义实际上是确定ROC曲线上的threshold取多少个离散值
        super(ROCMetric, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.tp_arr = np.zeros(self.bins+1)
        self.pos_arr = np.zeros(self.bins+1)
        self.fp_arr = np.zeros(self.bins+1)
        self.neg_arr = np.zeros(self.bins+1)
        self.class_pos=np.zeros(self.bins+1)
        # self.reset()

    def update(self, preds, labels):
        for iBin in range(self.bins+1):
            score_thresh = (iBin + 0.0) / self.bins
            # print(iBin, "-th, score_thresh: ", score_thresh)
            i_tp, i_pos, i_fp, i_neg,i_class_pos = cal_tp_pos_fp_neg(preds, labels, self.nclass,score_thresh)
            self.tp_arr[iBin]   += i_tp
            self.pos_arr[iBin]  += i_pos
            self.fp_arr[iBin]   += i_fp
            self.neg_arr[iBin]  += i_neg
            self.class_pos[iBin]+=i_class_pos

    def get(self):

        tp_rates    = self.tp_arr / (self.pos_arr + 0.001)
        fp_rates    = self.fp_arr / (self.neg_arr + 0.001)

        recall      = self.tp_arr / (self.pos_arr   + 0.001)
        precision   = self.tp_arr / (self.class_pos + 0.001)


        return tp_rates, fp_rates, recall, precision

    def reset(self):

        self.tp_arr   = np.zeros([self.bins+1])
        self.pos_arr  = np.zeros([self.bins+1])
        self.fp_arr   = np.zeros([self.bins+1])
        self.neg_arr  = np.zeros([self.bins+1])
        self.class_pos= np.zeros([self.bins+1])



class PD_FA():
    def __init__(self, nclass, bins):
        super(PD_FA, self).__init__()
        self.nclass = nclass
        self.bins = bins
        self.image_area_total = []
        self.image_area_match = []
        self.FA = np.zeros(self.bins+1)
        self.PD = np.zeros(self.bins + 1)
        self.target= np.zeros(self.bins + 1)
    def update(self, preds, labels):
        W = preds.shape[3]
        for iBin in range(self.bins+1):
            score_thresh = iBin * (255/self.bins)
            predits  = np.array((preds > score_thresh).cpu()).astype('int64')
            if W == 512:
                predits  = np.reshape (predits,  (512,512))#512
                labelss = np.array((labels).cpu()).astype('int64') # P
                labelss = np.reshape (labelss , (512,512))#512
            elif W==384:
                predits = np.reshape(predits, (384, 384))  # 512
                labelss = np.array((labels).cpu()).astype('int64')  # P
                labelss = np.reshape(labelss, (384, 384))  # 512
            else:
                predits = np.reshape(predits, (512//2, 512//2))  # 512
                labelss = np.array((labels).cpu()).astype('int64')  # P
                labelss = np.reshape(labelss, (512//2, 512//2))  # 512

            image = measure.label(predits, connectivity=2)
            coord_image = measure.regionprops(image)
            label = measure.label(labelss , connectivity=2)
            coord_label = measure.regionprops(label)

            self.target[iBin]    += len(coord_label)
            self.image_area_total = []
            self.image_area_match = []
            self.distance_match   = []
            self.dismatch         = []

            for K in range(len(coord_image)):
                area_image = np.array(coord_image[K].area)
                self.image_area_total.append(area_image)

            for i in range(len(coord_label)):
                centroid_label = np.array(list(coord_label[i].centroid))
                for m in range(len(coord_image)):
                    centroid_image = np.array(list(coord_image[m].centroid))
                    distance = np.linalg.norm(centroid_image - centroid_label)
                    area_image = np.array(coord_image[m].area)
                    if distance < 3:
                        self.distance_match.append(distance)
                        self.image_area_match.append(area_image)

                        del coord_image[m]
                        break

            self.dismatch = [x for x in self.image_area_total if x not in self.image_area_match]
            self.FA[iBin]+=np.sum(self.dismatch)
            self.PD[iBin]+=len(self.distance_match)
        # print(len(self.image_area_total))
    def get(self,img_num):

        Final_FA =  self.FA / ((512*512) * img_num)#512
        Final_PD =  self.PD /self.target

        return Final_FA,Final_PD


    def reset(self):
        self.FA  = np.zeros([self.bins+1])
        self.PD  = np.zeros([self.bins+1])

class mIoU():

    def __init__(self, nclass):
        super(mIoU, self).__init__()
        self.nclass = nclass
        self.reset()

    def update(self, preds, labels):
        # print('come_ininin')

        correct, labeled = batch_pix_accuracy(preds, labels)
        inter, union = batch_intersection_union(preds, labels, self.nclass)
        self.total_correct += correct
        self.total_label += labeled
        self.total_inter += inter
        self.total_union += union


    def get(self):

        pixAcc = 1.0 * self.total_correct / (np.spacing(1) + self.total_label)
        IoU = 1.0 * self.total_inter / (np.spacing(1) + self.total_union)
        mIoU = IoU.mean()
        return pixAcc, mIoU

    def reset(self):

        self.total_inter = 0
        self.total_union = 0
        self.total_correct = 0
        self.total_label = 0




def cal_tp_pos_fp_neg(output, target, nclass, score_thresh):

    predict = (torch.sigmoid(output) > score_thresh).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    intersection = predict * ((predict == target).float())

    tp = intersection.sum()
    fp = (predict * ((predict != target).float())).sum()
    tn = ((1 - predict) * ((predict == target).float())).sum()
    fn = (((predict != target).float()) * (1 - predict)).sum()
    pos = tp + fn
    neg = fp + tn
    class_pos= tp+fp

    return tp, pos, fp, neg, class_pos

def batch_pix_accuracy(output, target):

    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")

    assert output.shape == target.shape, "Predict and Label Shape Don't Match"
    predict = (output > 0).float()
    pixel_labeled = (target > 0).float().sum()
    pixel_correct = (((predict == target).float())*((target > 0)).float()).sum()



    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled


def batch_intersection_union(output, target, nclass):

    mini = 1
    maxi = 1
    nbins = 1
    predict = (output > 0).float()
    if len(target.shape) == 3:
        target = np.expand_dims(target.float(), axis=1)
    elif len(target.shape) == 4:
        target = target.float()
    else:
        raise ValueError("Unknown target dimension")
    intersection = predict * ((predict == target).float())

    area_inter, _  = np.histogram(intersection.cpu(), bins=nbins, range=(mini, maxi))
    area_pred,  _  = np.histogram(predict.cpu(), bins=nbins, range=(mini, maxi))
    area_lab,   _  = np.histogram(target.cpu(), bins=nbins, range=(mini, maxi))
    area_union     = area_pred + area_lab - area_inter

    assert (area_inter <= area_union).all(), \
        "Error: Intersection area should be smaller than Union area"
    return area_inter, area_union
class TestSetLoader(Dataset):
    """Iceberg Segmentation dataset."""
    NUM_CLASS = 1

    def __init__(self, dataset_dir, img_id,transform=None,base_size=512,crop_size=480,suffix='.png'):
        super(TestSetLoader, self).__init__()
        self.transform = transform
        self._items    = img_id
        self.masks     = dataset_dir+'/'+'masks'
        self.images    = dataset_dir+'/'+'images'
        self.base_size = base_size
        self.crop_size = crop_size
        self.suffix    = suffix

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img  = img.resize ((base_size, base_size), Image.BICUBIC)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        # final transform
        img, mask = np.array(img), np.array(mask, dtype=np.float32)  # img: <class 'mxnet.ndarray.ndarray.NDArray'> (512, 512, 3)
        return img, mask

    def __getitem__(self, idx):
        # print('idx:',idx)
        cv2.setNumThreads(0)

        img_id = self._items[idx]  # idx：('../SIRST', 'Misc_70') 成对出现，因为我的workers设置为了2
        img_path   = self.images+'/'+img_id+self.suffix    # img_id的数值正好补了self._image_path在上面定义的2个空
        label_path = self.masks +'/'+img_id+self.suffix
        img  = Image.open(img_path).convert('RGB')  ##由于输入的三通道、单通道图像都有，所以统一转成RGB的三通道，这也符合Unet等网络的期待尺寸
        mask = Image.open(label_path)
        # synchronized transform
        img, mask = self._testval_sync_transform(img, mask)

        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        mask = np.expand_dims(mask, axis=0).astype('float32') / 255.0


        return img, torch.from_numpy(mask)  # img_id[-1]

    def __len__(self):
        return len(self._items)
def load_dataset (root, dataset, split_method):
    test_txt  = root + '/' + dataset + '/' + split_method + '/' + 'valtest.txt'
    val_img_ids = []
    with open(test_txt, "r") as f:
        line = f.readline()
        while line:
            val_img_ids.append(line.split('\n')[0])
            line = f.readline()
        f.close()
    return val_img_ids,test_txt

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

def save_Pred_GT(pred, labels, target_image_path, val_img_ids, num, suffix, size):

    predsss = np.array((pred > 0).cpu()).astype('int64') * 255
    predsss = np.uint8(predsss)
    labelsss = labels * 255
    labelsss = np.uint8(labelsss.cpu())

    img = Image.fromarray(predsss.reshape(size, size))
    img.save(target_image_path + '/' + '%s_Pred' % (val_img_ids[num]) +suffix)
    img = Image.fromarray(labelsss.reshape(size, size))
    img.save(target_image_path + '/' + '%s_GT' % (val_img_ids[num]) + suffix)

def total_visulization_generation(dataset_dir, test_txt, suffix, target_image_path, target_dir, list, size):
    source_image_path = dataset_dir + '/images'
    target_dir = target_dir + '/fuse'
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    txt_path = test_txt
    ids = []
    with open(txt_path, 'r') as f:
        ids += [line.strip() for line in f.readlines()]

    for i in range(len(ids)):
        source_image = source_image_path + '/' + ids[i] + suffix
        target_image = target_image_path + '/' + ids[i] + suffix
        shutil.copy(source_image, target_image)
    for i in range(len(ids)):
        source_image = target_image_path + '/' + ids[i] + suffix
        img = Image.open(source_image)
        img = img.resize((size, size), Image.ANTIALIAS)
        img.save(source_image)
    for m in range(len(ids)):
        iou = list[m]
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 3, 1)
        img = plt.imread(target_image_path + '/' + ids[m] + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Raw Imamge", size=11)

        plt.subplot(1, 3, 2)
        img = plt.imread(target_image_path + '/' + ids[m] + '_GT' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Ground Truth", size=11)

        plt.subplot(1, 3, 3)
        img = plt.imread(target_image_path + '/' + ids[m] + '_Pred' + suffix)
        plt.imshow(img, cmap='gray')
        plt.xlabel("Predicts"+str(iou)[:4], size=11)

        plt.savefig(target_dir + '/' + ids[m].split('.')[0] + "_fuse" + suffix, facecolor='w', edgecolor='red')
        plt.close()



@torch.no_grad()
def visual(model, data_loader, device, epoch, iou_metric, nIoU_metric, PD_FA, mIoU, ROC, len_val, path, ids, suffix, dataset_dir, test_txt, size):

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
                # loss += loss_function(p, labels, None)
                pass
            loss /= len(pred)
            pred = pred[-1]
        else:
            # loss = loss_function(pred, labels, None)
            pass
        pred0 = torch.sigmoid(pred)
        pred0[pred0 > 0.5] = 1
        pred0[pred0 <= 0.5] = 0
        label = labels.float()
        label[label > 0] = 1
        intersection = pred0 * label
        loss_iou = (intersection.sum() + 1e-6) / (pred0.sum() + label.sum() - intersection.sum() + 1e-6)
        # loss_iou = 1.-loss_iou
        iou_list.append(loss_iou.tolist())
        # losses.update(loss.item(), pred.size(0))
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
        save_Pred_GT(pred, labels, path, ids, step, suffix, size)
    total_visulization_generation(dataset_dir, test_txt, suffix, path, path, iou_list, size)
    return losses.avg, IoU, nIoU, mean_IOU, ture_positive_rate, false_positive_rate, recall, precision, PD, FA


def main(args):

    dataset = args.dataset

    root, split_method, size, batch = args.root, args.split_method, args.size, 1
    dataset_dir = root + '/' + dataset
    val_img_ids, test_txt = load_dataset(root, dataset, split_method)
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(root_dir, model_name + "_visual")
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])
    ])
    valset = TestSetLoader(dataset_dir, img_id=val_img_ids, base_size=size, crop_size=size,
                           transform=input_transform, suffix=args.suffix)
    val_data = DataLoader(dataset=valset, batch_size=1, num_workers=0,drop_last=False)
    model = WindowPBTNet(num_classes=1, input_channels=3, num_blocks=[2, 2, 2, 2, 2],
                         nb_filter=[16, 32, 64, 128, 256, 512],
                         deep_supervision=False, depth=[1, 2, 2, 2, 2], drop=0.1, attn_drop=0.1, drop_path=0.1,
                         mlp_ratio=4.,
                         heads=[[1, 1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1], [1, 1], [1]], token_projection='linear',
                         token_mlp='leff', win_size=8, img_size=size, )

    model = model.to(device)
    ckpt = torch.load(model_dir, map_location='cpu')
    model_weights = ckpt['state_dict']
    try:
        model.load_state_dict(ckpt['state_dict'], strict=True)
    except:
        new_dict = OrderedDict()
        for k, v in model_weights.items():
            name = k[7:]
            new_dict[name] = v
        model.load_state_dict(new_dict, strict=True)
    print('fine')
    # print(ckpt['mean_IOU'])

    iou_metric = SigmoidMetric()
    niou_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
    roc = ROCMetric(1, 10)
    pdfa = PD_FA(1, 10)
    miou = mIoU(1)

    save_folder = log_dir
    save_visual = os.path.join(save_folder, 'visual')
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    if not os.path.exists(save_visual):
        os.mkdir(save_visual)
    val_loss, iou_, niou_, miou_, ture_positive_rate, false_positive_rate, recall, precision, pd, fa = \
        visual(model, val_data, device, 0, iou_metric, niou_metric, pdfa, miou, roc, len(valset),
               save_visual, val_img_ids, args.suffix, dataset_dir, test_txt, args.size)
    tags = ['train_loss', 'val_loss', 'IoU', 'nIoU', 'mIoU', 'PD', 'FA', 'tp', 'fa', 'rc', 'pr', 'roc']
    note = open(os.path.join(save_folder, args.dataset+args.split_method+'.txt'), mode='w')
    note.write('IoU:\n')
    note.write('{}\n'.format(iou_))
    note.write('nIoU:\n')
    note.write('{}\n'.format(niou_))
    note.write('TP:\n')
    note.write('{}\n'.format(ture_positive_rate))
    note.write('FP:\n')
    note.write('{}\n'.format(false_positive_rate))
    note.write('recall:\n')
    note.write('{}\n'.format(recall))
    note.write('precision:\n')
    note.write('{}\n'.format(precision))
    note.write('PD:\n')
    note.write('{}\n'.format(pd))
    note.write('FA:\n')
    note.write('{}\n'.format(fa))

if __name__ == '__main__':

    args = parse_args()
    main(args)