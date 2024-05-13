import torch

from model.utils import save_ckpt

model_dir=f'weights/Best_nIoU_IoU-0.7896_nIoU-0.7943.pth.tar'
# model_dir=f'weights/Best_nIoU_IoU-0.9096_nIoU-0.9187.pth.tar'
ckpt = torch.load(model_dir, map_location='cpu')

save_ckpt({
                        'state_dict': ckpt['state_dict'],
                    }, save_path='./weights',
                        filename='Best_mIoU_' + 'Best_nIoU_IoU-0.7896_nIoU-0.7943-state_dict.pth.tar')