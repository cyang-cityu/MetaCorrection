import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import ttach as tta
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
import _init_paths
from nets.deeplab_multi import DeeplabMulti
from nets.deeplab_vgg import DeeplabVGG
from nets.meta_deeplab_multi import Res_Deeplab
from datasets.cityscapes_dataset import cityscapesDataSet
from collections import OrderedDict
import os
import time
from PIL import Image
import json
from os.path import join
import torch.nn as nn
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/cyang53/CED/Data/UDA_Natural/Cityscapes'
DATA_LIST_PATH = '/home/cyang53/CED/Ours/MetaCorrection-CVPR/datasets/cityscapes_list/val.txt'
SAVE_PATH = '/home/cyang53/CED/Ours/MetaCorrection-CVPR/result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_mIoU(gt_dir, pred_dir, devkit_dir='/home/cyang53/CED/Baseline/AdaptSegNet-CVPR2018/dataset/cityscapes_list'):
    """
    Compute IoU given the predicted colorized images and 
    """
    with open(join(devkit_dir, 'info.json'), 'r') as fp:
      info = json.load(fp)
    num_classes = np.int(info['classes'])
    print('Num classes', num_classes)
    name_classes = np.array(info['label'], dtype=np.str)
    mapping = np.array(info['label2train'], dtype=np.int)
    hist = np.zeros((num_classes, num_classes))

    image_path_list = join(devkit_dir, 'val.txt')
    label_path_list = join(devkit_dir, 'label.txt')
    gt_imgs = open(label_path_list, 'r').read().splitlines()
    gt_imgs = [join(gt_dir, x) for x in gt_imgs]
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        label = np.array(Image.open(gt_imgs[ind]))
        label = label_mapping(label, mapping)
        if len(label.flatten()) != len(pred.flatten()):
            print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
            continue
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # if ind > 0 and ind % 10 == 0:
        #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100*np.mean(per_class_iu(hist))))
    
    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))
    return mIoUs

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def evaluate(seg_model, pred_dir='/home/cyang53/CED/Ours/MetaCorrection-CVPR/result/cityscapes', post=False):
    """Create the model and start the evaluation process."""

    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    T = np.load('/home/cyang53/CED/Ours/MetaCorrection-CVPR/snapshots/Source_500.npy')
    device = torch.device("cuda")
    #print(device)
    model = seg_model.to(device)

    model.eval()

    testloader = data.DataLoader(cityscapesDataSet(DATA_DIRECTORY, DATA_LIST_PATH, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=SET),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    for index, batch in enumerate(testloader):
        # if index % 100 == 0:
        #     print('%d processd' % index)
        image, _, name = batch
        image = image.to(device)
        output1, output2 = model(image)

        #tta_model = tta.SegmentationTTAWrapper(model, tta.aliases.d4_transform(), merge_mode='mean')
        if post:
            output = torch.softmax(interp(0.55 * output1 + 0.45 * output2), dim=1).cpu().data[0].numpy()
            output = np.reshape(output, (NUM_CLASSES, -1))
            output = np.dot(np.linalg.inv(T), output)
            # print(T)
            # output = np.dot(np.linalg.inv(T), output)
            # output = np.dot(np.linalg.inv(T), output)
            output = np.reshape(output, (NUM_CLASSES, 1024, 2048))
        else:
            output = interp(0.45 * output2 + 0.55 * output1).cpu().data[0].numpy()

        output = output.transpose(1,2,0)
        output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
        output_col = colorize_mask(output)
        output = Image.fromarray(output)

        name = name[0].split('/')[-1]
        output.save('%s/%s' % (pred_dir, name))
        output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    gt_dir ='/home/cyang53/CED/Data/UDA_Natural/Cityscapes/label'
    # pred_dir = args.save
    mIoUs = compute_mIoU(gt_dir, pred_dir)
    return round(np.nanmean(mIoUs) * 100, 2)


if __name__ == '__main__':
    print('Begin Evaluation: '+time.asctime(time.localtime(time.time())))
    # model = Res_Deeplab(num_classes=19)
    # pretrained_dict = torch.load('/home/cyang53/CED/Ours/MetaCorrection-CVPR/snapshots/Meta_final.pth')
    # pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model.state_dict()}
    # model.load_state_dict(pretrained_dict)
    model = DeeplabMulti(num_classes=19)
    model.load_state_dict(torch.load('/home/cyang53/CED/Ours/MetaCorrection-CVPR/snapshots/Past/GTA5_best.pth'))
    # model.load_state_dict(torch.load('/home/cyang53/CED/Ours/MetaCorrection-CVPR/snapshots/Pseudo_LTIR_best.pth'))

    # new_params = model.state_dict().copy()
    # saved_state_dict = torch.load('/home/cyang53/CED/Ours/MetaCorrection-CVPR/snapshots/LTIR/ResNet_GTA_50.2.pth')
    # for i in saved_state_dict:
    #     i_parts = i.split('.')
    #     if not i_parts[0] == 'layer5' and not i_parts[0] == 'layer6':
    #         new_params[i] = saved_state_dict[i]
    #     else:
    #         new_params[i.replace('layer5','layer6')] = saved_state_dict[i]
    # model.load_state_dict(new_params)
    evaluate(model, pred_dir= '/home/cyang53/CED/Ours/MetaCorrection-CVPR/log/ltir/result', post=True)
    print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
