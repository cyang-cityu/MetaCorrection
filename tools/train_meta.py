import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random
from tensorboardX import SummaryWriter
import _init_paths
from evaluate_cityscapes import evaluate
from nets.deeplab_multi import DeeplabMulti
from nets.meta_deeplab_multi import Res_Deeplab
from nets.discriminator import FCDiscriminator
from utils.loss import CrossEntropy2d
from datasets.gta5_dataset import GTA5DataSet
from datasets.cityscapes_dataset import cityscapesPseudo
import datetime
import time

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

MODEL = 'LTIR'
BATCH_SIZE = 1
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/home/cyang53/CED/Data/UDA_Natural/Cityscapes'
DATA_LIST_PATH = '/home/cyang53/CED/Ours/MetaCorrection-CVPR/datasets/gta5_list/meta.lst'
IGNORE_LABEL = 255
INPUT_SIZE = '1024, 512'
DATA_DIRECTORY_TARGET = '/home/cyang53/CED/Data/UDA_Natural/Cityscapes'
DATA_LIST_PATH_TARGET = '/home/cyang53/CED/Ours/MetaCorrection-CVPR/datasets/cityscapes_list/pseudo_ltir_new.lst'
INPUT_SIZE_TARGET = '1024, 512'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_CLASSES = 19
NUM_STEPS = 250000
NUM_STEPS_STOP = 150000  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '/home/cyang53/CED/Ours/MetaCorrection-CVPR/snapshots/Pseudo_LTIR_best.pth'
SAVE_PRED_EVERY = 1000
WEIGHT_DECAY = 0.0005
LOG_DIR = '/home/cyang53/CED/Ours/MetaCorrection-CVPR/log/ltir_meta_debug'

LAMBDA_SEG = 0.1
GPU = '1'
TARGET = 'cityscapes'
SET = 'train'
T_WEIGHT = 0.11
IS_META = True
UPDATA_F = 1

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--is-meta", type=bool, default=IS_META, 
                        help="Whether to update T")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--t-weight", type=float, default=T_WEIGHT,
                        help="grad weight to correct T.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--tensorboard", action='store_true', default=True, help="choose whether to use tensorboard.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    parser.add_argument("--gpu", type=str, default=GPU,
                        help="gpu id to run.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--update-f", type=int, default=UPDATA_F,
                        help="update frequency for T.")
    parser.add_argument("--uncertainty", type=bool, default=True,
                        help="choose adaptation set.")
    return parser.parse_args()


args = get_arguments()


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def adjust_learning_rate_D(optimizer, i_iter):
    lr = lr_poly(args.learning_rate_D, i_iter, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

def build_model(args):

    net = Res_Deeplab(num_classes=args.num_classes)
    #print(net)

    if torch.cuda.is_available():
        net.cuda()
        torch.backends.cudnn.benchmark=True

    return net

def to_var(x, requires_grad=True):
    x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d(is_softmax=False).cuda()

    return criterion(pred, label)

def main():
    """Create the model and start the training."""
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    if not os.path.exists(args.log_dir + '/result'):
        os.makedirs(args.log_dir + '/result')

    best_mIoU = 0
    mIoU = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    metaloader = data.DataLoader(GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
    crop_size=input_size_target, scale=False, mirror=args.random_mirror, mean=IMG_MEAN), batch_size=args.update_f * args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    targetloader = data.DataLoader(cityscapesPseudo(args.data_dir_target, args.data_list_target,
    max_iters=args.num_steps * args.iter_size * args.batch_size,
    crop_size=input_size_target,
    scale=False, mirror=args.random_mirror, mean=IMG_MEAN),
    batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
    pin_memory=True)

    targetloader_iter = enumerate(targetloader)

    main_model = build_model(args)
    saved_state_dict = torch.load(args.restore_from)
    pretrained_dict = {k:v for k,v in saved_state_dict.items() if k in main_model.state_dict()}
    main_model.load_state_dict(pretrained_dict)

    optimizer = optim.SGD(main_model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer.zero_grad()



    interp = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(h, w), mode='bilinear', align_corners=True)


    # set up tensor board
    if args.tensorboard:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)

        writer = SummaryWriter(args.log_dir)

    for i_iter in range(args.num_steps):
        if args.is_meta:
            main_model.train()
            l_f_meta = 0
            l_g_meta = 0
            l_f = 0

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, i_iter)

            meta_net = Res_Deeplab(num_classes=args.num_classes)
            meta_net.load_state_dict(main_model.state_dict())
            meta_net.cuda()

            _, batch = targetloader_iter.__next__()
            image, label, _, _ = batch

            image = to_var(image, requires_grad=False)
            label = to_var(label, requires_grad=False)

            T1 = to_var(torch.eye(19, 19))
            T2 = to_var(torch.eye(19, 19))

            y_f_hat1, y_f_hat2 = meta_net(image)
            y_f_hat1 = torch.softmax(interp_target(y_f_hat1), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            y_f_hat2 = torch.softmax(interp_target(y_f_hat2), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)

            pre1 = torch.mm(y_f_hat1, T1).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            pre2 = torch.mm(y_f_hat2, T2).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            l_f_meta = loss_calc(pre2, label) + 0.1 * loss_calc(pre1, label)

            meta_net.zero_grad()

            grads = torch.autograd.grad(l_f_meta, (meta_net.params()), create_graph=True)
            meta_net.update_params(1e-3, source_params=grads)

            x_val, y_val, _, _ = next(iter(metaloader))
            x_val = to_var(x_val, requires_grad=False)
            y_val = to_var(y_val, requires_grad=False)

            y_g_hat1, y_g_hat2 = meta_net(x_val)
            y_g_hat1 = torch.softmax(interp(y_g_hat1), dim=1)
            y_g_hat2 = torch.softmax(interp(y_g_hat2), dim=1)

            l_g_meta = loss_calc(y_g_hat2, y_val) + 0.1 * loss_calc(y_g_hat1, y_val)
            grad_eps1 = torch.autograd.grad(l_g_meta, T1, only_inputs=True, retain_graph=True)[0]
            grad_eps2 = torch.autograd.grad(l_g_meta, T2, only_inputs=True)[0]
            #print(torch.max(grad_eps1), torch.max(grad_eps2))

            # norm_grad1 = torch.max(torch.abs(grad_eps1), 1)[0].unsqueeze(1)
            # one = torch.ones_like(norm_grad1)
            # norm_grad1 = torch.where(norm_grad1 == 0, one, norm_grad1)
            # grad_eps1 = torch.div(grad_eps1, norm_grad1)
            grad_eps1 = grad_eps1 / torch.max(grad_eps1)
            T1 = torch.clamp(T1-0.11*grad_eps1,min=0)
            # T1 = torch.softmax(T1, 1)
            norm_c = torch.sum(T1, 1)

            for j in range(args.num_classes):
                if norm_c[j] != 0:
                    T1[j, :] /= norm_c[j]


            # norm_grad2 = torch.max(torch.abs(grad_eps2), 1)[0].unsqueeze(1)
            # one = torch.ones_like(norm_grad2)
            # norm_grad2 = torch.where(norm_grad2 == 0, one, norm_grad2)
            # grad_eps2 = torch.div(grad_eps2, norm_grad2)
            grad_eps2 = grad_eps2 / torch.max(grad_eps2)
            T2 = torch.clamp(T2-0.11*grad_eps2,min=0)
            # T2 = torch.softmax(T2, 1)

            norm_c = torch.sum(T2, 1)


            for j in range(args.num_classes):
                if norm_c[j] != 0:
                    T2[j, :] /= norm_c[j]

            # print(T2)
            # norm_grad1 = torch.max(torch.abs(grad_eps1), 1)[0].unsqueeze(1)
            
            # one = torch.ones_like(norm_grad1)
            # norm_grad1 = torch.where(norm_grad1 == 0, one, norm_grad1)
            
            # # print(torch.count_nonzero(norm_grad1))
            # # print(grad_eps1.size(), norm_grad1.size())
            # grad_eps1 = torch.div(grad_eps1, norm_grad1)
            # T1 = torch.clamp(T1 - args.t_weight * grad_eps1, min=0)
            # norm_c = torch.sum(T1, 1).unsqueeze(1)

            # T1 = torch.div(T1, norm_c)

            # # for j in range(19):
            # #     if norm_c[:, j] != 0:
            # #         T1[:, :, j] /= norm_c[:, j]
            # norm_grad2 = torch.max(torch.abs(grad_eps2), 1)[0].unsqueeze(1)
            # norm_grad2 = torch.where(norm_grad2 == 0, one, norm_grad2)
            
            # grad_eps2 = torch.div(grad_eps2, norm_grad2)
            # T2 = torch.clamp(T2 - args.t_weight * grad_eps2, min=0)
            # norm_c = torch.sum(T2, 1).unsqueeze(1)

            # T2 = torch.div(T2, norm_c)
            #print(T1, T2)
            # for j in range(19):
            #     if norm_c[:, j] != 0:
            #         T2[:, :, j] /= norm_c[:, j]

            # print(T1, T2)

            y_f_hat1, y_f_hat2 = main_model(image)
            y_f_hat1 = torch.softmax(interp_target(y_f_hat1), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            y_f_hat2 = torch.softmax(interp_target(y_f_hat2), dim=1).permute(0, 2, 3, 1).contiguous().view(-1, args.num_classes)
            pre1 = torch.mm(y_f_hat1, T1).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)
            pre2 = torch.mm(y_f_hat2, T2).view(args.batch_size, h, w, args.num_classes).permute(0, 3, 1, 2)

            l_f = loss_calc(pre2, label) + 0.1 * loss_calc(pre1, label)
            optimizer.zero_grad()
            l_f.backward()
            optimizer.step()

            if args.tensorboard:
                scalar_info = {
                    'loss_g_meta': l_g_meta.item(),
                    'loss_f_meta': l_f_meta.item(),
                    'loss_f': l_f.item(),
                }

                if i_iter % 10 == 0:
                    for key, val in scalar_info.items():
                        writer.add_scalar(key, val, i_iter)

            print('exp = {}'.format(args.log_dir))
            print(
            'iter = {0:8d}/{1:8d}, loss_g_meta = {2:.3f} loss_f_meta = {3:.3f} loss_f = {4:.3f}'.format(
                i_iter, args.num_steps, l_g_meta.item(), l_f_meta.item(), l_f.item()))

        if i_iter >= args.num_steps_stop - 1:
            print('save model ...')
            torch.save(main_model.state_dict(), osp.join(args.log_dir, 'GTA5_' + str(args.num_steps_stop) + '.pth'))
            break
        if i_iter % args.save_pred_every == 0 and i_iter > 0:
            now = datetime.datetime.now()
            print (now.strftime("%Y-%m-%d %H:%M:%S"), '  Begin evaluation on iter {0:8d}/{1:8d}  '.format(i_iter, args.num_steps))
            mIoU = evaluate(main_model, pred_dir=args.log_dir + '/result')
            writer.add_scalar('mIoU', mIoU, i_iter)
            print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
            if mIoU > best_mIoU:
                best_mIoU = mIoU
                torch.save(main_model.state_dict(), osp.join(args.log_dir, 'Uncertainty_LTIR_best.pth'))

    if args.tensorboard:
        writer.close()


if __name__ == '__main__':
    main()
