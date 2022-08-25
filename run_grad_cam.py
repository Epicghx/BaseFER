import sys,os,json,ipdb, glob
import cv2
import random
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
import utils
from utils import *
import torch.nn.functional as F

from utils import set_logger
import models
from thop import profile
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM

from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

warnings.simplefilter(action="ignore", category=FutureWarning)

parser = argparse.ArgumentParser()

### ------------------------ modify -------------------------  ####
parser.add_argument('--dataset', type=str,    default = 'RAFDB')
parser.add_argument('--seed',    type=int,    default = 42)
parser.add_argument('--mode',    choices=['finetune', 'prune', 'inference'],
                                              default= 'inference', help='Run mode')
parser.add_argument('--arch',    type=str,    default='sge_resnet18',
                    help='')


#### ---------Paths--------- ####
parser.add_argument('--train_path', type=str, default='./data/{}/train/',
                    help='train path')
parser.add_argument('--val_path', type=str, default='./data/{}/val/',
                    help='train path')
parser.add_argument('--save_folder', type=str, default='./saved/',
                    help='location to save model')
parser.add_argument('--load_folder', type=str, default='',
                    help='')
parser.add_argument('--checkpoint_format', type=str,
                    default='./{save_folder}/{arch}/{dataset}/checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--jsonfile', type=str, default='logs/baseline_face_acc.txt',
                    help='file to restore baseline validation accuracy')
parser.add_argument('--log_path', type=str, default='./saved/{arch}/{dataset}/run.log',
                    help='')

#### ---------Input--------- ####
parser.add_argument('--epochs', type=int, default=200,
                    help='')
parser.add_argument('--batch_size', type=int, default=64,
                    help='')
parser.add_argument('--val_batch_size', type=int, default=64,
                    help='')
parser.add_argument('--image_size', type=int, default=224,
                    help='')
parser.add_argument('--in_channels', type=int, default=3,
                    help='')
parser.add_argument('--num_classes', type=int, default=7,
                    help='')

#### ---------Dataloader--------- ####
parser.add_argument('--workers', type=int, default=8,
                    help='')
parser.add_argument('--restore_epoch', type=int, default=0,
                    help='')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--use_pretrained', action='store_true', default=True,
                    help='')


# Optimizer parameters
parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                    help='Optimizer (default: "sgd"')
parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                    help='Optimizer Epsilon (default: None, use opt default)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='Optimizer momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=2e-5,
                    help='weight decay (default: 2e-5)')

# Learning rate schedule parameters
parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                    help='LR scheduler (default: "step"')
parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                    help='learning rate (default: 0.05)')
parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                    help='learning rate noise on/off epoch percentages')
parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                    help='learning rate noise limit percent (default: 0.67)')
parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                    help='learning rate noise std-dev (default: 1.0)')
parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                    help='learning rate cycle len multiplier (default: 1.0)')
parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                    help='amount to decay each learning rate cycle (default: 0.5)')
parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                    help='learning rate cycle limit, cycles enabled if > 1')
parser.add_argument('--lr-k-decay', type=float, default=1.0,
                    help='learning rate k-decay for cosine/poly (default: 1.0)')
parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                    help='warmup learning rate (default: 0.0001)')
parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                    help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
parser.add_argument('--decay-epochs', type=float, default=2.4, metavar='N',
                    help='epoch interval to decay LR')
parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                    help='epochs to warmup LR, if scheduler supports')
parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                    help='epochs to cooldown LR at min_lr, after cyclic schedule ends')

def main():

    args = parser.parse_args()

    if args.save_folder and not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    if not os.path.isdir(os.path.join(args.save_folder, args.arch, args.dataset)):
        os.makedirs(os.path.join(args.save_folder, args.arch, args.dataset))

    set_logger(args.log_path.format(arch=args.arch, dataset=args.dataset))

    if not torch.cuda.is_available():
        logging.info('no gpu device available\n')
        args.cuda = False

    if args.mode == 'finetune':
        logging.info('\n')
        logging.info("args: {}".format(args))

    # random.seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    #
    # cudnn.deterministic = True
    # cudnn.benchmark = False

    resume_from_epoch = 0
    for try_epoch in range(200, 0, -1):
        if os.path.exists(args.checkpoint_format.format(
                save_folder=args.save_folder, arch=args.arch, dataset=args.dataset, epoch=try_epoch)):
            resume_from_epoch = try_epoch
            break

    if args.restore_epoch:
        resume_from_epoch = args.restore_epoch

    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)

    model = models.__dict__[args.arch](in_channels=args.in_channels, num_classes=args.num_classes, pretrained=True)

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    input1 = torch.randn(1, 3, 224, 224)
    flops, params = profile(model, inputs=(input1,))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')


    if args.cuda:
        # Move model to GPU
        model = nn.DataParallel(model)
        model = model.cuda()

    if (resume_from_epoch != 0) & (args.mode == 'inference'):
        logging.info(f'second-time\n')
        print('Second time')
        filepath = args.checkpoint_format.format(save_folder=args.save_folder, arch=args.arch, dataset=args.dataset, epoch=resume_from_epoch)
        checkpoint = torch.load(filepath)
        state_dict = checkpoint['model_state_dict']
        model.load_state_dict(state_dict)

    target_layers = model.module.layer4[1].sge
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM, }
    for dir in range(7):
        for p in glob.glob('./data/RAFDB/val/{}/test*.jpg'.format(dir)):

            cam = methods['gradcam++'](model=model,
                            target_layers=target_layers,
                            use_cuda=True)

            rgb_img = cv2.imread(p, 1)[:, :, ::-1]
            rgb_img = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_CUBIC)
            rgb_img = np.float32(rgb_img) / 255
            input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

            target_category = None
            cam.batch_size = 32

            grayscale_cam = cam(input_tensor=input_tensor,
                                target_category=target_category,
                                aug_smooth=False,
                                eigen_smooth=False)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

            # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=True)
            # gb = gb_model(input_tensor, target_category=target_category)
            #
            # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
            # cam_gb = deprocess_image(cam_mask * gb)
            # gb = deprocess_image(gb)

            cv2.imwrite(f'./cam_result4/{dir}/{os.path.basename(p)}_name_cam.jpg', cam_image)


if __name__ == '__main__':
    main()