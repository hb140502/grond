import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np

import argparse
import os
from pprint import pprint

from utils_grond import *
from train import train_model, eval_model
from poison_loader import *

def CLP_head(net, u):
    print('CLP_head.')
    print('net: ', net.state_dict())
    params = net.state_dict()
    num_heads = net.blocks[0].attn.num_heads
    head_dim = net.blocks[0].attn.head_dim
    
    for name, m in net.named_modules():
        print('name: ', name)
        head_lips = []
        if 'qkv' in name:
            head_weights = m.weight.reshape(3, num_heads, head_dim, -1)

            for i in range(3):
                for j in range(num_heads):
                    weight = head_weights[i][j]
                    head_lips.append(torch.svd(weight)[1].max())
            head_lips = torch.Tensor(head_lips)
            index = torch.where(head_lips>head_lips.mean() + u*head_lips.std())[0]

            for i in index:
                qkv_index = i//num_heads
                heads_index = i%num_heads
                head_weights[qkv_index][heads_index].data *= 0.5

def CLP(net, u=3.0):
    params = net.state_dict()
    for name, m in net.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            std = m.running_var.sqrt()
            weight = m.weight

            channel_lips = []
            for idx in range(weight.shape[0]):
                # Combining weights of convolutions and BN
                if idx >= conv.weight.shape[0]:
                    continue
                w = conv.weight[idx].reshape(conv.weight.shape[1], -1) * (weight[idx]/std[idx]).abs()
                channel_lips.append(torch.svd(w.cpu())[1].max())
            channel_lips = torch.Tensor(channel_lips)

            index = torch.where(channel_lips>channel_lips.mean() + u*channel_lips.std())[0]

            params[name+'.weight'][index] = 0
            params[name+'.bias'][index] = 0
        
       # Convolutional layer should be followed by a BN layer by default
        elif isinstance(m, nn.Conv2d):
            conv = m

    net.load_state_dict(params)


def main(args):
    set_seed(args.seed)

    if args.dataset=='cifar10':
        args.num_classes = 10
        poi_test = POI_TEST(args.dataset, args.clean_data_path, target_cls=args.target_cls, transform=transform_test[args.dataset], upgd_path=args.upgd_path)
        test_set = datasets.CIFAR10(args.clean_data_path, train=False, transform=transform_test[args.dataset])
    elif args.dataset=='imagenet200':
        args.num_classes = 200
        poi_test = ImageNet200_POI_TEST(args.clean_data_path, target_cls=args.target_cls, transform=imagenet_transform_test, upgd_path=args.upgd_path)
        test_set = test_set = datasets.ImageFolder(root=args.clean_data_path+'/imagenet200/val', transform=imagenet_transform_test)
    elif args.dataset=='gtsrb':
        args.num_classes = 43
        poi_test = GTSRB_POI_TEST(args.clean_data_path, target_cls=args.target_cls, transform=gtsrb_transform_test, upgd_path=args.upgd_path)
        test_set = datasets.ImageFolder(root=args.clean_data_path+'/GTSRB/val4imagefolder', transform=gtsrb_transform_test)
    else:
        print('check dataset.')
        exit(0)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=10)
    poi_test_loader = DataLoader(poi_test, batch_size=args.batch_size, shuffle=False, num_workers=10)
    
    model = make_and_restore_model(args, resume_path=args.model_save_path)
    args.num_steps = 20
    args.step_size = args.eps * 2.5 / args.num_steps
    args.random_restarts = 5
    val_loader = None
    eval_model(args, model, val_loader, test_loader, poi_test_loader=poi_test_loader, write_csv=False)

    print('CLP pruning...')
    if args.arch == 'vit_small':
        CLP_head(model, 1.5)
    else:
        CLP(model)
    print('Pruned...')

    eval_model(args, model, val_loader, test_loader, poi_test_loader=poi_test_loader, write_csv=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Training classifiers for CIFAR10')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', default=0.01, choices=[0.1, 0.05, 0.01], type=float)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-4, choices=[0, 1e-4, 5e-4, 1e-3], type=float)

    parser.add_argument('--train_loss', default='ST', choices=['ST', 'AT'], type=str)
    parser.add_argument('--pr', default=0.01, type=float)
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)

    parser.add_argument('--arch', default='ResNet18', type=str, choices=['VGG16', 'VGG19', 'ResNet18', 
        'ResNet50', 'DenseNet121', 'EfficientNetB0', 'inception_next_tiny', 'inception_next_small', 'vit_small'])
    
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--out_dir', default='results/', type=str)
    parser.add_argument('--clean_data_path', default='../data/cifar10', type=str)
    parser.add_argument('--upgd_path', default='./results/upgd-cifar10-ResNet18-Linf-eps8.0', type=str)
    parser.add_argument('--target_cls', default=2, type=int)
    parser.add_argument('--model_save_path', default='./results/ResNet18-cifar10-STonupgd_backdoor-lr0.01-bs128-wd0.0005-pr0.1-seed0-/checkpoint.pth', type=str)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--image_size', default=32, type=int)
    parser.add_argument('--gpuid', default=0, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)

    args = parser.parse_args()
    
    args.tensorboard_path = os.path.join(args.out_dir, 'tensorboard')
    # args.model_save_path = os.path.join(args.out_dir, 'checkpoint.pth')
    args.epochs = 200
    args.lr_milestones = [100, 150]
    args.lr_step = 0.1
    args.log_gap = 1
    args.step_size = args.eps / 5
    args.num_steps = 7
    args.random_restarts = 1
    args.val_num_examples = 1000

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)

