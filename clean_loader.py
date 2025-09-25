#!/usr/bin/env python

import os

# torch package
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from Tiny import TinyImageNet


def build_cleanset(args):
    # Setting Dataset Required Parameters
    transforms_list = []
    transforms_list_test = []
    if args.dataset == "cifar10":
        args.num_classes = 10
        args.img_size  = 32
        args.channel   = 3
        args.mean = [0.4914, 0.4822, 0.4465]
        args.std = [0.247, 0.243, 0.261]
    elif args.dataset == "cifar100":
        args.num_classes = 100
        args.img_size  = 32
        args.channel   = 3
        args.mean = [0.5071, 0.4865, 0.4409]
        args.std = [0.2673, 0.2564, 0.2762]
    elif args.dataset == "gtsrb":
        args.num_classes = 43
        args.img_size  = 32
        args.channel   = 3
        args.mean = None
        args.std = None
        transforms_list_test.append(transforms.Resize(32))
        transforms_list_test.append(transforms.CenterCrop(32))
    elif args.dataset == "tiny":
        args.num_classes = 200
        args.img_size  = 64
        args.channel   = 3
        args.mean = [0.4802, 0.4481, 0.3975]
        args.std = [0.2302, 0.2265, 0.2262]
    elif args.dataset == "imagenette":
        args.num_classes = 10
        args.img_size  = 80
        args.channel   = 3
        args.mean = [0.4671, 0.4593, 0.4306], 
        args.std = [0.2692, 0.2657, 0.2884]
    elif args.dataset == "imagenet200":
        args.num_classes = 200
        args.img_size  = 224
        args.channel   = 3
        args.mean = [0.4802, 0.4481, 0.3975]
        args.std = [0.2302, 0.2265, 0.2262]
        transforms_list_test.append(transforms.Resize(256))
        transforms_list_test.append(transforms.CenterCrop(224))

    if args.dataset == "imagenet200":
        transforms_list.append(transforms.RandomResizedCrop(args.img_size))
    else:
        transforms_list.append(transforms.RandomCrop(args.img_size, padding=4))
        transforms_list.append(transforms.RandomRotation(10))

    if args.dataset == "cifar10":
        transforms_list.append(transforms.RandomHorizontalFlip())
        
    transforms_list.append(transforms.ToTensor())        
    transforms_list_test.append(transforms.ToTensor())

    if args.mean is not None and args.std is not None:
        transforms_list.append(transforms.Normalize(args.mean, args.std))
        transforms_list_test.append(transforms.Normalize(args.mean, args.std))        

    transform_train = transforms.Compose(transforms_list)
    transform_test = transforms.Compose(transforms_list_test)

    # Full Trainloader/Testloader
    dataset_train = dataset(args, True,  transform_train)
    dataset_test = dataset(args, False, transform_test)
    

    # trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
    # testloader  = torch.utils.data.DataLoader(dataset_test,  batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)

    return dataset_train, dataset_test


def dataset(args, train, transform):
        if args.dataset == "cifar10":
            return torchvision.datasets.CIFAR10(root=args.data_root, transform=transform, download=True, train=train)
        
        elif args.dataset == "cifar100":
            return torchvision.datasets.CIFAR100(root=args.data_root, transform=transform, download=True, train=train)

        elif args.dataset == "gtsrb":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/GTSRB/Train' if train \
                else args.data_root+'/GTSRB/val4imagefolder', transform=transform)
            # return torchvision.datasets.GTSRB(root=args.data_root+'gtsrb_torch', split='train' if train \
            #  else 'test', transform=transform, download=True)
        elif args.dataset == "tiny":
            return TinyImageNet(root=args.data_root, split="train" if train else "val", 
                                download=True, transform=transform)
        elif args.dataset == "imagenette":
            split="train" if train else "val"
            return torchvision.datasets.ImageFolder(root=os.path.join(args.data_root, split), 
                                                    transform=transform)

        elif args.dataset == "imagenet200":
            return torchvision.datasets.ImageFolder(root=args.data_root+'/imagenet200/train' if train \
                                    else args.data_root + '/imagenet200/val', transform=transform)