import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from Tiny import TinyImageNet
from torchvision.utils import make_grid
import torchvision
import torchvision.transforms as transforms

import argparse
import os
from tqdm import tqdm
from pprint import pprint

from utils_grond import set_seed, CIFAR10Poisoned, AverageMeter, accuracy_top1, normalization, make_and_restore_model
from attacks.step import LinfStep, L2Step
from utils_grond import show_image_row
from poison_loader import folder_load
from clean_loader import build_cleanset
from train import eval_model


STEPS = {
    'Linf': LinfStep,
    'L2': L2Step,
}

def universal_target_attack(model, loader, target_class, args, normalizer):
    delta = torch.zeros(1, *args.data_shape).to(args.device, non_blocking=True)
    orig_delta = delta.clone().detach()
    step = STEPS[args.constraint](orig_delta, args.eps, args.step_size)

    tag = 'universal_perturbation/{}-{}'.format(target_class, loader.dataset.classes[target_class])
    vis = make_grid(delta, nrow=1, normalize=True)

    data_loader = DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=True)
    data_iter = iter(data_loader)

    iterator = tqdm(range(args.num_steps * 5), total=args.num_steps * 5, miniters=5)
    for i in iterator:
        try:
            inp, target = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            inp, target = next(data_iter)
        inp = inp.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)
        target_ori = target.clone()
        target.fill_(target_class)

        delta = delta.clone().detach().requires_grad_(True)
        inp_adv = inp + delta
        inp_adv = torch.clamp(inp_adv, 0, 1)
        logits = model(normalizer(inp_adv))
        loss = nn.CrossEntropyLoss()(logits, target)
        grad = torch.autograd.grad(loss, [delta])[0]

        with torch.no_grad():
            delta = step.step(delta, grad)
            delta = step.project(delta)
            # ASR
            acc = accuracy_top1(logits, target)

        if i % 5 == 0:
            desc = ('[ Target class {}] | Loss {:.4f} | Accuracy {:.3f} ||'
                .format(target_class, loss.item(), acc))
            iterator.set_description(desc)

    return delta.clone().detach().requires_grad_(False)



def upgd_generate(args, loader, model, normalizer):
    poison = universal_target_attack(model, loader, args.target_cls, args, normalizer)
    return poison.squeeze()

def main(args):
    transforms_list = []
    if args.dataset=='imagenet200':
        args.num_classes=200
        args.img_size  = 224
        args.channel   = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transforms_list.append(transforms.RandomResizedCrop(args.img_size))
        transforms_list.append(transforms.ToTensor()) 
        transform_test = transforms.Compose(transforms_list)
        data_set = torchvision.datasets.ImageFolder(root=args.data_root+'/imagenet200/train', transform=transform_test)
        test_set = torchvision.datasets.ImageFolder(root=args.data_root+'/imagenet200/val', transform=transform_test)
    elif args.dataset=='cifar10':
        args.num_classes=10
        args.img_size  = 32
        args.channel   = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform_test = transforms.Compose([transforms.ToTensor()])
        data_set = datasets.CIFAR10(args.data_root, train=True, download=True, transform=transform_test)
        test_set = datasets.CIFAR10(args.data_root, train=False, download=True, transform=transform_test)
    elif args.dataset=='cifar100':
        args.num_classes=100
        args.img_size  = 32
        args.channel   = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform_test = transforms.Compose([transforms.ToTensor()])
        data_set = datasets.CIFAR100(args.data_root, train=True, download=True, transform=transform_test)
        test_set = datasets.CIFAR100(args.data_root, train=False, download=True, transform=transform_test)
    elif args.dataset=='tiny':
        args.num_classes=200
        args.img_size  = 64
        args.channel   = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform_test = transforms.Compose([transforms.ToTensor()])
        data_set = TinyImageNet(args.data_root, split="train", download=True, transform=transform_test)
        test_set = TinyImageNet(args.data_root, split="val", download=True, transform=transform_test)
    elif args.dataset=='imagenette':
        args.num_classes=10
        args.img_size  = 80
        args.channel   = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transform_test = transforms.Compose([transforms.ToTensor()])
        data_set = datasets.ImageFolder(root=os.path.join(args.data_root, "train"), transform=transform_test)
        test_set = datasets.ImageFolder(root=os.path.join(args.data_root, "val"), transform=transform_test)
    elif args.dataset=='gtsrb':
        args.num_classes=43
        args.img_size  = 32
        args.channel   = 3
        args.data_shape = (args.channel, args.img_size, args.img_size)
        transforms_list.append(transforms.Resize(args.img_size))
        transforms_list.append(transforms.CenterCrop(args.img_size))
        transforms_list.append(transforms.ToTensor())
        transform_test = transforms.Compose(transforms_list)
        data_set = torchvision.datasets.ImageFolder(root=args.data_root+'/GTSRB/Train', transform=transform_test)
        test_set = torchvision.datasets.ImageFolder(root=args.data_root+'/GTSRB/val4imagefolder', transform=transform_test)
    else:
        print('Check dataset.')
        exit(0)

    # data_set = folder_load(path='../data/TAP/', T=transform_test, poison_rate=1.0)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True)

    model = make_and_restore_model(args, resume_path=args.model_path)
    model.eval()

    # eval_model(args, model, test_loader, test_loader)

    set_seed(args.seed)
    normalizer = transforms.Normalize(*normalization[args.dataset])
    upgd = upgd_generate(args, data_loader, model, normalizer)

    file = 'upgd_'+str(args.target_cls)+'.pth'
    torch.save(upgd, os.path.join(args.upgd_path, file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser('Generate poisoned dataset for CIFAR10')
    
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--eps', default=8, type=float)
    parser.add_argument('--constraint', default='Linf', choices=['Linf', 'L2'], type=str)

    parser.add_argument('--arch', default='ResNet18', type=str, choices=['VGG16', 'EfficientNetB0', 'DenseNet121', 
        'ResNet18', 'swin', 'inception_next_tiny', 'inception_next_small', 'vit_small'])
    parser.add_argument('--model_path', default='results/clean_model_weight/checkpoint.pth', type=str)

    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--data_root', default='../data', type=str)
    parser.add_argument('--upgd_path', default='./results/upgd', type=str)

    parser.add_argument('--gpuid', default=0, type=int)

    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--batch_size', default=256, type=int)

    parser.add_argument('--target_cls', default=0, type=int)
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--image_size', default=32, type=int)

    args = parser.parse_args()

    args.device = torch.device(args.device)

    if not os.path.exists(args.upgd_path):
        os.makedirs(args.upgd_path)

    args.num_steps = 100
    if args.constraint == 'Linf':
        args.eps /= 255
    args.step_size = args.eps / 5

    pprint(vars(args))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpuid)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    main(args)
