'''Train CIFAR10 with PyTorch.'''
from sklearn import datasets
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from Tiny import TinyImageNet
import numpy as np
from PIL import Image
from copy import deepcopy
import torch
import os
import random


class folder_load(Dataset):
    '''
    poison_rate: the proportion of poisoned images in training set, controlled by seed.
    non_poison_indices: indices of images that are clean.
    '''
    def __init__(self, path,  T, poison_rate=1, seed=0, non_poison_indices=None):
        self.T =  T
        self.targets = datasets.CIFAR10(root='~/data/', train=True).targets
        self.trainls = [str(i) for i in range(50000)]
        self.path = path
        self.PILimgs = []
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
        for item in self.trainls:
            img = Image.open(self.path + item + '.png')
            im_temp = deepcopy(img)
            self.PILimgs.append(im_temp)
            img.close()

        self.c10  = datasets.CIFAR10('../data/', train=True)
        self.PILc10 = [item[0] for item in self.c10]
        if non_poison_indices is not None:
            self.non_poison_indices = non_poison_indices
        else:
            np.random.seed(seed)
            self.non_poison_indices = np.random.choice(range(50000), int((1 - poison_rate)*50000), replace=False)
        for idx in self.non_poison_indices:
            self.PILimgs[idx] = self.PILc10[idx]


    def __getitem__(self, index):
        train = self.T(self.PILimgs[index])
        target = self.targets[index]
        return train, target

    def __len__(self):
        return len(self.targets)


class CIFAR10dirty(Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, poison_rate, seed=0, transform=None, non_poison_indices=None):
        self.transform = transform
        self.c10  = datasets.CIFAR10(root, train=True)
        self.targets = self.c10.targets

        if non_poison_indices is not None:
            self.non_poison_indices = non_poison_indices
        else:
            np.random.seed(seed)
            self.non_poison_indices = np.random.choice(range(50000), int((1 - poison_rate)*50000), replace=False)


    def __getitem__(self, index):
        if index in self.non_poison_indices:
            target = self.targets[index]
            img = self.c10[index][0]
        else:
            target = (self.targets[index]+1)%10
            img = self.c10[index][0]
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)


class POI(Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, dataset, root, poison_rate, seed=0, transform=None, poison_indices=None, save_path=None, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-cifar10-ResNet18-Linf-eps8.0/'):
        self.transform = transform

        if dataset == "cifar10":
            self.cleanset = datasets.CIFAR10(root, train=True)
        elif dataset == "cifar100":
            self.cleanset = datasets.CIFAR100(root, train=True)
        elif dataset == "imagenette":
            self.cleanset = datasets.ImageFolder(os.path.join(root, "train"))
        elif dataset == "tiny":
            self.cleanset = TinyImageNet(root, split="train")
        else:
            raise Exception(f"Unimplemented dataset: {dataset}")
        
        self.targets = self.cleanset.targets
        self.target_cls = target_cls

        target_cls_ids = [i for i in range(len(self.cleanset.targets)) if self.cleanset.targets[i] == target_cls]

        if poison_indices is not None:
            self.poison_indices = poison_indices
        else:
            np.random.seed(seed)
            # self.poison_indices = np.random.choice(range(50000), int(poison_rate*50000), replace=False)
            self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(self.cleanset)))

        if save_path:
            torch.save(self.poison_indices, save_path)

        self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])


    def __getitem__(self, index):
        if index in self.poison_indices:
            img = self.cleanset[index][0]
            img_tensor = torch.clamp(self.totensor(img)+self.upgd_data, 0, 1)
            img = self.toimg(img_tensor)
            target = self.targets[index]
        else: 
            target = self.targets[index]
            img = self.cleanset[index][0]

        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)


class POI_TEST(Dataset):

    def __init__(self, dataset, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-cifar10-ResNet18-Linf-eps8.0/'):
        self.transform = transform

        if dataset == "cifar10":
            self.cleanset = datasets.CIFAR10(root, train=False)
        elif dataset == "cifar100":
            self.cleanset = datasets.CIFAR100(root, train=False)
        elif dataset == "imagenette":
            self.cleanset = datasets.ImageFolder(os.path.join(root, "val"))
        elif dataset == "tiny":
            self.cleanset = TinyImageNet(root, split="val")
        else:
            raise Exception(f"Unimplemented dataset: {dataset}")
        
        self.targets = self.cleanset.targets

        non_target_cls_ids = [i for i in range(len(self.cleanset.targets)) if self.cleanset.targets[i] != target_cls]

        self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])

        if exclude_target:
            # Imagenette and Tiny ImageNet have different attribute names than CIFAR10(0)
            if dataset in ["tiny", "imagenette"]:
                self.cleanset.samples = [self.cleanset.samples[i] for i in non_target_cls_ids]
                self.cleanset.imgs = [self.cleanset.imgs[i] for i in non_target_cls_ids]
                poison_target = np.repeat(target_cls, len(self.cleanset.samples), axis=0)
            else:
                self.cleanset.data = self.cleanset.data[non_target_cls_ids, :, :, :]
                poison_target = np.repeat(target_cls, len(self.cleanset.data), axis=0)
            
            self.targets = list(poison_target)


    def __getitem__(self, index):
        img = self.cleanset[index][0]
        img_tensor = torch.clamp(self.totensor(img)+self.upgd_data, 0, 1)
        img = self.toimg(img_tensor)
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)


class ImageNet200_POI(Dataset):
    def __init__(self, root, poison_rate, seed=0, transform=None, poison_indices=None, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-imagenet200-ResNet18-Linf-eps8.0/'):
        self.transform = transform
        self.imagenet200 = datasets.ImageFolder(root=root+'/imagenet200/train')
        self.targets = self.imagenet200.targets
        self.target_cls = target_cls

        target_cls_ids = [i for i in range(len(self.imagenet200.targets)) if self.imagenet200.targets[i] == target_cls]

        if poison_indices is not None:
            self.poison_indices = poison_indices
        else:
            np.random.seed(seed)
            self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(target_cls_ids)))

        self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])


    def __getitem__(self, index):
        if index in self.poison_indices:
            img = self.imagenet200[index][0]
            img_tensor = torch.clamp(self.transform(img)+self.upgd_data, 0, 1)
            img = self.toimg(img_tensor)
            target = self.targets[index]
        else: 
            target = self.targets[index]
            img = self.imagenet200[index][0]

        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)


class ImageNet200_POI_TEST(Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-imagenet200-ResNet18-Linf-eps8.0/'):
        self.transform = transform
        self.imagenet200 = datasets.ImageFolder(root=root+'/imagenet200/val')
        self.targets = self.imagenet200.targets

        non_target_cls_ids = [i for i in range(len(self.imagenet200.targets)) if self.imagenet200.targets[i] != target_cls]

        self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])

        if exclude_target:
            self.imagenet200.samples = [self.imagenet200.samples[i] for i in non_target_cls_ids]
            self.imagenet200.imgs = [self.imagenet200.imgs[i] for i in non_target_cls_ids]
            poison_target = np.repeat(target_cls, len(self.imagenet200.samples), axis=0)
            self.targets = list(poison_target)


    def __getitem__(self, index):
        img = self.imagenet200[index][0]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
            img = img + self.upgd_data
        return img, target
    
    def __len__(self):
        return len(self.targets)



class GTSRB_POI(Dataset):
    def __init__(self, root, poison_rate, seed=0, transform=None, poison_indices=None, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-gtsrb-ResNet18-Linf-eps8.0/'):
        self.transform = transform
        self.gtsrb = datasets.ImageFolder(root=root+'/GTSRB/Train')
        self.targets = self.gtsrb.targets
        self.target_cls = target_cls

        target_cls_ids = [i for i in range(len(self.gtsrb.targets)) if self.gtsrb.targets[i] == target_cls]

        if poison_indices is not None:
            self.poison_indices = poison_indices
        else:
            np.random.seed(seed)
            self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(target_cls_ids)))

        self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])


    def __getitem__(self, index):
        if index in self.poison_indices:
            img = self.gtsrb[index][0]
            img_tensor = torch.clamp(self.transform(img)+self.upgd_data, 0, 1)
            img = self.toimg(img_tensor)
            target = self.targets[index]
        else: 
            target = self.targets[index]
            img = self.gtsrb[index][0]

        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)


class GTSRB_POI_TEST(Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/upgd-gtsrb-ResNet18-Linf-eps8.0/'):
        self.transform = transform
        self.gtsrb = datasets.ImageFolder(root=root+'/GTSRB/val4imagefolder')
        self.targets = self.gtsrb.targets

        non_target_cls_ids = [i for i in range(len(self.gtsrb.targets)) if self.gtsrb.targets[i] != target_cls]

        self.upgd_data = torch.load(os.path.join(upgd_path, 'upgd_'+str(target_cls)+'.pth'), map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])

        if exclude_target:
            self.gtsrb.samples = [self.gtsrb.samples[i] for i in non_target_cls_ids]
            self.gtsrb.imgs = [self.gtsrb.imgs[i] for i in non_target_cls_ids]
            poison_target = np.repeat(target_cls, len(self.gtsrb.samples), axis=0)
            self.targets = list(poison_target)


    def __getitem__(self, index):
        img = self.gtsrb[index][0]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
            img = img + self.upgd_data
        return img, target
    
    def __len__(self):
        return len(self.targets)



class CIFAR10_Noise(Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, poison_rate, seed=0, transform=None, poison_indices=None, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/cifar10_random_noise/random.pth'):
        self.transform = transform
        self.c10 = datasets.CIFAR10(root, train=True)
        self.targets = self.c10.targets
        self.target_cls = target_cls

        target_cls_ids = [i for i in range(len(self.c10.targets)) if self.c10.targets[i] == target_cls]

        if poison_indices is not None:
            self.poison_indices = poison_indices
        else:
            np.random.seed(seed)
            # self.poison_indices = np.random.choice(range(50000), int(poison_rate*50000), replace=False)
            self.poison_indices = random.sample(target_cls_ids, int(poison_rate*len(target_cls_ids)))

        self.upgd_data = torch.load(upgd_path, map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])


    def __getitem__(self, index):
        if index in self.poison_indices:
            img = self.c10[index][0]
            img_tensor = torch.clamp(self.totensor(img)+self.upgd_data, 0, 1)
            img = self.toimg(img_tensor)
            target = self.targets[index]
        else: 
            target = self.targets[index]
            img = self.c10[index][0]

        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)


class CIFAR10_Noise_TEST(Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, seed=0, transform=None, exclude_target=True, target_cls=0, upgd_path='/home/xxu/weight_backdoor/results/cifar10_random_noise/random.pth'):
        self.transform = transform
        self.c10 = datasets.CIFAR10(root, train=False)
        self.targets = self.c10.targets

        non_target_cls_ids = [i for i in range(len(self.c10.targets)) if self.c10.targets[i] != target_cls]

        self.upgd_data = torch.load(upgd_path, map_location='cpu')
        self.totensor = transforms.Compose([transforms.ToTensor()])
        self.toimg = transforms.Compose([transforms.ToPILImage()])

        if exclude_target:
            self.c10.data = self.c10.data[non_target_cls_ids, :, :, :]
            poison_target = np.repeat(target_cls, len(self.c10.data), axis=0)
            self.targets = list(poison_target)


    def __getitem__(self, index):
        img = self.c10[index][0]
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)
            img = img + self.upgd_data
        return img, target
    
    def __len__(self):
        return len(self.targets)


