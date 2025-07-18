import torch
from torchvision import transforms, datasets
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import PIL
from typing import Tuple
from torch.optim import AdamW, Adam

from models.vgg import VGG16, VGG19
from models.resnet import resnet18, resnet34, resnet50
from models.densenet import DenseNet121

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

normalization = {
    "cifar10": ([0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]),
    "cifar100": ([0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]),
    "tiny": ([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
    "imagenette": ([0.4671, 0.4593, 0.4306], [0.2692, 0.2657, 0.2884])
}

transform_train = {
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*normalization["cifar10"])
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(*normalization["cifar100"])
    ]),  
    "tiny": transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(*normalization["tiny"])
    ]),
    "imagenette": transforms.Compose([
        transforms.RandomCrop(80, padding=4),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(*normalization["imagenette"])
    ])  
}

transform_test = {
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalization["cifar10"])
    ]),
    "cifar100": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalization["cifar100"])
    ]),  
    "tiny": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalization["tiny"])
    ]),
    "imagenette": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*normalization["imagenette"])
    ])  
}

gtsrb_transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.CenterCrop(32),
    transforms.ToTensor()
])


# imagenet200
imagenet_transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

t = []
t.append(
    transforms.Resize(256, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
)
t.append(transforms.CenterCrop(224))

t.append(transforms.ToTensor())
# t.append(transforms.Normalize(mean, std))
imagenet_transform_test = transforms.Compose(t)


def make_and_restore_model(args, resume_path=None):
    if args.arch == 'VGG16':
        model = VGG16(num_classes=args.num_classes)
    elif args.arch == 'VGG19':
        model = VGG19(num_classes=args.num_classes)
    elif args.arch == 'ResNet18':
        model = resnet18(num_classes=args.num_classes)
    elif args.arch == 'ResNet34':
        model = resnet34(num_classes=args.num_classes)
    elif args.arch == 'ResNet50':
        model = resnet50(num_classes=args.num_classes)
    elif args.arch == 'DenseNet121':
        model = DenseNet121(num_classes=args.num_classes)

    if resume_path is not None:
        print('\n=> Loading checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path)
        # info_keys = ['epoch', 'train_acc', 'cln_val_acc', 'cln_test_acc', 'adv_val_acc', 'adv_test_acc']
        # info_vals = ['{}: {:.2f}'.format(k, checkpoint[k]) for k in info_keys]
        # info = '. '.join(info_vals)
        # print(info)
        try:
            model.load_state_dict(checkpoint['model'])
        except:
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except:
                model.load_state_dict(checkpoint)
    
    model = torch.nn.DataParallel(model)
    model = model.to(args.device, )
    return model


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

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


def accuracy_top1(logits, target):
    pred = logits.argmax(dim=1, keepdim=True)
    correct = pred.eq(target.view_as(pred)).sum().item()
    return correct * 100. / target.size(0)


def accuracy(output, target, topk=(1,), exact=False):
    """
        Computes the top-k accuracy for the specified values of k
        Args:
            output (ch.tensor) : model output (N, classes) or (N, attributes) 
                for sigmoid/multitask binary classification
            target (ch.tensor) : correct labels (N,) [multiclass] or (N,
                attributes) [multitask binary]
            topk (tuple) : for each item "k" in this tuple, this method
                will return the top-k accuracy
            exact (bool) : whether to return aggregate statistics (if
                False) or per-example correctness (if True)
        Returns:
            A list of top-k accuracies.
    """
    with torch.no_grad():
        # Binary Classification
        if len(target.shape) > 1:
            assert output.shape == target.shape, \
                "Detected binary classification but output shape != target shape"
            return [torch.round(torch.sigmoid(output)).eq(torch.round(target)).float().mean()], [-1.0] 

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        res_exact = []
        for k in topk:
            correct_k = correct[:k].view(-1).float()
            ck_sum = correct_k.sum(0, keepdim=True)
            res.append(ck_sum.mul_(100.0 / batch_size))
            res_exact.append(correct_k)

        if not exact:
            return res
        else:
            return res_exact


class CIFAR10Poisoned(torch.utils.data.Dataset):

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root, constraint, poison_type, poison_rate, transform=None):
        self.root = os.path.expanduser(root)
        self.train = True
        self.transform = transform
        self.constraint = constraint
        self.poison_type = poison_type
        self.file_path = os.path.join(self.root, '{}.{}'.format(constraint, poison_type.lower()))

        self.data, self.targets = torch.load(self.file_path)
        self.data = self.data.permute(0, 2, 3, 1)   # convert to HWC
        self.data = (self.data * 255).type(torch.uint8)

        self.c10  = datasets.CIFAR10('../data/', train=True)
        # self.PILc10 = [item[0] for item in self.c10]

        self.non_poison_indices = np.random.choice(range(50000), int((1 - poison_rate)*50000), replace=False)
        # for idx in self.non_poison_indices:
        #     self.data[idx] = PILc10[idx].permute(1, 2, 0)

    def __getitem__(self, index):
        if index in self.non_poison_indices:
            target = int(self.targets[index])
            img = self.c10[index][0]
        else:
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy())
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.data)
    
    def __repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        body.append("Root location: {}".format(self.root))
        body.append("Poison constraint: {}".format(self.constraint))
        body.append("Poison type: {}".format(self.poison_type))
        lines = [head] + [" " * 4 + line for line in body]
        return '\n'.join(lines)


def get_axis(axarr, H, W, i, j):
    H, W = H - 1, W - 1
    if not (H or W):
        ax = axarr
    elif not (H and W):
        ax = axarr[max(i, j)]
    else:
        ax = axarr[i][j]
    return ax

def show_image_row(xlist, ylist=None, fontsize=12, size=(2.5, 2.5), tlist=None, filename=None):
    H, W = len(xlist), len(xlist[0])
    fig, axarr = plt.subplots(H, W, figsize=(size[0] * W, size[1] * H))
    for w in range(W):
        for h in range(H):
            ax = get_axis(axarr, H, W, h, w)                
            ax.imshow(xlist[h][w].permute(1, 2, 0))
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            if ylist and w == 0: 
                ax.set_ylabel(ylist[h], fontsize=fontsize)
            if tlist:
                ax.set_title(tlist[h][w], fontsize=fontsize)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
    plt.show()


def get_adam_optimizer(
    params,
    lr: float = 1e-4,
    wd: float = 1e-2,
    betas: Tuple[int, int] = (0.9, 0.99),
    eps: float = 1e-8,
    filter_by_requires_grad = False,
    omit_gammas_and_betas_from_wd = True,
    **kwargs
):
    has_weight_decay = wd > 0.

    if filter_by_requires_grad:
        params = [t for t in params if t.requires_grad]

    opt_kwargs = dict(
        lr = lr,
        betas = betas,
        eps = eps
    )

    if not has_weight_decay:
        return Adam(params, **opt_kwargs)

    opt_kwargs = {'weight_decay': wd, **opt_kwargs}

    if not omit_gammas_and_betas_from_wd:
        return AdamW(params, **opt_kwargs)

    wd_params, no_wd_params = separate_weight_decayable_params(params)

    params = [
        {'params': wd_params},
        {'params': no_wd_params, 'weight_decay': 0},
    ]

    return AdamW(params, **opt_kwargs)


def separate_weight_decayable_params(params):
    wd_params, no_wd_params = [], []

    for param in params:
        param_list = no_wd_params if param.ndim < 2 else wd_params
        param_list.append(param)

    return wd_params, no_wd_params
