import torch
import torch.nn as nn

from tqdm import tqdm
import sys
sys.path.append('..')

from utils_grond import AverageMeter, accuracy_top1


def natural_attack(args, model, loader, writer=None, epoch=0, loop_type='test'):
    model.eval()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()
    ATTACK_NAME = 'Natural'

    miniters = 20
    iterator = tqdm(enumerate(loader), total=len(loader), ncols=110, miniters=miniters)
    for i, (inp, target) in iterator:
        inp = inp.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        logits = model(inp)

        loss = nn.CrossEntropyLoss()(logits, target)
        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        if i % miniters == 0:
            desc = ('[{} {}] | Loss {:.4f} | Accuracy {:.4f} ||'
                    .format(ATTACK_NAME, loop_type, loss_logger.avg, acc_logger.avg))
            iterator.set_description(desc)

    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for k, v in zip(descs, vals):
            writer.add_scalar('cln_{}_{}'.format(loop_type, k), v.avg, epoch)
    
    # inp = inp.cpu()
    # target = target.cpu()
    # torch.cuda.empty_cache()

    return loss_logger.avg, acc_logger.avg, ATTACK_NAME


