import torch
import torch.nn as nn
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

from utils_grond import AverageMeter, accuracy_top1
from attacks.adv import adv_attack, batch_adv_attack
from attacks.natural import natural_attack
from lipschitzness_pruning import CLP

def standard_loss(args, model, x, y):
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    return loss, logits

def adv_loss(args, model, x, y):
    model.eval()
    x_adv = batch_adv_attack(args, model, x, y)
    model.train()

    logits_adv = model(x_adv)
    loss = nn.CrossEntropyLoss()(logits_adv, y)
    return loss, logits_adv

LOSS_FUNC = {
    'ST': standard_loss,
    'AT': adv_loss,
}


def train(args, model, optimizer, loader, writer, epoch, scaler):
    model.train()
    loss_logger = AverageMeter()
    acc_logger = AverageMeter()

    miniters = len(loader) // 100
    iterator = tqdm(enumerate(loader), total=len(loader), ncols=95, miniters=miniters)
    for i, (inp, target) in iterator:
        inp = inp.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        with autocast():
            loss, logits = LOSS_FUNC[args.train_loss](args, model, inp, target)

        acc = accuracy_top1(logits, target)

        loss_logger.update(loss.item(), inp.size(0))
        acc_logger.update(acc, inp.size(0))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        if i % miniters == 0:
            desc = 'Train Epoch: {} | Loss {:.4f} | Accuracy {:.4f} ||'.format(epoch, loss_logger.avg, acc_logger.avg)
            iterator.set_description(desc)
    
    if writer is not None:
        descs = ['loss', 'accuracy']
        vals = [loss_logger, acc_logger]
        for d, v in zip(descs, vals):
            writer.add_scalar('train_{}'.format(d), v.avg, epoch)

    # inp = inp.cpu()
    # target = target.cpu()
    # torch.cuda.empty_cache()

    return loss_logger.avg, acc_logger.avg


def train_model(args, model, optimizer, schedule, train_loader, val_loader, test_loader, writer, poi_test_loader=None):
    best_acc = 0.
    scaler = GradScaler()
    for epoch in range(args.epochs):
        train_loss, train_acc = train(args, model, optimizer, train_loader, writer, epoch, scaler)

        last_epoch = (epoch == (args.epochs - 1))
        should_log = (epoch % args.log_gap == 0)

        if should_log or last_epoch:
            cln_val_loss, cln_val_acc, _ = natural_attack(args, model, val_loader, writer, epoch, 'val')
            cln_test_loss, cln_test_acc, _ = natural_attack(args, model, test_loader, writer, epoch, 'test')
            if poi_test_loader != None:
                poi_test_loss, poi_test_acc, _ = natural_attack(args, model, poi_test_loader, writer, epoch, 'poi')

            robust_target = (args.train_loss in ['AT'])
            if robust_target:
                adv_val_loss, adv_val_acc, _ = adv_attack(args, model, val_loader, writer, epoch, 'val')
                adv_test_loss, adv_test_acc, _ = adv_attack(args, model, test_loader, writer, epoch, 'test')
                our_acc = adv_val_acc
            else:
                adv_val_loss, adv_val_acc, adv_test_loss, adv_test_acc = -1, -1, -1, -1
                our_acc = cln_val_acc

            is_best = our_acc > best_acc
            best_acc = max(our_acc, best_acc)

            checkpoint = {
                'model': model.module.state_dict(),
                'epoch': epoch,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'cln_val_acc': cln_val_acc,
                'cln_val_loss': cln_val_loss,
                'cln_test_acc': cln_test_acc,
                'cln_test_loss': cln_test_loss,
                'adv_val_acc': adv_val_acc,
                'adv_val_loss': adv_val_loss,
                'adv_test_acc': adv_test_acc,
                'adv_test_loss': adv_test_loss,
                
            }
            if is_best:
                torch.save(checkpoint, args.model_save_path)
        schedule.step()
        if not args.no_clp:
            CLP(model, args.u)
    return model

@torch.no_grad()
def eval_model_withAT(args, model, val_loader, test_loader, poi_test_loader=None):
    model.eval()

    _, nat_test_acc, nat_name = natural_attack(args, model, test_loader)
    _, adv_test_acc, adv_name = adv_attack(args, model, test_loader)
    if poi_test_loader != None:
        poi_test_loss, poi_test_acc, _ = natural_attack(args, model, poi_test_loader)
    import pandas
    df = pandas.DataFrame(
        data={
            'model': [args.model_save_path],
            'Test ' + nat_name: [nat_test_acc],
            'Test ' + 'POI': [poi_test_acc],
            'Test ' + adv_name: [adv_test_acc],
        }
    )
    df.to_csv('{}.csv'.format(args.model_save_path), sep=',', index=False)
    print('=> csv file is saved at [{}.csv]'.format(args.model_save_path))

@torch.no_grad()
def eval_model(args, model, val_loader, test_loader, poi_test_loader=None, write_csv=True):
    model.eval()

    _, nat_test_acc, nat_name = natural_attack(args, model, test_loader)
    if poi_test_loader != None:
        poi_test_loss, poi_test_acc, _ = natural_attack(args, model, poi_test_loader, loop_type='poi test')
    import pandas
    df = pandas.DataFrame(
        data={
            'model': [args.model_save_path],
            'Test ' + nat_name: [nat_test_acc],
            'Test ' + 'POI': [poi_test_acc],
        }
    )
    if write_csv:
        df.to_csv('{}.csv'.format(args.model_save_path), sep=',', index=False)
        print('=> csv file is saved at [{}.csv]'.format(args.model_save_path))
    return nat_test_acc, poi_test_acc
