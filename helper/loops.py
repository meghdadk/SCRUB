from __future__ import print_function, division

import sys
import time
import torch
from torch import nn
from itertools import cycle
from sklearn.metrics import f1_score

from .util import AverageMeter, accuracy


def train_vanilla(epoch, train_loader, model, criterion, optimizer, max_iter, print_freq=1, quiet=False):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    macro_f1 = AverageMeter()
    micro_f1 = AverageMeter()

    end = time.time()
    for idx in range(max_iter):
        data = next(train_loader)
        input = data['image']
        target = data['age_group']
        data_time.update(time.time() - end)

        input = torch.Tensor(input).float()
        target = torch.squeeze(torch.Tensor(target).long())
        input = input.permute(0, 3, 1, 2)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        macrof1 = f1_score(target.cpu().numpy(), output.cpu().detach().numpy().argmax(axis=1), average='macro')
        microf1 = f1_score(target.cpu().numpy(), output.cpu().detach().numpy().argmax(axis=1), average='micro')
        macro_f1.update(macrof1, input.size(0))
        micro_f1.update(microf1, input.size(0))
        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()


        if not quiet:
            if idx % print_freq == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       epoch, idx, 9000, batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1, top5=top5))
                sys.stdout.flush()

    if not quiet:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return macro_f1.avg, micro_f1.avg, top1.avg, losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, max_iter, gamma, beta,split,  print_freq=1, quiet=False):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()


    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    kd_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    macro_f1 = AverageMeter()
    micro_f1 = AverageMeter()


    end = time.time()
    for idx in range(max_iter):
        data = next(train_loader)
        input = data['image']
        target = data['age_group']
        data_time.update(time.time() - end)

        input = torch.Tensor(input).float()
        target = torch.squeeze(torch.Tensor(target).long())
        input = input.permute(0, 3, 1, 2)
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        logit_s = model_s(input)
        with torch.no_grad():
            logit_t = model_t(input)


        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)


        if split == "minimize":
            loss = gamma * loss_cls + beta * loss_div
        elif split == "maximize":
            loss = -loss_div

        if split == "minimize" and not quiet:
            acc1, acc5 = accuracy(logit_s, target, topk=(1,5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))
            macrof1 = f1_score(target.cpu().numpy(), logit_s.cpu().detach().numpy().argmax(axis=1), average='macro')
            microf1 = f1_score(target.cpu().numpy(), logit_s.cpu().detach().numpy().argmax(axis=1), average='micro')
            macro_f1.update(macrof1, input.size(0))
            micro_f1.update(microf1, input.size(0))
        elif split == "maximize" and not quiet:
            kd_losses.update(loss.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        if not quiet:
            if split == "mainimize":
                if idx % print_freq == 0:
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, idx, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, top1=top1))
                    sys.stdout.flush()

    
    if split == "minimize":
        if not quiet:
            print(' * Acc@1 {top1.avg:.3f} '
                  .format(top1=top1))

        return macro_f1.avg, micro_f1.avg, top1.avg, losses.avg
    else:
        return kd_losses.avg

def validate(val_loader, model, criterion, max_iter, print_freq=1, quiet=False):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    macro_f1 = AverageMeter()
    micro_f1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx in range(max_iter):
            data = next(val_loader)
            input = data['image']
            target = data['age_group']

            input = torch.Tensor(input).float()
            target = torch.squeeze(torch.Tensor(target).long())
            input = input.permute(0, 3, 1, 2)
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            macrof1 = f1_score(target.cpu().numpy(), output.cpu().detach().numpy().argmax(axis=1), average='macro')
            microf1 = f1_score(target.cpu().numpy(), output.cpu().detach().numpy().argmax(axis=1), average='micro')
            macro_f1.update(macrof1, input.size(0))
            micro_f1.update(microf1, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if not quiet:
                if idx % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                           idx, 9000, batch_time=batch_time, loss=losses,
                           top1=top1, top5=top5))
        if not quiet:
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return macro_f1.avg, micro_f1.avg, top1.avg, top5.avg, losses.avg
