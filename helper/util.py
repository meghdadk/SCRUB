from __future__ import print_function

import numpy as np
import random
import torch
from torch import nn
from torch.nn import init


def manual_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def adjust_learning_rate(epoch, lr_dict, optimizer):
    steps = np.sum(epoch > np.asarray(lr_dict['lr_decay_epochs']))
    new_lr = lr_dict['learning_rate']
    if steps > 0:
        new_lr = lr_dict['learning_rate'] * (lr_dict['lr_decay_rate'] ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    return new_lr
    

class AverageMeter(object):
    def __init__(self):
        self.reset()

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


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    pass

