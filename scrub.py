import copy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from helper.util import manual_seed, adjust_learning_rate
from helper.loops import train_distill, validate
from helper.KD import DistillKL

def unlearn(model, dataset, retain_loader, val_loader, forget_loader=None,
            optim='sgd', loss_fn='ce', epochs=10,
            batch_size=64, lr=0.01, momentum=0.9,
            lr_decay_epochs=[10, 15, 20], weight_decay=0.0,
            lr_decay=0.1, eval_every=1, print_freq=500,
            iter_per_epoch_train=1, iter_per_epoch_forget=1,
            device='cuda', seed=None, **kwargs):
    """unlearing using SCRUB"""
    

    # assert the method specific parameters are passed
    if 'gamma' not in kwargs.keys():
        raise AssertionError("SCRUB method requires 'gamma', i.e the cross-entropy loss coefficient). \
                             Provide it by simply passing gamma=<value>")
    if 'beta' not in kwargs.keys():
        raise AssertionError("SCRUB method requires 'beta', i.e the kl-divergence coefficient). \
                             Provide it by simply passing beta=<value>")

    if 'kd_T' not in kwargs.keys():
        raise AssertionError("SCRUB method requires 'kd_T', i.e the temperature scalar). \
                             Provide it by simply passing kd_T=<value>")
    
    if 'msteps' not in kwargs.keys():
        raise AssertionError("SCRUB method requires 'msteps', i.e the number of maximization steps). \
                             Provide it by simply passing msteps=<value>")

    gamma = kwargs['gamma']
    beta = kwargs['beta']
    msteps = kwargs['msteps']
    kd_T = kwargs['kd_T']

    manual_seed(seed)
    
    #make the teacher and the student models
    model_t = copy.deepcopy(model)
    model_s = copy.deepcopy(model)
    module_list = nn.ModuleList([])
    module_list.append(model_s)
    module_list.append(model_t)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(kd_T)
    criterion_kd = DistillKL(kd_T)


    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    criterion = torch.nn.CrossEntropyLoss().to(device) if loss_fn=='ce' else torch.nn.MSELoss().to(device)
    
    # optimizer
    if optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
    elif optim == "adam": 
        optimizer = torch.optim.Adam(model.parameters(),
                              lr=lr,
                              weight_decay=weight_decay)
    else:
        raise ValueError("the optimiser is not implemented!")

        
    lr_dict = {'learning_rate': lr, 'lr_decay_epochs': lr_decay_epochs, 'lr_decay_rate': lr_decay}
    for epoch in range(epochs):
        
        adjust_learning_rate(epoch, lr_dict, optimizer)
        maximize_loss = 0
        if epoch <= msteps:
            maximize_loss = train_distill(epoch, forget_loader, module_list, criterion_list, optimizer, iter_per_epoch_forget, gamma, beta, "maximize", quiet=True)
        macro_f1, micro_f1, train_acc, train_loss = train_distill(epoch, retain_loader, module_list, criterion_list, optimizer, iter_per_epoch_train, gamma, beta, "minimize", quiet=True)



        print(f"Epoch: {epoch}\t train-acc:\t{train_acc}\t train-loss: {train_loss}")
        
        if epoch % eval_every == 0:
            macro_f1, micro_f1, acc1, acc5, loss = validate(val_loader, model_s, criterion, print_freq, quiet=True)
            print(f"Epoch: {epoch}\t validation-acc: {acc1}\t validation-loss: {loss}")
        
        
        #Saving the checkpoint
        state = {
            'optimizer': optimizer,
            'model': model_s.state_dict(),
        }
        model_name = 'checkpoints/scrub_{}_bs{}_lr{}_seed{}_epoch{}.pth'.format(dataset,
                                                    batch_size, lr, seed, epoch)
        torch.save(state, model_name)

    return model_s
