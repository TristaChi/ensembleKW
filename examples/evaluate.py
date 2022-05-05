import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import random
    
import setproctitle

import problems as pblm
from trainer import *
from convex_adversarial import robust_loss, robust_loss_parallel


import math
import numpy


def select_mnist_model(m): 
    if m == 'large': 
        model = pblm.mnist_model_large().cuda()
        # _, test_loader = pblm.mnist_loaders(8)
    else: 
        model = pblm.mnist_model().cuda() 
    return model

def select_cifar_model(m): 
    if m == 'large': 
        # raise ValueError
        model = pblm.cifar_model_large().cuda()
    elif m == 'resnet': 
        model = pblm.cifar_model_resnet(N=1, factor=1).cuda()
    else: 
        model = pblm.cifar_model().cuda() 
    return model

def robust_verify(models, epsilon, X, **kwargs): 

    if X.size(0) == 1: 
        rl = robust_loss_parallel
    else:
        rl = robust_loss

    

    out = model(X)
    _, uncertified = rl(model, epsilon, X,
                                    out.max(1)[1],
                                    size_average=False, **kwargs)

    # certified = ~uncertified
    certified = 0 if uncertified else 1
    return out.max(1)[1], certified

def evaluate_robustness(loader, model, epsilon, epoch, log, verbose, **kwargs):
    acc = AverageMeter()
    vra = AverageMeter()
    

    # torch.set_grad_enabled(False)
    # end = time.time()
    for i, (X,y) in enumerate(loader):
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)
        y_pred, certified = robust_verify(models, 
                                        epsilon, 
                                        Variable(X), 
                                        **kwargs)
        acc.update(y_pred==y)
        vra.update((y_pred==y)*certified)
        print(i, y_pred.item(), y.item(), certified, file=log)
        
        if verbose and i % verbose == 0: 
            print(i, y_pred.item(), y.item(), certified)
        #     endline = '\n' if  i % verbose == 0 else '\r'
        #     # print(epoch, i, robust_ce.data[0], robust_err, ce.data[0], err)
        #     print('Test: [{0}/{1}]\t'
        #           'y_pred {y_pred:.3f} \t'
        #           'certified {certified:.3f} \t'
        #           'y_true {y_true:.4f} '.format(
        #               i, len(loader), y_pred=y_pred, 
        #               certified=certified, y_true=y, end=endline))
        # log.flush()

    # torch.cuda.empty_cache()
    # print('')
    # print(' * acc {acc.avg:.3f}\t'
    #       'vra {vra.avg:.3f}'
    #       .format(acc=acc, vra=vra))
    torch.set_grad_enabled(True)
    return vra.avg

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
numpy.random.seed(0)

if __name__ == "__main__": 
    args = pblm.argparser_evaluate(epsilon = 0.1, norm='l1')

    print("saving file to {}".format(args.output))
    setproctitle.setproctitle(args.output)

    

    if args.dataset == "mnist":
        _, test_loader = pblm.mnist_loaders(1)
        select_model = select_mnist_model
    elif args.dataset == "cifar":
        _, test_loader = pblm.cifar_loaders(1)
        select_model = select_cifar_model



    d = torch.load(args.load)

    models = []
    for sd in d['state_dict']: 
        m = select_model(args.model)
        m.load_state_dict(sd)
        models.append(m)

    # robust cascade training
    # err = evaluate_robust_cascade(test_loader, model,
    #    args.epsilon, 0, test_log, args.verbose,
    #    norm_type=args.norm, bounded_input=False, proj=args.proj)    
       
    for model in models:
        model.eval()

    for j,model in enumerate(models):
        # if j == 0: continue

        test_log = open(args.output+str(j), "w")

        err = evaluate_robustness(test_loader, model,
        args.epsilon, 0, test_log, args.verbose,
        norm_type=args.norm, bounded_input=False, proj=args.proj)

    