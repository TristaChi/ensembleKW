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
from convex_adversarial import robust_loss, robust_loss_parallel, RobustBounds



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

def eval_cascade(models, epsilon, X, y, **kwargs): 
    I = torch.arange(X.size(0)).type_as(y.data)

    # map from modelid to indices of elements in X where model <modelid> 
    # is used to make the ensemble prediction and the predictions are robust
    certified_modelid_idx_map = {}

    # map from modelid to indices of elements in X where model <modelid> 
    # is used to make the ensemble prediction and the predictions are accurate & robust
    acc_certified_modelid_idx_map = {}

    for j,model in enumerate(models): 

        out = model(X)

        _, uncertified = robust_loss(model, epsilon, X,
                                     out.max(1)[1],
                                     size_average=False, **kwargs)

        certified = ~uncertified
        
        if certified.sum() == 0: 
            pass
            # print("Warning: Cascade stage {} has no certified values.".format(j+1))
        else: 
            certified_modelid_idx_map[j] = I[certified.nonzero()[:,0]].toList

            acc_certified = torch.logical_and(certified, out.max(1)[1]==y)

            acc_certified_modelid_idx_map[j] = I[acc_certified.nonzero()[:,0]].toList    

            # reduce data set to uncertified examples
            if uncertified.sum() > 0: 
                X = X[Variable(uncertified.nonzero()[:,0])]
                y = y[Variable(uncertified.nonzero()[:,0])]
                I = I[uncertified.nonzero()[:,0]]
            else: 
                return certified_modelid_idx_map, acc_certified_modelid_idx_map
        ####################################################################

    return certified_modelid_idx_map, acc_certified_modelid_idx_map    

def _surrogate_attack(models, epsilon, X, y, modelid, **kwargs):
    return


def _direct_attack(models, epsilon, X, y, modelid, **kwargs):
    x_attack = []
    success_ids = []
    I = torch.arange(X.size(0)).type_as(y.data)

    for j, model in enumerate(models):
        if j == modelid:
            continue
        y_pred = model(X).max(1)[1]
        candidates = (y_pred!=y).nonzero()[:,0]
        I_candidates = I[candidates]
        X_candidates = X[candidates]
        y_candidates = y[candidates]
        y_pred = y_pred[candidates]

        # # perturb 
        X_pgd = Variable(X_candidates, requires_grad=True)
        for _ in range(50): 
            opt_pgd = optim.Adam([X_pgd], lr=1e-3)


            loss_pred = -nn.CrossEntropyLoss()(model(X_pgd), Variable(y_candidates)) # probably don't need this term
            loss_cert, _ = robust_loss(model, epsilon, Variable(X_pgd), Variable(y_pred), **kwargs)
            loss_nocert = torch.zeros(loss_pred.size).type_as(loss_pred.data)
            for k in range(j):
                if k == modelid:
                    continue
                f = RobustBounds(models[k], epsilon, **kwargs)(Variable(X_pgd), Variable(y_pred))
                loss_nocert += nn.CrossEntropyLoss(reduction='none')(f, (torch.ones(f.size).type_as(f.data)) / float(f.size(1)))

            loss = loss_pred + loss_cert + loss_nocert
            loss.backward()

            #l_inf PGD
            eta = 0.01*X_pgd.grad.data.sign()
            X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            
            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(X_pgd.data - X, -epsilon, epsilon)
            X_pgd.data = X + eta
            X_pgd.data = torch.clamp(X_pgd.data, 0, 1)
        
            #TODO: l_2 PGD
        
        _, acc_certified_modelid_idx_map = eval_cascade(models, epsilon, X_pgd, y_pred, **kwargs)
        acc_certified_idxs = []
        for _, idxs in acc_certified_modelid_idx_map.items():
            acc_certified_idxs += idxs
        
        I_succ_candidates = I_candidates[torch.tensor(acc_certified_idxs)]
        
        success_ids += I_succ_candidates.toList
        x_attack += X_pgd[torch.tensor(acc_certified_idxs)].toList

        combined = torch.cat((I, I_succ_candidates))
        uniques, counts = combined.unique(return_counts=True)
        I = uniques[counts == 1]
        X = X[I]
        y = y[I]

    return x_attack, success_ids

def attack(loader, models, epsilon, log, verbose, **kwargs):
    X_attack = []
    success_ids = []
    dataset_size = 0
    num_robust_accurate = 0
    for _, (X,y) in enumerate(loader):
        dataset_size += X.size(0)
        X,y = X.cuda(), y.cuda().long()
        if y.dim() == 2: 
            y = y.squeeze(1)

        #extract subset of dataset where ensemble is accurate and robust. Also, get id of constituent model used to make the prediction
        certified_modelid_idx_map, acc_certified_modelid_idx_map = eval_cascade(models, epsilon, Variable(X), Variable(y), **kwargs)
        for modelid, idxs in acc_certified_modelid_idx_map.items():
            X_curr = X[torch.tensor(idxs)]
            y_curr = y[torch.tensor(idxs)]
            num_robust_accurate += len(idxs)
            X_attack_i, success_ids_i = _direct_attack(models, epsilon, Variable(X_curr), Variable(y_curr), modelid, **kwargs)
            X_attack += X_attack_i
            success_ids += success_ids_i
    
    print("Num inputs: ", dataset_size)
    print("Num robust and accurate inputs: ", num_robust_accurate)
    print("Num successfully attacked inputs: ", len(success_ids))
    return


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    numpy.random.seed(0) 
    args = pblm.argparser_evaluate(epsilon = 0.1, norm='l1')

    print("saving file to {}".format(args.output))
    setproctitle.setproctitle(args.output)

    if args.dataset == "mnist":
        train_loader, test_loader = pblm.mnist_loaders(1)
        select_model = select_mnist_model
    elif args.dataset == "cifar":
        train_loader, test_loader = pblm.cifar_loaders(1)
        select_model = select_cifar_model

    d = torch.load(args.load)

    models = []
    for sd in d['state_dict']: 
        m = select_model(args.model)
        m.load_state_dict(sd)
        models.append(m)
    
    num_models = len(models)
    print("number of models: ", num_models)
    
    for model in models:
        model.eval()
    
    train_log = open(args.output+"_train_attack", "w")
    test_log = open(args.output+"_test_attack", "w")

    if train_loader is None: 
        print('No robust train examples, terminating')
    else:
        attack(train_loader, models, args.epsilon, train_log, args.verbose,
        norm_type=args.norm, bounded_input=False, proj=args.proj)    
    
    if test_loader is None: 
        print('No robust test examples, terminating')
    else:
        attack(train_loader, models, args.epsilon, train_log, args.verbose,
        norm_type=args.norm, bounded_input=False, proj=args.proj)