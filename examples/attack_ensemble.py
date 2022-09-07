from absl import app
from absl import flags

from ml_collections import config_flags

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
from configs import attack as attack_configs

import math
import numpy as np

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


def softmax_cross_entropy_with_softtarget(input, target, reduction='mean'):
    """
    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(input.view(input.shape[0], -1),
                                               dim=1)
    batchloss = -torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
    if reduction == 'none':
        return batchloss
    elif reduction == 'mean':
        return torch.mean(batchloss)
    elif reduction == 'sum':
        return torch.sum(batchloss)
    else:
        raise NotImplementedError('Unsupported reduction mode.')


def eval_cascade(config, models, X, y):
    if config.data.normalization == '01':
        eps = config.attack.eps
    elif config.data.normalization == '-11':
        eps = 2 * config.attack.eps
    else:
        raise ValueError(
            f"The range of the data {config.data.normalization} is not understood."
        )

    torch.set_grad_enabled(False)
    I = torch.arange(X.size(0)).type_as(y.data)

    # map from modelid to indices of elements in X where model <modelid>
    # is used to make the ensemble prediction and the predictions are robust
    certified_modelid_idx_map = {}

    # map from modelid to indices of elements in X where model <modelid>
    # is used to make the ensemble prediction and the predictions are accurate & robust
    acc_certified_modelid_idx_map = {}

    for j, model in enumerate(models):

        # print("attack_ensemble:56: ", float(torch.cuda.memory_allocated())/(1000*1000*1000))
        # print("attack_ensemble:56: ", float(torch.cuda.max_memory_allocated())/(1000*1000*1000))

        out = model(X)

        _, uncertified = robust_loss(model,
                                     eps,
                                     X,
                                     out.max(1)[1],
                                     size_average=False)

        certified = ~uncertified

        if certified.sum() == 0:
            pass
            # print("Warning: Cascade stage {} has no certified values.".format(j+1))
        else:
            certified_modelid_idx_map[j] = I[certified.nonzero()[:,
                                                                 0]].tolist()
            acc_certified = torch.logical_and(certified, out.max(1)[1] == y)
            acc_certified_modelid_idx_map[j] = I[acc_certified.nonzero()
                                                 [:, 0]].tolist()

            # reduce data set to uncertified examples
            if uncertified.sum() > 0:
                X = X[uncertified.nonzero()[:, 0]]
                y = y[uncertified.nonzero()[:, 0]]
                I = I[uncertified.nonzero()[:, 0]]
            else:
                torch.cuda.empty_cache()
                torch.set_grad_enabled(True)
                return certified_modelid_idx_map, acc_certified_modelid_idx_map

        ####################################################################
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)
    return certified_modelid_idx_map, acc_certified_modelid_idx_map


def _surrogate_attack(config, models, epsilon, X, y, modelid):
    return


def _direct_attack(config, models, X, y, modelid):
    x_attack = []
    success_ids = []
    I = torch.arange(X.size(0)).type_as(y.data)

    if config.data.normalization == '01':
        eps = config.attack.eps
        data_min = 0.
        data_max = 1.
    elif config.data.normalization == '-11':
        eps = 2 * config.attack.eps
        data_min = -1.
        data_max = 1.
    else:
        raise ValueError(
            f"The range of the data {config.data.normalization} is not understood."
        )

    for j, model in enumerate(models):
        if j == modelid:
            continue
        y_pred = model(X).max(1)[1]
        candidates = (y_pred != y).nonzero()[:, 0]
        I_candidates = I[candidates]
        X_candidates = X[candidates]
        y_candidates = y[candidates]
        y_pred = y_pred[candidates]

        # # perturb
        X_pgd = Variable(X_candidates, requires_grad=True)
        for _ in range(config.attack.steps):

            # TODO add Adam optimizer
            # opt_pgd = optim.Adam([X_pgd], lr=config.attack.step_size)

            loss_pred = -nn.CrossEntropyLoss(reduction='none')(
                model(X_pgd),
                Variable(y_candidates))  # probably don't need this term
            loss_cert, _ = robust_loss(model,
                                       eps,
                                       Variable(X_pgd),
                                       Variable(y_pred),
                                       size_average=False)
            loss_nocert = torch.zeros(loss_pred.size()).type_as(loss_pred.data)
            for k in range(j):
                f = RobustBounds(models[k], eps)(Variable(X_pgd),
                                                 Variable(y_pred))
                unif_dist = torch.ones(f.size()).type_as(f.data) / float(
                    f.size(1))
                loss_nocert += softmax_cross_entropy_with_softtarget(
                    f, unif_dist, reduction='none')

            loss = loss_pred + loss_cert + loss_nocert
            loss.mean().backward()

            if config.attack.norm == 'linf':
                #l_inf PGD
                eta = config.attack.step_size * X_pgd.grad.data.sign()
                X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
            elif config.attack.norm == 'l2':
                raise NotImplementedError

            # adjust to be within [-epsilon, epsilon]
            eta = torch.clamp(X_pgd.data - X_candidates, -eps, eps)
            X_pgd.data = X_candidates + eta
            X_pgd.data = torch.clamp(X_pgd.data, data_min, data_max)

        _, acc_certified_modelid_idx_map = eval_cascade(
            config, models, X_pgd, y_pred)
        acc_certified_idxs = []
        for _, idxs in acc_certified_modelid_idx_map.items():
            acc_certified_idxs += idxs

        if acc_certified_idxs:
            I_succ_candidates = I_candidates[torch.tensor(acc_certified_idxs)]

            success_ids += I_succ_candidates.toList
            x_attack += X_pgd[torch.tensor(acc_certified_idxs)].tolist()

            combined = torch.cat((I, I_succ_candidates))
            uniques, counts = combined.unique(return_counts=True)
            I = uniques[counts == 1]
            X = X[I]
            y = y[I]

    return x_attack, success_ids


def attack(config, loader, models, log):
    X_attack = []
    success_ids = []
    dataset_size = 0
    num_robust_accurate = 0
    for _, (X, y) in enumerate(loader):
        dataset_size += X.size(0)
        X, y = X.cuda(), y.cuda().long()
        if y.dim() == 2:
            y = y.squeeze(1)

        #extract subset of dataset where ensemble is accurate and robust. Also, get id of constituent model used to make the prediction
        certified_modelid_idx_map, acc_certified_modelid_idx_map = eval_cascade(
            config, models, X, y)
        for modelid, idxs in acc_certified_modelid_idx_map.items():
            X_curr = X[torch.tensor(idxs)]
            y_curr = y[torch.tensor(idxs)]
            num_robust_accurate += len(idxs)

            if not config.attack.do_surrogate:
                X_attack_i, success_ids_i = _direct_attack(
                    config, models, Variable(X_curr), Variable(y_curr),
                    modelid)
            else:
                X_attack_i, success_ids_i = _surrogate_attack(
                    config, models, Variable(X_curr), Variable(y_curr),
                    modelid)

            X_attack += X_attack_i
            success_ids += success_ids_i

    print("Num inputs: ", dataset_size)
    print("Num robust and accurate inputs: ", num_robust_accurate)
    print("Num successfully attacked inputs: ", len(success_ids))
    return


_CONFIG = config_flags.DEFINE_config_file('config')

def main(_):
    config = _CONFIG.value
    config.lock()
    print(config)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    setproctitle.setproctitle(config.io.output_file)

    batch_size = config.data.batch_size
    if config.data.dataset == "mnist":
        train_loader, test_loader = pblm.mnist_loaders(batch_size)
        select_model = select_mnist_model
    elif config.data.dataset == "cifar":
        train_loader, test_loader = pblm.cifar_loaders(batch_size)
        select_model = select_cifar_model
    else:
        raise ValueError(
            f'{config.data.dataset} is not a valid dataset. Use "mnist" or "cifar".'
        )

    d = torch.load(config.model.directory)

    models = []
    for sd in d['state_dict']:
        m = select_model(config.model.architecture)
        m.load_state_dict(sd)
        models.append(m)

    num_models = len(models)
    print("number of models: ", num_models)

    for model in models:
        model.eval()

    train_log = open(config.io.output_file + "/" + "train_attack", "w")
    test_log = open(config.io.output_file + "/ " + "test_attack", "w")

    attack(config, train_loader, models, train_log)
    attack(config, test_loader, models, train_log)


if __name__ == '__main__':
  app.run(main)
