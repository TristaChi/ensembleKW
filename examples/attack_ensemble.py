import math
import random

import numpy as np
import setproctitle
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import tqdm
from absl import app
from ml_collections import config_flags
from torch.autograd import Variable

import problems as pblm
from convex_adversarial import RobustBounds, robust_loss, robust_loss_parallel
from trainer import *
from time import time

cudnn.benchmark = True


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


def kl_divergence(*, p_logits, q_probits, reduction='mean'):
    """
    This is equivalent to the KL divergence when requiring gradients only on p_logits. 

    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a valid distribution: target[i, :].sum() == 1.
    """
    logprobs = torch.nn.functional.log_softmax(p_logits.view(
        p_logits.shape[0], -1),
                                               dim=1)
    batchloss = -torch.sum(q_probits.view(q_probits.shape[0], -1) * logprobs,
                           dim=1)
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
            value = I[certified.nonzero()[:, 0]].tolist()
            if len(value) > 0:
                certified_modelid_idx_map[j] = value

            acc_certified = torch.logical_and(certified, out.max(1)[1] == y)
            value = I[acc_certified.nonzero()[:, 0]].tolist()
            if len(value) > 0:
                acc_certified_modelid_idx_map[j] = value

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

        # if model j is the one that makes the clean prediction,
        # we skip it
        if j == modelid:
            continue

        # The predicted label of model j
        y_pred = model(X).max(1)[1]

        # We are only interested in models that already make different predicitons
        # as the previous certifier, i.e. the first model that cerfifies its prediciton
        # for the clean input, because if model j makes the same prediciton as the
        # certifier, it is impossible to find an adversarial point within the eps-ball
        # that outputs a different label with cetificate.
        candidates = (y_pred != y).nonzero()[:, 0]

        # If there is no candidate, we skip this model j.
        if len(candidates) < 1:
            continue

        I_candidates = I[candidates]
        X_candidates = X[candidates]
        y_candidates = y[candidates]
        y_pred = y_pred[candidates]

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

            # for all model i < j, they should fail to certify
            loss_nocert = torch.zeros(loss_cert.size()).type_as(loss_cert.data)
            for k in range(j):
                f = RobustBounds(models[k], eps)(Variable(X_pgd),
                                                 Variable(y_pred))

                # To make model i, i<j, fail to certify the input,
                # we push the adversarial point closer to the decision boundary
                # by minimizing the KL divergence betweent its class probability
                # distribution and a uniform distribution.
                unif_dist = torch.ones(f.size()).type_as(f.data) / float(
                    f.size(1))
                loss_nocert += kl_divergence(p_logits=f,
                                             q_probits=unif_dist,
                                             reduction='none')

            loss = loss_pred + loss_cert + loss_nocert
            loss.mean().backward()

            if config.attack.norm == 'linf':
                #l_inf PGD
                eta = config.attack.step_size * X_pgd.grad.data.sign()
                X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
                eta = torch.clamp(X_pgd.data - X_candidates, -eps, eps)
                X_pgd.data = X_candidates + eta

            elif config.attack.norm == 'l2':
                raise NotImplementedError

            # Clip the input to a valid data range.
            X_pgd.data = torch.clamp(X_pgd.data, data_min, data_max)

        _, acc_certified_modelid_idx_map = eval_cascade(
            config, models, X_pgd, y_pred)
        acc_certified_idxs = []
        for _, idxs in acc_certified_modelid_idx_map.items():
            acc_certified_idxs += idxs

        if acc_certified_idxs:
            I_succ_candidates = I_candidates[torch.tensor(acc_certified_idxs)]

            success_ids += I_succ_candidates.tolist()
            x_attack += X_pgd[torch.tensor(acc_certified_idxs)].tolist()

            combined = torch.cat((I, I_succ_candidates))
            uniques, counts = combined.unique(return_counts=True)
            I = uniques[counts == 1]
            X = X[I]
            y = y[I]

    return x_attack, success_ids


def attack(config, loader, models, log):
    X_attack = []
    dataset_size = 0
    total_num_robust_accurate = 0
    total_num_adv_robust_accurate = 0

    pbar = enumerate(loader)

    if config.verbose:
        pbar = tqdm.tqdm(pbar)

    for batch_id, (X, y) in pbar:

        start = time()

        dataset_size += X.size(0)
        X, y = X.cuda(), y.cuda().long()
        if y.dim() == 2:
            y = y.squeeze(1)

        # Extract a subset of batch where ensemble is accurate and robust.
        # Also, get the id of constituent model used to make the prediction
        # acc_certified_modelid_idx_map is a dictionary mapping from 
        # model_id to the list of batch-leve indixces of points it certifies.  
        _, acc_certified_modelid_idx_map = eval_cascade(
            config, models, X, y)
        
        # acc_certified_modelid_idx_map is a empty dictionary,
        # which means no point is both accurate and robust. 
        if len(acc_certified_modelid_idx_map.keys()) == 0:
            continue

        num_robust_accurate = 0
        num_deny_of_service = 0
        for modelid, idxs in acc_certified_modelid_idx_map.items():

            X_curr = X[torch.tensor(idxs)]
            y_curr = y[torch.tensor(idxs)]

            per_mapping_num_robust_accurate = len(idxs)
            num_robust_accurate += per_mapping_num_robust_accurate

            if not config.attack.do_surrogate:
                X_attack_i, success_ids_i = _direct_attack(
                    config, models, Variable(X_curr), Variable(y_curr),
                    modelid)
            else:
                raise NotImplementedError

            X_attack += X_attack_i
            num_deny_of_service += len(success_ids_i)

        num_adv_robust_accurate = num_robust_accurate - num_deny_of_service

        total_num_robust_accurate += num_robust_accurate
        total_num_adv_robust_accurate += num_adv_robust_accurate

        duration = time() - start

        if config.verbose:
            pbar.set_description(
                f"Batch {batch_id}/{config.data.n_examples // config.data.batch_size}" +
                f"   |   VRA: {num_robust_accurate/config.data.batch_size:.3f} "
                +
                f"   |   Attack VRA: {num_adv_robust_accurate/config.data.batch_size:.3f}"
                +
                f"   |   ETA: {duration * (0.0167 * config.data.n_examples // config.data.batch_size - batch_id - 1):.4f} min"
            )

    print("Num inputs: ", dataset_size)
    print("VRA: ", total_num_robust_accurate / dataset_size)
    print("Adv VRA: ", total_num_adv_robust_accurate / dataset_size)
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

    # attack(config, train_loader, models, train_log)
    attack(config, test_loader, models, test_log)


if __name__ == '__main__':
    app.run(main)
