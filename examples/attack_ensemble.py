
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
from trainer import * # pylint: disable=import-error
from time import time # pylint: disable=import-error

cudnn.benchmark = True

device = 'cpu'



def select_mnist_model(m):
    if m == 'large':
        model = pblm.mnist_model_large().to(device)

        # _, test_loader = pblm.mnist_loaders(8)
    else:
        model = pblm.mnist_model().to(device)
    return model


def select_cifar_model(m):
    if m == 'large':
        # raise ValueError
        model = pblm.cifar_model_large().to(device)
    elif m == 'resnet':
        model = pblm.cifar_model_resnet(N=1, factor=1).to(device)
    else:
        model = pblm.cifar_model().to(device)
    return model


def cross_entropy(*, p_logits, q_probits, reduction='mean'):
    """
    This is equivalent to the KL divergence when requiring gradients only on p_logits.

    :param input: (batch, *)
    :param target: (batch, *) same shape as input, each item must be a
        valid distribution: target[i, :].sum() == 1.
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


def sparse_cross_entropy(*,  p_logits, q_sparse, reduction='mean'):
    num_classes = p_logits.shape[1]
    q_probits = torch.nn.functional.one_hot(q_sparse, num_classes=num_classes)
    return cross_entropy(p_logits=p_logits, q_probits=q_probits, reduction=reduction)

# Evaluates the casacade given data X and labels y. 
# match_y=True if the predictions of cascade count as accurate when they match label y,
# while match_y=False if the predictions of cascade count as accurate when they do not match label y
def eval_cascade(config, models, X, y, match_y=True):
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

    # Map from modelid to indices of elements in X where model <modelid>
    # is used to make the ensemble prediction and the predictions are robust.
    certified_modelid_idx_map = {}

    # Map from modelid to indices of elements in X where model <modelid>
    # is used to make the ensemble prediction and the predictions are accurate & robust
    acc_certified_modelid_idx_map = {}

    # List of indices of elements in X where ensemble predictions are accurate but not robust.
    acc_uncertified_idx = []

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
            value_certified = I[certified.nonzero()[:, 0]].tolist()
            if len(value_certified) > 0:
                certified_modelid_idx_map[j] = value_certified

            if match_y:
                acc_certified = torch.logical_and(certified, out.max(1)[1] == y)
            else:
                acc_certified = torch.logical_and(certified, out.max(1)[1] != y)
            value_acc_certified = I[acc_certified.nonzero()[:, 0]].tolist()
            if len(value_acc_certified) > 0:
                acc_certified_modelid_idx_map[j] = value_acc_certified
            
            if match_y:
                acc_uncertified = torch.logical_and(uncertified, out.max(1)[1] == y)
            else:
                acc_uncertified = torch.logical_and(uncertified, out.max(1)[1] != y)
            value_acc_uncertified = I[acc_uncertified.nonzero()[:, 0]].tolist()
            if len(value_acc_uncertified) > 0:
                acc_uncertified_idx += value_acc_uncertified

            # reduce data set to uncertified examples
            if uncertified.sum() > 0:
                X = X[uncertified.nonzero()[:, 0]]
                y = y[uncertified.nonzero()[:, 0]]
                I = I[uncertified.nonzero()[:, 0]]
            else:
                torch.cuda.empty_cache()
                torch.set_grad_enabled(True)
                return certified_modelid_idx_map, acc_certified_modelid_idx_map, acc_uncertified_idx

        ####################################################################
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)

    return certified_modelid_idx_map, acc_certified_modelid_idx_map, acc_uncertified_idx


def make_objective_fn(config):
    if not config.attack.do_surrogate:
        def objective_fn(j, model, all_models, eps, X_pgd, y_pred):

            # for model j, we encourage it to certify the current input.
            loss_cert, _ = robust_loss(model,
                                       eps,
                                       X_pgd,
                                       y_pred,
                                       size_average=False)

            # for all model i < j, they should fail to certify the current input
            loss_nocert = torch.zeros(loss_cert.size()).type_as(loss_cert.data)
            for k in range(j):
                worse_case_logit_k = RobustBounds(all_models[k],
                                                  eps)(X_pgd, y_pred)

                # To make model i, i<j, fail to certify the input,
                # we push the adversarial point closer to the decision boundary
                # by minimizing the KL divergence betweent its class probability
                # distribution and a uniform distribution.
                unif_dist = torch.ones(worse_case_logit_k.size()).type_as(
                    worse_case_logit_k.data) / float(
                        worse_case_logit_k.size(1))
                loss_nocert += cross_entropy(p_logits=worse_case_logit_k,
                                             q_probits=unif_dist,
                                             reduction='none')

            return loss_cert + loss_nocert
    else:
        def objective_fn(j, model, all_models, eps, X_pgd, y_pred):

            loss_cert = sparse_cross_entropy(
                p_logits=model(X_pgd), q_sparse=y_pred, reduction='none')
            # for all model i < j, they should fail to certify the current input
            loss_nocert = torch.zeros(loss_cert.size()).type_as(loss_cert.data)
            for k in range(j):

                output_k = all_models[k](X_pgd)

                # To make model i, i<j, fail to certify the input,
                # we push the adversarial point closer to the decision boundary
                # by minimizing the KL divergence betweent its class probability
                # distribution and a uniform distribution.
                unif_dist = torch.ones(output_k.size()).type_as(
                    output_k.data) / float(
                        output_k.size(1))
                loss_nocert += cross_entropy(p_logits=output_k,
                                             q_probits=unif_dist, reduction='none')
            return loss_cert + loss_nocert

    return objective_fn


def attack_step(config, models, data, labels, modelid):

    data_clone = torch.clone(data)
    labels_clone = torch.clone(labels)

    noisy_data = []
    idx_for_all_data = torch.arange(data_clone.size(0)).type_as(labels_clone.data)

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

    attack_objective_fn = make_objective_fn(config)

    # This array stores where we have already found an adversarial example for
    # each input. If an adversarial example is found, we stop the attack.
    keep_attack = torch.ones(data.size(0)).type_as(data.data)

    for j, model in enumerate(models):

        # if model j is the one that makes the clean prediction,
        # we skip it
        if j == modelid:
            continue

        # The predicted label of model j
        y_pred = model(data_clone).max(1)[1]

        # We are only interested in models that already make different predicitons
        # from the previous certifier, i.e. the first model that cerfifies its prediciton
        # for the clean input, because if model j makes the same prediciton as the
        # certifier (model modelid), it is impossible to find an adversarial point within the eps-ball
        # that outputs a different label with certificate.
        candidates = torch.logical_and((y_pred != labels_clone), keep_attack)
        candidates = candidates.nonzero().squeeze(1)

        # If there is no candidate, we skip this model j.
        if len(candidates) < 1:
            continue

        candidate_idx = idx_for_all_data[candidates]
        data_candidates = data_clone[candidates]
        candidate_pred = y_pred[candidates]
        candidate_labels = labels_clone[candidates]
        candidate_keep_attack = torch.clone(keep_attack[candidates])

        # TODO: implement random start
        data_pgd = Variable(data_candidates, requires_grad=True)
        for _ in range(config.attack.steps):

            # TODO add Adam optimizer
            # opt_pgd = optim.Adam([X_pgd], lr=config.attack.step_size)

            loss = attack_objective_fn(j, model, models, eps, data_pgd, candidate_pred)
            loss.mean().backward()

            if config.attack.norm == 'linf':
                # l_inf PGD
                eta = config.attack.step_size * data_pgd.grad.data.sign()
                data_pgd = Variable(data_pgd.data + eta, requires_grad=True)
                eta = torch.clamp(data_pgd.data - data_candidates, -eps, eps)
                data_pgd.data = data_candidates + eta * candidate_keep_attack.view(-1, 1, 1, 1)

            elif config.attack.norm == 'l2':
                # l_2 PGD
                # Assumes X_candidates and X_pgd are batched tensors where the first dimension is
                # a batch dimension, i.e., .view() assumes batched images as a 4D Tensor
                grad_norms = torch.linalg.norm(
                    data_pgd.grad.view(data_pgd.shape[0], -1), dim=1)
                eta = config.attack.step_size * \
                    data_pgd.grad / grad_norms.view(-1, 1, 1, 1)
                data_pgd = Variable(data_pgd.data + eta, requires_grad=True)
                delta = data_pgd.data - data_candidates

                mask = torch.linalg.norm(delta.view(
                    delta.shape[0], -1), dim=1) <= eps

                scaling_factor = torch.linalg.norm(
                    delta.view(delta.shape[0], -1), dim=1)
                scaling_factor[mask] = eps

                delta *= eps / scaling_factor.view(-1, 1, 1, 1)

                data_pgd.data = data_candidates + delta * candidate_keep_attack.view(-1, 1, 1, 1)

            # Clip the input to a valid data range.
            data_pgd.data = torch.clamp(data_pgd.data, data_min, data_max)

        # Check whether the model is certifiably robust on a different label after the attack.
        _, acc_certified_modelid_idx_map, _ = eval_cascade(
            config, models, data_pgd, candidate_labels, match_y=False)
        acc_certified_idxs = []

        for _, idxs in acc_certified_modelid_idx_map.items():
            acc_certified_idxs += idxs

        if acc_certified_idxs:
            # If we have found an adversarial example for an input, we stop the attack.
            candidate_keep_attack[torch.tensor(acc_certified_idxs)] = 0

        keep_attack[candidates] = torch.minimum(candidate_keep_attack, keep_attack[candidates])

    noisy_data = data[keep_attack == 0]

    return noisy_data, 1 - keep_attack


def attack(config, loader, models, log):
    # data_adv = []
    dataset_size = 0
    total_num_robust_accurate = 0
    total_num_adv_robust_accurate = 0
    total_num_succ_attack_on_certified = 0

    total_num_accurate_not_robust = 0
    total_num_succ_attack_on_uncertified = 0
    total_num_emp_robust_accurate = 0
    
    duration = 0
    num_batches = config.data.n_examples // config.data.batch_size

    for batch_id, (data, label) in enumerate(loader):

        start = time()

        dataset_size += data.size(0)
        data, label = data.to(device), label.to(device).long()
        if label.dim() == 2:
            label = label.squeeze(1)

        # Extract a subset of batch where ensemble is accurate and robust.
        # Also, get the id of constituent model used to make the prediction
        # acc_certified_modelid_idx_map is a dictionary mapping from
        # model_id to the list of batch-level indices of points it certifies accurately.
        # acc_uncertified_idx is a list of batch-level indices of points that ensemble
        #  predicts accurately but does not certify.
        _, acc_certified_modelid_idx_map, acc_uncertified_idx = eval_cascade(config, models, data, label)

        # acc_certified_modelid_idx_map is a empty dictionary,
        # which means no point is both accurate and robust.
        # Also, acc_uncertified_idx is an empty list,
        # which means no point is both accurate and not certified robust.
        if len(acc_certified_modelid_idx_map.keys()) == 0 and len(acc_uncertified_idx) == 0:
            continue

        num_robust_accurate = 0
        num_succ_attack_on_certified = 0
        if len(acc_certified_modelid_idx_map.keys()) != 0:
            for modelid, idxs in acc_certified_modelid_idx_map.items():

                # rc = robust and accurate
                # We take the subset of batch where ensemble is accurate and robust.
                rc_data = data[torch.tensor(idxs)]
                rc_label = label[torch.tensor(idxs)]

                per_mapping_num_robust_accurate = len(idxs)
                num_robust_accurate += per_mapping_num_robust_accurate

                data_adv_i, is_adv = attack_step(
                    config, models, Variable(rc_data), Variable(rc_label),
                    modelid)

                # data_adv += data_adv_i
                num_succ_attack_on_certified += len(data_adv_i)
        
        num_accurate_not_robust = 0
        num_succ_attack_on_uncertified = 0
        if len(acc_uncertified_idx) != 0:
            # nc = not robust and accurate
            # We take the subset of batch where ensemble is accurate and not robust.
            nc_data = data[torch.tensor(acc_uncertified_idx)]
            nc_label = label[torch.tensor(acc_uncertified_idx)]

            data_adv_nc, is_adv = attack_step(
                config, models, Variable(nc_data), Variable(nc_label),
                len(models)-1)

            # data_adv += data_adv_nc
            num_succ_attack_on_uncertified += len(data_adv_nc)

        num_adv_robust_accurate = num_robust_accurate - num_succ_attack_on_certified
        total_num_robust_accurate += num_robust_accurate
        total_num_adv_robust_accurate += num_adv_robust_accurate
        total_num_succ_attack_on_certified += num_succ_attack_on_certified

        num_emp_robust_accurate = (num_robust_accurate + num_accurate_not_robust) - (num_succ_attack_on_certified + num_succ_attack_on_uncertified)
        total_num_accurate_not_robust += num_accurate_not_robust
        total_num_succ_attack_on_uncertified += num_succ_attack_on_uncertified
        total_num_emp_robust_accurate += num_emp_robust_accurate

        if duration > 0:
            duration = (time() - start) * 0.05 + duration * 0.95
        else:
            duration = time() - start

        if config.verbose:
            print(
                f"Batch {batch_id}/{num_batches}" +
                f"   |   Ensemble Reported CRA: {num_robust_accurate/config.data.batch_size:.3f} "
                +
                f"   |   Post Attack CRA: {num_adv_robust_accurate/config.data.batch_size:.3f}"
                +
                f"   |   Attack Success Rate: {num_succ_attack_on_certified/num_robust_accurate:.3f}"
                +
                f"   |   Ensemble ERA: {num_emp_robust_accurate/config.data.batch_size:.3f}"
                +
                f"   |   ETA: {0.0167 * duration * (num_batches - batch_id - 1):.1f} min"
            )

    print("Num inputs: ", dataset_size)
    print("Ensemble Reported CRA: ", total_num_robust_accurate / dataset_size)
    print("Post Attack CRA: ", total_num_adv_robust_accurate / dataset_size)
    print("Attack Success Rate: ", total_num_succ_attack_on_certified / total_num_robust_accurate)
    print("Ensemble CRA: ", total_num_emp_robust_accurate / dataset_size)
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
