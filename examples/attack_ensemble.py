
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
    # is used to make the ensemble prediction and the predictions are certified robust.
    CR_modelid_idx_map = {}

    # Map from modelid to indices of elements in X where model <modelid>
    # is used to make the ensemble prediction and the predictions are certified robust & accurate.
    CRA_modelid_idx_map = {}

    # List of indices of elements in X where ensemble predictions are accurate but not certified robust.
    # Since all such predictions are made by the last model in the ensemble, we do not need to record the modelid. 
    A_idxs = []

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
            CR_idxs = I[certified.nonzero()[:, 0]].tolist()
            if len(CR_idxs) > 0:
                CR_modelid_idx_map[j] = CR_idxs

            if match_y:
                certified_acc = torch.logical_and(certified, out.max(1)[1] == y)
            else:
                certified_acc = torch.logical_and(certified, out.max(1)[1] != y)
            CRA_idxs = I[certified_acc.nonzero()[:, 0]].tolist()
            if len(CRA_idxs) > 0:
                CRA_modelid_idx_map[j] = CRA_idxs
            
            if j == len(models) - 1:
                if match_y:
                    uncertified_acc = torch.logical_and(uncertified, out.max(1)[1] == y)
                else:
                    uncertified_acc = torch.logical_and(uncertified, out.max(1)[1] != y)
                A_idxs += I[uncertified_acc.nonzero()[:, 0]].tolist()

            # reduce data set to uncertified examples
            if uncertified.sum() > 0:
                X = X[uncertified.nonzero()[:, 0]]
                y = y[uncertified.nonzero()[:, 0]]
                I = I[uncertified.nonzero()[:, 0]]
            else:
                torch.cuda.empty_cache()
                torch.set_grad_enabled(True)
                return CR_modelid_idx_map, CRA_modelid_idx_map, A_idxs

        ####################################################################
    torch.cuda.empty_cache()
    torch.set_grad_enabled(True)

    return CR_modelid_idx_map, CRA_modelid_idx_map, A_idxs


def make_objective_fn(config, cert_needed=True):
    if not cert_needed:
        def objective_fn(j, model, all_models, eps, X_pgd, y_pred):

            # for model j, we encourage it to have a different prediction at the current input.
            loss_cert = sparse_cross_entropy(
                p_logits=model(X_pgd), q_sparse=y_pred, reduction='none')

            # for all model i < j, they should fail to certify the current input.
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
            return loss_cert - loss_nocert
            


    elif not config.attack.do_surrogate:
        def objective_fn(j, model, all_models, eps, X_pgd, y_pred):

            # for model j, we encourage it to certify the current input.
            loss_cert, _ = robust_loss(model,
                                       eps,
                                       X_pgd,
                                       y_pred,
                                       size_average=False)

            # for all model i < j, they should fail to certify the current input.
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

            return -loss_cert - loss_nocert
    else:
        def objective_fn(j, model, all_models, eps, X_pgd, y_pred):

            # for model j, we encourage it to certify the current input.
            loss_cert = sparse_cross_entropy(
                p_logits=model(X_pgd), q_sparse=y_pred, reduction='none')

            # for all model i < j, they should fail to certify the current input.
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
            return -loss_cert - loss_nocert

    return objective_fn

def attack_step(config, models, data, labels, modelid):

    last_modelid = len(models) - 1
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

        if j == last_modelid:
            attack_objective_fn = make_objective_fn(config, cert_needed=False)
        else:
            # if model j is the one that makes the clean prediction,
            # we skip it
            if j == modelid:
                continue

        # The predicted label of model j
        y_pred = model(data_clone).max(1)[1]

        if j == last_modelid:
            candidates = keep_attack.nonzero().squeeze(1)
        else:
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
                # data_pgd.data = data_candidates + eta * candidate_keep_attack.view(-1, 1, 1, 1)
                data_pgd.data = data_candidates + eta

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

                # data_pgd.data = data_candidates + delta * candidate_keep_attack.view(-1, 1, 1, 1)
                data_pgd.data = data_candidates + delta

            # Clip the input to a valid data range.
            data_pgd.data = torch.clamp(data_pgd.data, data_min, data_max)

        # Check whether the model is certifiably robust on a different label after the attack.
        _, CRA_modelid_idx_map, A_idxs = eval_cascade(
            config, models, data_pgd, candidate_labels, match_y=False)
        CRA_idxs = []

        for _, idxs in CRA_modelid_idx_map.items():
            CRA_idxs += idxs

        if CRA_idxs:
            # If we have found an adversarial example for an input, we stop the attack.
            candidate_keep_attack[torch.tensor(CRA_idxs)] = 0

        if j == last_modelid and A_idxs:
            # If we have found an adversarial example for an input, we stop the attack.
            candidate_keep_attack[torch.tensor(A_idxs)] = 0

        keep_attack[candidates] = torch.minimum(candidate_keep_attack, keep_attack[candidates])

    #TODO: fix such that noisy data contains the perturbed inputs
    noisy_data = data[keep_attack == 0]

    return noisy_data, 1 - keep_attack


def attack(config, loader, models, log):
    # data_attackable = []
    dataset_size = 0

    # Number of samples in the dataset where the ensemble is certified robust and accurate
    total_num_CRA = 0

    # Number of samples in the dataset where the ensemble is certified robust and accurate
    # but our attack was successful
    total_num_attackable_CRA = 0

    # Number of samples in the dataset where the ensemble is certified robust and accurate
    # and our attack was not successful
    total_num_not_attackable_CRA = 0

    # Number of samples in the dataset where the ensemble is accurate but not certified robust
    total_num_A = 0

    # Number of samples in the dataset where the ensemble is accurate but not certified robust
    # and our attack was successful
    total_num_attackable_A = 0

    # Number of samples in the dataset where the ensemble is accurate and our attack was unsuccessful
    total_num_ERA = 0
    
    duration = 0
    num_batches = config.data.n_examples // config.data.batch_size

    for batch_id, (data, label) in enumerate(loader):

        start = time()

        dataset_size += data.size(0)
        data, label = data.to(device), label.to(device).long()
        if label.dim() == 2:
            label = label.squeeze(1)

        # CRA_modelid_idx_map is a dictionary mapping from modelid to 
        # the list of batch-level indices of points where the ensemble uses 
        # model modelid for prediction and the predictions are certified robust & accurate.
        # 
        # A_idx is a list of batch-level indices of points that the ensemble
        # predicts accurately but cannot certify robustness. Since all such points are predicted using
        # the last model in the ensemble, we don't need to record the modelid.
        _, CRA_modelid_idx_map, A_idx = eval_cascade(config, models, data, label)

        if len(CRA_modelid_idx_map.keys()) == 0 and len(A_idx) == 0:
            # CRA_modelid_idx_map is a empty dictionary, which means no point is both certified robust & accurate.
            # Also, A_idx is an empty list, which means no point is both not certified robust & accurate.
            continue

        num_CRA = 0
        num_attackable_CRA = 0
        if len(CRA_modelid_idx_map.keys()) != 0:
            for modelid, idxs in CRA_modelid_idx_map.items():

                # CRA = certified robust and accurate
                # We take the subset of batch where ensemble certified robust and accurate.
                CRA_data = data[torch.tensor(idxs)]
                CRA_label = label[torch.tensor(idxs)]

                per_mapping_num_CRA = len(idxs)
                num_CRA += per_mapping_num_CRA

                data_attackable_CRA, is_attackable_CRA = attack_step(
                    config, models, Variable(CRA_data), Variable(CRA_label),
                    modelid)

                # data_attackable += data_attackable_CRA
                num_attackable_CRA += len(data_attackable_CRA)
        
        num_A = 0
        num_attackable_A = 0
        if len(A_idx) != 0:
            # A = accurate (but not certified robust)
            # We take the subset of batch where ensemble is accurate but not certified robust.
            A_data = data[torch.tensor(A_idx)]
            A_label = label[torch.tensor(A_idx)]

            num_A += len(A_idx)

            data_attackable_A, is_attackable_A = attack_step(
                config, models, Variable(A_data), Variable(A_label),
                len(models)-1)

            # data_attackable += data_attackable_A
            num_attackable_A += len(data_attackable_A)

        num_not_attackable_CRA = num_CRA - num_attackable_CRA
        total_num_CRA += num_CRA
        total_num_not_attackable_CRA += num_not_attackable_CRA
        total_num_attackable_CRA += num_attackable_CRA

        num_ERA = (num_CRA + num_A) \
            - (num_attackable_CRA + num_attackable_A)
        total_num_A += num_A
        total_num_attackable_A += num_attackable_A
        total_num_ERA += num_ERA

        if duration > 0:
            duration = (time() - start) * 0.05 + duration * 0.95
        else:
            duration = time() - start

        if config.verbose:
            print(
                f"Batch {batch_id}/{num_batches}" +
                f"   |   Clean Accuracy: {(num_CRA+num_A)/config.data.batch_size:.3f} "
                +
                f"   |   Unsound CRA: {num_CRA/config.data.batch_size:.3f} "
                +
                f"   |   Post-Attack Unsound CRA: {num_not_attackable_CRA/config.data.batch_size:.3f}"
                +
                f"   |   Attackable Certificate Ratio: {num_attackable_CRA/num_CRA:.3f}"
                +
                f"   |   ERA: {num_ERA/config.data.batch_size:.3f}"
                +
                f"   |   ETA: {0.0167 * duration * (num_batches - batch_id - 1):.1f} min"
            )

    print("Num inputs: ", dataset_size)
    print("Clean Accuracy: ", (total_num_CRA + total_num_A) / dataset_size)
    print("Unsound CRA: ", total_num_CRA / dataset_size)
    print("Post-Attack Unsound CRA: ", total_num_not_attackable_CRA / dataset_size)
    print("Attackable Certificate Ratio: ", total_num_attackable_CRA / total_num_CRA)
    print("ERA: ", total_num_ERA / dataset_size)
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
