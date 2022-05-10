import numpy as np
import argparse
import csv


def readFile(count=2,
             dir="evalData/l_inf/mnist_small_0_1/mnist_small_0_1",
             data="test"):
    file = dir + "_" + str(count) + "_" + str(data)
    result = np.loadtxt(file)
    y_pred = result[:, 1]
    y_true = result[:, 2]
    certified = result[:, 3]
    return y_pred, y_true, certified


def robust_voting(y_pred, y_true, certified):
    acc = 0
    vra = 0
    for i in range(np.shape(y_pred)[1]):
        count_vra = np.zeros(int(max(y_pred[0])) + 1)
        bot = np.shape(certified)[0] - sum([row[i] for row in certified])
        count_vra[int(y_true[0][i])] -= bot  # add bottom to count_vra

        count_acc = np.zeros(int(max(y_pred[0])) + 1)

        for j in range(np.shape(y_pred)[0]):
            if certified[j][i] == 1: count_vra[int(y_pred[j][i])] += 1
            count_acc[int(y_pred[j][i])] += 1

        # j - j_true
        count_vra -= np.full_like(count_vra, count_vra[int(y_true[0][i])])
        count_vra[int(y_true[0][i])] = -1

        count_acc -= np.full_like(count_acc, count_acc[int(y_true[0][i])])
        count_acc[int(y_true[0][i])] = -1

        # j_true is j_max
        if sum(count_vra < 0) == len(count_vra): vra += 1
        if sum(count_acc < 0) == len(count_acc): acc += 1

    print("voting clean_acc: ", acc / np.shape(y_pred)[1], ", error: ",
          1 - acc / np.shape(y_pred)[1])
    print("voting vra: ", vra / np.shape(y_pred)[1], ", error: ",
          1 - vra / np.shape(y_pred)[1])


def matrix_op_robust_voting(y_pred,
                            y_true,
                            certified,
                            weights=None,
                            num_classes=10,
                            solve_for_weights=False,
                            eps=1e-6):
    ''' Compute the voting ensemble of robust models

    Args:
        y_pred: list of np.array. Pedictions for each models
        y_true: np.array of shape (n_smaples,). True labels.
        certified: list of np.array. Boolean array indicating whether the prediction is certified.
        weights: list of np.array. Weights for each models. If None, weight each model evenly. Default: None
    
    Returns:
        y_ensemble_clean: np.array of shape (n_samples,). Clean predictions for the ensemble
        y_ensemble_certificate: np.array of shape (n_samples,). Boolean array indicating whether the prediction is certified.
        acc: float. Accuracy of the ensemble
        vra: float. VRA of the ensemble

    '''

    # Construct a Y matrix of shape (n_models, n_sampels, n_classes)
    # as the one-hot encoding of the prediction and another C matrix of
    # of shape (n_models, n_sampels, n_classes+1) as the prediction includiing bot
    Y = []
    C = []
    for c, y in zip(certified, y_pred):
        y = np.array(y).astype(np.int32)
        Y.append(np_onehot(y, num_classes=num_classes)[None])

        # append \bot if the prediction is not certified
        C.append(
            np_onehot(np.where(c == np.ones_like(c), y,
                               (np.zeros_like(c) + num_classes).astype(
                                   np.int32)),
                      num_classes=num_classes + 1)[None])

    Y = np.vstack(Y)
    C = np.vstack(C)

    # Construct a groundtruth C_hat matrix with an extra column for the bottom classes
    C_hat = np_onehot(y_true.astype(np.int32), num_classes=num_classes + 1)

    if weights is None and solve_for_weights:
        w = find_weights(C, C_hat)
    else:
        w = np.ones((C.shape[0], ))

    # Do the voting to find the clean prediction
    # index with 0 to remove the redundant axis.
    votes = np.einsum('ij,jlk->ilk', w[None],
                      Y)[0]  # shape (n_sampels, n_classes)
    y_ensemble_clean = np.argmax(votes, axis=1)

    # Do the voting to find the robust prediction
    # index with 0 to remove the redundant axis.
    votes = np.einsum('ij,jlk->ilk', w[None],
                      C)[0]  # shape (n_sampels, n_classes+1)

    # Calculate the top class and the corresponding votes.
    j = np.argmax(votes[:, :-1], axis=1)
    votes_j = np.max(votes[:, :-1], axis=1, keepdims=True)

    # Add votes of bot to all classes except the top class
    # Also add a small number in case the bottom is 0 and all models vote
    # for different classes where np.argmax always return the prediction of the first model.
    votes_bot = np.where(votes_j == votes[:, :-1], votes[:, :-1],
                         votes[:, :-1] + votes[:, -1:] + eps)

    # Concatenate the votes of all classes and the votes of the bottom class
    votes_bot = np.concatenate([votes_bot, votes[:, -1:]], axis=1)

    # find the top class with votes for the bot added to all other classes.
    robust_j = np.argmax(votes_bot, axis=1)

    # the prediction is certified if votes for the top is still the one without adding votes for the bot.
    y_ensemble_certificate = (robust_j == j).astype(np.int32)

    acc = np.mean(y_ensemble_clean == y_true)
    vra = np.mean((y_ensemble_clean == y_true) * y_ensemble_certificate)

    print("mo voting clean_acc: ", acc)
    print("mo voting vra: ", vra)

    return y_ensemble_clean, y_ensemble_certificate, acc, vra


def find_weights(C, C_hat):
    return


def cascade(y_pred, y_true, certified):
    correct = 0
    vra = 0
    for i in range(np.shape(y_pred)[1]):
        for j in range(np.shape(y_pred)[0]):
            if y_pred[j][i] == y_true[j][i] and certified[j][i] == 1:
                correct = correct + 1
                vra = vra + 1
                break
            elif y_pred[j][i] == y_true[j][i] and j == np.shape(y_pred)[0] - 1:
                correct = correct + 1

    print("cascade clean_acc: ", correct / np.shape(y_pred)[1], ", error: ",
          1 - correct / np.shape(y_pred)[1])
    print("cascade vra: ", vra / np.shape(y_pred)[1], ", error: ",
          1 - vra / np.shape(y_pred)[1])


def np_onehot(vector, num_classes=None):
    if num_classes is None:
        num_classes = np.max(vector)
    return np.eye(num_classes)[vector]


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--filename', default=opt)
    # args = parser.parse_args()

    # readFile(args.filename)
    model_type = "cifar_small_2px"
    dir = "/home/chi/NNRobustness/ensembleKW/evalData/l_inf/" + model_type + "/" + model_type
    count = 3
    y_pred_all = []
    y_true_all = []
    certified_all = []
    print("results for model type ", model_type)
    for i in range(count):
        y_pred, y_true, certified = readFile(count=i, dir=dir, data="test")

        y_pred_all.append(y_pred)
        y_true_all.append(y_true)
        certified_all.append(certified)

    print("model 0 clean accuracy: ", np.mean(y_pred_all[0] == y_true_all[0]),
          ", error: ", 1 - np.mean(y_pred_all[0] == y_true_all[0]))
    print("model 0 vra: ",
          np.mean((y_pred_all[0] == y_true_all[0]) * certified_all[0]),
          ", error: ", 1 - np.mean(
              (y_pred_all[0] == y_true_all[0]) * certified_all[0]))

    # cascade(y_pred_all, y_true_all, certified_all)
    robust_voting(y_pred_all, y_true_all, certified_all)

    y_ensemble_clean, y_ensemble_certificate, acc, vra = matrix_op_robust_voting(
        y_pred_all, y_true_all[0], certified_all)
