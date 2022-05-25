import numpy as np
import argparse
import csv
import tensorflow as tf

from scriptify import scriptify


def readFile(count=2,
             dir="evalData/l_inf/mnist_small_0_1/mnist_small_0_1",
             data="test"):
    file = dir + "_" + str(count) + "_" + str(data)
    result = np.loadtxt(file)
    y_pred = result[:, 1]
    y_true = result[:, 2]
    certified = result[:, 3]
    return y_pred, y_true, certified


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

    if weights is None:
        if solve_for_weights:
            weights = find_weights(C, C_hat)
        else:
            weights = np.ones((C.shape[0], ))

    # Do the voting to find the clean prediction
    # index with 0 to remove the redundant axis.
    votes = np.einsum('ij,jlk->ilk', weights[None],
                      Y)[0]  # shape (n_sampels, n_classes)
    y_ensemble_clean = np.argmax(votes, axis=1)

    # Do the voting to find the robust prediction
    # index with 0 to remove the redundant axis.
    votes = np.einsum('ij,jlk->ilk', weights[None],
                      C)[0]  # shape (n_sampels, n_classes+1)

    # replace the votes for the top class to -1 to
    # differentiate the votes for the top class and
    # other classes that have the same votes
    votes_bot = votes[:, :-1].copy()
    np.put_along_axis(votes_bot, y_ensemble_clean[:, None], -1, axis=1)

    # Add the votes for the botom class to all classes except the top.
    # Also add eps to the votes for the bottom class in case it is 0
    votes_bot = np.where(votes_bot == -1, votes[:, :-1],
                         votes[:, :-1] + votes[:, -1:] + eps)

    # Concatenate the votes of all classes and the votes of the bottom class
    votes_bot = np.concatenate([votes_bot, votes[:, -1:]], axis=1)

    # find the top class with votes for the bot added to all other classes.
    robust_j = np.argmax(votes_bot, axis=1)

    # the prediction is certified if votes for the top is still the one without adding votes for the bot.
    y_ensemble_certificate = (robust_j == y_ensemble_clean).astype(np.int32)

    acc = np.mean(y_ensemble_clean == y_true)
    vra = np.mean((y_ensemble_clean == y_true) * y_ensemble_certificate)

    return y_ensemble_clean, y_ensemble_certificate, acc, vra, weights


def find_weights(C, C_hat):
    return optimize_find_weights(C, C_hat)


def normalize(x):
    return x / (tf.reduce_sum(x) + 1e-16)

def half_temp_sigmoid(x, temp):
    return tf.where(x<=0, tf.nn.sigmoid(x/temp), tf.nn.sigmoid(x))

def optimize_find_weights(Y_candidates,
                          Y_hat,
                          steps=1000,
                          lr=1e-2,
                          t1=1e5,
                          t2=1e-3):
    '''
    Y_candidates: shape: KxNx(C+1). The one-hot encoding of the predicitons, including \bot, of K models for N points. 
    Y_hat: shape: Nx(C+1). The one-hot encoding of the labels. 
    w: shape: K, weights
    '''

    Y_candidates = tf.cast(Y_candidates, tf.float32)
    Y_hat = tf.cast(Y_hat, tf.float32)

    K = Y_candidates.shape[0]
    w = tf.Variable(initial_value=tf.ones((K, )))

    vars = [w]

    B = np.zeros_like(Y_hat)
    B[:, -1] = 1
    B = tf.constant(B)

    opt = tf.keras.optimizers.Adam(learning_rate=lr)

    pbar = tf.keras.utils.Progbar(steps)

    softmax = tf.nn.softmax
    for _ in range(steps):
        with tf.GradientTape() as tape:
            # weighted votes (N, C+1)
            tape.watch(w)
            valid_w = softmax(w)

            Y = tf.squeeze(
                tf.einsum('ij,jlk->ilk', valid_w[None], Y_candidates))

            # the votes for the grountruth class
            y_j = tf.reduce_sum(Y * Y_hat, axis=1)
            # the votes for the bottom class
            y_bot = tf.reduce_sum(Y * B, axis=1)
            # the votes for the highest class except the groudtruth and the bottom classes
            # y_second = tf.reduce_sum(Y * softmax(Y * (1 - Y_hat - B) / t2))
            # y_second = tf.reduce_max(Y * (1 - Y_hat - B))
            y_second = tf.reduce_max(Y * (1 - Y_hat - B), axis=1)
            margin = y_j - y_bot - y_second
            loss = -tf.reduce_mean(half_temp_sigmoid(margin, t1))
            # loss = tf.reduce_mean(relu(-(y_j - y_bot - y_second)))

        grads = tape.gradient(loss, vars)
        pbar.add(1, [("loss", loss),
                     ('margin', tf.reduce_mean(margin))] + [(f"w{i}", w[i])
                                                            for i in range(K)])

        opt.apply_gradients(zip(grads, vars))

    valid_w = softmax(w)
    return valid_w.numpy()


def cascade(y_pred, y_true, certified):
    ''' use pre-evaluated data for cascade ensemble algorithm

    Args:
        y_pred: list of np.array. Pedictions for each models
        y_true: np.array of shape (n_smaples,). True labels.
        certified: list of np.array. Boolean array indicating whether the prediction is certified.
    
    Returns:
        acc: float. Accuracy of the ensemble
        vra: float. VRA of the ensemble

    '''
    correct = 0
    vra = 0
    for i in range(np.shape(y_pred)[1]):
        # for each input point
        for j in range(np.shape(y_pred)[0]):
            # for each model
            if certified[j][i] == 1:
                if y_pred[j][i] == y_true[j][i]:
                    correct = correct + 1
                    vra = vra + 1
                break
            elif y_pred[j][i] == y_true[j][i] and j == np.shape(y_pred)[0] - 1:
                correct = correct + 1

    acc = correct / np.shape(y_pred)[1]
    vra = vra / np.shape(y_pred)[1]

    return acc, vra


def np_onehot(vector, num_classes=None):
    if num_classes is None:
        num_classes = np.max(vector)
    return np.eye(num_classes)[vector]


if __name__ == "__main__":

    @scriptify
    def script(
            model_type="mnist_large_0_1",  # mnist_large_0_1, cifar_small_2px
            norm="l_2",
            count=3,
            weights=None,
            solve_for_weights=False,
            seq=False):
        if seq: dir = "./ensembleKW/evalData/seq_trained/" + norm + "/" + model_type + "/" + model_type
        else: dir = "./ensembleKW/evalData/non_seq_trained/" + norm + "/" + model_type + "/more_" + model_type
        y_pred_all = []
        y_true_all = []
        certified_all = []

        results = {}

        for i in range(count):
            # read all pre-evaluated files
            y_pred, y_true, certified = readFile(count=i, dir=dir, data="test")

            y_pred_all.append(y_pred)
            y_true_all.append(y_true)
            certified_all.append(certified)

            acc = np.mean(y_pred_all[-1] == y_true)
            vra = np.mean((y_pred_all[-1] == y_true) * certified_all[-1])

            results.update({
                f"model_{i}_acc": float(acc),
                f"model_{i}_vra": float(vra)
            })

        cas_acc, cas_vra = cascade(y_pred_all, y_true_all, certified_all)

        results.update({
            f"cas_acc": float(cas_acc),
            f"cas_vra": float(cas_vra)
        })

        if weights is not None and not solve_for_weights:
            # voting with given weights
            weights = np.array(list(map(float, weights.split(','))))

        elif solve_for_weights:
            # use evaluation on train dataset to find optimized weights
            train_y_pred_all = []
            train_certified_all = []
            for i in range(count):
                train_y_pred, train_y_true, train_certified = readFile(
                    count=i, dir=dir, data="train")

                train_y_pred_all.append(train_y_pred)
                train_certified_all.append(train_certified)

            _, _, _, _, weights = matrix_op_robust_voting(
                train_y_pred_all,
                train_y_true,
                train_certified_all,
                solve_for_weights=True,
                weights=None)

        _, _, vote_acc, vote_vra, weights = matrix_op_robust_voting(
            y_pred_all,
            y_true,
            certified_all,
            solve_for_weights=False,
            weights=weights)

        results.update({
            f"vote_acc": float(vote_acc),
            f"vote_vra": float(vote_vra)
        })

        weights = str(list(weights))

        results.update({'ensemble_weights': weights})

        print(results)

        return results
