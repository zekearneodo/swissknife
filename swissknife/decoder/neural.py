import numpy as np
import logging
from swissknife.decoder.core import datashape as ds

logger = logging.getLogger('decoder.neural')


def sup_to_keras(sup, target, history_bins, n_test=10):
    x, y = ds.data_arrange(sup, target, history_bins)
    x = x[:, 1:]
    norm_x = np.max(x)
    norm_y = np.max(y)

    trial_len = sup.shape[1] - history_bins
    n_feat = x.shape[1]

    x_train = x[:-(n_test * trial_len), :]
    y_train = y[:-(n_test * trial_len)]
    x_test = x[:(n_test * trial_len), :]
    y_test = y[:(n_test * trial_len)]
    logger.info('Train vector sizes {0} and {1}'.format(x_train.shape, y_train.shape))
    logger.info('Test vector sizes {0} and {1}'.format(x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test


def sup_list_to_keras(sv_list, target_list, history_bins, n_test=10):
    big_r_stack = []
    targets_stack = []
    for one_sv, one_target in zip(sv_list, target_list):
        big_r, target = ds.data_arrange(one_sv, one_target[0].reshape([1, -1]), history_bins)
        target_len = target.shape[0]
        three_targets = one_target[:, :target_len].T
        big_r_stack.append(big_r)
        targets_stack.append(three_targets)
    X = np.concatenate(big_r_stack, axis=0)
    Y = np.concatenate(targets_stack, axis=0)
    return X, Y


def sup_to_keras_hirank(sup, target, history_bins, n_test=10):
    y_stack = []
    [trials, bins, feats] = target.shape
    for f in range(feats):
        x, y_f = ds.data_arrange(sup, target[:, :, f], history_bins)
        #logger.info('yf shape {}'.format(y_f.shape))
        y_stack.append(y_f)

    y = np.stack(y_stack, axis=-1)

    x = x[:, 1:]
    norm_x = np.max(x)
    norm_y = np.max(y)

    trial_len = sup.shape[1] - history_bins
    n_feat = x.shape[1]
    logger.info('Total vector sizes {0} and {1}'.format(x.shape, y.shape))

    x_train = x[:-(n_test * trial_len), :]
    y_train = y[:-(n_test * trial_len)]
    x_test = x[:(n_test * trial_len), :]
    y_test = y[:(n_test * trial_len)]
    logger.info('Train vector sizes {0} and {1}'.format(x_train.shape, y_train.shape))
    logger.info('Test vector sizes {0} and {1}'.format(x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test