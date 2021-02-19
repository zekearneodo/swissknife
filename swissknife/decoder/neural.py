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

    x_train = x[(n_test * trial_len):, :]
    y_train = y[(n_test * trial_len):]
    x_test = x[:(n_test * trial_len), :]
    y_test = y[:(n_test * trial_len)]
    logger.info('Train vector sizes {0} and {1}'.format(x_train.shape, y_train.shape))
    logger.info('Test vector sizes {0} and {1}'.format(x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test

# a list of support vectors and their corresponding targets lists
# to a training and testing pair (x, y)
def sup_list_to_keras(sv_list, target_list, history_bins, n_test=5):
    big_r_stack = []
    targets_stack = []
    for one_sv, one_target in tqdm(zip(sv_list, target_list)):
        big_r, target = ds.data_arrange(one_sv, one_target[0].reshape([1, -1]), history_bins)
        target_len = target.shape[0]
        three_targets = one_target[:, :target_len].T
        big_r_stack.append(big_r)
        targets_stack.append(three_targets)
        #print(big_r.shape)

    x_train = np.concatenate(big_r_stack[:-n_test], axis=0)
    y_train = np.concatenate(targets_stack[:-n_test], axis=0)

    x_test = np.concatenate(big_r_stack[-n_test:], axis=0)
    y_test = np.concatenate(targets_stack[-n_test:], axis=0)
    return x_train, y_train, x_test, y_test


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

    x_train = x[-(n_test * trial_len):, :]
    y_train = y[-(n_test * trial_len):]
    x_test = x[:(n_test * trial_len), :]
    y_test = y[:(n_test * trial_len)]
    logger.info('Train vector sizes {0} and {1}'.format(x_train.shape, y_train.shape))
    logger.info('Test vector sizes {0} and {1}'.format(x_test.shape, y_test.shape))

    return x_train, y_train, x_test, y_test


def one_frame_to_window(one_fv, one_tv, n_lookback):
    # get an array of feature vector and target vector and spit out a
    # a window of features with lookback and
    # a window of targets
    ''''
    :param one_fv: ndarray([n_features, n_bins, *]), array with the binned features
    :param one_tv: ndarray(target_dim, target_n_bins), array with the targets
    :n_lookback: number of lookback bins (target_nbins+lookback <= n_bins)
    :return:
        one_frame_feature_window: ndarray(target_n_bins,  n_lookback, n_features,)
        one_frame_target_window: ndarray(target_n_bins, target_dim). Array with the data
    '''

    n_feat, f_bins, _ = one_fv.shape # n of features and feature bins
    t_dim, t_bins = one_tv.shape # n of time bins of target data, and target space dimension

    #logger.info('one_fv.shape = {}'.format(one_fv.shape))
    #logger.info('one_tv.shape = {}'.format(one_tv.shape))
    #logger.info('tbins {}, nlookback {}'.format(t_bins, n_lookback))
    assert((t_bins + n_lookback) <= f_bins)

    one_frame_feature_window = np.stack([one_fv.squeeze()[:, j:j+n_lookback].T for j in range(t_bins)], axis=0)
    one_frame_target_window = one_tv.T

    #print(one_frame_target_window.shape)
    #print(one_frame_feature_window.shape)

    return one_frame_feature_window, one_frame_target_window
