import numpy as np
import logging
import core.datashape as ds

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