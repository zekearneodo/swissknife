import numpy as np
import logging
import core.datashape as ds

logger = logging.getLogger('decoder.linear')




def data_arrange(sup, target, hist_bins):
    """
    :param sup: np array (t+history)xnu support vector (from -history to t)
                of nu features (features are the columns)
    :param target: np array (t) (from 0 to t)
    :param hist_bins: size of sliding window (in bins/time steps)
    :return: an array of n_points x n_features, an array of n_points x 1 (target)
    """
    logger.warn('This function is deprecated, use the one in datashape.')
    sup_correct = ds.sup_correct(sup, target, hist_bins)
    big_r = ds.make_big_r(sup_correct, hist_bins)
    return big_r, target.flatten()


def fit_kernel(sup, target, hist_bins):
    """
    :param sup: np array (t+history)xnu support vector (from -history to t)
                of nu features (features are the columns)
    :param target: np array (t) (from 0 to t)
    :param hist_bins: size of sliding window (in bins/time steps)
    :return:
    """
    [n_feat, ext_t, n_trial] = sup.shape
    [target_trial, t] = target.shape
    assert (target_trial == n_trial)
    try:
        assert ((t + hist_bins - 1) == ext_t)
    except:
        print('Size mismatch between target and support vector')
        logger.warn('Size mismatch between target and support vector')
        # drop the first bins
        sup = sup[:, :(t+hist_bins-1), :]
    big_r = ds.make_big_r(sup, hist_bins)
    #logger.info('Big_r shape {}'.format(big_r.shape))
    c = np.dot(big_r.T, big_r)
    rc = np.dot(big_r.T, target.flatten())
    f = np.dot(np.linalg.inv(c), rc)
    return f


@ds.trial_arrange_wrapper
def kernel_predict(sup, kernel):
    [n_feat, n_bin, n_trial] = sup.shape
    hist_bins = (kernel.size - 1)/n_feat

    big_r = ds.make_big_r(sup, hist_bins)
    #logger.info('Big_r shape {}'.format(big_r.shape))
    u = np.dot(big_r, kernel)
    return u.reshape(n_trial, -1)




