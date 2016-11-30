import numpy as np
import logging

logger = logging.getLogger('decoder.linear')

def trial_arrange_wrapper(sup_function):
    def array_checker(sup_array, *args, **kwargs):
        if sup_array.ndim==2:
            ret_array = np.reshape(sup_array,
                                   (sup_array.shape[0],
                                    sup_array.shape[1],
                                    1))
        else:
            ret_array = sup_array
        return sup_function(ret_array, *args, **kwargs)
    return array_checker

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
        sup = sup[:,:(t+hist_bins-1),:]
    big_r = make_big_r(sup, hist_bins)
    c = np.dot(big_r.T, big_r)
    rc = np.dot(big_r.T, target.flatten())
    f = np.dot(np.linalg.inv(c), rc)
    return f


@trial_arrange_wrapper
def kernel_predict(sup, kernel):
    [n_feat, n_bin, n_trial] = sup.shape
    hist_bins = (kernel.size - 1)/n_feat

    big_r = make_big_r(sup, hist_bins)
    u = np.dot(big_r, kernel)
    return u.reshape(n_trial, -1)


@trial_arrange_wrapper
def make_big_r(sup, hist_bins):
    """
    :param sup: n_feat x n_bin x n_trial array
    :param hist_bins: int (window size)
    :return: (n_bins-hist_bins)*n_trial x n_feat*hist_bins array
    """
    [n_feat, n_bin, n_trial] = sup.shape
    m = n_bin - hist_bins + 1
    r = np.vstack([sup[:, j:j + hist_bins, :].reshape(-1, n_trial)
                   for j in range(m)])

    return np.hstack([np.ones([n_trial*m, 1]),
                      np.reshape(r.T, (-1, n_feat*(hist_bins)))])

