from __future__ import division

import logging
import os

import numpy as np
from scipy.signal import resample
from scipy.signal import savgol_filter

from swissknife.bci.core import expstruct as et

logger = logging.getLogger('bci_pipeline.synthetic')


def normalize(x):
    x_normed = (x - x.min(0)) / x.ptp(0)
    return x_normed


def running_diff(x, y):
    # y is the larger
    n_x = x.shape[1]
    n_y = y.shape[1]
    return np.array([np.sum(abs(x - y[:, i: i + n_x])) for i in range(n_y - n_x)])


def np_mulog(x, mu=512):
    return np.multiply(np.sign(x), np.log(mu * np.fabs(x) + 1.) / np.log(1. + mu))


def np_mulog_inv(y, mu=512):
    return np.sign(y) / mu * (np.power((1. + mu) * np.ones_like(y), np.fabs(y)) - 1.)


def stream_resample(stream, s_f, new_s_f):
    n_samples = stream.shape[0]
    new_samples = int(n_samples * new_s_f / s_f)
    if int(s_f) == int(new_s_f):
        resampled = stream
    else:
        resampled = resample(stream, new_samples)
    return resampled


def resample_no_interp(x, s_f, new_s_f, axis=0):
    assert(new_s_f>=s_f)
    n_repeats = np.int(np.ceil(new_s_f/s_f))
    return np.repeat(x, n_repeats, axis=axis)


def resample_interp(x, s_f, new_s_f):
    t = np.linspace(0, x.size / s_f, x.size)
    # print t
    new_t = np.linspace(0, x.size / s_f, np.int(x.size * new_s_f / s_f))
    print(new_t.shape)
    return np.interp(new_t, t, x)


def latent_to_stream(onof, beta, alpha):
    onof[onof > .3] = .3
    onof[onof < 0] = 0
    alpha[alpha < 0] = 0
    return np.vstack([0.15 - onof, -beta, alpha]).T


def load_syn_stream(bird, syn_sess=1, s_f=44100, new_s_f=30000, file_base='synth_bos'):
    stim_folder = et.file_names(bird)['folders']['stim']
    syn_file = os.path.join(stim_folder, str(syn_sess).zfill(3), file_base + '.dat')
    return stream_resample(np.loadtxt(syn_file), s_f, new_s_f)


def save_syn_streams(streams, bird, syn_sess=1, s_f=30000, new_s_f=30000, file_base='synth_bos'):
    stim_folder = et.file_names(bird)['folders']['stim']
    syn_file = os.path.join(stim_folder, str(syn_sess).zfill(3), file_base + '.dat')
    np.savetxt(stream_resample(streams, s_f, new_s_f))


def load_alpha_beta(bird, syn_sess=1, s_f=44100, new_s_f=30000, first=None):
    syn_par = load_syn_stream(bird, syn_sess=syn_sess, s_f=s_f, new_s_f=new_s_f)
    bos = load_syn_stream(bird, syn_sess=syn_sess, s_f=s_f, new_s_f=new_s_f, file_base='bos')
    env_bos = load_syn_stream(bird, syn_sess=syn_sess, s_f=s_f, new_s_f=new_s_f, file_base='env_bos')

    first = syn_par.shape[0] if first is None else first

    alpha = syn_par[:first, 2]
    alpha[alpha > 0] = 0
    alpha[alpha < 0] = 0.3

    # equivalent to -alpha + 0.15 but getting rid of high frequency
    # env = st.envelope(bos[:alpha.shape[0]], window=300, conv_mode='same')

    env = env_bos[:first, 2]
    beta = savgol_filter(-(syn_par[:first, 1]), 43, 3)
    beta = -(syn_par[:first, 1])
    beta[beta < 0] = 0
    beta = savgol_filter(beta, 43, 3)
    beta[beta < 0] = 0

    return (alpha), (beta), env


def fitted_to_stream(onof, beta, alpha):
    return np.vstack([0.15 - onof, -beta, alpha * 1000]).T


def regularize_pars(x, beta_max=2.5):
    '''
    Get an array of parameters alpha, beta env and transforms to input to nn:
    alpha: 0/1 (off/on)
    beta: mulog(-beta/beta_max)
    env: mulog(env
    :param x: ndarray([3, n_ms]), array with ms sampled parameters in cols (alpha, beta, env)
    :param beta_max: divisor for beta column
    :return: ndarray([3, n_ms])
    '''
    # 3 cols: alpha, beta, env
    alpha = np.zeros_like(x[0])
    alpha[x[0] < 0] = 1
    beta = np_mulog(-x[1] / beta_max)
    env = np_mulog(x[2])
    return np.stack([alpha, beta, env])


def regularize_pars_inv(y, beta_max=2.5):
    '''
    Get an array of binned parameters alpha, beta env and transforms to:
    alpha: .15/-.15 (off/on)
    beta: -mulog_inv(beta)*beta_max
    env: mulog_inv(env)
    :param y: ndarray([n, 3]), array with ms sampled parameters in rows (alpha, beta, env)
    :param beta_max: divisor for beta column
    :return: ndarray([n, 3])
    '''
    # 3 cols: alpha, beta, env
    alpha = np.ones_like(y[0])*0.15
    alpha[y[0] > 0.5] = -.15

    beta = -np_mulog_inv(y[1]) * beta_max
    env = 1000*np_mulog_inv(y[2])
    env[env<0] = 0
    return np.stack([alpha, beta, env])

def pred_to_par_stream(y, bin_size, s_f=44100):
    '''
    Get an array of binned parameters alpha, beta env and transforms to:
    alpha: .15/-.15 (off/on)
    beta: -mulog_inv(beta)*beta_max
    env: mulog_inv(env)
    :param y: ndarray([n_bin, 3]), array with ms sampled parameters in rows (alpha, beta, env)
    :param beta_max: divisor for beta column
    :return: ndarray([n_bin*bin_size*s_f, 3]) array with parameters sampled at s_f in cols (alpha, beta, env)
    '''
    # resample (fourier interpolation) to ms bins
    yr = stream_resample(y, 1, bin_size)
    yr = regularize_pars_inv(yr.T)
    # interpolate parameters that are in bin chunks
    return resample_no_interp(yr.T, 1, int(s_f/1000.), axis=0)
