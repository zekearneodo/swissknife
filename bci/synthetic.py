from __future__ import division

import logging
import numpy as np
import os
from scipy.signal import resample

from bci.core import expstruct as et

logger = logging.getLogger('bci_pipeline.synthetic')


def normalize(x):
    x_normed = (x - x.min(0)) / x.ptp(0)
    return x_normed


def running_diff(x, y):
    # y is the larger
    n_x = x.shape[1]
    n_y = y.shape[1]
    return np.array([np.sum(abs(x - y[:, i: i + n_x])) for i in range(n_y - n_x)])


def mulog(x, mu=256):
    return np.sign(x) * np.log(1. + mu * np.fabs(x)) / np.log(1. + mu)


def mulog_vec(x, mu):
    return np.array([mulog(i) for i in x])


def inverse_mulog_vec(y, mu=256):
    return np.sign(y) / mu * (np.power((1. + mu) * np.ones_like(y), np.fabs(y)) - 1.)


def stream_resample(stream, s_f, new_s_f):
    n_samples = stream.shape[0]
    new_samples = int(n_samples * new_s_f / s_f)
    if int(s_f) == int(new_s_f):
        resampled = stream
    else:
        resampled = resample(stream, new_samples)
    return resampled


def resample_interp(x, s_f, new_s_f):
    t = np.linspace(0, x.size / s_f, x.size)
    # print t
    new_t = np.linspace(0, x.size / s_f, np.int(x.size * new_s_f / s_f))
    print new_t.shape
    return np.interp(new_t, t, x)


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
    beta = -(syn_par[:first, 1])
    beta[beta < 0] = 0

    return (alpha), (beta), env


def fitted_to_stream(onof, beta, alpha):
    return np.vstack([0.15 - onof, -beta, alpha*1000]).T

