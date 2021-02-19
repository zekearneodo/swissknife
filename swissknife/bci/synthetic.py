import logging
import pickle 
import os

import numpy as np

from scipy.signal import resample
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

from swissknife.bci.core import expstruct as et

logger = logging.getLogger('swissknife.bci.synthetic')


def normalize(x):
    x_normed = (x - x.min(0)) / x.ptp(0)
    return x_normed


def running_diff(x, y):
    # y is the larger
    n_x = x.shape[1]
    n_y = y.shape[1]
    return np.array([np.sum(abs(x - y[:, i: i + n_x])) for i in range(n_y - n_x)])

def resample_nofourier(x, s_f, new_s_f, kind='linear'):
    n = int(np.ceil(x.size * new_s_f/s_f))
    f = interp1d(np.linspace(0, 1, x.size), x, kind)
    return f(np.linspace(0, 1, n))

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

def resample_int1d(x, s_f, new_s_f, interp_kind='linear'):
    t = np.linspace(0, x.size / s_f, x.size)
    new_n = np.int(x.size * new_s_f / s_f)
    new_t = np.linspace(0, new_n/new_s_f, new_n)

    f = interp1d(t, x, interp_kind)
    return f(new_t)

def latent_to_stream(onof, beta, alpha):
    onof[onof > .3] = .3
    onof[onof < 0] = 0
    alpha[alpha < 0] = 0
    return np.vstack([0.15 - onof, -beta, alpha]).T


def load_syn_stream(bird, syn_sess=1, s_f=44100, new_s_f=30000, file_base='synth_bos'):
    stim_folder = et.file_names(bird)['folders']['stim']
    syn_file = os.path.join(stim_folder, str(syn_sess).zfill(3), file_base + '.dat')
    return stream_resample(np.loadtxt(syn_file), s_f, new_s_f)

def load_syn_pickle(bird, syn_sess=1, new_s_f=30000):
    stim_folder = et.file_names(bird)['folders']['stim']
    syn_file_path = os.path.join(stim_folder, str(syn_sess).zfill(3), 'syn_template.p')
    with open(syn_file_path, 'rb') as pf:
        syn_dict = pickle.load(pf)
    
    # resample the three parameters and the time
    resample_keys = ['alpha_ms', 'beta_ms', 'env_ms', 't_fits_ms']

    resampled_dict = {k.split('_ms')[0]: resample_int1d(v, 1000, new_s_f) for (k, v) in syn_dict.items() if k in resample_keys}
    syn_dict.update(resampled_dict)
    return syn_dict

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

def stream_to_fitted(alpha, beta, env):
    alpha[alpha > 0] = 0
    alpha[alpha < 0] = 0.3

    beta = -beta
    beta = beta - np.min(beta)
    beta[beta<0] = 0

    env = env/np.max(env)
    env[env<0]=0

    return alpha, beta, env

def regularize_pars(x: np.array, beta_max: float=2.5, scale: float=10) -> np.array:
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

    # add a jitter to break 0s
    epsilon = 1e-3
    # werks with 1e-10
    jitter_arr = epsilon * (np.random.rand(3, env.shape[0]) - 0.5)

    return np.stack([alpha, beta, env]) * scale + jitter_arr


def regularize_pars_inv(y: np.array, beta_max: float=2.5, scale: float=10) -> np.array:
    '''
    Get an array of binned parameters alpha, beta env and transforms to:
    alpha: .15/-.15 (off/on)
    beta: -mulog_inv(beta)*beta_max
    env: mulog_inv(env)
    :param y: ndarray([n, 3]), array with ms sampled parameters in rows (alpha, beta, env)
    :param beta_max: divisor for beta column
    :return: ndarray([n, 3])
    '''
    y_normalized = y / scale
    # 3 cols: alpha, beta, env
    alpha = np.ones_like(y_normalized[0])*0.15
    alpha[y_normalized[0] > 0.5] = -.15

    beta = -np_mulog_inv(y_normalized[1]) * beta_max
    
    env = 1000*np_mulog_inv(y_normalized[2])
    env[env<0] = 1e-15
    
    return np.stack([alpha, beta, env])


def regularize_pars_z017(x, beta_max=0.1):
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
    beta = (-x[1] / beta_max)
    env = (x[2])/np.max(x[2])

    return np.stack([alpha, beta, env])

def regularize_pars_inv_z017(y, beta_max=0.1):
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
    alpha = np.ones_like(y[0])*0.15
    alpha[y[0] > 0.5] = -.15

    beta = (-y[1] * beta_max)
    env = (y[2])*1000
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

def pred_to_par_stream_z017(y: np.array, bin_size: int, s_f: int=44100) -> np.array:
    '''
    Get an array of binned parameters alpha, beta env and transforms to:
    alpha: .15/-.15 (off/on)
    beta: -beta
    env: env/np.max(env)|env[env<0]=0
    :param y: ndarray([n_bin, 3]), array with ms sampled parameters in rows (alpha, beta, env)
    :param beta_max: divisor for beta column
    :return: ndarray([n_bin*bin_size*s_f, 3]) array with parameters sampled at s_f in cols (alpha, beta, env)
    '''
    alpha = np.ones_like(y[:, 0])*0.15
    alpha[y[:, 0] > 0.3] = -.15

    beta = -y[:, 1]*2

    env = y[:, 2]/np.max(y[:, 2])*1000
    env[env < 0] = 0

    # resample (no fourier interpolation) to ms bins
    mapped_y_fast = map(lambda x: resample_nofourier(x, 1000/bin_size, s_f, kind='linear'), [alpha, beta, env])
    y_fast = np.vstack(list(mapped_y_fast)).T
    
    # interpolate parameters that are in bin chunks
    return y_fast