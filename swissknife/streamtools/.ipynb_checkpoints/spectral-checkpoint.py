# from mock.mock import self
from scipy.signal import spectrogram
import scipy.signal as sg
import matplotlib.pyplot as plt
import numpy as np
import copy
import tensorflow as tf

from swissknife.streamtools.core.data import overlap
from swissknife.streamtools.core import rosa
from tqdm import tqdm

import logging

logger = logging.getLogger('h5tools')

class Spectrogram(object):
    def __init__(self, x, s_f,
                 n_window=192,
                 n_overlap=None,
                 sigma=None,
                 ax=None,
                 f_cut=10000,
                 db_cut=0.,
                 **kwargs):
        self.x = x
        self.before_ms = None
        self.after_ms = None
        self.s_f = s_f
        self.n_window = n_window
        self.n_overlap = int(n_window / 2) if n_overlap is None else n_overlap
        self.sigma = .001 * s_f if n_overlap is None else sigma
        self.ax = None
        self.f_cut = f_cut
        self.db_cut = db_cut

        self.f, self.t, self.Sxx = spectrogram(x, s_f,
                                               nperseg=n_window,
                                               noverlap=self.n_overlap,
                                               window=sg.gaussian(n_window, self.sigma),
                                               scaling='spectrum')

        self.t *= 1000.

        self.Sxx[[self.Sxx < np.max(self.Sxx * db_cut)]] = 1

    def __add__(self, other):
        assert (self.Sxx.shape == other.Sxx.shape)
        new_spec = copy.copy(self)

        new_spec.x = np.mean(np.array([self.x, other.x]), axis=0)
        new_spec.Sxx = np.mean(np.array([self.Sxx, other.Sxx]), axis=0)

        return new_spec

    def __radd__(self, other):
        return self + other

    def plot(self, before_ms=0, after_ms=None, ax=None, log_f=False, cmap='inferno', **kwargs):
        self.before_ms = before_ms
        self.after_ms = np.max(self.t, after_ms)

        self.ax = plot_spectrum(self.t, self.f[self.f < self.f_cut], self.Sxx[self.f < self.f_cut, :],
                                before_ms=self.before_ms,
                                after_ms=self.after_ms,
                                ax=ax,
                                f_cut=self.f_cut,
                                log_f=log_f,
                                cmap=cmap,
                                **kwargs)
        # self.ax.set_ylim(0, self.f_cut*1.1)
        return self.ax


def plot_spectrum(t, f, Sxx, before_ms=0, after_ms=None, ax=None, f_cut=10000, log_f=False, **kwargs):
    after_ms = np.max(t) if after_ms is None else after_ms
    print(after_ms)
    if ax is None:
        spec_fig = plt.figure()
        ax = spec_fig.add_axes([0, 0, 1, 1])

    f_plot = np.log(f) if log_f else f
    f_cut_plot = np.log(f_cut) if log_f else f_cut
    print('fcut_plot {}'.format(f_cut_plot))
    ax.pcolormesh((t - before_ms), f_plot[f_plot < f_cut_plot], np.log(Sxx[f_plot < f_cut_plot, :]), rasterized=True,
                  **kwargs)
    ax.set_xlim(-before_ms, after_ms - before_ms)
    ax.set_ylim(0, f_cut_plot)

    return ax


def plot_spectrogram(x, before_ms, after_ms, s_f, n_window=192, n_overlap=None, sigma=None, ax=None, **kwargs):
    if n_overlap is None:
        n_overlap = n_window / 2
    if sigma is None:
        sigma = 1. / 1000. * s_f

    # Make the spectrogram
    f, t, Sxx = spectrogram(x, s_f, nperseg=n_window, noverlap=n_overlap, window=sg.gaussian(n_window, sigma),
                            scaling='spectrum')

    Sxx[[Sxx < np.max((Sxx) * 0.000065)]] = 1

    span_before = np.zeros((Sxx.shape[0], np.int(before_ms / 1000. * s_f)))
    span_after = np.zeros((Sxx.shape[0], np.int(after_ms / 1000. * s_f) + x.size - Sxx.shape[1]))
    span_before[:] = np.nan
    span_after[:] = np.nan
    # Sxx = np.hstack((span_before, (Sxx), span_after))

    if ax is None:
        spec_fig = plt.figure()
        ax = spec_fig.add_axes([0, 0, 1, 1])

    ax.pcolormesh(((t - 0.5 * n_window / s_f) * 1000.), f, np.log(Sxx), rasterized=True, cmap='inferno')
    ax.set_xlim(-before_ms, after_ms + int(x.size / s_f * 1000.))
    ax.set_ylim(0, 10000)
    # ax.plot((span_before.shape[1], span_before.shape[1]), (np.min(f), np.max(f)), 'k--')

    return Sxx, ax


def make_butter_bandpass(s_f, lo_cut, hi_cut, order=4):
    hp_b, hp_a = sg.butter(order, lo_cut / (s_f / 2.), btype='high')
    lp_b, lp_a = sg.butter(order, hi_cut / (s_f / 2.), btype='low')
    return {'lo_cut': lo_cut,
            'hi_cut': hi_cut,
            'hp_b': hp_b,
            'hp_a': hp_a,
            'lp_b': lp_b,
            'lp_a': lp_a}


def apply_butter_bandpass(x, pars):
    x_hi = sg.filtfilt(pars['hp_b'], pars['hp_a'], x, axis=0)
    x_bp = sg.filtfilt(pars['lp_b'], pars['lp_a'], x_hi, axis=0)
    return x_bp


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    Parameters
    ----------
    X : ndarray, shape=(n_samples,)
        Input signal to window and overlap
    window_size : int
        Size of windows to take
    window_step : int
        Step size between windows
    Returns
    -------
    X_strided : shape=(n_windows, window_size)
        2D array of overlapped X
    """
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    x = np.hstack((X, append))

    valid = len(x) - window_size
    nw = valid // window_step
    out = np.ndarray((nw, window_size), dtype=x.dtype)

    for i in xrange(nw):
        # "slide" the window along the samples
        start = i * window_step
        stop = start + window_size
        out[i] = x[start: stop]
    return out

# functions using tensorflow
def real_fft(x, only_abs=True, logarithmic=False, window=None):
    """
    Computes fft of a stack of time series.
    :param x: ndarray, shape=(n_series, n_samples)
            Input stack of time series
    :param only_abs: boolean
            return only absolute values (power spectra)
    :param logarithmic: boolean
            return in logarithmic scale; ignored if only_abs==False
    :param window: ndarray, shape=(n_samples, )
            boxing window (default none), np.array (n_samples)
    :return: ndarray, shape=(n_series, n_samples/2)
            with fft computed for every row
             dtype=np.float32 (if only_abs==True) or dtype=np.complex64 (if only_abs==False)
    """
    n_series, n_samples = x.shape

    tensor_x = tf.Variable(x, dtype=tf.float32)
    if window is not None:
        assert (window.size == n_samples), 'Window must be size n_samples'
        vector_win = tf.Variable(window.flatten(), dtype=tf.float32)
        tensor_win = tf.reshape(tf.tile(vector_win, [n_series]), [n_series, n_samples])
        real_x = tf.multiply(tensor_x, tensor_win)
    else:
        real_x = tensor_x
    img_x = tf.Variable(np.zeros_like(x), dtype=tf.float32)
    complex_x = tf.complex(real_x, img_x)
    complex_y = tf.fft(complex_x)[:, :int(x.shape[-1] / 2)]

    if only_abs:
        amps_y = tf.abs(complex_y)
        if logarithmic:
            log_10 = tf.constant(np.log(10), dtype=tf.float32)
            fft = tf.multiply(tf.log(amps_y), log_10)
        else:
            fft = amps_y
    else:
        fft = complex_y

    model = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(model)
        s = sess.run(fft)
    return s

def pretty_spectrogram_tf(X, log=True, db_cut=65, fft_size=512, step_size=64,
                          window=None):
    """
    creates a spectrogram of the 1d time series X using tensorflow to compute the fft
    :param x: ndarray, shape=(n_samples, ) Input time series
    :param db_cut: float, threshold (in dB, relative to total max of the whole spectrogram)
    :param fft_size: int, (samples) size of the window to use for every spectral slice
    :param step_size: int, (samples) stride for the spectral slices
    :param log: boolean, whether to take the log
    :param window: ndarray, shape=(fft_size, ). If entered, the window function for the fft.
                   must be an array of the same size of fft_size
    :return: ndarray, shape=(n_steps, fft_size/2) with the spectrogram
    """
    x = overlap(X, fft_size, step_size)
    specgram = real_fft(x, only_abs=True, logarithmic=log, window=window)
    max_specgram = np.max(specgram)

    # do the cut_off. Values are amplitude, not power, hence db = -20*log(V/V_0)
    if log:
        # threshold = pow(10, max_specgram) * pow(10, -db_cut*0.05)
        # specgram[specgram < np.log10(threshold)] = np.log10(threshold)
        log10_threshhold = max_specgram - db_cut * 0.05
        specgram[specgram < log10_threshhold] = log10_threshhold
        # specgram /= specgram.max()  # volume normalize to max 1
    else:
        threshold = max_specgram * pow(10, -db_cut * 0.05)
        specgram[specgram < threshold] = threshold  # set anything less than the threshold as the threshold
    return specgram

def pretty_spectrogram(x, s_f, log=True, fft_size=512, step_size=64, window=None,
                       db_cut=65,
                       f_min=0.,
                       f_max=None):
    # db_cut=0 for no_trhesholding
    f, t, specgram = sg.spectrogram(x, fs=s_f, window=window,
                                       nperseg=fft_size,
                                       noverlap=fft_size - step_size,
                                       nfft=None,
                                       detrend='constant',
                                       return_onesided=True,
                                       scaling='spectrum',
                                       axis=-1)
                                       #mode='psd')

    if db_cut>0:
        specgram = spectrogram_db_cut(specgram, db_cut=db_cut, log_scale=False)
    if log:
        specgram = np.log10(specgram)

    if f_max is None:
        f_max = s_f/2.

    f_filter = np.where((f >= f_min) & (f < f_max))
    #return f, t, specgram
    return f[f_filter], t, specgram[f_filter]


def chunky_spectrogram(x, s_f, log=True, fft_size=256, step_size=64, window=None,
                       db_cut=65,
                       f_min=0.,
                       f_max=None,
                       chunk_size = 40):

    # chunk is n. of fft_sizes
    logger.info('Getting chunked spectrogram, total size {}'.format(x.size))
    if np.mod(fft_size, step_size) > 0:
        raise ValueError('fft_size has to be multiple of step_size for chunked spectrogram')
    logger.debug('Input size {}'.format(x.size))
    chunk_size = np.min([chunk_size * fft_size * step_size, x.size])
    #print('Chunk size {}'.format(chunk_size))
    chunk_size_samples = fft_size * chunk_size
    chunk_starts = np.arange(0, x.size, chunk_size_samples)
    #print('Chunk starts {}'.format(chunk_starts))

    d_t = step_size/s_f

    f_nyq = s_f/2
    d_f = s_f/fft_size
    all_f = np.arange(0, f_nyq, d_f)
    f_filter = np.where((all_f >= f_min) & (all_f < f_max))[0]
    select_f = all_f[f_filter]

    chunk_size_steps = int(np.floor( (chunk_size_samples - fft_size)/step_size + 1 )) #the output of spectrogram at each chunk
    logger.debug('Chunk size steps {}'.format(chunk_size_steps))
    spec_chunks_starts = np.arange(chunk_starts.size)*chunk_size_steps

    total_steps = int(np.floor( (x.size - fft_size)/step_size + 1 ))
    logger.debug('total_steps {}'.format(total_steps))
    all_specgram = np.zeros([select_f.size, total_steps])
    all_specgram[:] = np.nan
    all_t = np.arange(total_steps) * d_t
    #ctrl_t = np.zeros(total_steps)
    #ctrl_t[:] = np.nan
    logger.debug('All_spec {}'.format(all_specgram.shape))
    #print('All_t {}'.format(all_t.shape))
    out_sizes = []
    # db_cut=0 for no_trhesholding
    t_prev = 0
    for i_chunk, (i_sample, i_step) in tqdm(enumerate(zip(chunk_starts, spec_chunks_starts)),
                                 total=chunk_starts.size):
        #if i_sample > 0:
        i_sample -= (fft_size - step_size)*i_chunk

        x_t = x[i_sample: i_sample+chunk_size_samples]

        logger.debug('Enter chunk size {}'.format(x_t.shape))
        f, t, s = sg.spectrogram(x_t,
                                 fs=s_f, window=window,
                                           nperseg=fft_size,
                                           noverlap=fft_size - step_size,
                                           nfft=None,
                                           detrend='constant',
                                           return_onesided=True,
                                           scaling='spectrum',
                                           axis=-1)
        out_sizes.append(s.shape[1])
        logger.debug('Out size {}'.format(s.shape))
        logger.debug(i_step)
        logger.debug(chunk_size_steps)
        #all_specgram[:, i_step: i_step + s.shape ] = s[f_filter,:]
        all_specgram[:, i_step: i_step + chunk_size_steps] = s[f_filter,:]
        #ctrl_t[i_step: i_step + chunk_size_steps] = t + t_prev
        #t_prev += t[-1]

    if db_cut>0:
        all_specgram = spectrogram_db_cut(all_specgram, db_cut=db_cut, log_scale=False)
    if log:
        all_specgram = np.log10(all_specgram)

    if f_max is None:
        f_max = s_f/2.

    f_filter = np.where((f >= f_min) & (f < f_max))
    #return f, t, specgram
    return f[f_filter], all_t, all_specgram


def spectrogram_db_cut(s, db_cut=65, log_scale=False):
    specgram = np.copy(s)
    max_specgram = np.max(specgram)
    # do the cut_off. Values are amplitude, not power, hence db = -20*log(V/V_0)
    if log_scale:
        # threshold = pow(10, max_specgram) * pow(10, -db_cut*0.05)
        # specgram[specgram < np.log10(threshold)] = np.log10(threshold)
        log10_threshhold = max_specgram - db_cut * 0.05
        specgram[specgram < log10_threshhold] = log10_threshhold
        # specgram /= specgram.max()  # volume normalize to max 1
    else:
        threshold = max_specgram * pow(10, -db_cut * 0.05)
        specgram[specgram < threshold] = threshold  # set anything less than the threshold as the threshold
    return specgram

def spectrogram_rms(x, y):
    rms = np.linalg.norm(x-y)/np.sqrt(x.size)
    return rms


def array_spectrogram(x, pars, axis=-1):
    y = x.T if axis == 0 else x
    t_steps, n_ch = y.shape

    spec_tower = []
    for i_ch in range(n_ch):
        f, t, s = pretty_spectrogram(x[:, i_ch], pars['s_f'],
                                        fft_size=pars['win_samples'],
                                        log=True,
                                        step_size=pars['step_samples'], db_cut=pars['db_cut'],
                                        f_min=pars['f_min'], f_max=pars['f_max'],
                                        window=('gaussian', pars['win_samples'] // 4))

        spec_tower.append(s)
    return np.stack(spec_tower, axis=0)

def rosa_spectrogram(y, hparams):
  D = rosa._stft(rosa.preemphasis(y,hparams), hparams)
  S = rosa._amp_to_db(np.abs(D)) - hparams['ref_level_db']
  return rosa._normalize(S, hparams)


def inv_spectrogram(spectrogram, hparams):
  '''Converts spectrogram to waveform using librosa'''
  S = rosa._db_to_amp(rosa._denormalize(spectrogram, hparams) + hparams['ref_level_db'])  # Convert back to linear
  return rosa.inv_preemphasis(rosa._griffin_lim(S ** hparams['power'], hparams), hparams)          # Reconstruct phase