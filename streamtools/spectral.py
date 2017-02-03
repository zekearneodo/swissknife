from mock.mock import self
from scipy.signal import spectrogram
import scipy.signal as sg
import matplotlib.pyplot as plt
import numpy as np
import copy


class Spectrogram(object):
    def __init__(self, x, s_f,
                 n_window = 192,
                 n_overlap = None,
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
        self.n_overlap = int(n_window/2) if n_overlap is None else n_overlap
        self.sigma = .001 * s_f if n_overlap is None else sigma
        self.ax = None
        self.f_cut = f_cut
        self.db_cut = db_cut

        self.f, self.t, self.Sxx = spectrogram(x, s_f,
                                               nperseg=n_window,
                                               noverlap=self.n_overlap,
                                               window=sg.gaussian(n_window, self.sigma),
                                               scaling='spectrum')

        self.t*=1000.

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

        self.ax = plot_spectrum(self.t, self.f[self.f<self.f_cut], self.Sxx[self.f<self.f_cut, :],
                                before_ms=self.before_ms,
                                after_ms=self.after_ms,
                                ax=ax,
                                f_cut=self.f_cut,
                                log_f=log_f,
                                cmap=cmap,
                                **kwargs)
        #self.ax.set_ylim(0, self.f_cut*1.1)
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
    ax.pcolormesh((t-before_ms), f_plot[f_plot<f_cut_plot], np.log(Sxx[f_plot<f_cut_plot,:]), rasterized=True, **kwargs)
    ax.set_xlim(-before_ms, after_ms-before_ms)
    ax.set_ylim(0, f_cut_plot)

    return ax


def plot_spectrogram(x, before_ms, after_ms, s_f, n_window = 192, n_overlap = None, sigma=None, ax=None, **kwargs):

    if n_overlap is None:
        n_overlap = n_window/2
    if sigma is None:
        sigma = 1./1000.*s_f

    # Make the spectrogram
    f, t, Sxx = spectrogram(x, s_f, nperseg = n_window, noverlap=n_overlap, window = sg.gaussian(n_window, sigma),
                            scaling='spectrum')

    Sxx[[Sxx<np.max((Sxx)*0.000065)]]=1

    span_before = np.zeros((Sxx.shape[0], np.int(before_ms/1000. * s_f)))
    span_after = np.zeros((Sxx.shape[0], np.int(after_ms/1000. * s_f) + x.size - Sxx.shape[1]))
    span_before[:] = np.nan
    span_after[:] = np.nan
    #Sxx = np.hstack((span_before, (Sxx), span_after))

    if ax is None:
        spec_fig = plt.figure()
        ax = spec_fig.add_axes([0, 0, 1, 1])

    ax.pcolormesh(((t-0.5*n_window/s_f)*1000.), f, np.log(Sxx), rasterized=True, cmap='inferno')
    ax.set_xlim(-before_ms, after_ms + int(x.size/s_f * 1000.))
    ax.set_ylim(0,10000)
    #ax.plot((span_before.shape[1], span_before.shape[1]), (np.min(f), np.max(f)), 'k--')

    return Sxx, ax


def make_butter_bandpass(s_f, lo_cut, hi_cut, order=4):
    hp_b, hp_a = sg.butter(order, lo_cut/(s_f/2.), btype='high')
    lp_b, lp_a = sg.butter(order, hi_cut/(s_f/2.), btype='low')
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