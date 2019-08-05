from __future__ import division
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import math
from numba import jit

# fucntions for handling and plotting
def decim(x, q):
    # decimate a 1 x n array
    # x: 1xn matrix (float)
    # q: int (decimate ratio), 0<q<=x.size
    assert (x.size >= q and q > 0)
    pad_size = int(math.ceil(float(x.size) / q) * q - x.size)
    pad = np.empty(pad_size)
    pad[:] = np.nan
    x_padded = np.append(x, pad)
    return sp.nanmean(x_padded.reshape(-1, q), axis=1)


# bin the array in columns
def col_binned(a, bs):
    # a: rectangular array shape (n, m)
    # bs: size for bin columns
    # output : array (n, o=m//bs)
    # if m<o*bs, those columns are padded with zeros

    n = a.shape[0]
    m = a.shape[1]
    o = np.int(np.ceil(m / bs))

    pad = np.empty([n, o * bs - m]) * np.nan
    padded = np.append(a, pad, axis=1)
    return np.nansum(padded.reshape(n, o, bs), axis=2)

# bin the array in columns
def col_binned_max(a, bs):
    # a: rectangular array shape (n, m)
    # bs: size for bin columns
    # output : array (n, o=m//bs)
    # if m<o*bs, those columns are padded with zeros

    n = a.shape[0]
    m = a.shape[1]
    o = np.int(np.ceil(m / bs))

    pad = np.empty([n, o * bs - m]) * np.nan
    padded = np.append(a, pad, axis=1)
    return np.max(padded.reshape(n, o, bs), axis=2)

def plot_raster(x, t1=0, t2=-1, t0=0, ax=None, bin_size=0):
    # plot a raster
    # x: spikes matrix:
    # nxt matrix with 1 where there is a spikes.
    # cols: time stamps (ms)
    # rows: trials

    # t1 from beggining of x to plot: default 0, dont cut
    # t2 time after begginning of x to plot: default -1, all range
    # t0 where to put the 0 (stimulus mark) relative to the range t1:t2
    # ax: axes object where to put the plot in (default = None, create a new one)
    # bin_size: int

    # Returns:
    # raster: a PathCollection (if bin_size=0) or a Line2D object (if bin_size=1)
    # ax    : Axes object

    # prepare the axis
    # if no axis, make a new plot
    if ax is None:
        raster_fig = plt.figure()
        ax = raster_fig.add_axes([0, 0, 1, 1])

    # pdb.set_trace()
    # if bin_size was entered, we want a psth
    if bin_size > 0:
        psth, t_dec = make_psth(x, t1=t1, t2=t2, t0=t0, bin_size=bin_size)
        raster = ax.plot(t_dec, psth, color='C5')
        ax.set_ylim(0, max(psth) * 1.2)
        #stim = ax.plot((0, 0), (0, max(psth) * 1.2), 'k--')
        #ax.axvline(x=0, color='C6', linestyle=':')
        t_max = max(t_dec)
        ax.set_ylabel('F.R. (Hz)')
        ax.yaxis.set_ticks([int(max(psth)*0.8)])

    else:
        # Chop the segment
        if t2 > 0:
            assert (t2 > t1)
            x = x[:, t1:t2]
        else:
            x = x[:, t1:]

        # get dimensions and time
        events = x.shape[0]
        t_stamps = x.shape[1]
        t = np.arange(t_stamps) - t0

        # mask the zeros (no spike)
        nsp = x[:] == 0
        # x[nsp]=np.nan

        # make the frame for plotting
        row = np.ones(t_stamps, dtype=np.float)
        col = np.arange(events, dtype=np.float)
        frame = col[:, np.newaxis] + row[np.newaxis, :]

        raster = ax.scatter(t * x, frame * x, marker='|', linewidth=0.2, 
                            rasterized=True, color='C3')
        ax.set_ylim(0, events + 1)
        #ax.plot((0, 0), (0, events + 1), 'k--')
        t_max = t_stamps - t0
        ax.set_ylabel('trial')
        ax.yaxis.set_ticks([events - 1])

    ax.set_xlim(0 - t0, t_max)
    return raster, ax


# make a psth from a spikes matrix
def make_psth(x, t1=0, t2=-1, t0=0, bin_size=1):
    # x: spikes matrix:
    # nxt matrix with 1 where there is a spikes.
    # cols: time stamps (ms)
    # rows: trials

    # t1 from beginning of x: default 0, dont cut
    # t2 time after beginning of x to cut: default -1, all range
    # bin_size: int

    # Returns:
    # psth: an array with the frequency (counts/(bin_size*n_trials))

    # Chop the segment
    if t2 > 0:
        assert (t2 > t1)
        x = x[:, t1:t2]
    else:
        x = x[:, t1:]

    # get dimensions and time
    events = x.shape[0]
    t_stamps = x.shape[1]
    t = np.arange(t_stamps) - t0

    # pdb.set_trace()
    # if bin_size was entered, we want a psth
    # x = x[:t_stamps, :]

    t_dec = decim(t, bin_size)
    n_bins = t_dec.shape[0]
    pad_size = n_bins * bin_size - x.shape[1]
    pad = np.zeros(pad_size, dtype=np.int)

    psth = np.sum(np.append(pad, np.sum(x, axis=0)).reshape(n_bins, bin_size), axis=1) / (events * bin_size * 0.001)
    return psth, t_dec


# grab a raster in format row=timestamps, col=trials
# and turn it into a matrix n_trials x t_samples with a one wherever there is a spike
def sparse_raster(x, nan=False):
    n_t = x.shape[0] # n of trials
    # n_s = x.shape[1] # n of samples
    raster = np.empty_like(x)
    raster[:] = np.nan

    for trial in np.arange(n_t):
        r = x[trial, :]-1
        raster[trial, np.array(r[~np.isnan(r)], dtype=np.int)] = 1

    if not nan:
        raster[np.isnan(raster)] = 0
    return raster

@jit(nopython=True)
def plottable_array(x:np.ndarray, scale:np.ndarray, offset:np.ndarray) -> np.ndarray:
    """ Rescale and offset an array for quick plotting multiple channels, along the 
        1 axis, for each jth axis
    Arguments:
        x {np.ndarray} -- [n_col x n_row] array (each col is a chan, for instance)
        scale {np.ndarray} -- [n_col] vector of scales (typically the ptp values of each row)
        offset {np.ndarray} -- [n_col] vector offsets (typycally range (row))

    Returns:
        np.ndarray -- [n_row x n_col] scaled, offsetted array to plot
    """
    # for each row [i]:
    # - divide by scale_i
    # - add offset_i
    n_row, n_col = x.shape
    for col in range(n_col):
        for row in range(n_row):
            x[row, col] = x[row, col] * scale[col] + offset[col]
    return x