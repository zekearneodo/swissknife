import numpy as np


def overlap(X, window_size, window_step):
    """
    Create an overlapped version of X
    taken from Tim Sainburg or Marvin Theilk
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

    for i in range(nw):
        # "slide" the window along the samples
        start = i * window_step
        stop = start + window_size
        out[i] = x[start: stop]
    return out
