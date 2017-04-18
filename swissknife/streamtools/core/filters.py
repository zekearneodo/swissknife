import numpy as np
import scipy.signal as sg


class Filter:
    def __init__(self, *filter_args, **filter_kwargs):
        self.filter_fun = sg.filtfilt
        self.args = filter_args
        self.kwargs = filter_kwargs

    def do_filter(self, data, **kwargs):
        return apply_filter(data, self.filter_fun, *self.args, **kwargs)


def apply_filter(data, filter_fun,  filter_b, filter_a, **kwargs):
    y = filter_fun(filter_b, filter_a, data, **kwargs)
    return y


def make_butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sg.butter(order, [low, high], btype='band')
    return b, a