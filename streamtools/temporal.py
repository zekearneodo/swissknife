import numpy as np
import peakutils as pk

import logging

from h5tools import tables as h5t
from matplotlib import pyplot as plt

# TODO:  Make DatSound a subclass of WavData
#        Save chunk as wav file (using wave)

logger = logging.getLogger("streamtools.temporal")


def find_spikes(arr, thresholds, min_dist=10):
    """
    find spike events in an array
    :param arr: numpy arran
    :param thresholds:
    :param min_dist:
    :return:
    """
    chans = np.arange(thresholds.shape[0])
    ranges = np.ptp(arr, axis=0)
    maxima = np.max(arr, axis=0)

    all_indexes = []
    for ch, t, m, r in zip(chans,
                           thresholds,
                           maxima,
                           ranges):
        all_indexes.append(pk.indexes(-arr[:, ch],
                                      thres=(t + m) / r,
                                      min_dist=min_dist))

    return all_indexes
