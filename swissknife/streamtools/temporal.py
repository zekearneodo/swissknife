import logging

import numpy as np
import peakutils as pk

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


def spikes_array(data, thresholds, min_dist=10):
    '''
    :param data: np. array [n_samples, n_chans]
    :param thresholds:
    :param min_dist:
    :return:
    '''
    # logger.info('Getting spikes from chunk with data sized {}'.format(chunk.data.shape))
    spk_lst = find_spikes(data, thresholds, min_dist=min_dist)
    spk_arr = np.zeros_like(data)
    assert (len(spk_lst) == spk_arr.shape[1])
    for ch in range(len(spk_lst)):
        spk_arr[spk_lst[ch], ch] = 1
    return spk_arr
