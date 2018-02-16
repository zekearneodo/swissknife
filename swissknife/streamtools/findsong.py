import scipy.signal as sg
import numpy as np
from itertools import groupby
from operator import itemgetter
import wave
import struct
import logging
import os
import sys

from swissknife.streamtools.streams import WavData, Chunk

logger = logging.getLogger('findsong')


def rms(x):
    return np.std(x, axis=0)


def mad(x):
    med = np.median(x)
    dev = np.abs(x - np.median(x))
    return np.median(dev)


def band_pass_filter(chunk, hp_b, hp_a, lp_b, lp_a):
    chunk_hi = sg.filtfilt(hp_b, hp_a, chunk)
    chunk_filt = sg.filtfilt(lp_b, lp_a, chunk_hi)
    return chunk_filt


# decorator for getting the rms after applying a filter to a segment
def rms_after_filter(filter_func):
    def out_of_filtered(*args, **kwargs):
        # print "Arguments were: %s, %s" % (args, kwargs)
        # plt.plot(np.transpose(filter_func(*args, **kwargs)))
        return np.linalg.norm(filter_func(*args, **kwargs))

    return out_of_filtered


@rms_after_filter
def rms_band_pass_filter(chunk, hp_b, hp_a, lp_b, lp_a):
    # plt.plot(np.transpose(chunk))
    chunk_hi = sg.filtfilt(hp_b, hp_a, chunk)
    chunk_filt = sg.filtfilt(lp_b, lp_a, chunk_hi)
    return chunk_filt


@rms_after_filter
def rms_identity(chunk):
    return (chunk)


def get_bouts(all_sound, bit_size=70, refractory=5, bits_bout=4, threshold=20,
              filt_lo=1000, filt_hi=500):
    s_f = all_sound.s_f

    # make the filter
    hp_b, hp_a = sg.butter(4, filt_hi / (s_f / 2.), btype='high')
    lp_b, lp_a = sg.butter(4, filt_lo / (s_f / 2.), btype='low')

    bit_size_sample = np.int(bit_size * all_sound.s_f * 0.001)
    all_starts = np.arange(0, all_sound.n_samples - bit_size_sample, bit_size_sample)

    # get all the rms
    all_rms = all_sound.apply_repeated(all_starts, bit_size_sample, rms_band_pass_filter, hp_b, hp_a, lp_b, lp_a)

    thresh_rms = np.median(all_rms) + threshold * mad(all_rms)

    supra_bits = np.where(all_rms > thresh_rms)[0]
    # plt.plot(np.arange(supra_bits.size), supra_bits, '*')

    group_bound = np.array([0, 0], dtype=np.int32)

    for k, g in groupby(enumerate(supra_bits), lambda i_x: i_x[1] - i_x[0]):
        group = list(map(itemgetter(1), g))
        if len(group) >= bits_bout:
            group_bound = np.vstack((group_bound, np.array([group[0], group[-1]])))
            # print group

    if group_bound.size > 2:
        found_bouts = True
        group_bound = np.delete(group_bound, 0, axis=0)
        # segments has to be in samples
        segments = np.array([all_starts[group_bound[:, 0]], all_starts[group_bound[:, 1]]])
        assert (np.all(np.diff(segments, axis=0)[0] > 0))

    else:
        segments = None

    return segments


def write_segments(segments, all_sound, bout_file_path, bit_size, more=2, filt_lo=10000, filt_hi=350):
    bit_size_sample = np.int(bit_size * all_sound.s_f * 0.001)
    n_seg = segments.shape[1]
    chopped_segments = np.zeros_like(segments.T)
    # open the file
    logger.debug('Saving bouts in {}'.format(bout_file_path))
    bouts_out = wave.open(bout_file_path, 'w')
    bouts_out.setparams((all_sound.n_chans, 2, all_sound.s_f, 0, 'NONE', 'not compressed'))
    # make the filter
    hp_b, hp_a = sg.butter(4, filt_hi / (all_sound.s_f / 2.), btype='high')
    lp_b, lp_a = sg.butter(4, filt_lo / (all_sound.s_f / 2.), btype='low')
    try:
        for i_seg, seg in enumerate(segments.T):
            seg = seg + np.array([-more, more]) * bit_size_sample
            start = seg[0] if seg[0] > 0 else 0
            end = seg[1] if seg[1] < all_sound.n_samples else all_sound.n_samples
            chopped_segments[i_seg] = np.array([start, end])
            # print start, end
            sg_chunk = Chunk(all_sound, segment=[start, end])
            # save the chunk to a data file
            # filter for debugging
            sg_chunk.apply_filter(band_pass_filter, hp_b, hp_a, lp_b, lp_a)
            data_out = np.array(sg_chunk.data, dtype=np.int16)[0]
            packed_data = struct.pack('<' + str((end - start)) + 'h', *data_out)
            bouts_out.writeframes(data_out)
        bouts_out.close()

    except Exception as inst:
        print(inst)
        bouts_out.close()
        raise (inst)

    return chopped_segments


def get_all_bouts(raw_file_path, bout_file_path=None,
                  bit_size=50, refractory=5, bits_bout=2,
                  threshold=20, filt_lo=10000, filt_hi=300, more=2):
    logger.info('Will get bouts for file {}'.format(raw_file_path))
    # set bout and stamps file names
    if bout_file_path is None:
        raw_path, raw_fname = os.path.split(raw_file_path)
        bout_file_path = os.path.join(raw_path,
                                      '{}_auto.wav'.format(raw_fname.split('.')[0]))
        logger.debug('Auto assigned bout_file_name')

    # set the file_name for the stamps relative to original file
    bout_path, bout_fname = os.path.split(bout_file_path)
    stamp_file_path = os.path.join(bout_path,
                                   '{}.stamps.npy'.format(bout_fname.split('.')[0]))

    all_sound = WavData(raw_file_path)
    try:
        found_segments = get_bouts(all_sound, bit_size=bit_size, refractory=refractory, bits_bout=bits_bout,
                                   threshold=threshold, filt_lo=filt_lo, filt_hi=filt_hi)
    except:
        e = sys.exc_info()[0]
        logger.warning('Could not get bouts, maybe file is corrupted? Returning error')
        all_sound.close()
        return e

    if found_segments is None:
        logger.info('No segments found')

    else:
        logger.info('Writing segments to {}'.format(bout_file_path))
        chopped_segments = write_segments(found_segments, all_sound,
                                          bout_file_path, bit_size,
                                          more=more)
        logger.debug('Saving stamps to {}'.format(stamp_file_path))
        np.save(stamp_file_path, chopped_segments)
    logger.debug('Closing raw file')
    all_sound.close()
    return found_segments
