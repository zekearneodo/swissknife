# Objects and helper functions to do stuff with sound
# The logic of using wave is to be able to page through long files.
import numpy as np
import wave
import struct
import matplotlib.pyplot as plt
import scipy.stats as ss
import scipy.signal as sg
from scipy.signal import spectrogram
from scipy.stats import pearsonr
from scipy.signal import hilbert
import pandas as pd
import tf_cluster as tfc
import logging
import os
# TODO:  Make DatSound a subclass of WavData
#        Save chunk as wav file (using wave)

logger = logging.getLogger('swissknife.bci.streamtools.soundtools')


def rms(x):
    return np.linalg.norm(x)


def get_rms_threshold(full_sound, window_size_samples, rms_threshold_factor):
    all_starts = np.arange(0, full_sound.n_samples - window_size_samples, window_size_samples)
    all_rms = full_sound.apply_repeated(all_starts, window_size_samples, rms)
    thresh_rms = np.mean(all_rms) + rms_threshold_factor * np.std(all_rms)
    return thresh_rms


def scalar_correlation(x, y):
    return ss.pearsonr(x, y)[0]


def envelope_correlation(x, template_env, window=200, rms_thresh=None):
    if x.ndim > 1:
        x = x[0, :]
    if (rms_thresh is not None) and rms(x) < rms_thresh:
        scalar_corr = 0.
    else:
        scalar_corr = ss.pearsonr(envelope(x, window=window), template_env)[0]
    return scalar_corr


def envelope(x, window=500, conv_mode='valid'):
    """
    :param x: one-d numpy array with data to get envelope
    :param window: number of samples for the smooothing window
    :return:
    """
    analytic_signal = sg.hilbert(x)
    amplitude_envelope = np.abs(analytic_signal)
    w = np.ones(window, 'd')
    return np.convolve(w / w.sum(), amplitude_envelope, mode=conv_mode)


def bandpass_filter(x, s_f, hi=400, lo=7000):
    """
    :param x: one d numpy arrays
    :param s_f: sampling frequency (in hz)
    :param hi: hi-pass cutoff
    :param lo: lo-pass cutoff
    :return:
    """
    hp_b, hp_a = sg.butter(4, hi / (s_f / 2.), btype='high')
    lp_b, lp_a = sg.butter(4, lo / (s_f / 2.), btype='low')
    x_hi = sg.filtfilt(hp_b, hp_a, x, axis=0)
    x_lo = sg.filtfilt(lp_b, lp_a, x_hi, axis=0)
    return x_lo


def get_redundant_peaks(x, peaks_ind, min_peak_distance):
    """
    :param x: vector where the peaks were selected from
    :param peaks_ind: indices of the found peaks
    :param min_peak_distance: minimum distance between peaks
    :return:
    """
    closer = np.where(np.diff(peaks_ind) < min_peak_distance)[0]
    degenerates = consecutive(closer)
    redundant_peaks_ind = np.array([], dtype=np.int)

    for degenerate in degenerates:
        multiple_peaks_ind = peaks_ind[degenerate[0]:degenerate[-1] + 2]
        abs_peak_ind = multiple_peaks_ind[np.argmax(x[multiple_peaks_ind])]
        redundant_peaks_ind = np.append(redundant_peaks_ind,
                                        multiple_peaks_ind[multiple_peaks_ind != abs_peak_ind])
    return redundant_peaks_ind


def filter_peaks_chunked(x, peaks_ind, min_peak_distance):
    redundant_peaks_ind = get_redundant_peaks(x, peaks_ind, min_peak_distance)
    return np.delete(peaks_ind, np.searchsorted(peaks_ind, redundant_peaks_ind))


def consecutive(x, stepsize=1):
    return np.split(x, np.where(np.diff(x) != stepsize)[0] + 1)


def filter_peaks_ranking(x, peaks_ind, min_peak_distance):
    """
    :param x: vector where the peaks were selected from
    :param peaks_ind: indices of the found peaks
    :param min_peak_distance: minimum distance between peaks
    :return: list of peak positions separated more than min_peak_distance apart,
            sorted in descending order according to the value of x at each position
    """
    ranked_peaks_ind = peaks_ind[np.argsort(x[peaks_ind])[::-1]]
    standing_peaks = np.array([], np.int)

    while ranked_peaks_ind.size > 0:
        p_0 = ranked_peaks_ind[0]
        standing_peaks = np.append(standing_peaks, p_0)
        ranked_peaks_ind = np.delete(ranked_peaks_ind,
                                     np.where((ranked_peaks_ind >= p_0) &
                                              (ranked_peaks_ind < (p_0 + min_peak_distance))
                                              )
                                     )
    return standing_peaks


def find_peaks(x, threshold=0., min_peak_distance=None, filter_method='ranked'):
    """
    :param x: vector
    :param threshold: peaks higher than this value
    :param min_peak_distance: minimum distance between consecutive peaks
    :param filter_method: function to use to filter out neighboring peaks:
            'ranked': makes a ranking of the values at the peaks and recursively clears a window
            of min_peak_distance after each highest value.
            'chunked': gets clusters of peaks closer than min_peak_distance and picks the single highest one.
    :return: index of the positions of the peaks.
    """

    logger.info('Finding peaks ...')
    filter_methods = {'ranked': filter_peaks_ranking,
                      'chunked': filter_peaks_chunked}

    # find the peaks naively
    a = x[1:-1] - x[2:]
    b = x[1:-1] - x[:-2]
    c = x[1:-1]
    max_pos = np.where((a > 0) & (b > 0) & (c > threshold))[0]
    logger.info('{} peaks found'.format(max_pos.size))
    max_pos = max_pos if min_peak_distance is None else filter_methods[filter_method](x, max_pos, min_peak_distance)
    logger.info('{} peaks left after filtering redundant'.format(max_pos.size))
    return max_pos


def compare_spectrogram(sxx, pattern_sxx):
    return np.sum(np.abs(pattern_sxx - sxx) / np.sum(np.abs(sxx)))


def spectrogram_rms(x, y):
    rms = np.linalg.norm(x-y)/np.sqrt(x.size)
    return rms

def spectrogram_diff(x, pattern_sxx, s_f, spectrogram_kwargs):
    """
    Get the difference between the spectrogram of a vector and a reference spectrogram
    :param x: vector
    :param pattern_sxx: spectrogram (as in the output of spectrogram(y), where y is the same length as x
    :param s_f: sampling rate
    :param spectrogram_kwargs: arguments for function spectrogram (the same as were used to get spectrogram pattern_sxx
    :return: a scalar comparison of the spectrograms.
    """
    f, t, sxx = spectrogram(x, s_f, **spectrogram_kwargs)
    assert (sxx.shape == pattern_sxx.shape)
    return compare_spectrogram(np.log(sxx), np.log(pattern_sxx))


def match_by_envelope(full_sound, pattern, window_size=500, corr_thresh=0.85, rms_threshold=None, window_step=1,
                      rolling_method='pandas'):
    """
    :param full_sound: Sound with the full sound
    :param pattern: Chunk with the pattern to find
    :param window_size: size of window for the envelope
    :param corr_thresh: threshold for peaks of correlation
    :param rms_threshold: threshold for rms (in standard deviations)
    :param window_step: sub_sampling rate for the rolling window (i.e a window every window_step samples)
    :return:
    """
    logger.info('Matching envelope across sound with {0} samples'.format(full_sound.n_samples))
    logger.info('Pattern has {} samples'.format(pattern.samples))

    if rolling_method == 'no_pandas':
        logger.info('Will use function find_envelope_sound')
        find_envelope = find_envelope_sound
        sound_search = full_sound
    elif rolling_method == 'pandas':
        if window_step > 1:
            raise ValueError('Can\'t use rolling_pandas with window_step bigger than 1')
        logger.info('Will use function find_envelope_pandas')
        find_envelope = find_envelope_pandas
        sound_search = Chunk(full_sound).data[:, 0]

    if rms_threshold is not None:
        logger.info('Getting rms threshold with {} std'.format(rms_threshold))
        min_rms = get_rms_threshold(full_sound, pattern.samples, rms_threshold)
        logger.info('Correlation threshold set to {}'.format(min_rms))
    else:
        logger.info('No rms threshold')
        min_rms = None

    match_points = find_envelope(sound_search, (pattern.data[:, 0]),
                                 corr_thresh=corr_thresh,
                                 window_size=window_size,
                                 min_rms=min_rms,
                                 window_step=window_step)
    logger.debug('Returning match_points')
    return match_points


def find_envelope_sound(full_sound, pattern, corr_thresh=0., window_size=500, min_rms=None, window_step=1):
    """
    :param full_sound: DatSoundCompat object with the whole stream for searching
    :param pattern: 0xN array
    :param corr_thresh:
    :param window_size:
    :param min_rms: minimum value of rms for the window to compute envelope
    :param window_step: step of the rolling window searching for the pattern
    :return:
    """
    logger.info('Env has {} samples'.format(full_sound.n_samples))
    logger.info('Rolling correlation with envelope...')
    pattern_env = envelope(pattern, window=window_size)
    logger.info(pattern_env.shape)

    all_starts = np.arange(0, full_sound.n_samples - pattern.size, window_step)
    all_corr = full_sound.apply_repeated(all_starts, pattern.size,
                                         envelope_correlation,
                                         pattern_env,
                                         window_size, min_rms)

    logger.info('Found {} segments above correlation threshold'.format(all_corr.size))
    # find the peaks semi-naively
    max_pos = find_peaks(all_corr,
                         threshold=corr_thresh,
                         min_peak_distance=int(pattern.size / window_step),
                         filter_method='ranked')

    logger.info('Found {0} peaks'.format(max_pos.shape[0]))
    return {'corr': all_corr, 'peaks': max_pos}


def find_envelope_pandas(stream, pattern, corr_thresh=0., window_size=200, min_rms=None, window_step=None):
    """
    :param stream: 0xN array
    :param pattern: 0xN array
    :param corr_thresh:
    :param window_size:
    :param min_rms: minimum value of rms for the window to compute envelope
    :param window_step: unused for now
    :return:
    """
    logger.info('Env has {} samples'.format(pattern.size))
    logger.info('Rolling correlation with envelope...')
    stream_df = pd.DataFrame(stream)
    pattern_env = envelope(pattern, window=window_size)
    logger.info(pattern_env.shape)
    rc = stream_df.rolling(pattern.size).apply(envelope_correlation, args=(pattern_env,),
                                               kwargs={'window': window_size,
                                                       'rms_thresh': min_rms})
    logger.info('Found {} segments above correlation threshold'.format(len(rc.index)))
    # find the peaks semi-naively
    max_pos = find_peaks(rc.values[:, 0],
                         threshold=corr_thresh,
                         min_peak_distance=pattern.size,
                         filter_method='ranked')

    logger.info('Found {0} peaks'.format(max_pos.shape[0]))
    del stream_df
    return {'corr': rc.values[:, 0], 'peaks': max_pos}


def spectral_score(chunk, pattern_sxx, spectrogram_kwargs):
    f, t, sxx = spectrogram(chunk.data, chunk.sound.s_f, **spectrogram_kwargs)
    assert (sxx.shape == pattern_sxx.shape)
    return compare_spectrogram(sxx, pattern_sxx)


def spectrogram_scores(pattern_chunk, chan_sound, candidates):
    s_f = chan_sound.s_f
    n_window = 256
    n_overlap = 192
    sigma = 1. / 1000. * s_f

    spectrogram_kwargs = {'nperseg': n_window,
                          'noverlap': n_overlap,
                          'window': sg.gaussian(n_window, sigma),
                          'scaling': 'spectrum'}

    pattern_spectrogram = spectrogram(pattern_chunk.data[:, 0], s_f, **spectrogram_kwargs)

    logger.info('Getting spectrogram difference score for {} candidates'.format(len(candidates.index)))
    for (i, start) in enumerate(candidates['start'][:]):
        logger.debug('Start {0}: {1}'.format(i, start))
        motif_start = start
        series = chan_sound.get_chunk(motif_start, motif_start + pattern_chunk.samples)
        # f, t, sxx = spectrogram(bandpass_filter(series[:, 0], s_f), s_f, **spectrogram_kwargs)

        candidates.set_value(i, 'spectral_diff',
                             spectrogram_diff(series[:, 0],
                                              pattern_spectrogram[2],
                                              s_f,
                                              spectrogram_kwargs)
                             )
        # plt.figure()
        # plt.pcolormesh(t, f[f < f_cut], np.log(sxx[f < f_cut, :]))


def correct_candidates_offset(candidates, offset):
    for i in range(candidates.index.size):
        candidates.set_value(i, 'start', candidates.start[i] - offset)


def cluster_spectrograms(candidates):
    logger.info('Attempting to cluster bout candidates by spectrogram similarity')
    diff_vector = candidates['spectral_diff'].values.reshape([candidates['spectral_diff'].values.shape[0], 1])
    clustered = tfc.means_cluster(diff_vector, 2)
    candidates['bout'] = pd.Series((clustered[1] == np.argmin(clustered[0])))


def fix_offset_issue(candidates, chan_sound, pattern_chunk):
    correct_candidates_offset(candidates, pattern_chunk.samples)
    spectrogram_scores(pattern_chunk, chan_sound, candidates)
    cluster_spectrograms(candidates)


def find_happy_song(chan_sound, pattern_chunk, search_pars, cand_file_path=None, cand_grp=None):
    """
    :param chan_sound: DatSoundCompat object with the whole sound where to search
    :param pattern_chunk: Chunk object with the pattern to search for
    :param search_pars: a dictionary with the parameters for the search. Has to have
            window_env: window for the envelope
            rms_threshold: rms_threshold (deviations from the mean)
            onset_resolution: search every n windows
            rolling_method: use pandas or no_pandas (apply_repeated method of chan_sound)
    :param cand_file_path: path to write the candidate files (.mot.h5 fle) for later curation
    :param cand_grp: group to make
    :return:
    """
    window_env = search_pars['window_env']
    if cand_file_path is not None:
        logger.debug('Will save candidates pandas df channel in file {}'.format(os.path.split(cand_file_path)[-1]))
        logger.debug('Will save them in group {}'.format(cand_grp))

    onset_resolution = search_pars['onset_resolution']
    rolling_method = search_pars['rolling_method']
    rms_thresh = search_pars['rms_threshold']

    logger.info('Begin to search')
    envelope_matches = match_by_envelope(chan_sound, pattern_chunk,
                                         window_size=window_env,
                                         rms_threshold=rms_thresh,
                                         window_step=onset_resolution,
                                         rolling_method=rolling_method)

    offset = pattern_chunk.samples if not rolling_method != 'pandas' else 0
    candidates = pd.DataFrame({'start': (envelope_matches['peaks'] + 1) * onset_resolution - offset,
                               'env_corr': envelope_matches['corr'][envelope_matches['peaks']],
                               'spectral_diff': np.zeros_like(envelope_matches['peaks'], dtype=np.float)})

    n_candidates = len(candidates.index)
    logger.info('Found {0} candidates'.format(n_candidates))

    if n_candidates > 0:
        logger.info('Getting spectrogram scores')
        spectrogram_scores(pattern_chunk, chan_sound, candidates)
        if n_candidates > 2:
            logger.info('Clustering candidates')
            cluster_spectrograms(candidates)
    else:
        pass

    if cand_file_path is not None:
        logger.info('Saving candidates pandas df channel in file {}'.format(os.path.split(cand_file_path)[-1]))
        try:
            cand_file = pd.HDFStore(cand_file_path)
            cand_grp = 'candidates' if cand_grp is None else cand_grp
            cand_file[cand_grp] = candidates
            cand_file.close()
            logger.info('done')
        except:
            logger.warn('could not save')

    logger.debug('Returning clustered candidates')
    return candidates
