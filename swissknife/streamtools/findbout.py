import scipy.signal as sg
import numpy as np
import pandas as pd
import glob
from itertools import groupby
from operator import itemgetter
import wave
import struct
import logging
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
from swissknife.streamtools.streams import WavData, WavData2, Chunk
import swissknife.streamtools.spectral as sp
import swissknife.streamtools.streams as st
import scipy.stats as ss
from sklearn.cluster import KMeans

logger = logging.getLogger('swissknife.streamtools.findbout')


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

def cluster_spectrograms(candidates):
    logger.info('Attempting to cluster bout candidates by spectrogram similarity')
    diff_vector = np.array(candidates['spectral_diff'].tolist())
    k_means = KMeans(n_clusters=2).fit(diff_vector.reshape(-1, 1))
    assignments = k_means.labels_
    low_cluster = np.argmin(k_means.cluster_centers_)
    candidates['bout'] = pd.Series((assignments == low_cluster))
    return candidates

def cluster_spectrograms_2d(candidates):
    logger.info('Attempting to cluster bout candidates by spectrogram similarity')
    diff_vector = np.array([candidates['spectral_diff'].tolist(), candidates['env_corr'].tolist()]).T
    k_means = KMeans(n_clusters=2).fit(diff_vector.reshape(-1, 1))
    assignments = k_means.labels_
    centers = k_means.cluster_centers_
    low_cluster = np.argmin(np.linalg.norm(centers, axis=1))
    candidates['bout_2d'] = pd.Series((assignments == low_cluster))
    return candidates

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


def norm_corr(x, y):
    return np.corrcoef(x, y)[0][1]


def spec_vs_spec(s_x, s_pattern, rms_thresh=None):
    #print(s_x.shape)
    s_rms = sp.spectrogram_rms(s_x, s_pattern)
    s_env = np.sum(s_x, axis=0)
    rms_env = st.rms(s_env)

    if (rms_thresh is not None) and rms_env < rms_thresh:
        scalar_corr = 0.
    else:
        scalar_corr = ss.pearsonr(s_env, np.sum(s_pattern, axis=0))[0]
    return s_rms, scalar_corr, rms_env


def spec_scores(x, x_pattern, s_pars, chunked_specgram=True):
    logger.info('Getting the spectral scores')

    s_f = s_pars['s_f']
    filt_lo = s_pars['filt_lo'] #Hz
    filt_hi = s_pars['filt_hi'] #Hz
    n_window = s_pars['n_window']
    step_ms = s_pars['step_ms']
    db_cut = s_pars['db_cut']
    sigma = s_pars['sigma_factor']*s_f
    specgram_step = np.int(0.001 * s_f)

    if chunked_specgram:
        if int(s_f) == 30000:
            specgram_step = 32
        else:
            raise NotImplementedError('Dont know what specgram_step to use when chunked for s_f {}'.format(s_f))
        specgram_func = sp.chunky_spectrogram
        logger.debug('chunked specgram')
    else:
        specgram_step = int(s_f*.001)
        specgram_func = sp.pretty_spectrogram
        logger.debug('No chunked specgram')


    logger.info('Getting pattern spectrogram')
    f_p, t_p, s_p = specgram_func(x_pattern, s_f, log=True,
                                          fft_size=n_window, step_size=specgram_step,
                                          window=sg.gaussian(n_window, sigma),
                                          db_cut=db_cut,
                                          f_min=filt_hi,
                                          f_max=filt_lo
                                          )
    s_p_len = s_p.shape[1]
    plt.pcolormesh(t_p, f_p, s_p, cmap='inferno')

    logger.info('Getting stream spectrogram')
    f_x, t_x, s_x = specgram_func(x, s_f, log=True,
                                          fft_size=n_window, step_size=specgram_step,
                                          window=sg.gaussian(n_window, sigma),
                                          db_cut=db_cut,
                                          f_min=filt_hi,
                                          f_max=filt_lo
                                          )


    d_t = np.unique(np.diff(t_x))[0]
    actual_step_ms = step_ms * np.unique(np.diff(t_x))[0]
    start_points = np.arange(0, s_x.shape[1] - s_p_len, step_ms) #rough correct for last chunk
    logger.info('Getting all scores in {0} points'.format(start_points.size))
    all_scores = map(lambda i: spec_vs_spec(sp.spectrogram_db_cut(s_x[:,i: i+s_p_len],
                                                               db_cut=s_pars['db_cut'],
                                                              log_scale=True),
                                         s_p),
                  start_points)
    logger.info('done collecting scores')
    scores_pd = pd.DataFrame(list(all_scores), columns=['s_rms', 'env_corr', 'rms_env'])
    return scores_pd, s_p, d_t


def find_the_bouts(x, x_pattern, search_pars, cand_file_path=None, cand_grp=None,
                   chunked_specgram=False, debug=False):
    if cand_file_path is not None:
        logger.debug('Will save candidates pandas df channel in file {}'.format(os.path.split(cand_file_path)[-1]))
        logger.debug('Will save them in group {}'.format(cand_grp))

    logger.info('Begin to get scores (rms, env_corr)')
    all_scores, s_pattern, actual_d_t = spec_scores(x.flatten(), x_pattern.flatten(), search_pars,
                                                    chunked_specgram=chunked_specgram)
    all_corr = np.array(all_scores['env_corr'].tolist())
    all_s_rms = np.array(all_scores['s_rms'].tolist())
    logger.info('Getting spectrogram correlation peaks')
    max_pos = find_peaks(all_corr,
                         threshold=search_pars['corr_thresh'],
                         min_peak_distance=int(s_pattern.shape[0]/search_pars['step_ms']),
                         filter_method='ranked'
                         )
    logger.info('Found {0} correlation peaks'.format(max_pos.shape[0]))
    logger.debug('actual dt {}'.format(actual_d_t))
    # start goes in samples
    s_f = search_pars['s_f']
    step_ms = search_pars['step_ms']
    candidates = pd.DataFrame({'start': np.array(
                                                (max_pos + 1) * step_ms * actual_d_t *s_f).astype(np.int64),
                               'env_corr': all_corr[max_pos + 1],
                               'spectral_diff': all_s_rms[max_pos + 1]
                               })
    n_candidates = len(candidates.index)
    logger.info('Found {0} candidates'.format(n_candidates))

    if n_candidates > 2:
        logger.info('Clustering candidates')
        cluster_spectrograms(candidates)
        cluster_spectrograms_2d(candidates)
    else:
        candidates['bout_2d'] = False
        candidates['bout'] = False

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
    if debug:
        return candidates, all_scores
    else:
        return candidates


# a stripped-down acces for just three paths, retunrs the pd dataframe
def search_bouts(wav_path, pattern_path, search_pars, chunked_specgram=False, debug=False):
    logger.info('searching for pattern in {}'.format(wav_path))

    all_sound = WavData2(wav_path)
    pattern_sound = WavData2(pattern_path)

    s_f = all_sound.s_f
    search_pars['s_f'] = s_f

    pattern_chunk = st.Chunk(pattern_sound)
    all_chunk = st.Chunk(all_sound)

    logger.info('{0} samples loaded at {1} Hz'.format(all_chunk.samples, s_f))
    logger.info('Calling find_the_bouts')
    candidates = find_the_bouts(all_chunk.data.flatten(), pattern_chunk.data.flatten(),
                                   search_pars,
                                   cand_file_path=None,
                                   cand_grp=None,
                                   chunked_specgram=chunked_specgram,
                                   debug=debug)

    if debug:
        logger.info('Returning candidates panda, all_scores_panda')
    else:
        logger.info('Returning candidates panda')
        candidates['src_file'] = wav_path
        candidates['pattern_file'] = pattern_path
        candidates['bout_check'] = False
    return candidates


def search_bouts_sess(sess_folder, pattern_path, search_pars, chunked_specgram=False):
    logger.info('will search for pattern in all waves of folder {}'.format(sess_folder))
    all_waves = glob.glob(os.path.join(sess_folder, '*.wav'))
    all_waves.sort()
    logger.info('found {} wav files'.format(len(all_waves)))
    all_candy = map(lambda x: search_bouts(x, pattern_path, search_pars,
                                              chunked_specgram=chunked_specgram),
                    all_waves)
    return pd.concat(list(all_candy))


def collect_bouts_waveforms(cand_pd, pattern_path, border_samples=3000):
    bouts_starts = cand_pd.start.values
    bouts_files = cand_pd.src_file.tolist()
    bouts_idx = cand_pd.index.tolist()

    pattern_sound = st.WavData2(pattern_path)
    pattern_chunk = st.Chunk(pattern_sound)

    logger.info('Will collect the waveforms of {} bouts'.format(bouts_starts.size))

    all_x = np.zeros([bouts_starts.size, pattern_chunk.samples + border_samples * 2])
    logger.info(all_x.shape)
    for i, start in enumerate(bouts_starts):
        motif_start = start
        src_file = bouts_files[i]
        idx = bouts_idx[i]
        # print all_candidates[i]
        chan_sound = st.WavData2(src_file)
        try:
            x = chan_sound.get_chunk(motif_start - border_samples,
                                     motif_start + pattern_chunk.samples + border_samples)
            all_x[i, :] = x.flatten()
        except AssertionError as err:
            logger.warning('Could not get the one chunk {} in file {}'.format(i, src_file))

        # f_x, t_x, s_x = sp.pretty_spectrogram(x.flatten(), s_f, **spec_kwargs)
        # get the rms
        # plt.figure()
        # plt.imshow(s_x[::-1], cmap='inferno', aspect='auto')
        # plt.title('idx = {}'.format(idx))
        # plt.grid(False)

    cand_pd['border_samples'] = border_samples
    cand_pd['waveform'] = pd.DataFrame({'waveform': all_x.tolist()})
    return cand_pd