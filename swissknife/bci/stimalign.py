import logging

import h5py
import numpy as np
import pandas as pd
import warnings
from swissknife.bci.core import expstruct as et
from swissknife.bci.core import kwefunctions as kwe
from swissknife.bci.core.file import h5_functions as h5f

from swissknife.h5tools import tables as h5t

from swissknife.streamtools import spectral as sp

module_logger = logging.getLogger("stimalign")


def append_rows(table, new_data):
    rows = table.shape[0]
    more_rows = new_data.shape[0]
    table.resize(rows + more_rows, axis=0)
    if table.size == (rows + more_rows):
        table[rows:] = new_data
    else:
        table[rows:, :] = new_data


def unlimited_rows_data(group, table_name, data):
    try:
        table = group.create_dataset(table_name,
                                     data=data,
                                     dtype=data.dtype,
                                     maxshape={None, None})
    except RuntimeError as e:
        if 'Name already exists' in str(e):
            table = group[table_name]
            append_rows(table, data)
        else:
            raise
    return table


def find_first_peak(x, thresh_factor=0.3):
    x = x - np.mean(x)
    thresh = np.max(x) * thresh_factor
    # find the peaks naively
    a = x[1:-1] - x[2:]
    b = x[1:-1] - x[:-2]
    c = x[1:-1]
    max_pos = np.where((a >= 0) & (b > 0) & (c > thresh))[0] + 1
    return max_pos[0]


def find_wav_onset(dset, chan, stamp, tr_df):
    [start, end] = get_trial_bounds(stamp, tr_df)
    # module_logger.debug('Finding onset around {0}-{1} for {2}'.format(start, end, stamp))
    trial_frame = h5t.load_table_slice(dset, np.arange(start, end), [chan])
    onset_in_trial = find_first_peak(trial_frame)
    return start + onset_in_trial


def store_motiff(ev_file, rec, bout_starts, motiff_name):
    motiff_group = ev_file.require_group('/event_types/singing/{0}'.format(motiff_name))
    t_table = unlimited_rows_data(motiff_group, 'time_samples', bout_starts)
    r_table = unlimited_rows_data(motiff_group, 'recording',
                                  data=np.ones_like(bout_starts) * int(rec))


def get_wave_times(kwe_file, wave_file=None):
    all_msg = kwe.get_messages(kwe_file)
    wav_df = all_msg[all_msg['text'].str.contains('play_wav')]

    wav_times = wav_df.t.values.astype(np.int)
    wav_names = [(x.split()[-1].split('/')[-1].split('.')[0]) for x in wav_df.text.values]

    if not wav_names:
        module_logger.info('Empty list of play_wav in file {}'.format(kwe_file.filename))
        wav_table = pd.DataFrame()
    else:
        wav_table = pd.DataFrame.from_items([('t', wav_times - get_start_rec_times(kwe_file)),
                                             ('stim_name', wav_names),
                                             ('rec', wav_df['rec'].values)])
        if wave_file is not None:
            wav_table = wav_table[wav_table['stim_name'] == wave_file]
    return wav_table


def get_stims_list(kwe_file):
    module_logger.info('getting stim lists of {}'.format(kwe_file))
    stims_table = get_wave_times(kwe_file)
    #module_logger.info(stims_table)

    if stims_table.empty:
        stim_names = []
    else:
        stim_names = np.unique(stims_table.stim_name)

    module_logger.debug('Stim_names: {}'.format(stim_names))
    return stim_names


def get_start_rec_times(kwe_file):
    all_msg = kwe.get_messages(kwe_file)
    start_rec_df = all_msg[all_msg['text'].str.contains('start time')]
    start_times = [int(x.split()[-1].split('@')[0]) for x in start_rec_df.text.values]
    assert (len(start_times) == 1), "more or less than one recording start time"
    return start_times[0]


def get_trial_times(kwe_file):
    all_msg = kwe.get_messages(kwe_file)
    trials_df = all_msg[all_msg['text'].str.contains('trial_number')]
    trial_times = trials_df.t.values.astype(np.int)
    trial_numbers = [int(x.split()[-1]) for x in trials_df.text.values]
    return pd.DataFrame.from_items([('t', trial_times - get_start_rec_times(kwe_file)),
                                    ('trial', trial_numbers),
                                    ('rec', trials_df['rec'].values)])


def get_trial_bounds(stamp, tr_df):
    trial_n = max(tr_df[tr_df.t < stamp].trial)
    t_start = tr_df[tr_df.trial == trial_n].t.values[0]
    try:
        t_end = tr_df[tr_df.trial == trial_n + 1].t.values[0]
    except:
        t_end = None
    return [t_start, t_end]


def get_stim_starts(kwe_file, kwd_file, rec, tag_chan, stim_name):
    data_set = h5f.get_data_set(kwd_file, rec)
    trials = get_trial_times(kwe_file)
    stim_times = get_wave_times(kwe_file, stim_name)
    precise_starts = np.array([find_wav_onset(data_set, tag_chan, stamp, trials) for stamp in stim_times.t.values])
    return precise_starts


def align_stim(bird_id, super_sess_id, raw_location='rw', ss_location='ss'):
    ss_fn = et.file_names(bird_id, super_sess_id)
    mot_file_path = et.file_path(ss_fn, ss_location, 'sng')
    super_sess_path = et.file_path(ss_fn, ss_location, 'ss_raw')
    rec_list = et.get_rec_list(bird_id, super_sess_id, location=ss_location)
    module_logger.info('Aligning stim of sess {}, rec_list={}'.format(super_sess_path, rec_list))

    for rec in rec_list:
        # get the rec events file
        rec_origin = h5f.get_rec_origin(super_sess_path, rec)
        module_logger.debug('Rec origin: {0}'.format(rec_origin))
        origin_fn = et.file_names(bird_id, rec_origin['sess'], base=rec_origin['structure'])

        rec_ev_file_path = et.file_path(origin_fn, raw_location, 'evt')
        rec_kwd_file_path = et.file_path(origin_fn, raw_location, 'ss_raw')

        # read the raw parameters file and get the tag channel
        pars = et.get_parameters(bird_id, rec_origin['sess'], location=raw_location)
        tag_chan = int(pars['channel_config']['sts'])
        #module_logger.debug('ev file path: {}'.format(rec_ev_file_path))

        with h5py.File(rec_ev_file_path, 'r') as rec_ev_file:
            module_logger.debug('File {}'.format(rec_ev_file.filename))
            for stim_id in get_stims_list(rec_ev_file):
                module_logger.info('Getting starts of stim {}'.format(stim_id))
                store_starts = get_stim_starts(rec_ev_file, rec_kwd_file_path,
                                               rec_origin['rec'], tag_chan, stim_id)
                module_logger.info('Found {} starts'.format(store_starts.size))
                with h5py.File(mot_file_path, 'a') as mot_file:
                    store_motiff(mot_file, rec, store_starts, stim_id)
                    module_logger.info('Stored in {}'.format(mot_file_path))

        module_logger.info('Done')


# Functions for the intan board

def dig_pulses(dig_chan):
    # get start:end in dig_chan (chan uint64 with [0,1])
    x = dig_chan.astype(np.short)
    onsets = np.where(np.diff(x) == 1)[0]
    offsets = np.where(np.diff(x) == -1)[0]

    # for debugging
    # plt.plot(rrx)
    # plt.plot(onsets, np.ones_like(onsets), 'r*');
    # plt.plot(offsets, np.ones_like(offsets), 'k*')

    # print(onsets.shape)
    if (onsets.size > 0) & (offsets.size > 0):
        if onsets[0] > offsets[0]:
            offsets = offsets[1:]
        if offsets[-1] < onsets[-1]:
            onsets = onsets[:-1]
    else:
        onsets = np.nan
        offsets = np.nan
    return np.vstack([onsets, offsets]).T


def get_sine_freq(x, s_f, samples=1024):
    f, t, s = sp.pretty_spectrogram(x[:samples].astype(np.float), s_f,
                                    log=False,
                                    fft_size=samples,
                                    step_size=samples,
                                    window=('tukey', 0.25))
    # s should be just one slice
    # get the peak frequency
    f_0 = f[np.argmax(s[:, 0])]

    return f_0


def get_sine(x, s_f, trial_bound):
    # onset ref to the beginning of the rec
    onset = find_first_peak(x[trial_bound[0]:trial_bound[1]]) + trial_bound[0]
    sin_chunk = x[onset: trial_bound[1]]

    f_0 = get_sine_freq(sin_chunk, s_f)
    #module_logger.debug('f0 {}'.format(f_0))

    if (np.isfinite(f_0) and f_0 > 0):
        # correct the onset with the 1/4 wave
        wave_samples = float(s_f) / f_0
        samples_correction = int(wave_samples * 0.25)
        # print(samples_correction)
    else:
        msg = 'Invalid f_0 around {}'.format(trial_bound)
        warnings.warn(msg, RuntimeWarning)
        samples_correction = 0

    return onset - samples_correction, onset, f_0


def get_all_wav_starts(sine_chan, mark_chan, s_f):
    module_logger.debug('Getting all markers from digital chan')

    all_on_off = dig_pulses(mark_chan)

    if np.isnan(all_on_off).any():
        module_logger.warning('No events found')
        all_starts = None
    else:
        module_logger.info('found {} events'.format(all_on_off.shape[0]))
        all_starts = list(map(lambda tb: get_sine(sine_chan, s_f, tb), all_on_off))

    all_starts_pd = pd.DataFrame(all_starts, columns=['t', 't_uncorrected', 'tag_f'])
    return all_starts_pd


def chan_dict_lookup(chan_name, chan_dict_list):
    id_found = [i for i, chan_dict in enumerate(chan_dict_list) if chan_dict['name'] == chan_name]
    assert len(id_found) == 1, 'Found many or none channels of name {}'.format(chan_name)
    return chan_dict_list[id_found[0]]


def lookup_freq(freq, stim_freqs_dict, precision=100):
    freq_to_find = round(freq / precision) * precision
    # module_logger.info('lookup freq {}'.format(lookup_freq))
    found_name = [k for (k, v) in stim_freqs_dict.items() if v == freq_to_find]
    # print(found_name)

    assert len(found_name) == 1, 'Either not found or found many stim with freq {} ({})'.format(freq, freq_to_find)

    return found_name[0]
