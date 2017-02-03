import logging
import numpy as np
import pandas as pd
import h5py
import yaml
from h5tools import tables as h5t
from bci.core import kwefunctions as kwe
from bci.core.file import h5_functions as h5f
from bci.core import expstruct as et

import matplotlib.pyplot as plt

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
    #module_logger.debug('Finding onset around {0}-{1} for {2}'.format(start, end, stamp))
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

    wav_table = pd.DataFrame.from_items([('t', wav_times - get_start_rec_times(kwe_file)),
                                         ('stim_name', wav_names),
                                         ('rec', wav_df['rec'].values)])
    if wave_file is not None:
        wav_table = wav_table[wav_table['stim_name'] == wave_file]
    return wav_table


def get_stims_list(kwe_file):
    stims_table = get_wave_times(kwe_file)
    return np.unique(stims_table.stim_name)


def get_start_rec_times(kwe_file):
    all_msg = kwe.get_messages(kwe_file)
    start_rec_df = all_msg[all_msg['text'].str.contains('start time')]
    start_times = [long(x.split()[-1].split('@')[0]) for x in start_rec_df.text.values]
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


def align_stim(bird_id, super_sess_id):
    ss_fn = et.file_names(bird_id, super_sess_id)
    mot_file_path = et.file_path(ss_fn, 'ss', 'sng')
    super_sess_path = et.file_path(ss_fn, 'ss', 'ss_raw')
    rec_list = et.get_rec_list(bird_id, super_sess_id)

    for rec in rec_list:
        # get the rec events file
        rec_origin = h5f.get_rec_origin(super_sess_path, rec)
        module_logger.debug('Rec origin: {0}'.format(rec_origin))
        origin_fn = et.file_names(bird_id, rec_origin['sess'], base=rec_origin['structure'])

        rec_ev_file_path = et.file_path(origin_fn, 'rw', 'evt')
        rec_kwd_file_path = et.file_path(origin_fn, 'rw', 'ss_raw')

        # read the raw parameters file and get the tag channel
        par_file_path = et.file_path(et.file_names(bird_id, rec_origin['sess']), 'rw', 'par')
        pars = et.get_parameters(bird_id, rec_origin['sess'], location='rw')
        tag_chan = int(pars['channel_config']['sts'])

        with h5py.File(rec_ev_file_path, 'r') as rec_ev_file:
            for stim_id in get_stims_list(rec_ev_file):
                module_logger.info('Getting starts of stim {}'.format(stim_id))
                store_starts = get_stim_starts(rec_ev_file, rec_kwd_file_path,
                                               rec_origin['rec'], tag_chan, stim_id)
                module_logger.info('Found {} starts'.format(store_starts.size))
                with h5py.File(mot_file_path, 'a') as mot_file:
                    store_motiff(mot_file, rec, store_starts, stim_id)
                    module_logger.info('Stored in {}'.format(mot_file_path))

        module_logger.info('Done')