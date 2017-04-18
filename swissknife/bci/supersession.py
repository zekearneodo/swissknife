from __future__ import division
import numpy as np
import h5py
import errno
import os
import glob
import logging
import yaml
import wave
import scipy.signal as sg
import shutil as sh

from core.file import h5_functions as h5
from core import expstruct as et

module_logger = logging.getLogger("supersession")


def list_flatten(x):
    result = []
    for el in x:
        if hasattr(el, "__iter__") and not isinstance(el, basestring):
            result.extend(list_flatten(el))
        else:
            result.append(el)
    return result


def create_neural_data_set(data_set, parent_group, channel_list,
                           frame_size=None,
                           processing=None,
                           *args, **kwargs):
    # make the new dataset
    d_type = data_set.dtype
    nd_cols = np.array(channel_list).size
    nd_rows = data_set.shape[0]
    d_chunks = np.array(data_set.chunks)

    if d_chunks[0] > nd_rows:
        d_chunks[0] = nd_rows

    if frame_size is None:
        frame_size = d_chunks[0] if d_chunks[0] < nd_rows else nd_rows

    neural_dset = parent_group.create_dataset("data", (nd_rows, nd_cols),
                                              chunks=(d_chunks[0], nd_cols), dtype=d_type)

    copy_frame = np.zeros([frame_size, nd_cols], dtype=d_type)
    n_full_frames = int(data_set.shape[0] / frame_size)
    # print neural_dset.shape
    # Fill the dataset:
    for i in range(n_full_frames):
        copy_frame = h5.load_table_slice(data_set,
                                         np.arange(i * frame_size, (i + 1) * frame_size),
                                         channel_list)
        neural_dset[i * frame_size: (i + 1) * frame_size, 0:nd_cols] = copy_frame

    if frame_size * n_full_frames < nd_rows:
        last_frame_size = nd_rows - frame_size * n_full_frames
        copy_frame = np.zeros([last_frame_size, nd_cols], dtype=d_type)
        copy_frame = h5.load_table_slice(data_set,
                                         np.arange(n_full_frames * frame_size,
                                                   n_full_frames * frame_size + last_frame_size),
                                         channel_list)
    # copy the attributes:
    neural_dset.attrs.create('valid_samples',
                             np.array(data_set.attrs['valid_samples'][channel_list]))


def copy_application_data(source_rec, dest_rec, chan_list,
                          resize_keys=['channel_bit_volts', 'channel_sample_rates']):
    dest_rec.create_group('application_data')
    for key, attrib in source_rec['application_data'].attrs.iteritems():
        if not key in resize_keys:
            dest_rec['application_data'].attrs.create(key, attrib, dtype=attrib.dtype)
        else:
            # print key
            # print attrib[chan_list]
            dest_rec['application_data'].attrs.create(key, attrib[chan_list], dtype=attrib.dtype)


def check_rec_data(source_rec):
    # check the data table of a rec group
    data_set = source_rec['data']
    valid_samples = None
    try:
        valid_samples = data_set.attrs.get('valid_samples')
    except:
        module_logger.warn('No valid samples in data of {}'.format(source_rec.name))
        pass

    has_data = True if valid_samples is not None and valid_samples[0] > 0 else False
    return has_data


def get_valid_samples(source_rec):
    # get the valid samples declared by the data table attributes
    data_set = source_rec['data']
    valid_samples = None
    try:
        valid_samples = data_set.attrs.get('valid_samples')[0]
    except:
        pass
    return valid_samples


def create_data_groups(raw_file, new_file, chan_list):
    for rec_name, rec_group in raw_file['/recordings'].iteritems():
        new_file['/recordings'].create_group(rec_name)
        h5.copy_attribs(rec_group, new_file['/recordings'][rec_name])
        copy_application_data(rec_group, new_file['/recordings'][rec_name], chan_list)
        create_neural_data_set(rec_group['data'], new_file['/recordings'][rec_name], chan_list)


def make_shank_kwd(raw_file, out_file_path, chan_list):
    ss_file = h5py.File(out_file_path, 'w')
    h5.copy_attribs(raw_file, ss_file)
    ss_file.create_group('/recordings')
    create_data_groups(raw_file, ss_file, chan_list)


def new_channel_config(chan_config):
    new_chan_config = {}
    chan_list = list_flatten([chans for chans in chan_config.itervalues()])
    chan_list.sort()
    raw_chans = np.array(chan_list)
    for chan_group, channels in chan_config.iteritems():
        if hasattr(channels, "__iter__") and not isinstance(channels, basestring):
            new_chan_config[chan_group] = [int(np.where(raw_chans == r_ch)[0][0]) for r_ch in channels]
        else:
            new_chan_config[chan_group] = int(np.where(raw_chans == channels)[0][0])
    return chan_list, new_chan_config


def create_data_group(raw_file, new_file, chan_list, new_rec_name):
    for rec_name, rec_group in raw_file['/recordings'].iteritems():
        new_file['/recordings'].create_group(rec_name)
        h5.copy_attribs(rec_group, new_file['/recordings'][rec_name])
        copy_application_data(rec_group, new_file['/recordings'][rec_name], chan_list)
        create_neural_data_set(rec_group['data'], new_file['/recordings'][rec_name], chan_list)


def insert_neural_rec_group(dest_file, raw_rec_group, chan_list, new_group_name=None):
    rec_name = raw_rec_group.name if new_group_name is None else new_group_name
    dest_file['/recordings'].create_group(rec_name)
    new_rec = dest_file['/recordings'][rec_name]
    copy_application_data(raw_rec_group, new_rec, chan_list)
    h5.copy_attribs(raw_rec_group, new_rec)
    create_neural_data_set(raw_rec_group['data'], new_rec, chan_list)


def modify_rec_group_attribs(kwd_file, rec_name, attr_dict, new_attr_dict=None):
    for key, value in attr_dict.iteritems():
        kwd_file['/recordings'][rec_name].attrs.modify(key, value)

    if new_attr_dict is not None:
        for key, value in new_attr_dict.iteritems():
            kwd_file['/recordings'][rec_name].attrs.create(key, value)


def get_experiment_endpoints(experiment_file):
    recs_list = h5.get_rec_list(experiment_file)
    if recs_list.size > 0:
        last_rec_num = np.max(recs_list)
        last_rec_name = str(last_rec_num)
        last_rec = experiment_file['/recordings'][last_rec_name]
        last_rec_start = last_rec.attrs['start_sample']
        last_rec_nsamples = last_rec['data'].shape[0]
        next_rec_num = last_rec_num + 1
        next_sample = last_rec_start + last_rec_nsamples
    else:
        next_rec_num = 0
        next_sample = 0

    return [next_rec_num, next_sample]


def insert_experiment_groups(dest_file, raw_file, chan_list):
    # all the recs in an experiment file if they pass check
    for raw_rec_name, raw_rec_group in raw_file['/recordings'].iteritems():
        # print 'rec {0}'.format(raw_rec_name)
        s_f = raw_rec_group.attrs['sample_rate']
        if check_rec_data(raw_rec_group):
            target_rec_num, target_start_sample = get_experiment_endpoints(dest_file)
            target_rec_name = str(target_rec_num)
            target_start_time = int(target_start_sample / (0.001 * s_f))
            target_source = '{0}:''/''recordings/{1}'.format(raw_file.filename, raw_rec_name)
            insert_neural_rec_group(dest_file, raw_rec_group, chan_list, new_group_name=target_rec_name)
            modify_rec_group_attribs(dest_file, target_rec_name,
                                     {'start_sample': target_start_sample},
                                     new_attr_dict={'name': target_source,
                                                    'start_time': target_start_time})
        else:
            # print "Skipping rec {0} with no data".format(raw_rec_name)
            pass


def make_super_file(path):
    new_file = h5py.File(path, 'w')
    new_file.create_group('/recordings')
    new_file.close()


def save_ss_par(par, bird, sess, ss_location='ss'):
    fn = et.file_names(bird, sess)
    par_path = et.file_path(fn, ss_location, 'par')
    with open(par_path, 'w') as f:
        written = yaml.dump(par, f)
    return written


def band_pass_filter_pars(s_f, exp_par):
    filt_hi = exp_par['search_motiff']['filt_hi']
    filt_lo = exp_par['search_motiff']['filt_lo']
    hp_b, hp_a = sg.butter(4, filt_hi / (s_f / 2.), btype='high')
    lp_b, lp_a = sg.butter(4, filt_lo / (s_f / 2.), btype='low')
    return hp_b, hp_a, lp_b, lp_a


def filter_bp(data, hp_b, hp_a, lp_b, lp_a):
    x_hi = sg.filtfilt(hp_b, hp_a, data, axis=0)
    x_lo = sg.filtfilt(lp_b, lp_a, x_hi, axis=0)
    return x_lo


def get_dset_group_attr(data_set, attr_name):
    return data_set.parent.attrs[attr_name]


def export_audio(data_set, chan_number, out_file_path, filter_func=None, args=(), **kwargs):
    frame_size = 60  # seconds
    s_f = get_dset_group_attr(data_set, 'sample_rate')
    samples_frame = int(frame_size * s_f)
    samples_data = data_set.shape[0]

    frame_buffer = np.zeros((samples_frame, 1), dtype=np.float)
    frame_starts = np.arange(0, samples_data, samples_frame)

    wavefile = wave.open(out_file_path, 'wb')
    wavefile.setparams((1, 2, s_f, 0, 'NONE', 'not compressed'))

    for start in frame_starts:
        end = min(start + samples_frame, samples_data)
        frame_buffer[0: end - start, :] = h5.load_table_slice(data_set,
                                                              np.arange(start, end),
                                                              [chan_number])
        frame_buffer[0:end - start, :] = filter_func(frame_buffer[0:end - start, :],
                                                     *args, **kwargs)
        wavefile.writeframes(frame_buffer[0: end - start].astype('h').tostring())

    wavefile.close()


def extract_wav_chans(bird_id, sess, ch_name='mic', location='ss'):
    fn = et.file_names(bird_id, sess, 0)
    super_file_path = os.path.join(fn['folders'][location], fn['structure']['ss_raw'])
    exp_par = et.get_parameters(bird_id, sess)
    super_file = h5py.File(super_file_path, 'r')
    rec_list = super_file['/recordings'].keys()
    module_logger.info('Extract {0} chan to wav for recs {1}'.format(ch_name, rec_list))
    for rec, rec_grp in super_file['/recordings'].iteritems():
        fn = et.file_names(bird_id, sess, int(rec))
        data_set = rec_grp['data']
        s_f = rec_grp.attrs['sample_rate']
        wav_file_path = os.path.join(fn['folders']['ss'], fn['structure'][ch_name])
        module_logger.info('Rec {0}: {1}'.format(rec, wav_file_path))
        export_audio(data_set, exp_par['channel_config'][ch_name], wav_file_path,
                     filter_func=filter_bp,
                     args=band_pass_filter_pars(s_f, exp_par))

    module_logger.info('Done extracting channels')


def list_experiment_files(bird_id, sess_str, depth=None, raw_location='rw', file_type='data'):
    """
    List experiment files for a bird and a session base string (usually a day) and post string
    (usually a depth)
    :param bird_id: str, bird id
    :param sess_str: str, base identifier for a group of sessions
    :param depth: str, post_identifier
    :param raw_location: what is the location of the raw data (for the file_names returned structure)
    :return:
    """
    raw_data_folder = et.file_names(bird_id)['folders'][raw_location]
    str_depth = '' if depth is None else '_{}'.format(depth)
    sessions = glob.glob(os.path.join(raw_data_folder, sess_str + '*' + str_depth))
    sess_par = et.get_parameters(bird_id, os.path.split(sessions[0])[-1], location=raw_location)
    processor = sess_par['rec_config']['processors'][file_type]
    search_str = '*_{}.raw.kwd'.format(processor)
    module_logger.info('searching {}'.format(search_str))
    experiments = list_flatten(
        [glob.glob(os.path.join(s, search_str))[:] for s in sessions])
    experiments.sort()
    return experiments


def make_super_session(bird_id, sess_str, depth='', raw_location='rw', ss_location='ss'):
    raw_data_folder = et.file_names(bird_id)['folders'][raw_location]
    sessions = glob.glob(os.path.join(raw_data_folder, sess_str + '*' + str(depth)))
    exp_files = list_experiment_files(bird_id, sess_str,
                                      depth=depth,
                                      raw_location=raw_location,
                                      file_type='data')
    sess_par = et.get_parameters(bird_id, os.path.split(sessions[0])[-1], location=raw_location)

    super_sess_par = sess_par.copy()
    super_sess_name = 'day-' + sess_str
    if depth != '':
        super_sess_name += '_{0}'.format(depth)
    fn = et.file_names(bird_id, super_sess_name, 0)
    super_sess_path = fn['folders'][ss_location]
    super_file_path = os.path.join(super_sess_path, fn['structure']['ss_raw'])
    module_logger.info("Making supersession {}".format(super_sess_name))
    module_logger.info('super file path: {}'.format(super_file_path))
    module_logger.info('Found {} experiment files'.format(len(exp_files)))
    et.mkdir_p(super_sess_path)
    make_super_file(super_file_path)

    with h5py.File(super_file_path, 'a') as super_file:
        for experiment_path in exp_files:
            module_logger.info('Inserting file {0}'.format(experiment_path))
            sess_fold = os.path.split(os.path.split(experiment_path)[0])[1]
            sess_par = et.get_parameters(bird_id, sess_fold, location=raw_location)
            kwd_chan_list, new_par_chan_config = new_channel_config(sess_par['channel_config'])
            with h5py.File(experiment_path, 'r') as raw_file:
                insert_experiment_groups(super_file, raw_file, kwd_chan_list)
            super_file.flush()
    super_sess_par['channel_config'] = new_par_chan_config.copy()
    save_ss_par(super_sess_par, bird_id, super_sess_name, ss_location=ss_location)
    return sessions


def make_copies(sess_list, dest_path, keep=True):
    module_logger.info("Backup of {0} sessions in {1}".format(len(sess_list), dest_path))
    et.mkdir_p(dest_path)
    for session_path in sess_list:
        session_name = os.path.split(session_path)[-1]
        module_logger.debug('Sess {}'.format(session_name))
        dest_bkp = os.path.join(dest_path, session_name)
        operation = sh.copytree if keep else sh.move
        try:
            operation(session_path, dest_bkp)
        except OSError as exc:  # Python >2.5
            if exc.errno == 17:
                pass
            else:
                raise


def make_raw_bkp(bird_id, sess_list, raw_location='rw', locations=None):
    locations = et.get_locations() if locations is None else locations
    exp_path = os.path.join(locations['experiment'], 'raw_data', bird_id)
    store_path = os.path.join(locations['store'], 'raw_data', bird_id)
    source_path = os.path.split(sess_list[0])[0]

    if source_path == exp_path:
        module_logger.info('Data is already in experiment folder {}'.format(source_path))
    else:
        module_logger.info('Copying data to experiment folder {}'.format(exp_path))
        make_copies(sess_list, exp_path, keep=True)

    if raw_location == 'raw':
        module_logger.info('Should be Moving data out of local drive {}'.format(source_path))
        make_copies(sess_list, store_path, keep=False)
    else:
        module_logger.info('Data is not in local drive but in {}, doing nothing'.format(source_path))


def process_awake_recording(bird_id, sess_day_id, depth, raw_location='raw'):
    raw_data_folder = et.file_names(bird_id)['folders'][raw_location]
    sessions = glob.glob(os.path.join(raw_data_folder, sess_day_id + '*' + str(depth)))
    sess_par = et.get_parameters(bird_id, os.path.split(sessions[0])[-1], location=raw_location)
    data_processor = sess_par['rec_config']['processors']['data']
    experiments = list_flatten(
        [glob.glob(os.path.join(s, '*_{}.raw.kwd'.format(data_processor)))[:] for s in sessions])
    experiments.sort()

    super_sess_name = 'day-' + sess_day_id + '_' + depth
    fn = et.file_names(bird_id, super_sess_name, 0)
    super_sess_path = fn['folders']['ss']
    super_file_path = os.path.join(super_sess_path, fn['structure']['ss_raw'])
    module_logger.info('Super session path {}'.format(super_file_path))

    sess_list = make_super_session(bird_id, sess_day_id, depth, raw_location=raw_location)
    make_raw_bkp(bird_id, sess_list, raw_location=raw_location)
    extract_wav_chans(bird_id, super_sess_name)
    module_logger.info('Done making supersession')


def process_asleep_recording(bird_id, sess_day_id, depth, raw_location='raw', ss_location='ss'):
    raw_data_folder = et.file_names(bird_id)['folders'][raw_location]
    sessions = glob.glob(os.path.join(raw_data_folder, sess_day_id + '*' + str(depth)))
    sess_par = et.get_parameters(bird_id, os.path.split(sessions[0])[-1], location=raw_location)
    data_processor = sess_par['rec_config']['processors']['data']
    experiments = list_flatten(
        [glob.glob(os.path.join(s, '*_{}.raw.kwd'.format(data_processor)))[:] for s in sessions])
    experiments.sort()

    super_sess_name = 'day-' + sess_day_id + '_' + depth
    fn = et.file_names(bird_id, super_sess_name, 0)
    super_sess_path = fn['folders'][ss_location]
    super_file_path = os.path.join(super_sess_path, fn['structure']['ss_raw'])
    module_logger.info('Super session path {}'.format(super_file_path))

    sess_list = make_super_session(bird_id, sess_day_id, depth, raw_location=raw_location)
    make_raw_bkp(bird_id, sess_list, raw_location=raw_location)
    extract_wav_chans(bird_id, super_sess_name)
    module_logger.info('Done making supersession')


def process_recording_realtime(bird_id, sess_day_id, depth, raw_location='raw', ss_location='ss'):
    raw_data_folder = et.file_names(bird_id)['folders'][raw_location]
    sessions = glob.glob(os.path.join(raw_data_folder, sess_day_id + '*' + str(depth)))
    sess_par = et.get_parameters(bird_id, os.path.split(sessions[0])[-1], location=raw_location)
    data_processor = sess_par['rec_config']['processors']['data']
    experiments = list_flatten(
        [glob.glob(os.path.join(s, '*_{}.raw.kwd'.format(data_processor)))[:] for s in sessions])
    experiments.sort()

    super_sess_name = 'day-' + sess_day_id + '_' + depth
    fn = et.file_names(bird_id, super_sess_name, 0)
    super_sess_path = fn['folders'][ss_location]
    super_file_path = os.path.join(super_sess_path, fn['structure']['ss_raw'])
    module_logger.info('Super session path {}'.format(super_file_path))

    sess_list = make_super_session(bird_id, sess_day_id, depth, raw_location=raw_location)

    return super_sess_path


def process_recording_realtime(bird_id, sess_day_id, depth, raw_location='raw', ss_location='ss'):
    raw_data_folder = et.file_names(bird_id)['folders'][raw_location]
    sessions = glob.glob(os.path.join(raw_data_folder, sess_day_id + '*' + str(depth)))
    sess_par = et.get_parameters(bird_id, os.path.split(sessions[0])[-1], location=raw_location)
    data_processor = sess_par['rec_config']['processors']['data']
    experiments = list_flatten(
        [glob.glob(os.path.join(s, '*_{}.raw.kwd'.format(data_processor)))[:] for s in sessions])
    experiments.sort()

    super_sess_name = 'day-' + sess_day_id + '_' + depth
    fn = et.file_names(bird_id, super_sess_name, 0)
    super_sess_path = fn['folders'][ss_location]
    super_file_path = os.path.join(super_sess_path, fn['structure']['ss_raw'])
    module_logger.info('Super session path {}'.format(super_file_path))

    sess_list = make_super_session(bird_id, sess_day_id, depth, raw_location=raw_location, ss_location=ss_location)
    return sess_list