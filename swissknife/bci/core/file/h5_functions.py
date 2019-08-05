# Functions to do stuff with h5 open files
import numpy as np
from numpy.lib import recfunctions as rf
import logging
import h5py
import logging
import os
from tqdm import tqdm_notebook as tqdm

#from mdaio import writemda16i

logger = logging.getLogger('bci.h5_functions')

def h5_wrap(h5_function):
    """
    Decorator to open h5 structure if the path was provided to a function.
    :param h5_function: a function that receives an h5file as first argument
    :return: decorated function that takes open('r' mode) or path as first argument
    """

    def file_checker(h5_file, *args, **kwargs):
        if type(h5_file) is not h5py._hl.files.File:
            h5_file = h5py.File(h5_file, 'r')
        #logging.debug('H5 file: {}'.format(h5_file))
        return_value = h5_function(h5_file, *args, **kwargs)
        # h5_file.close()
        return return_value

    return file_checker

def h5_decorator(default_mode='r'):
    """
    Decorator to open h5 structure if the path was provided to a function.
    :param h5_function: a function that receives an h5file as first argument
    :param default_mode: what mode to open the file by default.
        It is overriden when file is entered open and when option 'mode' is set
        in h5_function (if it exists)
    :return: decorated function that takes open or path as first argument
    """
    def wrap(h5_function):
        def file_checker(h5_file, *args, **kwargs):
            if 'mode' in kwargs.keys():
                mode = kwargs['mode']
            else:
                mode = default_mode
                #logger.debug('mode {}'.format(mode))
            try:
                if type(h5_file) is not h5py._hl.files.File:
                    with h5py.File(h5_file, mode) as h5_file:
                        return_value = h5_function(h5_file, *args, **kwargs)
                else:
                    return_value = h5_function(h5_file, *args, **kwargs)
                return return_value
            except UnboundLocalError as err:
                logger.error(err)
                raise
        return file_checker
    return wrap

def list_subgroups(h5_group):
    return [key for key, val in h5_group.items() if isinstance(val, h5py.Group)]


def h5_unicode_hack(x):
    if isinstance(x, str):
        x = x.encode('utf8')
    elif isinstance(x, bytes):
        x = x.decode('utf-8')
    return x

# gets the sampling frequency of a recording
@h5_wrap
def get_record_sampling_frequency(h5, recording=0):
    path = 'recordings/{0:d}'.format(recording)
    return h5_unicode_hack(h5[path].attrs.get('sample_rate'))


@h5_wrap
def get_rec_list(k_file):
    """
    :param k_file:
    :return: list of recordings in an h5file (kwik/kwd) as a sorted numpy array
    """
    return np.sort(list(map(int, list(k_file['/recordings'].keys()))))


@h5_wrap
def get_shank_list(k_file):
    """
    :param k_file:
    :return: list of recordings in an h5file (kwik/kwd) as a sorted numpy array
    """
    #print(list(k_file['/channel_groups'].keys().items()))
    return np.sort(list(map(int, list(k_file['/channel_groups'].keys()))))


@h5_wrap
def get_rec_group(kwd_file, rec):
    """
    :param kwd_file:
    :param rec: number of rec
    :return: h5 group
    """
    logging.debug('Getting group of rec {}'.format(rec))
    return kwd_file['/recordings/{}'.format(int(rec))]


@h5_wrap
def get_rec_origin(kwd_file, rec):
    """
    :param kwd_file:
    :param rec: number of rec
    :return: dictionary with bird, sess, rec
    """
    group = get_rec_group(kwd_file, rec)
    origin_strings = h5_unicode_hack(group.attrs.get('name')).split(':')
    recording = int(origin_strings[1].split('/')[-1])
    path = os.path.split(origin_strings[0])[0]
    base_name = os.path.split(origin_strings[0])[-1].split('.')[0]
    sess = os.path.split(path)[-1]
    bird = os.path.split(os.path.split(path)[0])[-1]
    return {'bird': bird, 'sess': sess, 'rec': recording, 'structure': base_name}


@h5_wrap
def get_data_set(kwd_file, rec):
    """
    :param kwd_file:
    :param rec: number of rec
    :return: h5 dataset object with
    """
    logging.debug('Getting dataset from rec {}'.format(rec))
    return kwd_file['/recordings/{}/data'.format(int(rec))]


@h5_wrap
def get_data_size(kwd_file, rec):
    return get_data_set(kwd_file, rec).shape[0]


@h5_wrap
def get_rec_sizes(kwd_file):
    rec_list = get_rec_list(kwd_file)
    rec_sizes = {i: get_data_size(kwd_file, rec_list[i])
                 for i in range(0, rec_list.size)}
    return rec_sizes


@h5_wrap
def get_corresponding_rec(kwd_file, sample):
    # returns the order of rec in which this position belongs (not the rec_id)
    rec_starts = get_rec_starts(kwd_file)
    rec_index = np.min(np.where(sample > rec_starts))
    return rec_index


@h5_wrap
def get_rec_starts(kwd_file):
    logger.debug('Getting rec_starts')
    rec_sizes = get_rec_sizes(kwd_file)
    #logger.debug('rec sizes {}'.format(rec_sizes))
    starts_vec = np.array(list(rec_sizes.values())).cumsum()
    #logger.debug('starts vector {}'.format(starts_vec))
    starts_vec = np.hstack([0, starts_vec[:-1]])
    rec_starts = {rec: r_start for r_start, rec in zip(starts_vec, rec_sizes.keys())}
    return rec_starts


# H5 functions
def copy_attribs(source, dest):
    for key, attrib in source.attrs.items():
        dest.attrs.create(key, attrib)

# Table functions
def get_dset_group_attr(data_set, attr_name):
    return data_set.parent.attrs[attr_name]


def load_table_slice(table, row_list=None, col_list=None):
    """
    Loads a slice of a h5 dataset.
    It can load sparse columns and rows. To do this, it first grabs the smallest chunks that contains them.
    :param table: dataset of an h5 file.
    :param row_list: list of rows to get (int list)
    :param col_list: list of cols to get (int list)
    :return: np.array of size row_list, col_list with the concatenated rows, cols.
    """
    table_cols = table.shape[1]
    table_rows = table.shape[0]
    d_type = table.dtype

    col_list = np.arange(table_cols) if col_list is None else np.array(col_list)
    row_list = np.arange(table_rows) if row_list is None else np.array(row_list)

    raw_table_slice = np.empty([np.ptp(row_list) + 1, np.ptp(col_list) + 1], dtype=np.dtype(d_type))
    table.read_direct(raw_table_slice,
                      np.s_[np.min(row_list): np.max(row_list) + 1, np.min(col_list): np.max(col_list) + 1])
    # return raw_table_slice
    return raw_table_slice[row_list - np.min(row_list), :][:, col_list - np.min(col_list)]


# passing stuff to binary
def dset_to_binary_file(data_set, out_file, chan_list=None, chunk_size=8000000):
    """
    :param data_set: a table from an h5 file to write to a binary. has to be daughter of a rec
    :param out_file: binary file - has to be open in 'w' mode.
    :param chan_list: list of channels (must be list or tuple). Default (None) will do the whole table
    :param chunk_size: size in samples of the chunk
    :return:
    """
    samples_data = data_set.shape[0]
    channels_data = data_set.shape[1]
    data_type =np.dtype(data_set.dtype)
    logging.info('Ripping dataset from {}'.format(data_set.parent.name))
    if chan_list is None:
        logging.debug('Counting channels')
        chan_list = range(channels_data)
    logging.info('Channel list: {}'.format(chan_list))

    samples_chunk = min(chunk_size, samples_data)
    channels_chunk = len(chan_list)

    chunk_buffer = np.empty((samples_chunk, channels_chunk), dtype=data_type)
    chunk_starts = np.arange(0, samples_data, samples_chunk)
    n_chunks = chunk_starts.size

    logging.info('About to store {} entire chunks'.format(n_chunks - 1))
    for start in chunk_starts:
        logging.info('Chunk start: {0}'.format(start))
        end = min(start + samples_chunk, samples_data)
        chunk_buffer[0: end - start, :] = load_table_slice(data_set,
                                                           np.arange(start, end),
                                                           chan_list)
        #logging.info('Chunk buffer dtype {}'.format(chunk_buffer.dtype))
        #logging.info('Chunk dtype dtype {}'.format(data_type))
        out_file.write(chunk_buffer[0: end - start].astype(data_type).tostring())

    stored = n_chunks * chunk_buffer.size + chunk_buffer[0: end - start, :].size
    logging.info('{} elements written'.format(stored))
    return stored


@h5_wrap
def kwd_to_binary(kwd_file, out_file_path, chan_list=None, chunk_size=8000000):
    """
    :param kwd_file: kwd file or kwd file
    :param out_file_path: path to the bin file that will be created
    :param chan_list: list of channels (must be list or tuple). Default (None) will do the whole table
    :param chunk_size: size in samples of the chunk
    :return:
    """
    # get the dataset of each recording and concatenateit to the out_file_path
    logging.info('Writing kwd_file {} to binary'.format(kwd_file.filename))
    logging.info('Channels to extract: {}'.format(chan_list))
    logging.info('Creating binary file {}'.format(out_file_path))
    if chan_list is not None:
        if (type(chan_list) is not list) and (type(chan_list) is not tuple):
            assert (type(chan_list) is int)
            chan_list = [chan_list]
        chan_list = list(chan_list)
    rec_list = get_rec_list(kwd_file)
    logging.info('Will go through recs {}'.format(rec_list))
    with open(out_file_path, 'wb') as out_file:
        stored_elements = list(map(lambda rec_name: dset_to_binary_file(get_data_set(kwd_file, rec_name),
                                                                   out_file,
                                                                   chan_list=chan_list,
                                                                   chunk_size=chunk_size
                                                                   ),
                              rec_list))

    elements_in = np.array(list(stored_elements)).sum()
    logging.info('{} elements written'.format(elements_in))

# @h5_wrap
# def kwd_to_mda(kwd_file, out_file_path, chan_list=None, chunk_size=None):
#     """
#     :param kwd_file: kwd file or kwd file
#     :param out_file_path: path to the bin file that will be created
#     :param chan_list: list of channels (must be list or tuple). Default (None) will do the whole table
#     :param chunk_size: size in samples of the chunk
#     :return:
#     """
#     # get the dataset of each recording and concatenateit to the out_file_path
#     logging.info('Writing kwd_file {} to binary'.format(kwd_file.filename))
#     logging.info('Channels to extract: {}'.format(chan_list))
#     logging.info('Creating mountainsort mda file {}'.format(out_file_path))
#
#     if chan_list is not None:
#         raise NotImplementedError
#         if (type(chan_list) is not list) and (type(chan_list) is not tuple):
#             assert (type(chan_list) is int)
#             chan_list = [chan_list]
#         chan_list = list(chan_list)
#
#     rec_list = get_rec_list(kwd_file)
#     logging.info('Will read through recs {}'.format(rec_list))
#
#     n_chans = -1
#     n_samples = 0
#     with h5.File(kwd, 'r') as kwd_f:
#         for recording in rec_list:
#             assert n_chans == -1 or n_chans == kwd_f['recordings'][recording]['data'].shape[1]
#             n_chans = kwd_f['recordings'][recording]['data'].shape[1]
#             n_samples += kwd_f['recordings'][recording]['data'].shape[0]
#         logging.debug("total number of samples %d" % (n_samples))
#         data = np.empty((n_samples, n_chans), dtype='int16')
#         idx = 0
#         for recording in recordings:
#             rec_len = kwd_f['recordings'][recording]['data'].shape[0]
#             logging.debug("loading recording %s with length of %d" % (recording, rec_len))
#             data[idx:idx + rec_len, :] = kwd_f['recordings'][recording]['data']
#             idx += rec_len
#
#     logging.info("writing data")
#     writemda16i(data.T, out_mda)
#     # read everything first
#
#     with open(out_file_path, 'wb') as out_file:
#         stored_elements = list(map(lambda rec_name: dset_to_binary_file(get_data_set(kwd_file, rec_name),
#                                                                    out_file,
#                                                                    chan_list=chan_list,
#                                                                    chunk_size=chunk_size
#                                                                    ),
#                               rec_list))
#
#     elements_in = np.array(list(stored_elements)).sum()
#     logging.info('{} elements written'.format(elements_in))


@h5_wrap
def get_events_one_name(kwe_file, ev_type, ev_name, rec=None):
    times_table = kwe_file['event_types'][ev_type][ev_name]['time_samples'][:]
    rec_table = kwe_file['event_types'][ev_type][ev_name]['recording'][:]
    type_table = np.array(['{}'.format(ev_type) for i in range(times_table.size)],
                          dtype='|S32')
    name_table = np.array(['{}'.format(ev_name) for i in range(times_table.size)],
                          dtype='|S32')
    events_recarray = np.rec.fromarrays((times_table,
                                         rec_table,
                                         name_table,
                                         type_table),
                                        dtype=[('t', 'i8'),
                                               ('rec', 'i2'),
                                               ('name', '|S32'),
                                               ('type', '|S32')])
    events_recarray = events_recarray if rec is None else events_recarray[events_recarray.rec == rec]
    return events_recarray


def get_events_one_type(kwe_file, ev_type, ev_names=[], rec=None):
    if ev_names == []:
        ev_names = list_events(kwe_file, ev_type)
    ev_stack = [get_events_one_name(kwe_file, ev_type, ev_name, rec=rec) for ev_name in ev_names]
    return rf.stack_arrays(ev_stack, asrecarray=True, usemask=False)


def get_all_events(kwe_file, rec=None):
    ev_types = list_event_types(kwe_file)
    ev_stack = [get_events_one_type(kwe_file, ev_type, rec=rec) for ev_type in ev_types]
    return rf.stack_arrays(ev_stack, asrecarray=True, usemask=False)


@h5_wrap
def list_events(kwe_file, ev_type):
    ev_type_group = kwe_file['event_types'][ev_type]
    return list_subgroups(ev_type_group)


@h5_wrap
def list_event_types(kwe_file):
    ev_group = kwe_file['event_types']
    return list_subgroups(ev_group)


def count_events(kwe_file, ev_type, ev_name, rec=None):
    return get_events_one_type(kwe_file, ev_type, ev_name, rec=rec).size

@h5_decorator(default_mode='r')
def collect_frames_fast(kwd_file, recs_list, starts, span, chan_list):
    recs = np.unique(recs_list)
    all_frames_list = []
    for i_rec, rec in tqdm(enumerate(recs), total=recs.size):
        starts_from_rec = starts[recs_list == rec]
        dset = get_data_set(kwd_file, rec)
        n_samples = dset.shape[0]
        valid_starts = starts_from_rec[(starts_from_rec > 0)
                                       & (starts_from_rec + span < n_samples)]
        if valid_starts.size < starts_from_rec.size:
            logger.warn('Some frames were out of bounds and will be discarded')
            logger.warn('will collect only {0} events...'.format(
                valid_starts.size))

        # get the dataset slices for only the channel list
        this_rec_frames = get_slice_array(dset, valid_starts, span, chan_list)
        all_frames_list.append(this_rec_frames)

    try:
        all_frames_array = np.concatenate(all_frames_list, axis=0)
    except ValueError:
        raise
        # logger.warn('Failed to collect stream frames, return is nan array')
        # zero_dset_shape = get_data_set(kwd_file, rec).shape
        # all_frames_array = np.empty([1, *zero_dset_shape])
        # all_frames_array[:] = np.nan
    return all_frames_array

def get_slice_array(dset: np.ndarray, starts: np.ndarray, span: np.int, chan_list) -> np.ndarray:
    n_slices = starts.size
    #n_chan = dset.shape[1]
    n_chan = chan_list.size
    chan_list = np.array(chan_list)
    #logger.info('nslice {}, span {}, chan {}'.format(n_slices,n_chan,chan_list))
    slices_array = np.zeros([n_slices, span, n_chan])
    #logger.info('dset {}, starts {}, span {}, chan_list {}'.format(dset.shape, starts, span, chan_list))
    #logger.info('starts {}'.format(starts.dtype))
    for i, start in enumerate(starts.astype(np.int64)):
        #logger.info('start {}'.format(start.shape))
        #logger.info('chan_list {}'.format(chan_list))
        #logger.info('i {}'.format(i))
        #logger.info('span'.format(span))
        #start = np.int(start)
        #aux = dset[5: 5 + span, chan_list]
        slices_array[i, :, :] = dset[start: start + span, chan_list]
    return slices_array