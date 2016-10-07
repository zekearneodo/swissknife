# Set of functions for appending data and reformatting the kwik file (and kwe)
from scipy.io import wavfile
from scipy import signal as ss
import numpy as np
import logging
import h5_functions as h5f
import file_tools as ft
import os
import h5py

module_logger = logging.getLogger("kwik_functions")


# add wave stimuli
# Insert a wav file with a sound into a group


def insert_sound(kf, sound_name, sound_file_path,
                 parent_group='/event_types/Stimulus',
                 stim_meta_data=None,
                 waveform_meta_data=None):
    """
    Inserts a waveform stimulus into a kwik file.

    Inserts the wav file in sound_file_path into a group under parent_group.
    It will create a group named sound_name. The group will have as default attributes:
    sampling_rate: the sampling frequency of the wav file.
    Extra parameters can be entered as a dictionary in meta_data
    The waveform is written in the dataset 'waveform' within that group.
    In principle, the name of the sound coincides with the identifier of the ../Stimulus/text dataset

    :param kf: the kwik file
    :param sound_name: name of the sound
    :param sound_file_path: path to the wavefile
    :param parent_group: stimulus group in the kwik file
    :param sitm_meta_data: dictionary of meta_data {attrib_name: value} to insert as attributes of the group.
    :yelds: inserts group sound_name with attributes and dataset with the waveform into parent_group

    """

    [s_f, data] = wavfile.read(sound_file_path)
    assert (kf[parent_group])
    sound_group = kf[parent_group].create_group(sound_name)
    waveform_group = sound_group.create_group('waveforms')
    waveform_group.attrs.create('sampling_rate', s_f, dtype='f4')

    if waveform_meta_data is not None:
        for key, val in waveform_meta_data.iteritems():
            waveform_group.attrs.create(key, val)

    if stim_meta_data is not None:
        for key, val in stim_meta_data.iteritems():
            sound_group.attrs.create(key, val)

    waveform_group.create_dataset('stimulus', data=data)


def append_stream(kf, sound_name, stream, stream_name, parent_group='/event_types/Stimulus', meta_data=None,
                  resample=False):
    waveform_group = kf[parent_group + '/' + sound_name + '/' + 'waveforms']
    assert waveform_group
    if meta_data is not None:
        for key, val in meta_data.iteritems():
            waveform_group.attrs.create(key, val)

    if resample:
        stream = ss.resample(stream, waveform_group['stimulus'].size)

    waveform_group.create_dataset(stream_name, data=stream)


def append_table_in_stim(kf, sound_name, table, table_name, parent_group='/event_types/Stimulus', meta_data=None,
                         table_fcn=None, **kwargs):
    stim_group = kf[parent_group + '/' + sound_name]

    if table_fcn is None:
        dset = stim_group.create_dataset(table_name, data=table)
        dict2attrs(meta_data, dset)
    else:
        table_fcn(kf, sound_name, table, table_name, parent_group, meta_data, **kwargs)


def read_sound(kf, sound_name, parent_group='/event_types/Stimulus'):
    """
    Reads the waveform of a sound stimulus and its meta_data

    :param kf: the kwik file
    :param sound_name: name of the sound
    :param parent_group: stimulus group in the kwik file
    :returns:
        data: n_samples x 1 numpy array with the waveform
        meta_data: dictionary of meta_data {attrib_name: value} to insert as attributes of the group.
    """
    waveforms_group = kf[parent_group + '/' + sound_name + '/' + 'waveforms']
    data = np.array(waveforms_group['stimulus'], dtype=waveforms_group['stimulus'].dtype)
    # read the meta_data
    meta_data = attrs2dict(waveforms_group)
    return data, meta_data


def list_stim_streams(kf, sound_name, parent_group='/event_types/Stimulus'):
    sound_group = kf[parent_group + '/' + sound_name + '/' + 'waveforms']
    return sound_group.keys()


def read_stim_stream(kf, sound_name, stream_name, parent_group='/event_types/Stimulus'):
    sound_wf_group = kf[parent_group + '/' + sound_name + '/' + 'waveforms']
    data = np.array(sound_wf_group[stream_name], dtype=sound_wf_group[stream_name].dtype)
    # read the meta_data
    meta_data = {'stim': attrs2dict(sound_wf_group), 'stream': attrs2dict(sound_wf_group[stream_name])}
    return data, meta_data


def read_stim_subtable(kf, sound_name, table_name, parent_group='/event_types/Stimulus'):
    stim_group = kf[parent_group + '/' + sound_name]
    data = np.array(stim_group[table_name], dtype=stim_group[table_name].dtype)
    # read the meta_data
    meta_data = attrs2dict(stim_group[table_name])
    return data, meta_data


def read_stim_groups(kf, parent_group='/event_types/Stimulus'):
    stim_group = kf[parent_group]
    return stim_group


# gets the sampling frequency of a recording
@h5f.h5_wrap
def get_record_sampling_frequency(h5, recording=0):
    path = 'recordings/{0:d}'.format(recording)
    return h5[path].attrs.get('sample_rate')


@h5f.h5_wrap
def get_rec_attrs(h5, recording):
    return h5['/recordings/{}'.format(int(recording))].attrs


@h5f.h5_wrap
def get_rec_start_sample(h5, recording):
    return get_rec_attrs(h5, recording)['start_sample']


@h5f.h5_wrap
def get_rec_list(k_file):
    """
    :param k_file: (kwik or kwd)
    :return: list of recordings in an h5file (kwik/kwd) as a sorted numpy array
    """
    return np.sort(map(int, k_file['/recordings'].keys()))


@h5f.h5_wrap
def rec_start_array(kwik):
    rec_list = map(int, get_rec_list(kwik))
    rec_array = np.arange(max(rec_list) + 1)
    start_array = np.zeros_like(rec_array)
    for i_rec in rec_list:
        start_array[i_rec] = get_rec_start_sample(kwik, i_rec)
    return start_array


@h5f.h5_wrap
def get_corresponding_rec(kwik, stamps):
    '''
    Get the vector of recordings to which an array of absolute stamps belong
    :param kwik: kwik_file (open in r mode), to get the rec starts
    :param stamps: a list of stamps, absolute, refering to beginning of rec 0
    :return: recs
    '''
    rec_list = get_rec_list(kwik)
    rec_starts = rec_start_array(kwik)
    stamps_recs = np.array([rec_list[np.where(rec_starts < start)[0][-1]] for start in stamps])
    return stamps_recs


def apply_rec_offset(h5, stamp_array, rec_array):
    rec_offsets = rec_start_array(h5)
    return apply_offsets(stamp_array, rec_array, rec_offsets)


def apply_offsets(stamps, recs, offset_array):
    offsets = map(lambda i: offset_array[i], recs)
    return stamps + offsets


# List all the units in a file
def list_units(kf, group=0, sorted=True):
    # get the unit group
    qlt_path = "/channel_groups/{0:d}/clusters/main".format(group)

    g_dtype = np.int
    clu_dtype = np.int
    qlt_dtype = np.int

    clu_list = kf[qlt_path].keys()
    qlt_list = [kf["{0:s}/{1:s}".format(qlt_path, c)].attrs.get('cluster_group') for c in clu_list]
    n_spikes = len(clu_list)
    clu_dt = np.dtype([('group', g_dtype, 1), ('clu', clu_dtype, 1), ('qlt', qlt_dtype, 1)])
    clu = np.recarray(n_spikes, dtype=clu_dt)

    clu['group'] = group
    clu['clu'] = map(int, clu_list)
    clu['qlt'] = map(int, qlt_list)

    if sorted:
        clu = clu[(clu['qlt'] == 1) | (clu['qlt'] == 2)]
    return clu


# List all the stimuli in a file
def list_sound_stimuli(h5, stim_table_path='/event_types/Stimulus/text'):
    datagroup = h5[stim_table_path]
    all_stim = np.array([s for s in datagroup[:] if not is_number(s)])
    return np.unique(all_stim)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def attrs2dict(node):
    return {key: val for key, val in node.attrs.iteritems()}


def dict2attrs(meta_dict, node):
    if meta_dict is not None:
        assert node
        for key, val in meta_dict.iteritems():
            node.attrs.create(key, val)


class KwikFile:
    def __init__(self, file_names, chan_group=0):
        self.file_names = file_names
        if file_names['clu']:
            self.clu = np.squeeze(np.load(file_names['clu']))
        elif file_names['temp']:
            self.clu = np.squeeze(np.load(file_names['temp']))
        else:
            raise IOError('both spike_clusters.npy and spike_templates.npy weren\'t found')
        self.spk = np.load(file_names['spk'])
        if file_names['grp']:
            self.grp = load_grp_file(file_names['grp'])
        else:
            self.grp = [(i, 'unsorted') for i in np.unique(self.clu)]
        self.rec_kwik = None
        self.spk_kwik = None
        self.kwf = None
        self.chan_group = chan_group

        with open(file_names['par']) as f:
            exec (f.read())
            self.s_f = sample_rate

        self.create_kwf()

    def create_kwf(self):
        with h5py.File(self.file_names['kwk'], 'w') as kwf:
            kwf.create_group('/channel_groups')
            kwf.create_group('/recordings')

    def make_spk_tables(self):
        with h5py.File(self.file_names['kwd'], 'r') as kwd:
            rec_sizes = h5f.get_rec_sizes(kwd)
            self.rec_kwik, self.spk_kwik = ref_to_rec_starts(rec_sizes, self.spk)

        with h5py.File(self.file_names['kwk'], 'r+') as kwf:
            chan_group = kwf['/channel_groups'].require_group('{}'.format(self.chan_group))
            spikes_group = chan_group.require_group('spikes')
            insert_table(spikes_group, self.rec_kwik.flatten(), 'recording')
            insert_table(spikes_group, self.spk_kwik.flatten(), 'time_samples')
            insert_table(spikes_group, self.spk_kwik.flatten() / self.s_f, 'time_fractional')

            clusters_group = spikes_group.require_group('clusters')
            insert_table(clusters_group, self.clu, 'main')
            insert_table(clusters_group, self.clu, 'original')

    def make_rec_groups(self):
        rec_list = np.unique(self.rec_kwik)
        rec_start_samples = h5f.get_rec_starts(self.file_names['kwd'])

        with h5py.File(self.file_names['kwk'], 'r+') as kwf:
            rec_group = kwf.require_group('recordings')
            for rec in rec_list:
                rec_name = 'recording_{}'.format(rec)
                attribs = [{'name': 'name', 'data': rec_name, 'dtype': 'S{}'.format(len(rec_name))},
                           {'name': 'sample_rate', 'data': self.s_f, 'dtype': np.dtype(np.float64)},
                           {'name': 'start_sample', 'data': rec_start_samples[rec], 'dtype': np.int64},
                           {'name': 'start_time', 'data': rec_start_samples[rec] / self.s_f, 'dtype': np.float64}]
                insert_group(rec_group, str(rec), attribs)

    def make_clu_groups(self, name='main'):
        clu_grp_dict = {'good': 2,
                        'mua': 1,
                        'noise': 0,
                        'unsorted': 3}

        with h5py.File(self.file_names['kwk'], 'r+') as kwf:
            chan_group = kwf['/channel_groups'].require_group('{}'.format(self.chan_group))
            clusters_group = chan_group.require_group('clusters')
            desc_group = clusters_group.require_group(name)

            for clu in self.grp:
                attribs = [{'name': 'cluster_group',
                            'data': clu_grp_dict[clu[1]],
                            'dtype': np.int64}]

                insert_group(desc_group, str(clu[0]), attribs)


def make_shank_kwd(raw_file, out_file_path, chan_list):
    ss_file = h5py.File(out_file_path, 'w')
    copy_attribs(raw_file, ss_file)
    ss_file.create_group('/recordings')
    create_data_groups(raw_file, ss_file, chan_list)
    ss_file.close()


def insert_table(group, table, name, attr_dict=None):
    return group.create_dataset(name, data=table)


def insert_group(parent_group, name, attr_dict_list=None):
    new_group = parent_group.create_group(name)
    if attr_dict_list is not None:
        append_atrributes(new_group, attr_dict_list)
    return new_group


def append_atrributes(h5obj, attr_dict_list):
    for attr_dict in attr_dict_list:
        # print attr_dict['name'] + ' {0} - {1}'.format(attr_dict['data'], attr_dict['dtype'])
        h5obj.attrs.create(attr_dict['name'], attr_dict['data'], dtype=attr_dict['dtype'])
        # h5obj.attrs.create(attr['name'], attr['data'], dtype=attr['dtype'])


def load_grp_file(grp_file_path):
    if os.path.exists(grp_file_path):
        return np.loadtxt(grp_file_path,
                          dtype={'names': ('cluster_id', 'group'),
                                 'formats': ('i2', 'S8')},
                          skiprows=1)
    else:
        return None


# offset the recs
def ref_to_rec_starts(rec_sizes, spk_array):
    start = 0
    spk_rec = np.empty_like(spk_array)
    rec_array = np.empty_like(spk_array)

    for rec, size in rec_sizes.iteritems():
        end = start + size
        this_rec_spk = (spk_array > start) & (spk_array < end)
        spk_rec[this_rec_spk] = spk_array[this_rec_spk] - start
        rec_array[this_rec_spk] = rec
        start = end

    return rec_array, spk_rec


def kilo_to_kwik(bird, sess, file_names=None, location='ss'):
    module_logger.info('Creating kwik file for bird: {} sess: {}'.format(bird, sess))
    if file_names is None:
        file_names = dict(
            clu='spike_clusters.npy',
            spk='spike_times.npy',
            grp='cluster_groups.csv',
            par='params.py',
            temp='spike_templates.npy',
            kwd='experiment.raw.kwd',
            kwk='experiment.kwik')

    fn = ft.experiment.file_names(bird, sess)
    for key, value in file_names.iteritems():
        file_names[key] = os.path.join(fn['folders'][location], value)

    # Check whether there is manual sort or not:
    if not os.path.isfile(file_names['clu']):
        module_logger.info('Clu not found, will assume no manual sorting was done')
        file_names['clu'] = None
        file_names['grp'] = None
        module_logger.debug(file_names)
    else:
        module_logger.info('Found clu file, will attempt to unpack manual sorted data from kilosort')
        file_names['temp'] = None
        module_logger.debug(file_names)

    k = KwikFile(file_names)
    module_logger.info('Making spike tables')
    k.make_spk_tables()
    module_logger.info('Making rec tables')
    k.make_rec_groups()
    module_logger.info('Making cluster group tables')
    k.make_clu_groups()
    module_logger.info('Done')
