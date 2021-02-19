import errno
import glob
import logging
import os
import socket

import h5py
# for more than structure
import numpy as np
import pandas as pd
import yaml
from numpy.lib import recfunctions as rf

from swissknife.bci.core.file import file_functions as ff
from swissknife.bci.core.file import h5_functions as h5f

logger = logging.getLogger('bci.core.expstruct')


# file structure

def get_definitions_dictionaries():
    # packages is a dictionary with {'description': 'path relative to repos folder'}
    packages = {'sound': 'soundflow',
                'ephys': 'ephys_flow',
                'analysis': 'analysis_tools',
                'swiss': 'swissknife'}

    locations = {'passaro':
                     {'repos': os.path.abspath('/mnt/cube/earneodo/repos'),
                      'experiment': os.path.abspath('/mnt/cube/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/usr/local/experiment'),
                      'archive': os.path.abspath('/mnt/cube/earneodo/archive/bci_zf'),
                      'store': os.path.abspath('/Data/bci_zf'),
                      'scratch': os.path.abspath('/usr/local/experiment/scratchpad')},
                 'pakhi':
                     {'repos': os.path.abspath('/mnt/cube/earneodo/repos'),
                      'experiment': os.path.abspath('/mnt/cube/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/home/earneodo/bci_zf'),
                      'archive': os.path.abspath('/mnt/sphere/earneodo/archive/bci_zf'),
                      'scratch': os.path.abspath('/home/earneodo/scratchpad')},
                 'niao':
                     {'repos': os.path.abspath('/mnt/cube/earneodo/repos'),
                      'experiment': os.path.abspath('/mnt/cube/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/home/earneodo/bci_zf'),
                      'archive': os.path.abspath('/mnt/cube/earneodo/archive/bci_zf')},
                 'lookfar':
                     {'repos': os.path.abspath('/Users/zeke/repos'),
                      'experiment': os.path.abspath('/Volumes/gentner/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/Users/zeke/bci_zf')},
                 'txori':
                     {'repos': os.path.abspath('/mnt/cube/earneodo/repos'),
                      'experiment': os.path.abspath('/mnt/cube/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/home/earneodo/bci_zf'),
                      'archive': os.path.abspath('/mnt/cube/earneodo/archive/bci_zf'),
                      'scratch': os.path.abspath('/home/earneodo/scratchpad')},
                 }

    return {'packages': packages,
            'locations': locations}


def get_computer_name():
    return socket.gethostname()


def get_locations(dict_path=None, comp_name=None):
    if dict_path is None:
        if comp_name is None:
            comp_name = get_computer_name()
            locations = get_definitions_dictionaries()['locations'][comp_name]
    else:
        #
        raise NotImplementedError('Still dont know how to load a dictionary of locations')
    return locations


def set_paths(repos_root=None):
    if repos_root is None:
        repos_root
    pass


def is_none(x):
    return 'none' == str(x).lower()


def flex_file_names(bird, sess='', rec=0, experiment_folder=None, base='experiment', location='experiment'):
    fn = file_names(bird, sess, rec, experiment_folder, base)

    exp_base = fn['locations'][location]
    folders = {'raw': os.path.join(exp_base, 'raw_data', bird, sess),  # local raw
           'ss': os.path.join(exp_base, 'ss_data', bird, sess),
           'rw': os.path.join(exp_base, 'raw_data', bird, sess),  # stored raw
           'stim': os.path.join(exp_base, 'stim_data', bird, sess),
           'tmp': os.path.join(exp_base, 'tmp_data', bird, sess),
           'templ': os.path.join(exp_base, 'templates'),
           'prb': os.path.join(exp_base, 'probes')}

    fn['folders'] = folders
    return fn


def read_yml(file_path):
    with open(file_path, 'r') as f:
        contents = yaml.load(f)
    return contents

def file_names(bird, sess='', rec=0, experiment_folder=None, base='experiment'):
    computer_name = get_computer_name()

    if experiment_folder is None:
        experiment_folder = get_definitions_dictionaries()['locations'][computer_name]['experiment']
        experiment_local = get_definitions_dictionaries()['locations'][computer_name]['experiment_local']
        base_location = get_definitions_dictionaries()['locations'][computer_name]
    else:
        experiment_local = experiment_folder
        base_location = {'experiment': os.path.abspath(experiment_folder),
                         'experiment_local': os.path.abspath(experiment_folder),
                         'store': os.path.abspath(experiment_folder),
                         'archive': os.path.abspath(experiment_folder)}

    folders = {'raw': os.path.join(experiment_local, 'raw_data', bird, sess),  # local raw
               'ss': os.path.join(experiment_folder, 'ss_data', bird, sess),
               'rw': os.path.join(experiment_folder, 'raw_data', bird, sess),  # stored raw
               'proc': os.path.join(experiment_folder, 'proc_data', bird, sess), #processed data
               'stim': os.path.join(experiment_folder, 'stim_data', bird, sess),
               'tmp': os.path.join(experiment_local, 'tmp_data', bird, sess),
               'templ': os.path.join(experiment_folder, 'templates'),
               'prb': os.path.join(experiment_folder, 'probes'),
               'kai': os.path.join(os.path.abspath('/mnt/cube/kai/results'), bird, sess)}


    files = {'structure': base,
             'ss_raw': base + '.raw.kwd',
             'ss_lfp': base + '.lfp.kwd',
             'ss_bin': base + '.dat',
             'ss_par': base + '.par.yml',
             'par': base + '.par.yml',
             'sng': base + '.sng.kwe',
             'stm': base + '.stm.kwe',
             'cand': base + '.mot.h5',
             'evt': base.split('_')[0] + '.kwe',
             'mic': base + '-rec_{0:03}.mic.wav'.format(int(rec)),
             'sts': base + '-rec_{0:03}.sts.wav'.format(int(rec)),
             'kk_prb': '*.prb',
             'kk_par': 'params.prm',
             'ks_par': 'params.py',
             'ks_mas': 'master.m',
             'ks_map': 'chanMap.mat',
             'ks_cfg': 'config.m'}

    return {'folders': folders,
            'structure': files,
            'locations': base_location}


def file_path(fn_dict, folder_key, file_key):
    """
    :param fn_dict: dictionary of file_names (as output of file_names)
    :param folder_key: string, key to folders (folder type)
    :param file_key: string, key to structure (file type)
    :return:
    """
    return os.path.join(fn_dict['folders'][folder_key], fn_dict['structure'][file_key])


def mkdir_p(path):
    logger.debug('Creating directory {}'.format(path))
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            logger.debug('Directory existed, doing nothing')
            pass
        else:
            raise

def sub_dirs(path):
    return [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)]


def list_birds(folder, breed='z'):
    all_dirs = [os.path.split(d)[-1] for d in sub_dirs(folder)]
    all_birds = [b for b in all_dirs if b.startswith(breed)]
    all_birds.sort()
    return all_birds

def list_sessions(bird, experiment_folder=None, location='ss'):
    fn = file_names(bird, experiment_folder=experiment_folder)
    bird_folder = fn['folders'][location]
    sessions_bird = [s for s in os.listdir(bird_folder) if os.path.isdir(os.path.join(bird_folder, s))]
    return sessions_bird



def list_raw_sessions(bird, sess_day=None, depth='', experiment_folder=None, location='raw'):
    all_sessions = list_sessions(bird, experiment_folder=experiment_folder, location=location)
    if sess_day is not None:
        all_sessions = [s for s in all_sessions if sess_day in s]
    if depth != '':
        all_sessions = [s for s in all_sessions if int(s[0].split('_')[-1]) == int(depth)]
    all_depths = ['{}'.format(s.split('_')[-1]) for s in all_sessions]
    return all_sessions, all_depths


def get_sessions_info_pd(breed='z', location='rw'):
    folder = file_names('')['folders'][location]
    info_pd = pd.DataFrame(list_birds(folder, breed=breed), columns=['bird'])
    info_pd['sessions'] = info_pd['bird'].apply(lambda x: list_sessions(x, location=location))
    return info_pd


# Experiment structure
def get_parameters(bird, sess, rec=0, experiment_folder=None, location='ss'):
    fn = file_names(bird, sess, rec, experiment_folder=experiment_folder)
    pars = read_yml(os.path.join(fn['folders'][location], fn['structure']['par']))
    return pars


def update_parameters(new_par, bird, sess, rec=0, experiment_folder=None, location='ss'):
    fn = file_names(bird, sess, rec, experiment_folder=experiment_folder)
    par_file_path = os.path.join(fn['folders'][location], fn['structure']['par'])
    bkp_path = ff.make_backup(par_file_path)
    logger.info('Overwriting parameter file; backup in {}'.format(bkp_path))
    with open(par_file_path, 'w') as f:
        written = yaml.dump(new_par, f)
    return written


def get_stims_dict(bird, sess, rec=0, experiment_folder=None, location='ss'):
    """
    get the dictionary of stimuli {name: file_name.wav}
    :param bird:
    :param sess:
    :param rec:
    :param experiment_folder:
    :param location:
    :return: dict
    """
    exp_pars = get_parameters(bird, sess,
                              rec=rec,
                              experiment_folder=experiment_folder,
                              location=location)
    return exp_pars['search_motiff']['motiff_patterns']


def stim_id(exp_par, name):
    """
    Get a stimulus name's id in the .evt file of events (its group name)
    :param exp_par: parameters of the experiment (read .yml file)
    :param name: the name of the stimulus in the 'motiff_patterns' dictionary
    :return: the id of the parameter
    """
    patterns = exp_par['search_motiff']['motiff_patterns']
    return patterns[name].split('.wav')[0]


def load_probe(bird, sess, experiment_folder=None, location='ss', override_path=None):
    logger.debug('loading probe')

    if override_path:
        logger.debug('Will load probe from specified file {}'.format(override_path))
        prb_file_path = override_path
    else:
        fn = file_names(bird, sess, experiment_folder=experiment_folder)
        par = get_parameters(bird, sess, experiment_folder=experiment_folder, location=location)
        probe_serial = par['probe']['serial'] if 'serial' in par['probe'].keys() else None
        probe_model = par['probe']['model'] if 'model' in par['probe'].keys() else None
        logger.debug('Probe serial {0}, model {1}'.format(probe_serial, probe_model))
        try:
            if not is_none(probe_serial):
                logger.debug('Probe serial specified, using it to define file name')
                probe_name = probe_serial
            elif not is_none(probe_model):
                logger.debug('Probe model specified, using it to define file name')
                probe_name = probe_model
            else:
                raise ValueError('No valid descriptor of the probe')

            try:
                probe_rev = par['probe']['rev']
                logger.debug('Probe rev specified: {}'.format(probe_rev))
            except KeyError:
                probe_rev = '0'
                logger.debug('Probe rev not specified: default is {}'.format(probe_rev))

            prb_file_path = os.path.join(fn['folders']['prb'],
                                    '{0}_{1}.prb'.format(probe_name, probe_rev))
            logger.debug('Probe should be {}'.format(prb_file_path))
        except ValueError:
            logger.debug('probe not specified in par file, going for default in-folder .prb file')
            prb_file_path = file_path(fn, 'ss', 'kk_prb')

    assert len(glob.glob(prb_file_path)) == 1, "Error finding .prb file in {}".format(prb_file_path)
    prb_file = glob.glob(prb_file_path)[0]

    logger.info('probe file: {}'.format(prb_file))
    with open(prb_file, 'r') as f:
        contents = f.read()
    metadata = {}
    exec (contents, {}, metadata)
    return metadata


def open_kwd(bird_id, sess, location='ss'):
    fn = file_names(bird_id, sess, 0)
    kwd_file_path = file_path(fn, location, 'ss_raw')
    kwd_file = h5py.File(kwd_file_path, 'r')
    return kwd_file


def open_kwik(bird_id, sess, shank=None, location='ss'):
    fn = file_names(bird_id, sess, 0)
    ss_path = fn['folders'][location]
    if shank is None:
        kwik_file_name = '{0}.kwik'.format(fn['structure']['structure'])
    else:
        kwik_file_name = '{0}_{1:01}.kwik'.format(fn['structure']['structure'], int(shank))
    kwik_file_path = os.path.join(ss_path, kwik_file_name)
    kwik_file = h5py.File(kwik_file_path, 'r')
    return kwik_file

def open_kwe(bird_id, sess, location='ss'):
    fn = file_names(bird_id, sess, 0)
    kwe_file_file_path = file_path(fn, location, 'sng')
    kwe_file = h5py.File(kwe_file_file_path, 'r')
    return kwe_file


def get_shank_files_list(bird_id, sess, location='ss'):
    fn = file_names(bird_id, sess, 0)
    ss_path = fn['folders'][location]
    kwik_file_paths = glob.glob(os.path.join(ss_path, '*.kwik'))
    shanks = [os.path.split(path)[-1].split('.kwik')[0].split('_')[-1] for path in kwik_file_paths]

    return np.sort(np.array(map(int, shanks)))


def get_shanks_list(bird_id, sess, location='ss'):
    with open_kwik(bird_id, sess, shank=None, location=location) as kwik_file:
        shanks = h5f.get_shank_list(kwik_file)
    return shanks


def get_rec_list(bird, sess, location='ss'):
    kwd_file = open_kwd(bird, sess, location)
    rec_list = list(kwd_file['/recordings'].keys())
    kwd_file.close()
    return rec_list


def get_rec_attribs(bird, sess, location='ss'):
    rec_list = get_rec_list(bird, sess, location=location)

    return rec_list


# these should go to a different file
def get_bird_events(bird, sessions_list=[], event_type=None, event_names=[], experiment_folder=None, location='ss'):
    bird_sessions = list_sessions(bird, experiment_folder=None,
                                  location=location) if sessions_list == [] else sessions_list
    ev_stack = []
    for sess in bird_sessions:
        sess_events = get_one_sess_events(bird, sess,
                                          event_type=event_type,
                                          event_names=event_names,
                                          experiment_folder=experiment_folder,
                                          location=location)
        if sess_events is not None:
            ev_stack.append(sess_events)

    return rf.stack_arrays(ev_stack, asrecarray=True, usemask=False)


def get_one_sess_events(bird, sess, event_type=None, event_names=[], experiment_folder=None, location='ss'):
    fn = file_names(bird, sess, experiment_folder=experiment_folder)
    kwe_path = os.path.join(fn['folders'][location], fn['structure']['sng'])
    try:
        if event_type is None:
            events = h5f.get_all_events(kwe_path)
        else:
            events = h5f.get_events_one_type(event_type,
                                             ev_names=event_names,
                                             rec=None)
        sess_table = np.array(['{}'.format(sess) for i in range(events.size)], dtype='|S32')

        return rf.append_fields(events, 'sess', sess_table, asrecarray=True, usemask=False)
    except:
        return None
