import socket
import os
import sys
import logging
import yaml
import h5py
import glob
import errno

# for more than files
import numpy as np
from numpy.lib import recfunctions as rf

os.path.split(os.path.abspath(os.curdir))
import h5_functions as h5f

logger = logging.getLogger('expstruct')
# file structure

def get_definitions_dictionaries():
    # packages is a dictionary with {'description': 'path relative to repos folder'}
    packages = {'sound': 'soundflow',
                'ephys': 'ephys_flow',
                'analysis': 'analysis_tools'}

    locations = {'passaro':
                     {'repos': os.path.abspath('/mnt/cube/earneodo/repos'),
                      'experiment': os.path.abspath('/mnt/cube/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/usr/local/experiment')},
                 'txori':
                     {'repos': os.path.abspath('/mnt/cube/earneodo/repos'),
                      'experiment': os.path.abspath('/mnt/cube/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/mnt/cube/earneodo/bci_zf')},
                 'niao':
                     {'repos': os.path.abspath('/mnt/cube/earneodo/repos'),
                      'experiment': os.path.abspath('/mnt/cube/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/home/earneodo/bci_zf')},
                 'lookfar':
                     {'repos': os.path.abspath('/Users/zeke/repos'),
                      'experiment': os.path.abspath('/Volumes/gentner/earneodo/bci_zf'),
                      'experiment_local': os.path.abspath('/Users/zeke/bci_zf')}
                 }

    return {'packages': packages,
            'locations': locations}


def get_computer_name():
    return socket.gethostname()


def set_paths(repos_root=None):
    if repos_root is None:
        repos_root
    pass


def file_names(bird, sess='', rec=0, experiment_folder=None, base='experiment'):
    computer_name = get_computer_name()

    if experiment_folder is None:
        experiment_folder = get_definitions_dictionaries()['locations'][computer_name]['experiment']
        experiment_local = get_definitions_dictionaries()['locations'][computer_name]['experiment_local']
    else:
        experiment_local = experiment_folder

    folders = {'raw': os.path.join(experiment_local, 'raw_data', bird, sess),  # local raw
               'ss': os.path.join(experiment_folder, 'ss_data', bird, sess),
               'rw': os.path.join(experiment_folder, 'raw_data', bird, sess),  # stored raw
               'stim': os.path.join(experiment_folder, 'stim_data', bird, sess),
               'tmp': os.path.join(experiment_local, 'tmp_data', bird, sess),
               'templ': os.path.join(experiment_folder, 'templates'),
               'prb': os.path.join(experiment_folder, 'probes'),
               'kai': os.path.join(os.path.abspath('/mnt/cube/kai/results'), bird, sess)}

    files = {'base': base,
             'ss_raw': base + '.raw.kwd',
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
            'files': files}


def file_path(fn_dict, folder_key, file_key):
    """
    :param fn_dict: dictionary of file_names (as output of file_names)
    :param folder_key: string, key to folders (folder type)
    :param file_key: string, key to files (file type)
    :return:
    """
    return os.path.join(fn_dict['folders'][folder_key], fn_dict['files'][file_key])


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def list_sessions(bird, experiment_folder=None, location='ss'):
    fn = file_names(bird, experiment_folder=experiment_folder)
    bird_folder = fn['folders'][location]
    sessions_bird = [s for s in os.listdir(bird_folder) if os.path.isdir(os.path.join(bird_folder, s))]
    return sessions_bird


# Experiment files
def get_parameters(bird, sess, rec=0, experiment_folder=None, location='ss'):
    fn = file_names(bird, sess, rec, experiment_folder=experiment_folder)
    with open(os.path.join(fn['folders'][location], fn['files']['par']), 'r') as f:
        pars = yaml.load(f)
    return pars


def open_kwd(bird_id, sess, location='ss'):
    fn = file_names(bird_id, sess, 0)
    kwd_file_file_path = file_path(fn, location, 'ss_raw')
    kwd_file = h5py.File(kwd_file_file_path, 'r')
    return kwd_file


def open_kwik(bird_id, sess, shank=None, location='ss'):
    fn = file_names(bird_id, sess, 0)
    ss_path = fn['folders'][location]
    if shank is None:
        kwik_file_name = '{0}.kwik'.format(fn['files']['base'])
    else:
        kwik_file_name = '{0}_{1:01}.kwik'.format(fn['files']['base'], int(shank))
    kwik_file_path = os.path.join(ss_path, kwik_file_name)
    kwik_file = h5py.File(kwik_file_path, 'r')
    return kwik_file


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
    rec_list = kwd_file['/recordings'].keys()
    kwd_file.close()
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
    kwe_path = os.path.join(fn['folders'][location], fn['files']['sng'])
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
