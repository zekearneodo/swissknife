# kilosort a session on (niao)
# Definitions and functions

import argparse
import glob
import logging
import os
import shutil as sh
import subprocess
import sys
from string import Template
import numpy as np

import scipy.io as sio

#quick and dirty fix
sys.path.append(os.path.join('/mnt/cube/earneodo/repos', 'swissknife'))
from swissknife.bci.core import expstruct as et
from swissknife.bci.core.file import h5_functions as h5f
import numpy as np
logger = logging.getLogger('kilosort')


def get_args():
    parser = argparse.ArgumentParser(description='Run bci_pipeline on a computer (niao for now)')
    parser.add_argument('bird', default='', nargs='?',
                        help='bird that has ss data in the cube')
    parser.add_argument('sess', default='', nargs='?',
                        help='sessin')
    return parser.parse_args()


def make_kilo_scripts(bird, sess, n_filt=None,
                      kilo_dir=os.path.abspath('/home/earneodo/repos/KiloSort'),
                      npymat_dir=os.path.abspath('/home/earneodo/repos/npy-matlab'),
                      use_gpu=True,
                      filt_per_chan=2
                      ):
    fn = et.file_names(bird, sess)
    exp_par = et.get_parameters(bird, sess)  # load the yml parameter file
    local_sort_dir = fn['folders']['tmp']
    logger.debug('local sort dir: {}'.format(local_sort_dir))
    block_name = fn['structure']['structure']
    s_f = h5f.get_record_sampling_frequency(et.open_kwd(bird, sess))
    n_chan = len(exp_par['channel_config']['neural'])
    logger.debug('n_chan: {}'.format(n_chan))
    if n_filt is None:
        n_filt = np.int(np.ceil(16*3./32.)*32)
    params = {
        'kilodir': kilo_dir,
        'npy_matdir': npymat_dir,
        'datadir': local_sort_dir,
        'blockname': block_name,
        'fs': s_f,
        'Nchan': n_chan,
        'Nfilt': int(np.ceil(n_chan/32)*32)*filt_per_chan,
        'useGPU': int(use_gpu)

    }
    logger.debug(params)
    template_dir = fn['folders']['templ']
    print(template_dir)
    with open(os.path.join(template_dir, 'master.template'), 'r') as src:
        master_template = Template(src.read())
    with open(os.path.join(template_dir, 'config.template'), 'r') as src:
        config_template = Template(src.read())

    et.mkdir_p(local_sort_dir)
    with open(et.file_path(fn, 'tmp', 'ks_mas'), 'w') as f:
        f.write(master_template.substitute(params))
    with open(et.file_path(fn, 'tmp', 'ks_cfg'), 'w') as f:
        f.write(config_template.substitute(params))


def make_kilo_chanmap(bird, sess):
    fn = et.file_names(bird, sess)
    logger.debug('Making ChanMap file')
    par = et.get_parameters(bird, sess)
    try:
        probe_serial = par['probe']['serial']
        logger.debug('Probe serial specified: {}'.format(probe_serial))
        try:
            probe_rev = par['probe']['rev']
            logger.debug('Probe rev specified: {}'.format(probe_rev))
        except KeyError:
            probe_rev = '0'
            logger.debug('Probe rev not specified: default is {}'.format(probe_rev))
        prb_file = os.path.join(fn['folders']['prb'],
                                     '{0}_{1}.prb'.format(probe_serial, probe_rev))
        logger.debug('Probe should be {}'.format(prb_file))
    except KeyError:
        logger.debug('probe not specified in par file, going for default in-folder .prb file')
        prb_file_path = et.file_path(fn, 'ss', 'kk_prb')

        assert len(glob.glob(prb_file_path)) == 1, "Error finding .prb file in {}".format(prb_file_path)
        prb_file = glob.glob(prb_file_path)[0]

    logger.info('probe file: {}'.format(prb_file))
    with open(prb_file, 'r') as f:
        contents = f.read()
    metadata = {}
    exec(contents, {}, metadata)

    n_channels = 0
    for group in metadata['channel_groups']:
        n_channels = max(n_channels, np.max(metadata['channel_groups'][group]['channels']))
    n_channels += 1

    connected = np.array([True] * n_channels).reshape((n_channels, 1))
    chan_map = np.arange(n_channels) + 1
    chan_map_zero_ind = np.arange(n_channels)

    x_coords = np.ones((n_channels, 1)) * -1
    y_coords = np.ones((n_channels, 1)) * -1
    k_coords = np.ones((n_channels, 1)) * -1

    for group in metadata['channel_groups']:
        for channel in metadata['channel_groups'][group]['geometry']:
            x_coords[channel], y_coords[channel] = metadata['channel_groups'][group]['geometry'][channel]
            k_coords[channel] = group + 1

    chan_map_dict = {'Nchannels': n_channels,
                'connected': connected,
                'chanMap': chan_map,
                'chanMap0ind': chan_map_zero_ind,
                'xcoords': x_coords,
                'ycoords': y_coords,
                'kcoords': k_coords}

    sio.savemat(et.file_path(fn, 'tmp', 'ks_map'), chan_map_dict)
    

def run_kilosort(bird, sess, no_copy=False, use_gpu=True,
                 kilo_dir=os.path.abspath('/home/earneodo/repos/KiloSort'),
                 npymat_dir=os.path.abspath('/home/earneodo/repos/npy-matlab')
                ):
    logger.info("will run bci_pipeline on bird {0} - session {1}".format(bird, sess))
    fn = et.file_names(bird, sess)
    log_file = os.path.join(fn['folders']['ss'], 'kilosort_py.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.info('Saving output in log file: {}'.format(log_file))
    logger.addHandler(fh)
    logger.info("will run bci_pipeline on bird {0} - session {1}".format(bird, sess))
    if no_copy:
        logger.info('Will not pull data from cube')
    else:
        copyed = fetch_kilo_data(bird, sess)
        logger.info('Copied {}'.format(copyed))
    logger.info('Will create the scripts')
    make_kilo_scripts(bird, sess, kilo_dir=kilo_dir, npymat_dir=npymat_dir, use_gpu=use_gpu)
    logger.info('Will do the chanMap for matlab')
    make_kilo_chanmap(bird, sess)
    logger.info('Will do the sort')
    sort_out = do_the_sort(bird, sess)
    logger.info('Sorted {}'.format(sort_out))
    pushed = push_kilo_data(bird, sess)
    logger.info('Pushed back to cube {}'.format(pushed))
    print('Finished ')
    return


def do_the_sort(bird, sess):
    logger.info('Running kilosort on matlab')
    fn = et.file_names(bird, sess)
    sort_folder = fn['folders']['tmp']
    log_file = os.path.join(fn['folders']['ss'], 'kilosort_mat.log')
    mlb_cmd = '-r "cd(\'{}\'); dir; master; exit();"'.format(sort_folder)
    log_cmd = '-logfile {}'.format(log_file)
    logger.debug('Issue command {}'.format(mlb_cmd))
    logger.info('output to {}'.format(log_file))
    sorter = subprocess.check_output(['matlab', '-nodesktop', '-nosplash', '-noawt', mlb_cmd, log_cmd])
    return sorter


def fetch_kilo_data(bird, sess):
    logger.info('Fetching data for bci_pipeline')
    files = ['ss_bin']
    fn = et.file_names(bird, sess)
    #files += [f for f in fn['files'].iterkeys() if f.split('_')[0] == 'ks']
    logger.info(files)
    copied = copy_data(bird, sess, 'ss', 'tmp', only_files=files)
    logger.debug(copied)
    return copied


def push_kilo_data(bird, sess):
    logger.info('Pushing back bci_pipeline data')
    fn = et.file_names(bird, sess)
    source_folder = fn['folders']['tmp']
    dest_folder = fn['folders']['ss']
    extensions = ['npy', 'csv', 'py', 'm']
    #print extensions
    copied = []
    for ext in extensions:
        files = glob.glob(os.path.join(source_folder, '*.{}'.format(ext)))
        copied += [sh.copy2(os.path.join(source_folder, f), dest_folder) for f in files]
    return copied


def copy_data(bird, sess, orig, dest, only_files=[], exclude_files=[]):
    fn = et.file_names(bird, sess)
    source_folder = fn['folders'][orig]
    dest_folder = fn['folders'][dest]
    et.mkdir_p(dest_folder)

    if len(only_files) > 0:
        logger.debug('Grabbing all files in list {}'.format(only_files))
        try:
            copied = map(lambda x: sh.copy2(et.file_path(fn, orig, x), dest_folder), only_files)

        except:
            logger.error('Missing files')
            raise ValueError('Missing kilosort data', 'missingFiles')

    else:
        copied = [sh.copy2(et.file_path(fn, orig, f), dest_folder) for f in fn['structure']
                  if not f in exclude_files
                  and os.path.isfile(et.file_path(fn, orig, f))]

    return copied


def main():
    args = get_args()
    try:
        run_kilosort(args.bird, args.sess)
        logger.info('Finished sorting')
        exit(0)

    except:
        logger.error('Something went wrong')
        exit(1)


if __name__ == '__main__':
    main()
