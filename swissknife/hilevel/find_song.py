import argparse
import logging
import sys
import os
import glob

from swissknife.streamtools import findsong
from swissknife.bci.core import expstruct as et
import swissknife.streamtools.findbout as fb
from swissknife.threadtools.threadedfunction import threadedFunction


def get_args():
    parser = argparse.ArgumentParser(
        description='Search song of a bird (from usual locations on cube'
    )
    parser.add_argument('bird', default = '', nargs='?',
                       help='bird that has ss data in the cube')
    parser.add_argument('sess', default = '', nargs='?',
                       help='session')
    return parser.parse_args()


def list_flatten(lists):
    return [t for sublist in lists for l in sublist for t in l]


def all_bird_sessions(raw_folder):
    return list(os.walk(raw_folder))[0][1]


def all_day_wavs(day_folder):
    return glob.glob(os.path.join(day_folder, '*.wav'))


def get_day_files(raw_data_folder_bird, day):
    day_path = os.path.join(raw_data_folder_bird, day)
    logging.info('Getting all wav names for day {}'.format(day_path))
    all_wavs = all_day_wavs(day_path)
    return all_wavs


def get_bird_files(raw_data_folder_bird, only_days=None):
    if only_days:
        all_days = only_days
    else:
        all_days = all_bird_sessions(raw_data_folder_bird)
    all_bird_files = [get_day_files(raw_data_folder_bird, day) for day in all_days]
    return [x for sublist in all_bird_files for x in sublist]


def find_song(bird_id, sess_day, raw_location='rw'):
    # Parameters of the search
    bit_size = 50  # ms; size of sound bits
    refractory = 5  # bits; refractory period in bits
    bits_bout = 2  # bits; how many bouts together is considered a bout
    threshold = 20  # threshold in mads (median deviation of the median)

    # The band where we expect most of the energy of zf song to be in
    filt_lo = 10000  # Hz
    filt_hi = 300  # Hz

    raw_data_folder_bird = et.file_names(bird_id)['folders'][raw_location]
    raw_file_list = get_bird_files(raw_data_folder_bird, only_days=[sess_day])
    all_raw_file_list = [x for x in raw_file_list if 'auto' not in x]
    all_raw_file_list.sort()
    raw_file_path = all_raw_file_list[-1]
    raw_path, raw_fname = os.path.split(raw_file_path)
    #raw_path.replace('raw_data', 'ss_data')

    # create file handler which logs even debug messages
    logger = logging.getLogger()
    log_f_name = os.path.join(raw_data_folder_bird, 'search_song_{}.log'.format(sess_day))
    logger.info('Saving log to {}'.format(log_f_name))
    fh = logging.FileHandler(log_f_name)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    found_songs = []
    for raw_file_path in all_raw_file_list:
        raw_path, raw_fname = os.path.split(raw_file_path)
        ss_path = raw_path.replace('raw_data', 'ss_data')
        et.mkdir_p(ss_path)
        bout_file_path = os.path.join(ss_path,
                                      '{}_auto.wav'.format(raw_fname.split('.')[0]))
        found = findsong.get_all_bouts(raw_file_path,
                                       bout_file_path=bout_file_path,
                                       bit_size=bit_size,
                                       refractory=refractory,
                                       bits_bout=bits_bout,
                                       threshold=threshold,
                                       filt_lo=filt_lo,
                                       filt_hi=filt_hi)
        found_songs.append(found)
    logger.info('done')


def main():
    args = get_args()
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info('Will find songs on bird {}, sess {}'.format(args.bird, args.sess))
    try:
        find_song(args.bird, args.sess)
        logger.info('Finished searching')
    except:
        logger.error('Something went wrong')
        sys.exit(1)

    sys.exit(0)

if __name__ == '__main__':
    main()
