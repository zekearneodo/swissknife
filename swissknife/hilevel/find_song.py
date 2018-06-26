import argparse
import logging
import sys
import traceback
import os
import glob
import csv
import datetime
from joblib import Parallel, delayed

from swissknife.streamtools import findsong
from swissknife.bci.core import expstruct as et
import swissknife.streamtools.findbout as fb
from swissknife.threadtools.threadedfunction import threadedFunction

module_logger = logging.getLogger('swissknife.hilevel.find_song')


def get_args():
    parser = argparse.ArgumentParser(
        description='Search song of a bird (from usual locations on cube'
    )
    parser.add_argument('bird', default = '', nargs='?',
                       help='bird that has ss data in the cube')
    parser.add_argument('sess', default = '', nargs='?',
                       help='session. (If no session entered, '
                            'will assume that first argument is the path to a list, '
                            'containing (csv) pairs bird, sess')
    return parser.parse_args()


def list_flatten(lists):
    return [t for sublist in lists for l in sublist for t in l]

def load_csv_list(list_file_path):
    '''
    Loads a csv file with a two-column list of bird, sess
    :param list_file_path: str, path of the file
    :return: list of the contents of each rows (lists of 2 elems)
    '''
    with open(list_file_path) as list_file:
        csv_reader = csv.reader(list_file)
        stripped_read = [[x.strip() for x in row] for row in csv_reader]
    return stripped_read


def all_bird_sessions(raw_folder):
    return list(os.walk(raw_folder))[0][1]


def all_day_wavs(day_folder):
    return glob.glob(os.path.join(day_folder, '*.wav'))


def get_day_files(raw_data_folder_bird, day):
    day_path = os.path.join(raw_data_folder_bird, day)
    module_logger.info('Getting all wav names for day {}'.format(day_path))
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
    module_logger.info('Set to find song for bird {}, sess {}'.format(bird_id, sess_day))
    # Parameters of the search
    bit_size = 50  # ms; size of sound bits
    refractory = 5  # bits; refractory period in bits
    bits_bout = 2  # bits; how many bouts together is considered a bout
    threshold = 20  # threshold in mads (median deviation of the median)

    # The band where we expect most of the energy of zf song to be in
    filt_lo = 10000  # Hz
    filt_hi = 300  # Hz

    raw_data_folder_bird = et.file_names(bird_id)['folders'][raw_location]
    ss_data_folder_bird = et.file_names(bird_id)['folders']['ss']
    et.mkdir_p(ss_data_folder_bird)
    raw_file_list = get_bird_files(raw_data_folder_bird, only_days=[sess_day])
    all_raw_file_list = [x for x in raw_file_list if 'auto' not in x]
    all_raw_file_list.sort()
    raw_file_path = all_raw_file_list[-1]
    raw_path, raw_fname = os.path.split(raw_file_path)
    #raw_path.replace('raw_data', 'ss_data')

    # create file handler which logs even debug messages
    log_f_name = os.path.join(ss_data_folder_bird, 'search_song_{}.log'.format(sess_day))
    module_logger.info('Saving log to {}'.format(log_f_name))
    fh = logging.FileHandler(log_f_name)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    # add the handlers to the logger
    module_logger.addHandler(fh)
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
    module_logger.info('done with the session {} : {}'.format(bird_id, sess_day))
    return bird_id, sess_day

def try_find_song(bird_id, sess_day):
    '''
    A wrapped find_song that retunrs None if the function fails.
    (Had to do this instead of a neat wrapper because joblib was not working with wrapped functions as of 0.11
    :param bird_id:
    :param sess_day:
    :return:
    '''
    module_logger.info('Try find song {}/{}'.format(bird_id, sess_day))
    try:
        return find_song(bird_id, sess_day)
    except:
        module_logger.warning('Error finding song for bird {} : sess {}'.format(bird_id, sess_day))
        module_logger.warning(traceback.print_tb(sys.exc_info()[2]))
        return None


def find_song_list(list_file_path, n_jobs=6):
    module_logger.info('Wil search song in list of sessions contained in {}'.format(list_file_path))
    sess_list = load_csv_list(list_file_path)
    module_logger.info('The list has {} sessions'.format(len(sess_list)))
    module_logger.info('List of sessions is {}'.format(sess_list))
    done_sessions = Parallel(n_jobs=n_jobs)(delayed(try_find_song)(s[0], s[1]) for s in sess_list)
    if None in done_sessions:
        module_logger.warning('There were errors for some sessions')
    module_logger.info('Good for sessions {}'.format(done_sessions))

    today_str = datetime.date.today().isoformat()
    list_file_bk_path = '{}.{}.bk'.format(list_file_path, today_str)

    os.rename(list_file_path, list_file_bk_path)
    module_logger.info('List of sessions renamed to {}'.format(list_file_bk_path))
    return done_sessions


def main():
    args = get_args()
    module_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)
    module_logger.addHandler(handler)

    if args.sess == '':
        sess_list_path = args.bird
        module_logger.info('No sess entered, Will go through csv list of birds, sess in file {}'.format(sess_list_path))
        find_song_list(sess_list_path)
    else:
        module_logger.info('Will find songs on bird {}, sess {}'.format(args.bird, args.sess))
        find_song(args.bird, args.sess)

    #try:
    #find_song(args.bird, args.sess)
        #module_logger.info('Finished searching')
    #except:
    #    logger.error('Something went wrong')
    #    sys.exit(1)
    sys.exit(0)

if __name__ == '__main__':
    main()
