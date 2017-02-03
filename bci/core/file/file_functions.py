import logging
import os
import shutil
import glob
import pickle

logger = logging.getLogger('bci.core.file.file_functions')


def list_folders(folder_path):
    return [i for i in glob.glob(os.path.join(folder_path, '*')) if os.path.isdir(i)]


def make_backup(file_path):
    [f_p, f_n] = os.path.split(file_path)
    bk_f_n = '{}.bk'.format(f_n)
    shutil.copyfile(file_path, os.path.join(f_p, bk_f_n))
    return bk_f_n


def save_obj(obj, full_file_path):
    with open(full_file_path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(full_file_path):
    with open(full_file_path, 'rb') as f:
        return pickle.load(f)