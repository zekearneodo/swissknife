import sys
import os
import logging
import glob
import argparse
import subprocess
import numpy as np


logger = logging.getLogger('bci_pipeline.archive')


def archive_folder(folder, dest_full_path,
                   dest_file_list=None,
                   compression_pars='xz'):

    logger.info('Archiving folder {0} into {1}'.format(folder, dest_full_path))

    [dest_folder, dest_file] = os.path.split(dest_full_path)
    et.mkdir_p(dest_folder)

    if dest_file_list is None:
        dest_file_list = os.paht.join(dest_folder, '{0}.{1}'.format(dest_file, 'lst'))

    logger.info('')
    archiver = subprocess.check_output(
        ['tar', '-I \"{}\"'.format(compression_pars), '-cvf',
         dest_full_path, '> {}'.format(dest_file_list)]
    )

    logger.info('Done')
    return archiver


def archive_data(bird, sess, orig, dest=None,
                 compression_pars='pxz -9 -T4',
                 compression_ext='xz'):
    logger.info('Will archive {0} data from bird {1}, sess {2}'.format(orig, bird, sess))
    fn = et.file_names(bird, sess)
    source_folder = fn['folders'][orig]

    logger.info('Source folder: {}'.format(source_folder))
    if dest is None:
        fb = et.file_names('')
        dest_in_archive = os.path.split(os.path.split(fb['folders']['ss'])[0])
        dest = os.path.join(fn['folders']['archive'],
                            dest_in_archive,
                            bird)
    logger.info('Dest. folder: {}'.format(dest))

    zip_ext = 'tar.{0}'.format(compression_ext)
    dest_file = [os.path.join(dest, '{0}.{1}'.format(sess, ext)) for ext in [zip_ext, 'lst']]

    try:
        archive_folder(source_folder, dest_file[0],
                       dest_file_list=dest_file[0],
                       compression_pars=compression_pars)
    except:
        logger.warn('Error archiving {}'.format(dest_file[0]))
