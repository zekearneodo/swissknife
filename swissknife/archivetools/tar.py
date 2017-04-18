import os
import logging
import glob
import tarfile as trf
from contextlib import closing
import hashlib

logger = logging.getLogger("archivetools.tar")


def compare_lists(p, q):
    """
    Check that all the elements in list p exist in list q and vice versa
    :param p: list
    :param q: list
    :return: bool True/False
    """
    return all([i in p for i in q]) and all([i in q for i in p])


def isp_inq(p, q):
    """
    Check that all the elements in list p exist in list q
    :param p: list
    :param q: list
    :return: bool True/False
    """
    return all([i in q for i in p])


def md5_wrapper(func):
    """
    # Wrapper to open file in 'r' mode if path was entered instead of open file
    :param func: any function that takes a file/file_path as a first argument
    :return: the function decorated
    """

    def wrapped_func(in_file, *args, **kwargs):
        # input was a file path, try to open it as byte buffer and go for it
        logger.debug('md5_wrapper: entered {}'.format(in_file))
        if isinstance(in_file, str) and os.path.isfile(in_file):
            with open(in_file, 'rb') as f:
                ret_val = func(f, *args, **kwargs)
        # input was some objet, hopefully an open stream of bytes
        else:
            try:
                logger.debug('md5_wrapper: enterd a fileobject to an md5 function')
                ret_val = func(in_file, *args, **kwargs)
            except:
                logger.debug('md5_wrapper: Not a file, not a valid stream for the function; will blow up')
                raise
        return ret_val

    return wrapped_func


def tar_file_wrapper(func):
    """
    # Wrapper to open file in 'r|*' mode if path was entered instead of open file
    :param func: any function that takes a tarfile.Tarfile object/file_path as a first argument
    :return: the function decorated
    """

    # Wrapper to open file in  mode if path was entered instead of open file
    def wrapped_func(in_file, *args, **kwargs):
        if isinstance(in_file, trf.TarFile):
            ret_val = func(in_file, *args, **kwargs)
        elif os.path.isfile(in_file):
            with trf.open(in_file, mode='r|*') as f:
                ret_val = func(f, *args, **kwargs)
        else:
            raise RuntimeError('tar_file_wrapper:Not even a file {}'.format(in_file))
        return ret_val

    return wrapped_func


@md5_wrapper
def md5(f_in, buf_size=1 << 15):
    """
    Compute the md5 hash string of a file
    :param f_in: file open in 'br' mode
    :param buf_size: int size of the buffer (in bytes)
    :return:
    """
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: f_in.read(buf_size), b""):
        hash_md5.update(chunk)
    return hash_md5.hexdigest()


@tar_file_wrapper
def md5_tar(tar_obj):
    """
    Compute the md5 hash of all the files in a tar archive
    :param tar_obj: tarfile.Tarfile object open in any '|*' mode.
    :return: list of lists of strings [[name_in_file_1, md5_hash_1], ...]
    """
    logger.info('Getting checksums of files in archive {}'.format(tar_obj))
    md_list = []
    for member in tar_obj:
        with closing(tar_obj.extractfile(member)) as file:
            md_hash = md5(file)
            md_list.append([member.name, md_hash])
            logger.debug('{}, md5:{}'.format(member.name, md_hash))
    logger.info('Done')
    return md_list


@tar_file_wrapper
def find_in_tar(tar_obj, name):
    """
    SHOULD NOT USE; USE tarfile.getmember(name) INSTEAD!!
    Compute the md5 hash of all the files in a tar archive
    :param tar_obj: tarfile.Tarfile object open in any '|*' mode.
    :param name: name of the element within the tar archive
    :return: Return a TarInfo object for member name
    """
    found = None
    for member in tar_obj:
        # print(os.path.normpath(member.name))
        if os.path.normpath(member.name) == os.path.normpath(name):
            # print('found {}'.format(found))
            found = member
        else:
            pass
    return found


@tar_file_wrapper
def check_tar_archive(tar_obj, md_checklist=None):
    """
    Checks that all the md5 in a tar file  coincide with the ones in the checklist
    and that all the files in the checklist are in the archive
    and returns the checklist from the archive and True/False of consistence of the lists.
    If no checklist is entered, it just makes a [file, md5] list of all the files in the archive
    and returns None for verification of checklists.
    :param tar_obj: tarfile.Tarfile object open in any '|*' mode.
    :param md_checklist: list of lists of strings [[name_in_file_1, md5_hash_1], ...]
    :return: list of lists of strings [[name_in_file_1, md5_hash_1], ...], boolean True/False or None
    """

    logger.info('Getting checksums of files in archive {}'.format(tar_obj))
    md_list = []
    verified = None if md_checklist == None else True
    for member in tar_obj:
        with closing(tar_obj.extractfile(member)) as file:
            md_hash = md5(file)
            md_list.append([member.name, md_hash])
            logger.debug('{}, md5:{}'.format(member.name, md_hash))
            if md_checklist is not None:
                md_check = [member.name, md_hash] in md_checklist
                if not md_check:
                    verified = False
                    logger.warning('Md5 hash check failed {0}'.format([member.name, md_hash]))
                    # raise RuntimeWarning('Md5 hash check failed {0} vs. {1}'.format(md_check, [member.name, md_hash]))
                    break
    logger.info('Done checking')
    return md_list, verified


def compress_folder(source_fold, dest_path, mode='w:'):
    """
    Archives a folder into a tarfile into dest_paht, named as the last element of the folder path.
    Uses tarfile.Tarfile package to make the tar.
    :param source_fold: string path of the folder to compress.
    :param dest_path: string folder to create the tar archive in
    :param mode: string compression mode to pass tarfile ('w:' no compression, 'w:xz' for lhz, se tarfile documentation for more)
    :return: (string, string, list)
    """
    ext = mode.split(':')[-1]
    fold_orig, fold_name = os.path.split(source_fold)
    file_name = '{0}.tar'.format(fold_name)
    if not ext == '':
        file_name += '.{0}'.format(ext)
    dest_file = os.path.join(dest_path, file_name)
    dest_md5_list = os.path.join(dest_path, '{0}.{1}'.format(fold_name, 'mdl'))
    dest_md5 = os.path.join(dest_path, '{0}.{1}'.format(fold_name, 'md5'))
    logger.info('Will compress folder {0} from {1} into {2}'.format(fold_name, fold_orig, dest_file))

    md_list = []
    with trf.open(dest_file, mode=mode) as a:
        with open(dest_md5_list, 'w') as mdf:
            for root, dirs, files in os.walk(source_fold, topdown=True):
                for name in files:
                    f = format(os.path.join(root, name))
                    arcname = os.path.normpath(os.path.relpath(f, start=source_fold))
                    # arcname = (os.path.join('.', root.split(source_fold)[-1], name))
                    md5_f = md5(os.path.join(root, name))
                    mdf.write('{0}, {1}\n'.format(arcname, md5_f))
                    logger.debug('Adding {0} as {1}'.format(f, arcname))
                    a.add(f, arcname=arcname)
                    md_list.append(['{0}'.format(arcname), md5_f])
                    # pack in the list of files and checksums
                    # logger.debug('Adding {0} as {1}'.format(dest_md5_list, '{0}.{1}'.format(fold_name,'mdl')))
                    # a.add(dest_md5_list, arcname='{0}.{1}'.format(fold_name,'mdl'))

    logger.info('done archiving, computing md5 hash of archive')
    md5_arch = md5(dest_file)
    with open(dest_md5, 'w') as md5f:
        md5f.write('{}'.format(md5_arch))
    logger.info('done')
    return dest_file, md5_arch, md_list
