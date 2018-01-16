# Higher level functions for calling kilosort with pipeline arguments

import subprocess
import paramiko
import select
import logging
import sys
import os

from swissknife.bci.core import expstruct as et
from swissknife.bci.core.file import h5_functions as h5f
from swissknife.bci.core.pipeline.core import kilosort

logger = logging.getLogger('kilosort_functions')


def ssh_run(host, cmd):
    # logger.info('Will run ssh command {} on {}'.format(host, cmd))
    ssh = subprocess.Popen(["ssh", "%s" % host, '%s;' % cmd, 'exit;'],
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)

    result = ssh.stdout.readlines()
    if result == []:
        error = ssh.stderr.readlines()
        print >> sys.stderr, "ERROR: %s" % error
        # logger.error('something went wrong')
    else:
        print(result)
        # logger.info('finished')
    return (result)


def ssh_paramiko(host, cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host)
    print("Connected to %s" % host)
    stdin, stdout, stderr = ssh.exec_command(cmd)
    msg = []
    while not stdout.channel.exit_status_ready():
        # Only print data if there is data to read in the channel
        if stdout.channel.recv_ready():
            rl, wl, xl = select.select([stdout.channel], [], [], 0.0)
            if len(rl) > 0:
                # Print data from stdout
                print(msg.append(stdout.channel.recv(1024)))
    print("finished command, returning")
    ssh.close()
    return msg


def ssh_mine(host, cmd):
    # Ports are handled in ~/.ssh/config since we use OpenSSH

    logger.info('sending command ' + cmd)
    ssh = subprocess.Popen(["ssh", "%s" % host, cmd],
                           shell=False,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    result = ssh.stdout.readlines()
    if result == []:
        error = ssh.stderr.readlines()
        raise Exception('Error in ssh command')
        # print(>>sys.stderr, "ERROR: %s" % error)
    else:
        print(result)

    print("finished command, returning")
    return result


# Make one single binary file
def make_binary(bird, sess, port=0):
    print('Making binary')
    fn = et.file_names(bird, sess)
    params = et.get_parameters(bird, sess)
    kwd_path = et.file_path(fn, 'ss', 'ss_raw')
    bin_path = et.file_path(fn, 'ss', 'ss_bin')
    neural_key = 'neural_{}'.format(port)
    logger.info('port is ' + neural_key)
    try:
        chan_list = params['channel_config'][neural_key]
    except KeyError:
        logger.info('not found chan group ' + neural_key)
        if port == 0:
            logger.info('Using default (neural) chan list')
            chan_list = params['channel_config']['neural']
            logger.info('chan_list: {}'.format(chan_list))
        else:
            raise
    h5f.kwd_to_binary(kwd_path, bin_path, chan_list=chan_list)
    print('done')


# make all the parameters for sorting with kilosort

# sort with a script in niao (scripts gets the data from cube, puts back in cube)
def niao_kilosort(bird, sess, port=0):
    print('Sending sort to niao')
    return ssh_paramiko('niao', 'source activate tf11 && py_kilosort.sh {0} {1} {2}'.format(bird, sess, port))


# sort in passaro
def local_kilosort(bird, sess, port=0):
    logger.info('Will sort using kilosort on local computer on port {}'.format(port))
    kilosort.run_kilosort(bird, sess,
                          kilo_dir=os.path.abspath('/home/earneodo/repos/KiloSort'),
                          npymat_dir=os.path.abspath('/home/earneodo/repos/npy-matlab'),
                          no_copy=False,
                          use_gpu=True,
                          port=port)
