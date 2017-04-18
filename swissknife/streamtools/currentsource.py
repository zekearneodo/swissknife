import numpy as np
import wave
import struct
import copy

import logging

from pykCSD import pykCSD as cs
logger = logging.getLogger("streamtools.currentsource")


def csd_all_points(in_data, in_elec_pos, csd_pars={'gdX': 0.025, 'gdY': 0.025}, step=1):
    # in_data is [n_samples, n_chans]
    n_samp, n_ch = in_data.shape
    csd_stack = []
    for i in np.arange(n_samp, step=step):
        if i % 3000 == 0:
            logger.info("Sample {0}/{1} ...".format(i, n_samp))
        k_i = cs.KCSD(in_elec_pos, np.reshape(in_data[i, :], [-1, 1]), csd_pars)
        k_i.estimate_pots()
        k_i.estimate_csd()
        csd_stack.append(k_i.pass_estimation()[0][:,:,0])
    logger.info('Done with all csd samples')
    return np.stack(csd_stack, axis=2)


def plot_csd_point(in_data, sel_chans, ch_map, chan_pos_fun, csd_pars={'gdX': 0.025, 'gdY': 0.025}):
    aux_elec_pos = chan_pos_fun(sel_chans, ch_map)
    aux_elec_pos = aux_elec_pos/np.max(aux_elec_pos)
    aux_data = np.reshape(in_data, [-1, 1])
    k_aux = cs.KCSD(aux_elec_pos, aux_data, csd_pars)
    k_aux.estimate_pots()
    k_aux.estimate_csd()
    k_aux.plot_all()
    return k_aux