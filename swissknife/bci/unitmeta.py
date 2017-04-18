import numpy as np
import numpy.lib.recfunctions as rfn
from bci.core import expstruct as et

from core import kwik_functions as kwf


def list_shank_units(bird, sess, shank, shank_file=None, sorted=False):
    meta_dt = np.dtype([('sess', 'S32', 1), ('shank', np.int, 1), ('is_good', 'b', 1)])
    kwik_file = et.open_kwik(bird, sess) if shank_file is None else et.open_kwik(bird, sess, shank_file)
    group = int(shank)
    all_units = kwf.list_units(kwik_file, group=group, sorted=False)
    n_units = all_units.size
    all_meta = np.recarray(n_units, dtype=meta_dt)
    all_meta['sess'] = sess
    all_meta['shank'] = shank
    all_meta['is_good'] = True
    return rfn.merge_arrays((all_meta, all_units), asrecarray=True, flatten=True)


def list_sess_units(bird, sess, sorted=False):
    shanks = et.get_shanks_list(bird, sess)
    sess_units = None
    for shank in shanks:
        shank_units = list_shank_units(bird, sess, shank, sorted=False)
        if sess_units is None:
            sess_units = shank_units
        else:
            sess_units = rfn.stack_arrays((sess_units, shank_units))
    return sess_units