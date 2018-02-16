import swissknife.bci.core.expstruct as et

import swissknife.streamtools.streams as st
import swissknife.streamtools.spectral as sp
from swissknife.streamtools import findbout as fb
import swissknife.bci.core.file.h5_functions as h5

def search_pattern(bird_id, sess, rec, pattern_chunk):
    logger.info('Configuring search for rec {}'.format(rec))
    fn = et.file_names(bird_id, sess, rec)
    mic_file_path = et.file_path(fn, 'ss', 'mic')
    logger.info('Loading the data from rec {}'.format(rec))
    chan_sound = st.WavData2(mic_file_path)
    chan_chunk = st.Chunk(chan_sound)
    logger.info('{} samples loaded'.format(chan_sound.n_samples))
    logger.info('Calling find_the_bouts')

    exp_par = et.get_parameters(bird_id, sess, 0)
    search_pars = exp_par['search_motiff']
    search_pars['s_f'] = chan_sound.s_f

    cand_file_path_orig = et.file_path(fn, 'ss', 'cand')
    cand_file_parts = os.path.split(cand_file_path_orig)
    cand_file_path = os.path.join(cand_file_parts[0], cand_file_parts[1] + '.new')
    cand_grp = '/pattern_{0}/{1:03d}'.format(pattern, rec)
    candidates = fb.find_the_bouts(chan_chunk.data.flatten(), pattern_chunk.data.flatten(),
                                   search_pars,
                                   cand_file_path=cand_file_path,
                                   cand_grp=cand_grp)

    logger.info('Found {} candidates'.format(candidates.index.size))
    return candidates
