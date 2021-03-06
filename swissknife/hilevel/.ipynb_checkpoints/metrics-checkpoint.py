import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from swissknife.streamtools import spectral as sp
from swissknife.dynamo import finch as zf
from swissknife.bci import synthetic as syn

from swissknife.hilevel.ffnn import get_raw_stream, scale_pc_inv

logger = logging.getLogger('swissknife.hilevel.metrics')


def pred_to_song(y, bin_size, s_f=30000):
    par_streams = syn.pred_to_par_stream(y, bin_size, s_f=s_f)
    syn_song = zf.make_song(par_streams, s_f_override=s_f)[:, 0]
    return syn_song, par_streams


def pred_to_song_pc(y, bin_size, pca_dict, s_f=30000):
    # resample (fourier interpolation) to ms bins
    yr = syn.stream_resample(y, 1, bin_size)
    yr = scale_pc_inv(yr, pca_dict)

    # Those be the pc, now to the spectrogram
    pca = pca_dict['pc_obj']
    pca_spectrum = pca.inverse_transform(yr).T
    song = sp.inv_spectrogram(pca_spectrum, pca_dict['spec_pars'])

    return (song-np.mean(song))/np.ptp(song)*30000, yr


def rms_spec(u, v):
    spec_diff = u / np.linalg.norm(u) - v / np.linalg.norm(v)
    return np.linalg.norm(spec_diff) / np.sqrt(spec_diff.size)


def rms_per_slice(u, v):
    spec_diff = u / np.linalg.norm(u) - v / np.linalg.norm(v)
    root_n = np.sqrt(u.shape[0])
    rms_per_slice = [np.linalg.norm(sl) / root_n for sl in spec_diff.T]
    return np.array(rms_per_slice)


def normalize(x):
    return x / np.max(x)


def stream_normalize(x, axis=0):
    return x / np.max(x, axis=axis)


def full_shuffle(x):
    x_shape = x.shape
    x_flat = x.flatten()
    np.random.shuffle(x_flat)
    return x_flat.reshape(x_shape)


def rms_slices(s_1, s_2):
    # compare two spectrograms
    # normalize to [0, 1], compare every slice, return vector of slice rms and total rms
    s_norm = [normalize(s - np.min(s)) for s in [s_1, s_2]]
    s_dif = (s_norm[0] - s_norm[1])
    rms_slice = np.linalg.norm(s_dif, axis=0) / np.sqrt(s_dif.shape[0])
    # rms_total = np.linalg.norm(s_dif)/np.sqrt(s_dif.size)
    rms_total = np.linalg.norm(rms_slice) / np.sqrt(rms_slice.size)
    return rms_slice, rms_total


def compare_spectra_old(x, y, s_f=30000, n_perseg=1024, step_s=0.001, db_cut=65, f_min=300, f_max=7500,
                        log=False,
                        plots=None):
    fx, tx, sx = sp.pretty_spectrogram(x, s_f, fft_size=n_perseg, log=log,
                                       step_size=int(s_f * step_s), db_cut=db_cut,
                                       f_min=300, f_max=7500, window=('gaussian', 80))

    fy, ty, sy = sp.pretty_spectrogram(y, s_f, fft_size=n_perseg, log=log,
                                       step_size=int(s_f * step_s), db_cut=db_cut,
                                       f_min=300, f_max=7500, window=('gaussian', 80))

    # make sure sizes are right
    sxy = [sx, sy]
    lengths = np.array([a.shape[-1] for a in sxy])
    len_diff = np.diff(lengths)
    if (np.abs(len_diff) > 2):
        logger.warning('Spectrograms differ in {} ms'.format(len_diff))
    shorter_t = np.min(lengths)
    sorted_l = np.argsort(lengths)

    short = sxy[sorted_l[0]]
    long = sxy[sorted_l[1]]

    if plots:
        plt.imshow(((short))[::-1], aspect='auto', cmap='inferno')
        plt.grid(False)
        # plt.savefig(os.path.join(fn['folders']['proc'], 'spec_1_ffnn_th.eps'))
        plt.figure()
        plt.imshow(((long))[::-1], aspect='auto', cmap='inferno')
        plt.grid(False)
        # plt.savefig(os.path.join(fn['folders']['proc'], 'spec_2_ffnn_th.eps'))

    return rms_slices(short, long[:, :shorter_t])


def compare_spectra(x, y, s_f=30000, n_perseg=1024, step_s=0.001, db_cut=65, f_min=0, f_max=12000, plots=False):
    # make sure sizes are right
    xy = [x, y]
    lengths = np.array([a.shape[-1] for a in xy])
    len_diff = np.diff(lengths)
    #     if(np.abs(len_diff)>2):
    #         logger.warning('Spectrograms differ in {} ms'.format(len_diff))
    shorter_t = np.min(lengths)
    sorted_l = np.argsort(lengths)

    short = xy[sorted_l[0]]
    long = xy[sorted_l[1]]

    fx, tx, sx = sp.pretty_spectrogram(normalize(x[:shorter_t]), s_f, fft_size=n_perseg, log=True,
                                       step_size=int(s_f * step_s), db_cut=db_cut,
                                       f_min=f_min, f_max=f_max, window=('gaussian', 120))

    fy, ty, sy = sp.pretty_spectrogram(normalize(y[:shorter_t]), s_f, fft_size=n_perseg, log=True,
                                       step_size=int(s_f * step_s), db_cut=db_cut,
                                       f_min=f_min, f_max=f_max, window=('gaussian', 120))

    if plots:
        plt.imshow(((sx))[::-1], aspect='auto', cmap='inferno')
        plt.grid(False)
        # plt.savefig(os.path.join(fn['folders']['proc'], 'spec_1_ffnn_th.eps'))
        plt.figure()
        plt.imshow(((sy))[::-1], aspect='auto', cmap='inferno')
        plt.grid(False)
        # plt.savefig(os.path.join(fn['folders']['proc'], 'spec_2_ffnn_th.eps'))
        plt.figure()

    # normalize first spectrogram
    sx -= np.amin(sx)
    sx_max = np.amax(sx)

    sx /= sx_max

    # rescale second spectrogram
    sy -= np.amin(sy)
    sy /= np.amax(sx_max)

    rms = np.linalg.norm(sx - sy) / np.sqrt(sx.size)

    rho = np.corrcoef(sx, sy)
    return rms, sx, sy


def mot_scores(mot_id, Y, Z, mod_pred, sess_data, win_samples=64, other_pd=None):
    mot_idx = np.where(Z[:, 0] == mot_id)
    fp = sess_data.fp
    s_f = sess_data.s_f

    logger.info('Synthesizing the song corresponding to mot_id {}'.format(mot_id))
    syn_song, syn_stream = pred_to_song(Y[mot_idx], fp['bin_size'], s_f=s_f)
    neur_song, neur_stream = pred_to_song(mod_pred[mot_idx], fp['bin_size'], s_f=s_f)

    raw_song = sess_data.gimme_raw_stream(Z[mot_idx])
    ctl_song = np.random.normal(0, np.std(syn_song), len(raw_song))

    rms_control = compare_spectra(ctl_song, raw_song, s_f=sess_data.s_f, n_perseg=win_samples, plots=False)
    rms_comp = compare_spectra(neur_song, raw_song, s_f=sess_data.s_f, n_perseg=win_samples, plots=False)

    if other_pd is not None:
        rms_other = np.array([compare_spectra(neur_song, x, s_f=sess_data.s_f, n_perseg=win_samples, plots=False)[0] for
                              x in other_pd.x.tolist()])

        return rms_comp[0], rms_control[0], raw_song, syn_song, neur_song, syn_stream, neur_stream, Y[mot_idx], \
               mod_pred[mot_idx], rms_other

    return rms_comp[0], rms_control[0], raw_song, syn_song, neur_song, syn_stream, neur_stream, Y[mot_idx], mod_pred[
        mot_idx]


def all_mot_scores(model, sess_data, other_pd=None):
    x_t = sess_data.X_t
    y_t = sess_data.Y_t
    z, z_t = sess_data.gimme_raw()

    logger.info('Predicting parameters')
    mod_pred = np.array(model.predict(x_t, batch_size=sess_data.fp['batch_size'])).squeeze().transpose()
    mot_ids = np.unique(z_t[:, 0])

    all_rms_res = []

    logger.info('Generating songs and computing rms of {} motifs'.format(mot_ids.size))
    for m_id in mot_ids:
        score_results = mot_scores(m_id, y_t, z_t, mod_pred, sess_data, other_pd=other_pd)
        all_rms_res.append((m_id,) + score_results)

    if other_pd is not None:
        heathers = ['mot_id', 'rms', 'rms_ctrl', 'raw_song', 'syn_song', 'neur_song',
                    'syn_streams', 'neur_streams', 'pred_streams', 'target_streams', 'rms_other']
    else:
        heathers = ['mot_id', 'rms', 'rms_ctrl', 'raw_song', 'syn_song', 'neur_song',
                    'syn_streams', 'neur_streams', 'pred_streams', 'target_streams']

    pd_rms = pd.DataFrame(all_rms_res, columns=heathers)
    return pd_rms


def all_mot_scores_piecewise(y_t, z_t, y_p, sess_data, other_pd=None, mot_ids=None):
    if mot_ids is None:
        mot_ids = np.unique(z_t[:, 0])
    rms_list = []
    rms_slice_list = []
    logger = logging.getLogger()

    logger.info('Generating songs and computing rms of {} motifs'.format(mot_ids.size))

    logger.disabled = True
    all_rms_comp = []
    for m_id in tqdm(mot_ids):
        m_comp = mot_scores(m_id, y_t, z_t, y_p, sess_data, other_pd=other_pd)
        all_rms_comp.append((m_id,) + m_comp)

        ## all those rms to a pandas dataframe:
    if other_pd is not None:
        heathers = ['mot_id', 'rms', 'rms_ctrl', 'raw_song', 'syn_song', 'neur_song',
                    'syn_streams', 'neur_streams', 'pred_streams', 'target_streams', 'rms_other']
    else:
        heathers = ['mot_id', 'rms', 'rms_ctrl', 'raw_song', 'syn_song', 'neur_song',
                    'syn_streams', 'neur_streams', 'pred_streams', 'target_streams']
    pd_rms = pd.DataFrame(all_rms_comp, columns=heathers)
    logger.disabled = False
    return pd_rms


def all_mot_decoded_pcwise(y, z, y_p, sess_data):
    mot_ids = np.unique(z[:, 0])
    fp = sess_data.fp
    try:
        fit_target = fp['fit_target']
    except KeyError:
        fit_target = 'dyn'

    s_f = int(sess_data.s_f)
    logger.info('Getting streams and reconstructions of {} motifs, target fit is {}'.format(mot_ids.size, fit_target))

    all_decoded = []
    for i, m_id in tqdm(enumerate(mot_ids), total=mot_ids.size):
        mot_idx = np.where(z[:, 0] == m_id)
        mot_z = z[mot_idx]
        mot_raw = get_raw_stream(mot_z, sess_data.all_syl, fp['bin_size'], s_f)
        mot_y = y[mot_idx]
        mot_y_p = y_p[mot_idx]
        if fit_target == 'dyn':
            s_song, s_pars = pred_to_song(mot_y, fp['bin_size'])
            n_song, n_pars = pred_to_song(mot_y_p, fp['bin_size'])

        elif fit_target == 'pc':
            s_song, s_pars = pred_to_song_pc(mot_y, fp['bin_size'], fp['pca_dict'])
            n_song, n_pars = pred_to_song_pc(mot_y_p, fp['bin_size'], fp['pca_dict'])

        all_decoded.append([m_id, mot_idx, mot_z, mot_y, mot_y_p, s_pars, n_pars, s_song, n_song, mot_raw])

    head = ['m_id', 'mot_idx', 'mot_z', 'syn_par', 'neu_par', 'syn_par_stream', 'neu_par_stream',
            'syn_song', 'neu_song', 'raw_song']

    all_dec_pd = pd.DataFrame(all_decoded, columns=head)
    return all_dec_pd


def all_self_scores(one_pd, other_pd):
    # one_pd has a raw, a neu, and a syn
    logger = logging.getLogger()

    all_mots = one_pd['m_id'].tolist()
    logger.info('Found {} mots'.format(len(all_mots)))

    all_scores = []
    logger.disabled = True
    for m_id, x, x_syn, x_raw in tqdm(zip(one_pd['m_id'].tolist(),
                                          one_pd['neu_song'].tolist(),
                                          one_pd['syn_song'].tolist(),
                                          one_pd['raw_song'].tolist()),
                                      total=len(one_pd['m_id'].tolist())):
        rms_raw = compare_spectra(x, x_raw, n_perseg=64, db_cut=55)[0]
        rms_syn = compare_spectra(x, x_syn, n_perseg=64, db_cut=65)[0]
        cross_scores = np.array(
            list(map(lambda z: compare_spectra(x, z, n_perseg=128, db_cut=80)[0], other_pd['x'].tolist())))
        cross_mot_id = other_pd['m_id'].tolist()
        all_scores.append([m_id, rms_raw, rms_syn, cross_scores, cross_mot_id])

    logger.disabled = False

    pd_all_scores = pd.DataFrame(all_scores, columns=['m_id', 'rms_raw', 'rms_syn', 'rms_other', 'vs_id'])
    return pd_all_scores