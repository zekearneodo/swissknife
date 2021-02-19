import os
import logging
import itertools

import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm

from swissknife.bci.core import expstruct as et
from swissknife.streamtools import spectral as sp

from sklearn.decomposition import PCA
from sklearn.externals import joblib

logger = logging.getLogger('swissknife.hilevel.metricsfrompd')

default_sims_files_dict = {'model': 'train_pd_ffnn_keras_cluster_model.pkl',
                           'model_all': 'train_pd_ffnn_keras_cluster_model_all.pkl',
                           'model_some': 'train_pd_ffnn_keras_cluster_model_some.pkl',
                           'model_some_all': 'train_pd_ffnn_keras_cluster_model_some_all.pkl',
                           'model_some_other': 'train_pd_ffnn_keras_cluster_model_some_other.pkl',
                           'model_mua-int': 'train_pd_ffnn_keras_cluster_model_mua-int.pkl',
                           'model_mua-prj': 'train_pd_ffnn_keras_cluster_model_mua-prj.pkl',
                           'model_mua': 'train_pd_ffnn_keras_cluster_model_mua.pkl',
                           'model_prj': 'train_pd_ffnn_keras_cluster_model_prj.pkl',
                           'model_int': 'train_pd_ffnn_keras_cluster_model_int.pkl',
                           'threshold': 'train_pd_ffnn_keras_threshold_model.pkl',

                           'lstm': 'train_pd_lstm_keras_cluster_spectrogram.pkl',
                           'ffnn': 'train_pd_ffnn_keras_cluster_spectrogram.pkl',
                           'ffnn_linear': 'train_pd_ffnn_keras_linear_cluster_spectrogram.pkl'}

default_spec_pars = {'fft_size': 512,
                     'log': True,
                     's_f': 30000,
                     'step_size': int(30000*0.001),
                     'f_min': 300,
                     'f_max': 7500,
                     'db_cut': 45,
                     }

### loading and handling


def get_sims(sims_folder, sim_key='model', files_dict=default_sims_files_dict):
    pickle_path = os.path.join(sims_folder, files_dict[sim_key])
    sims_pd = pd.read_pickle(pickle_path)

    return sims_pd


def all_spec(model_pd, spec_pars: dict, streams=['syn', 'bos', 'neur']) -> pd.DataFrame:

    for stream_name in tqdm(streams):
        model_pd[stream_name + '_sxx'] = model_pd[stream_name +
                                                  '_t'].apply(lambda x: sp.pretty_spectrogram(x, **spec_pars)[2])

# projection/inversion from/to pca
def pca_on_sims_pd(sims_pd, n_components=3):
    pca = PCA(n_components=n_components)
    all_spectra = np.hstack(sims_pd['bos_sxx'].values)

    # Get the basis
    all_pc = pca.fit_transform(all_spectra.T)
    all_pc_min = np.amin(all_pc, axis=0)
    all_pc_ptp = np.ptp(all_pc, axis=0)
    all_pc_df = pd.DataFrame(all_pc)
    
    # Now one by one project onto the basis
    all_spectra = []
    sims_pd['pc'] = sims_pd['bos_sxx'].apply(lambda x: pca.transform(x.T))
    sims_pd['pc_sxx'] = sims_pd['pc'].apply(lambda x: pca.inverse_transform(x).T)

    return pca, all_pc_min, all_pc_ptp

# Distances
def warp_emd_dist(sx, sy):
    dist_fun = sp.spec_emd_dist
    #dist_fun = sp.spec_rms_dist

    dist, sz = sp.interp_spec_dist(sx, sy, dist_fun)
    return dist


def warp_rho_dist(sx, sy):
    #dist_fun = sp.spec_emd_dist

    dist, sz = sp.interp_spec_dist(sx, sy, sp.spec_rho_dist)
    dist[dist < 1e-8] = 1
    return dist


def all_spec_dist(model_pd, stream_pair, dist_fun, **dist_fun_kwargs):
    def two_row_dist(x): return dist_fun(
        x[stream_pair[0] + '_sxx'], x[stream_pair[1] + '_sxx'])
    model_pd['{}-{}'.format(*stream_pair)
             ] = model_pd.apply(two_row_dist, axis=1)


def all_pairs_dist(model_pd, stream, dist_fun, **dist_fun_kwargs):
    # compute all distances for all pairs within a pd
    dist_list = []
    sxx_key = stream + '_sxx'
    iter_pairs = itertools.combinations(model_pd.index.values, 2)
    for i_pair, pair in tqdm(enumerate(iter_pairs)):
        i, j = pair
        dist = dist_fun(model_pd.loc[i, sxx_key], model_pd.loc[j, sxx_key])
        dist_list.append(dist)
    return dist_list


def all_con_metrics(model_pd: pd.DataFrame, other_pd: pd.DataFrame, dist_fun, model_stream: str = 'bos', **dist_fun_kwargs,
                    ) -> pd:
    iter_pairs = itertools.product(model_pd['mot'], other_pd['mot'])

    mod_sxx_key = model_stream + '_sxx'
    con_sxx_key = 'x' + '_sxx'

    dist_list = []
    logger.info('getting all bos-con metrics')
    for i_pair, pair in tqdm(enumerate(iter_pairs)):
        i, j = pair
        dist = dist_fun(model_pd.loc[model_pd['mot'] == i, mod_sxx_key].values[0],
                        other_pd.loc[other_pd['mot'] == j, con_sxx_key].values[0])
        dist_list.append(dist)

    # make the metric summary
    sum_pd_con = pd.DataFrame({'emd': dist_list})
    sum_pd_con['pair'] = 'bos-con'

    return sum_pd_con


def cut_spectrogram(x, db_factor):
    y = x
    y[y < db_factor * np.max(y)] = -1
    return y

# computing for a bird and getting summary


def all_sims_metrics(model_pd,
                     dist_fun,
                     pairs: tuple = (['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos']),
                     do_bos: bool = True,
                     **dist_fun_kwargs
                     ):

    # compute all pairwise comparisons
    sum_pd_list = []
    logger.info('getting pairwise metrics for pairs {}'.format(pairs))
    for pair in (pairs):
        all_spec_dist(model_pd, pair, dist_fun)
        pair_key = '{}-{}'.format(*tuple(pair))

        sum_pd_pair = pd.DataFrame({'emd': model_pd[pair_key]})
        sum_pd_pair['pair'] = pair_key
        sum_pd_pair['bos_mot'] = model_pd['mot']
        sum_pd_list.append(sum_pd_pair)

    # compute all pairwise bos distances
    if do_bos:
        logger.info('getting all bos-bos metrics')
        bos_pairs = all_pairs_dist(model_pd, 'bos', dist_fun)
        bos_pd = pd.DataFrame({'emd': bos_pairs})
        bos_pd['pair'] = 'bos-bos'
        sum_pd_list.append(bos_pd)

    return pd.concat(sum_pd_list)


def all_bird_metrics(bird: str, sess: str,
                     spec_pars: dict = default_spec_pars,
                     spec_cut=0,
                     metrics_file='metrics_emd_ganz.pkl',
                     model_sim_type: list = ['model', 'threshold'],
                     spec_sim_type: list = ['ffnn', 'lstm']):

    exp_par = et.get_parameters(bird, sess)
    fn = et.file_names(bird, sess)

    #dist_fun = warp_rho_dist
    #metrics_file = 'metrics_rho_ganz.pkl'

    dist_fun = warp_emd_dist

    sims_folder = fn['folders']['ss']

    logger.info('loading pickles from folder {}'.format(sims_folder))

    other_pickles_path = '/mnt/cube/earneodo/bci_zf/stim_data/z_000/more_bird_mot_30khz.pickle'
    other_mot_pd = pd.read_pickle(other_pickles_path).rename(
        columns={'x': 'x_t', 'm_id': 'mot'})
    other_mot_pd.drop(
        other_mot_pd[(other_mot_pd['mot'].str.contains('z0'))].index, inplace=True)

    # get all spectrograms
    logger.info('getting all other spectrograms')
    all_spec(other_mot_pd, spec_pars=spec_pars, streams=['x'])

    # for the model/threshold simmulations
    all_pd_list = []

    # for the ffnn/lstm
    for sim_type in spec_sim_type:
        logger.info('Simmulation type {}'.format(sim_type))
        model_dist_pairs = (['neur', 'bos'], )
        # load the metrics
        logger.info('getting sims results pd')
        sims_pd = get_sims(sims_folder, sim_type)
        if spec_cut > 0:
            sims_pd['neur_sxx'] = sims_pd['neur_sxx'].apply(
                lambda x: cut_spectrogram(x, spec_cut))

        # to make up for missing 'mot' in the original pandas
        sims_pd['mot'] = sims_pd['mot'] = np.arange(sims_pd.index.size)
        # get all distances for the three pairs ['neur', 'bos']
        logger.info('getting pairwise metrics')
        sims_metrics_pd = all_sims_metrics(
            sims_pd, dist_fun, model_dist_pairs, do_bos=False)
        # get the bos-con metrics
        #con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, warp_emd_dist)
        sims_metrics_pd = pd.concat([sims_metrics_pd])
        sims_metrics_pd['sim_type'] = sim_type
        all_pd_list.append(sims_metrics_pd)

    for sim_type in model_sim_type:
        logger.info('Simmulation type {}'.format(sim_type))

        # only get the syn-bos and syn-neur for the 'model' and 'threshold'
        if sim_type in ['model', 'threshold', 'model_all']:
            model_dist_pairs = (
                ['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos'], ['pc', 'bos'])
            do_bos = True
        else:
            model_dist_pairs = (['neur', 'bos'],)
            do_bos = False

        # load the metrics
        logger.info('getting sims results pd')
        sims_pd = get_sims(sims_folder, sim_type)

        #clean up the badly fitted (all zeros)
        sims_pd.reset_index(inplace=True)
        sims_pd['bad'] = sims_pd['par_pred'].apply(lambda x: np.sum(x[:,2])==0)
        sims_pd.drop(sims_pd[sims_pd['bad']==True].index, inplace=True)

        if not ('mot' in sims_pd):
            # some older versions of the simulaiton didn't keep track of mot id
            sims_pd['mot'] = np.arange(sims_pd.index.size)
        # get the spectrograms
        logger.info('getting spectrograms')
        all_spec(sims_pd, spec_pars=spec_pars, streams=['syn', 'bos', 'neur'])
        # get the pca compression/decompression of the bos spectrograms
        logger.info('getting pc projections')
        pca_on_sims_pd(sims_pd) # now sims_pd has ['pc'] and ['pc_sxx']
        # get all distances for the three pairs ['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos']
        logger.info('getting pairwise metrics')
        sims_metrics_pd = all_sims_metrics(
            sims_pd, dist_fun, model_dist_pairs, do_bos=do_bos)
        # get the bos-con metrics
        if sim_type in ['model', 'model_all']:
            con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, dist_fun)
            sims_metrics_pd = pd.concat([sims_metrics_pd, con_metrics_pd])
        sims_metrics_pd['sim_type'] = sim_type
        all_pd_list.append(sims_metrics_pd)

    all_pd = pd.concat(all_pd_list)
    logger.info('Getting <emd> for all')
    all_pd['<emd>'] = all_pd['emd'].apply(np.mean)
    all_pd['bird'] = bird
    all_pd['sess'] = sess

    metrics_path = os.path.join(fn['folders']['proc'], metrics_file)
    logger.info('saving all metrics to {}'.format(metrics_path))
    et.mkdir_p(fn['folders']['proc'])
    all_pd.to_pickle(metrics_path)

    return all_pd


def all_bird_metrics_pw(bird: str, sess: str, spec_pars: dict = default_spec_pars, spec_cut=0, metrics_file='metrics_emd_pw.pkl'):
    exp_par = et.get_parameters(bird, sess)
    fn = et.file_names(bird, sess)

    #dist_fun = warp_rho_dist
    #metrics_file = 'metrics_rho_ganz.pkl'

    dist_fun = warp_emd_dist

    sims_folder = fn['folders']['ss']

    logger.info('loading pickles from folder {}'.format(sims_folder))

    other_pickles_path = '/mnt/cube/earneodo/bci_zf/stim_data/z_000/more_bird_mot_30khz.pickle'
    other_mot_pd = pd.read_pickle(other_pickles_path).rename(
        columns={'x': 'x_t', 'm_id': 'mot'})
    other_mot_pd.drop(
        other_mot_pd[(other_mot_pd['mot'].str.contains('z0'))].index, inplace=True)

    # get all spectrograms
    logger.info('getting all other spectrograms')
    all_spec(other_mot_pd, spec_pars=spec_pars, streams=['x'])

    # for the model/threshold simmulations
    all_pd_list = []

    # for the ffnn/lstm
    for sim_type in ['ffnn', 'lstm']:
        logger.info('Simmulation type {}'.format(sim_type))
        model_dist_pairs = (['neur', 'bos'], )
        # load the metrics
        logger.info('getting sims results pd')
        sims_pd = get_sims(sims_folder, sim_type)
        if spec_cut > 0:
            sims_pd['neur_sxx'] = sims_pd['neur_sxx'].apply(
                lambda x: cut_spectrogram(x, spec_cut))

        # to make up for missing 'mot' in the original pandas
        sims_pd['mot'] = sims_pd['mot'] = np.arange(sims_pd.index.size)
        # get all distances for the three pairs ['neur', 'bos']
        logger.info('getting pairwise metrics')
        sims_metrics_pd = all_sims_metrics(sims_pd, dist_fun, model_dist_pairs)
        # get the bos-con metrics
        #con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, warp_emd_dist)
        sims_metrics_pd = pd.concat([sims_metrics_pd])
        sims_metrics_pd['sim_type'] = sim_type
        all_pd_list.append(sims_metrics_pd)

    for sim_type in ['model', 'threshold']:
        logger.info('Simmulation type {}'.format(sim_type))
        model_dist_pairs = (['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos'])
        # load the metrics
        logger.info('getting sims results pd')
        sims_pd = get_sims(sims_folder, sim_type)

        if not ('mot' in sims_pd):
            # some older versions of the simulaiton didn't keep track of mot id
            sims_pd['mot'] = np.arange(sims_pd.index.size)
        # get the spectrograms
        logger.info('getting spectrograms')
        all_spec(sims_pd, spec_pars=spec_pars, streams=['syn', 'bos', 'neur'])
        # get all distances for the three pairs ['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos']
        logger.info('getting pairwise metrics')
        sims_metrics_pd = all_sims_metrics(sims_pd, dist_fun, model_dist_pairs)
        # get the bos-con metrics
        #con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, dist_fun)
        #sims_metrics_pd = pd.concat([sims_metrics_pd, con_metrics_pd])
        sims_metrics_pd['sim_type'] = sim_type
        all_pd_list.append(sims_metrics_pd)

    all_pd = pd.concat(all_pd_list)
    logger.info('Getting <emd> for all')
    all_pd['<emd>'] = all_pd['emd'].apply(np.mean)
    all_pd['bird'] = bird
    all_pd['sess'] = sess

    metrics_path = os.path.join(fn['folders']['proc'], metrics_file)
    logger.info('saving all metrics to {}'.format(metrics_path))
    et.mkdir_p(fn['folders']['proc'])
    all_pd.to_pickle(metrics_path)

    return all_pd


def all_bird_metrics_rho_pw(bird: str, sess: str, spec_pars: dict = default_spec_pars, spec_cut=0, metrics_file='metrics_rho_ganz.pkl'):
    exp_par = et.get_parameters(bird, sess)
    fn = et.file_names(bird, sess)

    dist_fun = warp_rho_dist

    #dist_fun = warp_emd_dist
    #metrics_file = 'metrics_emd_ganz.pkl'

    sims_folder = fn['folders']['ss']

    logger.info('loading pickles from folder {}'.format(sims_folder))

    other_pickles_path = '/mnt/cube/earneodo/bci_zf/stim_data/z_000/more_bird_mot_30khz.pickle'
    other_mot_pd = pd.read_pickle(other_pickles_path).rename(
        columns={'x': 'x_t', 'm_id': 'mot'})
    other_mot_pd.drop(
        other_mot_pd[(other_mot_pd['mot'].str.contains('z0'))].index, inplace=True)

    # get all spectrograms
    logger.info('getting all other spectrograms')
    all_spec(other_mot_pd, spec_pars=spec_pars, streams=['x'])

    # for the model/threshold simmulations
    all_pd_list = []

    # for the ffnn/lstm
    for sim_type in ['lstm', 'ffnn']:
        logger.info('Simmulation type {}'.format(sim_type))
        model_dist_pairs = (['neur', 'bos'], )
        # load the metrics
        logger.info('getting sims results pd')
        sims_pd = get_sims(sims_folder, sim_type)
        # to make up for missing 'mot' in the original pandas
        sims_pd['mot'] = sims_pd['mot'] = np.arange(sims_pd.index.size)
        # get all distances for the three pairs ['neur', 'bos']
        logger.info('getting pairwise metrics')
        sims_metrics_pd = all_sims_metrics(sims_pd, dist_fun, model_dist_pairs)
        # get the bos-con metrics
        #con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, warp_emd_dist)
        sims_metrics_pd = pd.concat([sims_metrics_pd])
        sims_metrics_pd['sim_type'] = sim_type
        all_pd_list.append(sims_metrics_pd)

        if spec_cut > 0:
            sims_pd['neur_sxx'] = sims_pd['neur_sxx'].apply(
                lambda x: cut_spectrogram(x, spec_cut))

    for sim_type in ['threshold', 'model']:
        logger.info('Simmulation type {}'.format(sim_type))
        model_dist_pairs = (['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos'])
        # load the metrics
        logger.info('getting sims results pd')
        sims_pd = get_sims(sims_folder, sim_type)

        if not ('mot' in sims_pd):
            # some older versions of the simulaiton didn't keep track of mot id
            sims_pd['mot'] = np.arange(sims_pd.index.size)
        # get the spectrograms
        logger.info('getting spectrograms')
        all_spec(sims_pd, spec_pars=spec_pars, streams=['syn', 'bos', 'neur'])
        # get all distances for the three pairs ['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos']
        logger.info('getting pairwise metrics')
        sims_metrics_pd = all_sims_metrics(sims_pd, dist_fun, model_dist_pairs)
        # get the bos-con metrics
        #con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, dist_fun)
        #sims_metrics_pd = pd.concat([sims_metrics_pd, con_metrics_pd])
        sims_metrics_pd['sim_type'] = sim_type
        all_pd_list.append(sims_metrics_pd)

    # get the con metrics only once
    spec_pars['db_cut'] = 125
    all_spec(other_mot_pd, spec_pars=spec_pars, streams=['x'])
    all_spec(sims_pd, spec_pars=spec_pars, streams=['syn', 'bos', 'neur'])
    con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, dist_fun)
    con_metrics_pd['sim_type'] = 'model'
    all_pd_list.append(con_metrics_pd)

    all_pd = pd.concat(all_pd_list)
    logger.info('Getting <emd> for all')
    all_pd['<emd>'] = all_pd['emd'].apply(np.mean)
    all_pd['bird'] = bird
    all_pd['sess'] = sess

    metrics_path = os.path.join(fn['folders']['proc'], metrics_file)
    logger.info('saving all metrics to {}'.format(metrics_path))
    et.mkdir_p(fn['folders']['proc'])
    all_pd.to_pickle(metrics_path)

    return all_pd, sims_pd


def all_bird_metrics_rho(bird: str, sess: str, spec_pars: dict = default_spec_pars, metrics_file='metrics_rho_ganz.pkl'):
    exp_par = et.get_parameters(bird, sess)
    fn = et.file_names(bird, sess)

    dist_fun = warp_rho_dist

    #dist_fun = warp_emd_dist
    #metrics_file = 'metrics_emd_ganz.pkl'

    sims_folder = fn['folders']['ss']

    logger.info('loading pickles from folder {}'.format(sims_folder))

    other_pickles_path = '/mnt/cube/earneodo/bci_zf/stim_data/z_000/more_bird_mot_30khz.pickle'
    other_mot_pd = pd.read_pickle(other_pickles_path).rename(
        columns={'x': 'x_t', 'm_id': 'mot'})
    other_mot_pd.drop(
        other_mot_pd[(other_mot_pd['mot'].str.contains('z0'))].index, inplace=True)

    # get all spectrograms
    logger.info('getting all other spectrograms')
    all_spec(other_mot_pd, spec_pars=spec_pars, streams=['x'])

    # for the model/threshold simmulations
    all_pd_list = []

    # for the ffnn/lstm
    for sim_type in ['lstm', 'ffnn']:
        logger.info('Simmulation type {}'.format(sim_type))
        model_dist_pairs = (['neur', 'bos'], )
        # load the metrics
        logger.info('getting sims results pd')
        sims_pd = get_sims(sims_folder, sim_type)
        # to make up for missing 'mot' in the original pandas
        sims_pd['mot'] = sims_pd['mot'] = np.arange(sims_pd.index.size)
        # get all distances for the three pairs ['neur', 'bos']
        logger.info('getting pairwise metrics')
        sims_metrics_pd = all_sims_metrics(sims_pd, dist_fun, model_dist_pairs)
        # get the bos-con metrics
        #con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, warp_emd_dist)
        sims_metrics_pd = pd.concat([sims_metrics_pd])
        sims_metrics_pd['sim_type'] = sim_type
        all_pd_list.append(sims_metrics_pd)

    for sim_type in ['threshold', 'model']:
        logger.info('Simmulation type {}'.format(sim_type))
        model_dist_pairs = (['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos'])
        # load the metrics
        logger.info('getting sims results pd')
        sims_pd = get_sims(sims_folder, sim_type)

        if not ('mot' in sims_pd):
            # some older versions of the simulaiton didn't keep track of mot id
            sims_pd['mot'] = np.arange(sims_pd.index.size)
        # get the spectrograms
        logger.info('getting spectrograms')
        all_spec(sims_pd, spec_pars=spec_pars, streams=['syn', 'bos', 'neur'])
        # get all distances for the three pairs ['syn', 'bos'], ['syn', 'neur'], ['neur', 'bos']
        logger.info('getting pairwise metrics')
        sims_metrics_pd = all_sims_metrics(sims_pd, dist_fun, model_dist_pairs)
        # get the bos-con metrics
        #con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, dist_fun)
        #sims_metrics_pd = pd.concat([sims_metrics_pd, con_metrics_pd])
        sims_metrics_pd['sim_type'] = sim_type
        all_pd_list.append(sims_metrics_pd)

    # get the con metrics only once
    spec_pars['db_cut'] = 125
    all_spec(other_mot_pd, spec_pars=spec_pars, streams=['x'])
    all_spec(sims_pd, spec_pars=spec_pars, streams=['syn', 'bos', 'neur'])
    #con_metrics_pd = all_con_metrics(sims_pd, other_mot_pd, dist_fun)
    #con_metrics_pd['sim_type'] = 'model'
    # all_pd_list.append(con_metrics_pd)

    all_pd = pd.concat(all_pd_list)
    logger.info('Getting <emd> for all')
    all_pd['<emd>'] = all_pd['emd'].apply(np.mean)
    all_pd['bird'] = bird
    all_pd['sess'] = sess

    metrics_path = os.path.join(fn['folders']['proc'], metrics_file)
    logger.info('saving all metrics to {}'.format(metrics_path))
    et.mkdir_p(fn['folders']['proc'])
    all_pd.to_pickle(metrics_path)

    return all_pd, sims_pd
