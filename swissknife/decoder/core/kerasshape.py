import numpy as np
import logging
import pandas as pd
import os

from tqdm.autonotebook import tqdm

from swissknife.bci.core import expstruct as et
from swissknife.bci.core import kwik_functions as kwkf
from swissknife.bci.core import basic_plot as bp
from swissknife.bci import unitmeta as um
from swissknife.bci import units as unitobjs
from swissknife.bci import synthetic as syn
from swissknife.decoder import neural as nd

logger = logging.getLogger('swissknife.decoder.core.kerasshape')

fit_pars_default = {'bin_size': 5,
           'history_bins': 20,
           'latency': 0,
           'test_fraction': 0.1,
            'num_ep': 100,
            'batch_size': 5,
           'par_reg_fcn': syn.regularize_pars,
           'par_reg_fcn_inv': syn.regularize_pars_inv,
           'network': 'ffnn',
           'syl_file': 'syllable_pandas_template',
           'syl_types': ['0'],
            'syl_filter': True,
           'unit_type': 'cluster', #'cluster', 'threshold', 'lfp'
           'target_type': 'model', #'spectrogram', 'model'
           'clu_list': None} 

def filter_syllables(all_syll, syl_type=None, test_frac=0):
    if syl_type:
        syl_selection = (all_syll['syl_type']==syl_type) & (all_syll['keep']==True)
    else:
        syl_selection = (all_syll['keep']==True)
        
    select_syl = all_syll[syl_selection]
    total_trials = select_syl.shape[0]
    test_trials = np.int(test_frac*total_trials)
    
    syl_train = select_syl[:total_trials-test_trials].reset_index(drop=True)
    syl_test = select_syl[-test_trials:]
    return syl_train, syl_test

def get_raw_stream(z_mot, sdf, bin_size, s_f):
    ms_samples = int(s_f*.001)
    bin_raw_stream_list = [sdf.iloc[z_bin[1]].rw_stream[z_bin[2]*ms_samples: (z_bin[2]+ bin_size)*ms_samples] 
                           for z_bin in z_mot]
    return np.concatenate(bin_raw_stream_list)

def flatten_feature_vector(x):
    # flatten a feature vector from keras (put it in ffnn shape as opposed to lstm)
    return x.reshape([x.shape[0], -1])

def keras_data_set(syl_df, fit_pars, units_list, flat_features=False, normalize_features=False):
    # all syl and units in unts_list are from the same bird and sess
    # (they share the kwd and kwik file)
    #syl_df = syl_df_in[:10]
    s_f = units_list[0].sampling_rate
    kw_file = units_list[0].h5_file
    ms_to_samp = s_f//1000.
    
    syl_starts = (syl_df.start.to_numpy() * ms_to_samp 
                  + syl_df.mot_start.to_numpy()).astype(np.int)
    #logger.info('syl starts {}'.format(syl_starts))
    syl_recs = syl_df.rec.to_numpy().astype(np.int)
    # the starts in absolute rel. to kw(k/d) file
    syl_starts_abs = kwkf.apply_rec_offset(kw_file, syl_starts, syl_recs)
    
    s_v = []
    target = []
    raw_target = []
    
    logger.info('Collecting neural features and target')
    for i_syl, syl in tqdm(syl_df.iterrows(), total=syl_df.shape[0]):
        #print(i_syl)
        len_ms = int(syl['duration'])
        
        len_samples = int(syl['duration'] * ms_to_samp)
        try:
            target_type = fit_pars['target_type']
        except KeyError:
            target_type = fit_pars_default['target_type']

        #logger.info('target_type {}'.format(target_type))

        if target_type == 'spectrogram':
            pars_ms =  syl['rw_spec'][:, :len_ms]
            pars_ms -= np.min(pars_ms)
        elif target_type == 'model':
            pars_ms = np.stack([np.array(syl[p_name]) for p_name in ['alpha', 'beta', 'env']])[:,:len_ms]
        
        #logger.info('target_type {}  - pars_ms shape {}'.format(target_type, pars_ms.shape))
        pars_ms = fit_pars['par_reg_fcn'](pars_ms)
        model_pars = bp.col_binned_max(pars_ms, fit_pars['bin_size'])     
        this_sv, ths_sv_u = unitobjs.support_vector(np.array([syl_starts_abs[i_syl]]), 
                                             len_samples, 
                                             units_list,
                                            bin_size = fit_pars['bin_size'],
                                            history_bins = fit_pars['history_bins']+1,
                                            no_silent=False)
        
        target_mot_id = np.array([syl['mot_id'] for i in np.arange(model_pars.shape[1])])
        target_syl_idx = np.array([i_syl for i in np.arange(model_pars.shape[1])])
        target_s_type = np.array([syl['syl_type'] for i in np.arange(model_pars.shape[1])])
        #start in ms relative to motiff
        target_start_in_syl = np.arange(model_pars.shape[1]) * fit_pars['bin_size']
        target_start_in_mot = target_start_in_syl + syl['start']
        #logger.info('this_sv shape {}'.format(this_sv.shape))
        s_v.append(this_sv)
        target.append(model_pars)
        raw_target.append(np.stack([target_mot_id, target_syl_idx, target_start_in_syl, 
                                    target_s_type, target_start_in_mot]))
    
    logger.info('stacking {} trials'.format(len(s_v)))
    fv_list = [] # feature neural
    tv_list = [] #target pars
    rv_list = [] # raw value marks (motif, start)
    for fv, tv, rv in tqdm(zip(s_v, target, raw_target)):
        f_win, t_win = nd.one_frame_to_window(fv, tv, fit_pars['history_bins'])
        fv_list.append(f_win)
        tv_list.append(t_win)
        rv_list.append(rv.T)
        
    feature_window = np.vstack(fv_list)

    if normalize_features:
        feature_window = feature_window/np.amax(feature_window)
    target_window = np.vstack(tv_list)
    raw_mark_window = np.vstack(rv_list)
    
    logger.info('Making Keras datasets')
    total_samples = feature_window.shape[0]
    test_samples = int(fit_pars['test_fraction']*total_samples)
    logger.info('Test samples {}'.format(test_samples))
    x = feature_window[:-test_samples, :, :]
    Y = target_window[:-test_samples, :]
    Z = raw_mark_window[:-test_samples, :]

    x_t = feature_window[-test_samples: , :, :]
    Y_t = target_window[-test_samples: , :]
    Z_t = raw_mark_window[-test_samples:, :]    
    logger.info('Done, train shapes are {0}, {1}'.format(x.shape, Y.shape))
    
    if flat_features:
        logger.info('Flattening the feature dimensions (for rrnn in keras, as opposed to lstm)')
        X = x.reshape([x.shape[0], -1])
        X_t = x_t.reshape([x_t.shape[0], -1])
    else:
        X = x
        X_t = x_t
    
    return X.astype(np.float32), Y.astype(np.float32), X_t.astype(np.float32), Y_t.astype(np.float32), Z, Z_t


class sessData():
    def __init__(self, bird, sess, fit_pars):
        logger.info('Init bird {0}, session {1}'.format(bird, sess))
        self.bird = bird
        self.sess = sess
        self.fp = fit_pars

        self.s_f = None
        self.all_syl = None
        self.normalize_features = False # whether to normalize the feature vector sets
        self.thresh_factor = 4.5
        
        self.X = None
        self.X_t = None
        self.Y = None
        self.Y_t = None
        self.Z = None
        self.Z_t = None
        
        # load metadatas and files
        self.fn = et.file_names(self.bird, self.sess)
        self.exp_par = et.get_parameters(bird, sess)

        self.kwik_file = et.open_kwik(bird, sess)
        self.kwd_file = et.open_kwd(bird, sess)
        
        self.set_properties_from_dict()
        self.get_s_f()
        self.load_syl()
        self.load_units()
        
    def set_properties_from_dict(self):
        # set properties from fit_pars dict if entered, leave defaults (as set in init) if not
        try:
            self.normalize_features = self.fp['normalize_features']
        except KeyError:
            pass

        try:
            self.thresh_factor = self.fp['thresh_factor']
        except KeyError:
            pass


    def get_s_f(self):
        self.s_f = kwkf.get_record_sampling_frequency(self.kwik_file)
        
    def load_syl(self):
        logger.debug('Loading syllable data for {0}-{1} with file descr {2}'.format(self.bird, self.sess, self.fp['syl_file']))
        syl_files = [os.path.join(self.fn['folders']['ss'], '{0}_{1}.pickle'.format(self.fp['syl_file'], s_type)) 
                     for s_type in self.fp['syl_types']]
                                 
        logger.debug(syl_files)
        loaded_syl = pd.concat([pd.read_pickle(syl_file) for syl_file in syl_files]).sort_values(by=['mot_id', 'start'])
        
        if self.fp['syl_filter']:
            logger.info('Filtering syllable data')
            self.all_syl, _ = filter_syllables(loaded_syl)
        else:
            self.all_syl = loaded_syl
        self.all_syl.reset_index(drop=True, inplace=True)

        
    def load_units(self):
        logger.debug('Loading units data for {0}-{1} with type {2}'.format(self.bird, self.sess, self.fp['unit_type']))
        u_type = self.fp['unit_type']

        if u_type is 'cluster':
            all_sess_units = um.list_sess_units(self.bird, self.sess)
            all_units = kwkf.list_units(self.kwik_file, group=0, sorted=False)    
            # If there is a specific list of clusters to use, use it.
            # Otherwise load the whole set as delivered by the spike sorting
            try:
                clu_list = self.fp['clu_list']
                if clu_list is None:
                     clu_list = all_units.clu
                     logger.info('No units list specified, using all clusters')
                else:
                    logger.info('Getting only {} units'.format(clu_list.size))
            except KeyError:
                logger.info('No units_list specified, using all clusters')
                clu_list = all_units.clu     
            self.units_list = [unitobjs.Unit(clu, kwik_file=self.kwik_file) for clu in clu_list]

        elif u_type is 'threshold':
            chans_list = np.array(self.exp_par['channel_config']['neural'])
            self.units_list = [unitobjs.threshUnit(ch, kwd_file=self.kwd_file, thresh_factor=self.thresh_factor) for ch in chans_list]
        
        elif u_type is 'lfp':     
            chans_list = np.array(self.exp_par['channel_config']['neural'])
            self.units_list = [unitobjs.lfpUnit(ch, kwd_file=self.kwd_file) for ch in chans_list]
        else:
            raise NotImplementedError('dont know how to handle unit type {}'.format(u_type))
            
        logger.debug('Loaded {} units'.format(len(self.units_list)))
        
    def init_set(self, test_trials=None):
        logger.info('Initializing data set for sess {}'.format(self.sess))
        #initialize a non-flat set
        self.X, self.Y, self.X_t, self.Y_t, self.Z, self.Z_t = keras_data_set(self.all_syl, 
                                                                              self.fp,
                                                                              self.units_list,
                                                                              normalize_features = self.normalize_features,
                                                                              flat_features=None)
        
    def gimme_set(self, test_trials=None):
        #flat_features = True if self.fp['network'] is 'ffnn' else False
        self.X, self.Y, self.X_t, self.Y_t, self.Z, self.Z_t = keras_data_set(self.all_syl, 
                                                                              self.fp,
                                                                              self.units_list,
                                                                              normalize_features = self.normalize_features,
                                                                              flat_features=False)
        return self.X, self.Y, self.X_t, self.Y_t
    
    def gimme_raw(self):
        return self.Z, self.Z_t
    
    def gimme_raw_stream(self, z=None):
        if z is None:
            z = self.Z_t
        return get_raw_stream(z, self.all_syl, self.fp['bin_size'], self.s_f)
    
    def max_mot_len(self):
        return np.max(self.Z[:, 4])
    
    def all_bins_list(self):
        mot_len = self.max_mot_len()
        return np.arange(0, mot_len, self.fp['bin_size'])
    
    def all_mots_list(self):
        all_Z = np.concatenate([self.Z, self.Z_t])
        
        return np.unique(all_Z[:, 0])
    
    def gimme_ranged_set(self, bins_list):
        # a training set of all but the listed bins
        # a testing set of the listed bins
        # join all the sets (should be ordered by mot, start)
        all_X = np.concatenate([self.X, self.X_t])
        all_Y = np.concatenate([self.Y, self.Y_t])
        all_Z = np.concatenate([self.Z, self.Z_t])
                
        # select all the bins in the range
        test = ((all_Z[:, 4] >= np.min(bins_list)) & (all_Z[:, 4] <= np.max(bins_list)))
        train = np.invert(test)
        
        return all_X[train], all_Y[train], all_X[test], all_Y[test], all_Z[train], all_Z[test]
    
    def gimme_motif_set(self, mot_list):
        all_X = np.concatenate([self.X, self.X_t])
        all_Y = np.concatenate([self.Y, self.Y_t])
        all_Z = np.concatenate([self.Z, self.Z_t])
        
        # select all the motifs in the range
        test = ((all_Z[:, 0] >= np.min(mot_list)) & (all_Z[:, 0] <= np.max(mot_list)))
        train = np.invert(test)
        
        return all_X[train], all_Y[train], all_X[test], all_Y[test], all_Z[train], all_Z[test]

