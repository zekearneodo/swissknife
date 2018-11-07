import logging
import os
import numpy as np

# for the session functions
from tqdm import tqdm
import pandas as pd

from swissknife.bci.core import expstruct as et
from swissknife.bci.core import kwik_functions as kwkf
from swissknife.bci import synthetic as syn
from swissknife.bci import unitmeta as um
from swissknife.bci import stimalign as sta
from swissknife.streamtools import spectral as sp

from swissknife.bci import units as unitobjs
from swissknife.bci.core.basic_plot import plot_raster, sparse_raster
from swissknife.bci.core import basic_plot as bp

from swissknife.decoder import linear as ld
from swissknife.decoder import neural as nd
##

# for the feedforward object
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

import tensorflow as tf

logger = logging.getLogger('swissknife.hilevel.ffnn')

class FeedForward(object):
    DEFAULTS = {
        "batch_size": 5,
        "validation_split": 0.1,
        "learning_rate": 5E-5,
        "early_stopping_patience": 0,
        "dropout": 0.9,
        "lambda_l2_reg": 1E-5,
        "nonlinearity": tf.nn.relu,
        "squashing": tf.nn.sigmoid,
        "regularization": tf.contrib.layers.l2_regularizer,
        "mu": 0,
        "sigma": 1.,
        "history_bins": 20,  # number of lookup bins
        "n_features": 64,  # number of channels of input (n of clusters, n of channels, n of lfp bands, ...)
        "n_target": 3,  # dimension of the target
        "initializer": tf.contrib.layers.xavier_initializer(),
        "log_dir": './log_feedforward_2'
    }
    RESTORE_KEY = "to_restore"

    def __init__(self, architecture=[], d_hyperparams={}, log_dir='./log_feedforward', tf_config=None):
        '''
        tf_config: a tf.ConfigProto() object, None by default
        '''
        self.architecture = architecture
        self.__dict__.update(FeedForward.DEFAULTS, **d_hyperparams)
        self.log_dir = self.log_dir

        tf.set_random_seed(1234)
        
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
        #tf_config = tf.ConfigProto(gpu_options=gpu_options)
        self.sesh = tf.Session(config=tf_config)

        # TODO: decide if load a model or build a new one. For now, build it
        handles = self._build_graph()

        # In any case, make a collection of variables that are restore-able
        for handle in handles:
            tf.add_to_collection(FeedForward.RESTORE_KEY, handle)

        # initialize all globals of the session
        self.sesh.run(tf.global_variables_initializer())

        # Unpack the tuple of handles created by the builder
        (self.x, self.y, self.y_, self.r_loss, self.c_solver,
         self.merged_summaries, self.global_step) = handles

        # Initialize the filewriter and write the graph (tensorboard)
        self.writer = tf.summary.FileWriter(self.log_dir, self.sesh.graph)

    #     def __del__(self):
    #         logger.info('FeedForward went out of scope, will die. Bye bye now.')
    #         self.clear()

    def clear(self):
        logger.info('Will clear the graph now')
        self.sesh.close()
        tf.reset_default_graph()

    def regressor(self, x_in):
        # r is a batch of neural features of dimensions [batch_size, hist_bin, n_units]
        # predictor gets a batch of r vectors and returns a batch of output values
        logger.info('Architecture X_in shape {}'.format(x_in.shape))
        with tf.variable_scope('regressor_common', reuse=tf.AUTO_REUSE):
            x = tf.reshape(x_in, [-1, self.n_features * self.history_bins], name='neural_input')
            logger.info('Architacture x shape {}'.format(x_in.shape))
            x = tf.layers.dense(inputs=x, units=(self.n_features * self.history_bins) // 2,
                                activation=self.nonlinearity,
                                kernel_initializer=self.initializer,
                                name='d_0')

            x = tf.layers.dense(inputs=x, units=128,
                                activation=self.nonlinearity,
                                kernel_initializer=self.initializer,
                                name='d_1')

            x = tf.layers.dense(inputs=x, units=32,
                                activation=self.nonlinearity,
                                kernel_initializer=self.initializer,
                                name='d_2')

            y_ = tf.layers.dense(inputs=x, units=self.n_target,
                                 activation=self.nonlinearity,
                                 kernel_initializer=self.initializer,
                                 name='y_0')

            return y_

    def regress(self, x):
        return self.sesh.run(self.y_, feed_dict={self.x: x})

    def regression_loss(self, y, y_):
        # The loss of the regression, assuming same loss function for everything
        loss = tf.reduce_sum(tf.square(y - y_) / (self.batch_size), name='regression_loss')
        return loss

    def _build_graph(self):
        # place holders for the batches of neural data, target data
        x = tf.placeholder(tf.float32, shape=[None, self.history_bins, self.n_features], name='x')
        y = tf.placeholder(tf.float32, shape=[None, self.n_target], name='y')
        logger.info('build_graph x placeholder shape {}'.format(x.shape))

        # Make a batch of predictions
        y_ = self.regressor(x)

        # Compute the losses
        r_loss = self.regression_loss(y, y_)

        # Minimize the gradients:
        # 1. Will use an trainable optimizer, rather than compute, clip, and apply (simpler but darker)
        # 2. They are all updated at the same intervals, hence only one global step
        # 3. There will be different optimizers for different groups of variables (3 for the alpha, beta, env)

        # Since they will all be run the same number of times for each minibatch, only one of them
        # gets to advance the global_step
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Regression minimize grads:
        # Solver has to 'train' (modify) the TRAINABLE variables within scope of 'common'
        # The mechanics of the gradients is
        # - Collect variables to update -trainable and within a given scope-
        # - Compute gradients (given a cost function -r_loss-, for a list of variables -c_vars-)
        # - Clip those gradients
        # - Apply them
        with tf.name_scope('common_optimizer'):
            c_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='regressor_common')
            c_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            c_grads_and_vars = c_optimizer.compute_gradients(r_loss, var_list=c_vars)

            clipped = []
            for grad, var in c_grads_and_vars:
                clipped.append((tf.clip_by_value(grad, -100, 100), var))
            c_solver = c_optimizer.apply_gradients(clipped, global_step=global_step,
                                                   name='common_apply_grad')
            # Alternatively, one can just let the optimizer do it's deed
            # c_solver = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(r_loss,
            #                                                                             var_list=c_vars,
            #                                                                             global_step=global_step)

        # append variables to summary
        with tf.name_scope('summaries'):
            with tf.name_scope('loss'):
                tf.summary.scalar('c_loss', r_loss)
        summaries = tf.summary.merge_all()

        return x, y, y_, r_loss, c_solver, summaries, global_step

    def train(self, X, Y, max_iter=np.inf, max_epochs=np.inf, cross_validate=True, verbose=False):
        # set aside train/validation split
        x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=self.validation_split)
        r_losses = []
        v_losses = []
        patience = self.early_stopping_patience
        logger.info('Will begin training with x:{}, y:{}'.format(x_train.shape, y_train.shape))
        logger.info('Early stopping patience {}'.format(self.early_stopping_patience))
        try:
            while True:
                # Get a batch of randomly ordered train data
                x, y = shuffle(x_train, y_train, n_samples=self.batch_size)
                # Run a step of the optimizer training for the regressor
                c_solver, r_loss, i, summary = self.sesh.run([self.c_solver,
                                                              self.r_loss,
                                                              self.global_step,
                                                              self.merged_summaries],
                                                             feed_dict={self.x: x, self.y: y})
                # output, logging and interruption
                if i % 10 == 0:  # Record summaries and test (valiate)
                    self.writer.add_summary(summary, i)
                    r_losses.append(r_loss)
                    # logger.info('Step {}'.format(i))
                    # manage the early stopping
                    if self.early_stopping_patience > 0:
                        if r_loss >= r_losses[-1]:
                            patience -= 1
                            if patience == 0:
                                logger.info('{} batches without loss improvement, early stopping')
                        else:
                            patience = self.early_stopping_patience

                if i % 100000 == 0 and verbose:
                    # pick a batch from the validation set and test performance
                    x_v, y_v = shuffle(x_val, y_val, n_samples=self.batch_size)
                    r_loss_cross = self.sesh.run([self.r_loss],
                                                 feed_dict={self.x: x_v, self.y: y_v})
                    v_losses.append(r_loss_cross)
                    logger.info('Round {}, r_loss {}, validation_loss {}'.format(i, r_loss, r_loss_cross))

                if i >= max_iter:
                    logger.info("final avg cost (@ step {} = {})".format(i, r_loss))
                    try:
                        self.writer.flush()
                        self.writer.close()
                    except(AttributeError):  # not logging
                        continue
                    break

        except KeyboardInterrupt:
            logger.info('User ended')

        plt.plot(np.array(r_losses))
        return r_losses, v_losses
###############################

def scale_pc(pc, pca_dict):
    '''scale an array of pc values to 0-1 for all components (using max/min for all dataset)'''
    mins = pca_dict['pc_min'].reshape((1, -1))
    ptps = pca_dict['pc_ptp'].reshape((1, -1))
    return (pc - mins)/ptps

def scale_pc_inv(pc, pca_dict):
    '''invert scaling of an array of pc values to 0-1 for all components (using max/min for all dataset)'''
    mins = pca_dict['pc_min'].reshape((1, -1))
    ptps = pca_dict['pc_ptp'].reshape((1, -1))
    return pc * ptps + mins


def filter_syllables(all_syll, syl_type=None, test_frac=0):
    if syl_type:
        syl_selection = (all_syll['syl_type'] == syl_type) & (all_syll['keep'] == True)
    else:
        syl_selection = (all_syll['keep'] == True)

    select_syl = all_syll[syl_selection]
    total_trials = select_syl.shape[0]
    test_trials = np.int(test_frac * total_trials)

    syl_train = select_syl[:total_trials - test_trials].reset_index(drop=True)
    syl_test = select_syl[-test_trials:]
    return syl_train, syl_test


def get_raw_stream(z_mot, sdf, bin_size, s_f):
    ms_samples = int(s_f * .001)
    bin_raw_stream_list = []
    for z_bin in z_mot:
        new_mot = sdf.iloc[z_bin[1]].rw_stream[z_bin[2] * ms_samples: (z_bin[2] + bin_size) * ms_samples]
        bin_raw_stream_list.append(new_mot)
    # bin_raw_stream_list = [sdf.iloc[z_bin[1]].rw_stream[z_bin[2] * ms_samples: (z_bin[2] + bin_size) * ms_samples]
    #                        for z_bin in z_mot]
    return np.concatenate(bin_raw_stream_list)


def flatten_feature_vector(x):
    # flatten a feature vector from keras (put it in ffnn shape as opposed to lstm)
    return x.reshape([x.shape[0], -1])


def keras_data_set(syl_df, fit_pars, units_list, flat_features=False, normalize_features=False):
    # all syl and units in unts_list are from the same bird and sess
    # (they share the kwd and kwik file)
    # syl_df = syl_df_in[:10]
    s_f = units_list[0].sampling_rate
    kw_file = units_list[0].h5_file
    ms_to_samp = s_f // 1000.

    syl_starts = (syl_df.start.as_matrix() * ms_to_samp
                  + syl_df.mot_start.as_matrix()).astype(np.int)
    # logger.info('syl starts {}'.format(syl_starts))
    syl_recs = syl_df.rec.as_matrix().astype(np.int)
    # the starts in absolute rel. to kw(k/d) file
    syl_starts_abs = kwkf.apply_rec_offset(kw_file, syl_starts, syl_recs)

    s_v = []
    target = []
    raw_target = []
    
    try:
        fit_target = fit_pars['fit_target']
    except KeyError:
        logger.debug('No fit target specified, going for dynamical model parameters')
        fit_target = 'dyn'

    try:
        #step != bin
        win_size = fit_pars['win_size']
        step_size = fit_pars['bin_size']
    except KeyError:
        # force step = bin
        logger.debug('Now win_size in fit_parameters, choosing win_size=bin_size (step = bin = win_size)')
        win_size = fit_pars['bin_size']
        step_size = fit_pars['bin_size']
    
    logger.info('Collecting neural features and target')
    logger.info('Fit target is {}'.format(fit_target))
    logger.info('bin_size = {} / win_size = {}'.format(win_size, step_size))
    for i_syl, syl in tqdm(syl_df.iterrows(), total=syl_df.shape[0]):
        # print(i_syl)
        len_ms = int(syl['duration'])
        len_samples = int(syl['duration'] * ms_to_samp)
        
        # Get parameters acording to target, in ms scale
        if fit_target == 'dyn':
            pars_ms = np.stack([np.array(syl[p_name]) for p_name in ['alpha', 'beta', 'env']])[:, :len_ms]
            pars_ms = fit_pars['par_reg_fcn'](pars_ms)
        elif fit_target == 'pc':
            pars_ms = np.array(syl['pc']).T[:, :len_ms]
            pars_ms = scale_pc(pars_ms.T, fit_pars['pca_dict']).T
        
        # get the neural data wheterh fixed or variable step
        if step_size == win_size:
            # Do the legacy thing
            model_pars = bp.col_binned(pars_ms, step_size)/step_size       
            this_sv, ths_sv_u = unitobjs.support_vector(np.array([syl_starts_abs[i_syl]]), 
                                             len_samples, 
                                             units_list,
                                            bin_size = step_size,
                                            history_bins = fit_pars['history_bins']+1,
                                            no_silent=False)

        else:
            # Get the stepped parameters and neural vectors convolved with window and get them in steps
            # Instead of n. of bins, do ms but with bin running avg (or savgol)
            model_pars = np.stack([np.convolve(x, np.ones(win_size), mode='same') for x in pars_ms])
            model_pars = model_pars[:, ::step_size]/win_size

            this_sv, _ = unitobjs.support_vector_ms(np.array([syl_starts_abs[i_syl]]),
                                                        len_samples,
                                                        units_list,
                                                        win_size=win_size,
                                                        step_size=step_size,
                                                        history_steps=fit_pars['history_bins'] + 1,
                                                        no_silent=False)

        #the shape of this_sv; this_sv_u are:
        # this_sv_u: list of used units
        #ms_sv: [n_bin, n_unit, n_trial], where n_bin is n_steps if support_vector_ms
        target_mot_id = np.array([syl['mot_id'] for i in np.arange(model_pars.shape[1])])
        target_syl_idx = np.array([i_syl for i in np.arange(model_pars.shape[1])])
        target_s_type = np.array([syl['syl_type'] for i in np.arange(model_pars.shape[1])])
        # start in ms relative to motiff
        target_start_in_syl = np.arange(model_pars.shape[1]) * step_size
        target_start_in_mot = target_start_in_syl + syl['start']
        logger.debug('this_sv shape {}'.format(this_sv.shape))
        s_v.append(this_sv)
        target.append(model_pars)
        raw_target.append(np.stack([target_mot_id, target_syl_idx, target_start_in_syl,
                                    target_s_type, target_start_in_mot]))

    logger.info('stacking {} trials'.format(len(s_v)))
    fv_list = []  # feature neural
    tv_list = []  # target pars
    rv_list = []  # raw value marks (motif, start)
    for fv, tv, rv in tqdm(zip(s_v, target, raw_target)):
        f_win, t_win = nd.one_frame_to_window(fv, tv, fit_pars['history_bins'])
        fv_list.append(f_win)
        tv_list.append(t_win)
        rv_list.append(rv.T)

    feature_window = np.vstack(fv_list)
    if normalize_features:
        feature_window = feature_window / np.amax(feature_window)
    target_window = np.vstack(tv_list)
    raw_mark_window = np.vstack(rv_list)

    logger.info('Making Keras datasets')
    total_samples = feature_window.shape[0]
    test_samples = int(fit_pars['test_fraction'] * total_samples)
    logger.info('Test samples {}'.format(test_samples))
    if test_samples > 0:
        x = feature_window[:-test_samples, :, :]
        Y = target_window[:-test_samples, :]
        Z = raw_mark_window[:-test_samples, :]

        x_t = feature_window[-test_samples:, :, :]
        Y_t = target_window[-test_samples:, :]
        Z_t = raw_mark_window[-test_samples:, :]
    else:
        x = feature_window
        Y = target_window
        Z = raw_mark_window

        x_t = np.empty([0, feature_window.shape[1], feature_window.shape[2]])
        Y_t = np.empty([0, target_window.shape[1]])
        Z_t = np.empty([0, raw_mark_window.shape[1]])

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
    def __init__(self, bird, sess, fit_pars, syl_data=None):
        logger.info('Init bird {0}, session {1}'.format(bird, sess))
        self.bird = bird
        self.sess = sess
        self.fp = fit_pars
        self.s_f = None
        self.all_syl = None

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

        self.get_s_f()
        self.load_syl(syl_data)
        self.load_units()

    def get_s_f(self):
        self.s_f = kwkf.get_record_sampling_frequency(self.kwik_file)

    def load_syl(self, syl_data):
        if syl_data is None:
            logger.debug('Loading syllable data for {0}-{1} with file descr {2}'.format(self.bird, self.sess, self.fp['syl_file']))
            syl_files = [os.path.join(self.fn['folders']['ss'], '{0}_{1}.pickle'.format(self.fp['syl_file'], s_type))
                         for s_type in self.fp['syl_types']]

            logger.debug(syl_files)
            loaded_syl = pd.concat([pd.read_pickle(syl_file) for syl_file in syl_files]).sort_values(by=['mot_id', 'start'])
        else:
            loaded_syl = syl_data

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
            #all_sess_units = um.list_sess_units(self.bird, self.sess)
            all_units = kwkf.list_units(self.kwik_file, group=0, sorted=False)
            self.units_list = [unitobjs.Unit(clu, kwik_file=self.kwik_file) for clu in all_units.clu]

        elif u_type is 'threshold':
            chans_list = np.array(self.exp_par['channel_config']['neural'])
            self.units_list = [unitobjs.threshUnit(ch, kwd_file=self.kwd_file) for ch in chans_list]

        elif u_type is 'lfp':
            chans_list = np.array(self.exp_par['channel_config']['neural'])
            self.units_list = [unitobjs.lfpUnit(ch, kwd_file=self.kwd_file) for ch in chans_list]
        else:
            raise NotImplementedError('dont know how to handle unit type {}'.format(u_type))

        logger.debug('Loaded {} units'.format(len(self.units_list)))

    def init_set(self, test_trials=None):
        logger.info('Initializing data set for sess {}'.format(self.sess))
        # initialize a non-flat set
        self.X, self.Y, self.X_t, self.Y_t, self.Z, self.Z_t = keras_data_set(self.all_syl,
                                                                              self.fp,
                                                                              self.units_list,
                                                                              flat_features=None)

    def gimme_set(self, test_trials=None):
        # flat_features = True if self.fp['network'] is 'ffnn' else False
        self.X, self.Y, self.X_t, self.Y_t, self.Z, self.Z_t = keras_data_set(self.all_syl,
                                                                              self.fp,
                                                                              self.units_list,
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

########


def train_and_test(x, y, x_t, y_t, sess_data, name_suffix='whl', tf_back_end=None):
    fit_pars = sess_data.fp
    fn = sess_data.fn

    try:
        fit_target = fit_pars['fit_target']
    except KeyError:
        fit_target = 'dyn'

    assert fit_pars['network'] is 'ffnn'
    logger.info('Network is FFNN (tf)')
    #logger.info('fit pars: {}'.format(fit_pars))
    model_name = 'tf_{0}_{1}_bs{2:02d}_hs{3:02d}_te{4:03d}_ut{5}_{6}.h5'.format(fit_target,
                                                                                fit_pars['network'],
                                                                            fit_pars['bin_size'],
                                                                            fit_pars['history_bins'],
                                                                            fit_pars['num_ep'],
                                                                            fit_pars['unit_type'],
                                                                            name_suffix)
    model_path = os.path.join(fn['folders']['ss'], 'model_' + model_name)
    logger.info('Model path: {}'.format(model_path))

    
    net_pars = {'batch_size': fit_pars['batch_size'],
                'history_bins': fit_pars['history_bins'],
                 'n_features': x.shape[2],
                'n_target': y.shape[-1]}

    # Define, train
    logger.info('Defining the network')
    try:
        tf_config = tf_back_end['config']
    except:
        logger.info('No or bad config in tf_back_end dictionary, will go with default session config')
        tf_config = None
        
    ffnn = FeedForward(d_hyperparams=net_pars, tf_config=tf_config)
    logger.info('Will begin network training')
    losses = ffnn.train(x, y, max_iter=fit_pars['max_iter'])
    logger.info('Done training. Will make a prediction batch now')
    y_r = ffnn.regress(x_t)
    # Close the session
    ffnn.clear()

    return model_path, y_r


def mot_wise_train(sess_data, mots_per_run=7, only_chunks='all', tf_back_end=None):
    all_mots = sess_data.all_mots_list()
    n_mots = all_mots.size
    chunks = [all_mots[i:i + mots_per_run] for i in np.arange(0, n_mots, mots_per_run)]
    n_chunks = len(chunks)

    all_model_path = []
    all_model_pred = []
    all_Z_t = []
    all_Y_t = []
    all_X_t = []

    if only_chunks == 'all':
        do_chunks = chunks
    else:
        do_chunks = [chunks[i] for i in only_chunks]

    logger.info('Will run for {}/{} chunks'.format(len(do_chunks), n_chunks))
    logger.info('Chunks {}'.format(do_chunks))

    for i_c, chunk in enumerate(do_chunks):
        logger.info('Chunk {0}/{1}'.format(i_c, len(do_chunks)))
        logger.info('Chunk comprises motifs {}'.format(chunk))

        X, Y, X_t, Y_t, Z, Z_t = sess_data.gimme_motif_set(chunk)
        mod_path, mod_pred = train_and_test(X, Y, X_t, Y_t, sess_data, 
                                            name_suffix='mot-chnk{:03d}'.format(i_c), 
                                            tf_back_end=tf_back_end)
        all_model_path.append(mod_path)
        all_model_pred.append(mod_pred)
        all_Z_t.append(Z_t)
        all_Y_t.append(Y_t)
        all_X_t.append(X_t)

    training_pd = pd.DataFrame({
        'model_path': all_model_path,
        'Y_p': all_model_pred,
        'Z_t': all_Z_t,
        'Y_t': all_Y_t,
        'X_t': all_X_t,
        })

    # save the dataframe to a pickle file
    pickle_path, base_name = os.path.split(mod_path)
    pickle_name_parts = base_name.split('mot-chnk')
    full_pickle_path = os.path.join(pickle_path, pickle_name_parts[0] + 'motwise.pickle')
    training_pd.to_pickle(full_pickle_path)
    logger.info('saved all network chunks data/metadata in {}'.format(full_pickle_path))

    [Y_t_arr, Z_t_arr, Y_p_arr] = map(np.vstack, [all_Y_t, all_Z_t, all_model_pred])

    for j in [0, 1, 2]:
        plt.figure()
        plt.plot(Y_t_arr[:200, j])
        plt.plot(Y_p_arr[:200, j])

    return Y_t_arr, Z_t_arr, Y_p_arr, training_pd


