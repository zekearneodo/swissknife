# objects to do quick stuff with clusters
from __future__ import division

import numpy as np
import logging
import os
import pickle
from tqdm import tqdm

from swissknife.bci.core.file import h5_functions as h5f
from swissknife.bci.core.file import file_functions as ff
from swissknife.bci.core import expstruct as et
from swissknife.bci.core import kwik_functions as kf
from swissknife.bci.core import basic_plot as bp

from swissknife.streamtools import streams as st
from swissknife.streamtools import spectral as sp
from swissknife.streamtools import temporal as tp

logger = logging.getLogger('swissknife.bci.core.units')


class superUnit:
    def __init__(self, clu, h5_file=None):
        '''
        :param clu: int,
        :param h5_file: kwik or kwd file usable with the methods
                        kf.rec_start_array
        '''

        self.clu = clu
        self.h5_file = h5_file

        self.type = None
        self.id = None
        self.bird_id = None
        self.sess_id = None

        self.qlt = None
        self.time_samples = None
        self.recordings = None
        self.recording_offsets = None
        self.sampling_rate = None

        if h5_file is not None:
            self.get_sampling_rate()
            self.get_rec_offsets()
            self.get_exp_pars()

            self.get_qlt()
            self.get_time_stamps()

    def get_rec_offsets(self):
        self.recording_offsets = kf.rec_start_array(self.h5_file)
        return self.recording_offsets

    def get_sampling_rate(self):
        assert (self.h5_file is not None)
        self.sampling_rate = h5f.get_record_sampling_frequency(self.h5_file)
        return self.sampling_rate

    def get_qlt(self):
        pass

    def get_time_stamps(self):
        pass

    def get_exp_pars(self):
        file_parts = ff.split_path(self.h5_file.filename)
        self.sess_id = file_parts[-2]
        self.bird_id = file_parts[-3]
        self.exp_pars = et.get_parameters(self.bird_id, self.sess_id)


class lfpUnit(superUnit):
    def __init__(self, ch, kwd_file=None, lfp_pars=None):
        super(lfpUnit, self).__init__(ch, h5_file=kwd_file)

        self.type = 'lfp'
        self.rec_dicts = None
        self.window_samples = None
        self.lfp_pars = lfp_pars
        self.set_lfp_pars()

    def set_lfp_pars(self):
        if self.lfp_pars is None:
            s_f = self.get_sampling_rate()
            self.lfp_pars = {'f_min': 0,
                             'f_max': 500,
                             'win_samples': 128 * 8,
                             'step_samples': int(0.001 * s_f),
                             'db_cut': 90,
                             's_f': s_f}
            self.window_samples = self.lfp_pars['win_samples']

    def get_raster(self):
        raise NotImplementedError

    def get_qlt(self):
        self.qlt = 5


class threshUnit(superUnit):
    def __init__(self, ch, kwd_file=None, thresh_factor=4.5):
        # init the superUnit class
        super(threshUnit, self).__init__(ch, h5_file=kwd_file)

        self.type = 'threshold'
        self.rec_dicts = None
        self.thresh_factor = None
        self.filter_pars = None
        self.filter_func = None
        self.filter_args = None
        self.filter_kwargs = None

        self.get_rec_dicts()
        self.set_thresh_factor(thresh_factor)
        self.set_filter_pars()
        self.set_filter_func()

    def set_thresh_factor(self, thf=4.5):
        self.thresh_factor = thf

    def set_filter_pars(self, filter_band=[500, 10000],
                        filter_gen_func=sp.make_butter_bandpass,
                        *args, **kwargs):

        self.filter_pars = filter_gen_func(self.sampling_rate,
                                           filter_band[0],
                                           filter_band[1],
                                           *args,
                                           **kwargs)

    def set_filter_func(self, filter_func=sp.apply_butter_bandpass, *args, **kwargs):
        self.filter_func = filter_func
        self.filter_args = args
        self.filter_kwargs = kwargs

    def get_rec_dicts(self, filter_band=[500, 10000], force=False):
        logger.debug('Getting rec dicts')
        recs_list = h5f.get_rec_list(self.h5_file)
        neural_chans = np.array(self.exp_pars['channel_config']['neural'])

        self.rec_dicts = []
        for i, rec in enumerate(recs_list):
            logger.debug('Rec {}:'.format(rec))
            dset = h5f.get_data_set(self.h5_file, rec)
            rec_stream_obj = st.H5Data(dset, self.sampling_rate,
                                       dtype=np.float,
                                       chan_list=neural_chans)
            one_dict = {'rec': rec,
                        'h5d': rec_stream_obj,
                        'rms': self.get_rms(rec_stream_obj,
                                            filter_band=filter_band,
                                            force=force)}
            self.rec_dicts.append(one_dict)

    def get_rms(self, stream_obj, filter_band=[500, 10000], force=False):
        '''
        get rms for a stream_obj
        param stream_obj: H5Data object
        '''
        logger.debug('Will get rms of dset {}'.format(stream_obj.data.name))
        dset_loc = os.path.split(stream_obj.data.parent.file.filename)[0]
        dset_rec = stream_obj.data.name.split('/')[-2].zfill(3)
        rms_filename = os.path.join(dset_loc, 'rms_{}.npy'.format(dset_rec))
        try:
            logger.debug('Looking for rms stored in {}'.format(rms_filename))
            assert (not force)
            rms = np.load(rms_filename)
            logger.debug('Found and loaded')
        except:
            logger.warning('Could not find rms npy file or forced to compute')

            self.filter_pars = sp.make_butter_bandpass(self.sampling_rate,
                                                       filter_band[0],
                                                       filter_band[1])

            rms = stream_obj.get_rms(window_size_samples=50000,
                                     n_windows=10000,
                                     rms_func=filter_rms,
                                     rms_args=(self.filter_pars,))
            np.save(rms_filename, rms)
        return rms

    def get_raster(self, starts, span, span_is_ms=True, return_ms=None, recs=None):
        """
        :param starts: start points of each event (in samples, absolute unles recs is provided)
        :param span: span of the raster (in samples or ms, depending on value of span_is_ms)
        :param span_is_ms: whether the span of the raster is given in samples or ms, default is ms
        :param return_ms: whether to return the raster in ms units, default is to do the units set by span_is_ms
        :param recs: if recs is provided, starts are referred to the beginning of each rec
                     otherwise, the method will identify which rec each start belongs to and offset accordingly
        :return: n x span
        """
        raise NotImplementedError
        if recs is None:
            recs = kf.get_corresponding_rec(self.h5_file, starts)
            start_rec_offsets = kf.rec_start_array(self.h5_file)
            rec_list = kf.get_rec_list(self.h5_file)
        else:
            assert (starts.size == recs.size)
            rec_list = kf.get_rec_list(self.h5_file)
            start_rec_offsets = np.ddzeros_like(rec_list)

        return_ms = span_is_ms if return_ms is None else return_ms
        span_samples = np.int(span * self.sampling_rate * 0.001) if span_is_ms else span
        span_ms = span if span_is_ms else np.int(span * 1000. / self.sampling_rate)

        rows = starts.shape[0]
        cols = span_ms if return_ms else span_samples
        raster = np.empty((rows, cols), dtype=np.float64)
        raster[:] = np.nan

        # do the raster in samples
        i_trial = 0
        for rec in np.unique(recs):
            rec_frames
            rec_time_samples = self.time_samples[self.recordings == rec]
            for start in starts[recs == rec]:
                start -= start_rec_offsets[rec_list == rec]
                end = np.int(start + span_samples)
                where = (rec_time_samples[:] >= start) & (rec_time_samples[:] <= end)
                n = np.sum(where)
                raster[i_trial, :n] = rec_time_samples[where] - start
                if return_ms:
                    raster[i_trial, :n] = np.round(raster[i_trial, :n] * 1000. / self.sampling_rate)
                i_trial += 1
        return raster

    def get_qlt(self):
        return 4


class Unit:
    def __init__(self, clu, group=0, kwik_file=None, sort=0):

        self.clu = clu
        self.group = group
        self.bird = None
        self.sess = None

        self.kwik_file = kwik_file
        self.h5_file = kwik_file
        self.sort = sort

        self.type = 'cluster'
        self.id = None
        self.metrics = None

        self.qlt = None
        self.time_samples = None
        self.recordings = None
        self.recording_offsets = None
        self.sampling_rate = None

        self.kwd_file = None
        self.all_waveforms = None # all of the waveforms
        self.n_waveforms = 1000 #sample of waveforms to show/compute
        self.waveforms = None
        self.avg_waveform = None
        self.main_chan = None
        self.main_wave = None
        self.waveform_pars = {}

        if kwik_file is not None:
            self.get_sampling_rate()
            self.get_qlt()
            self.get_time_stamps()
            self.get_rec_offsets()
            self.get_sess_par()

    # get time stamps of spiking events (in samples)
    def get_time_stamps(self):
        assert (self.kwik_file is not None)

        clu_path = "/channel_groups/{0:d}/spikes/clusters/main".format(self.group)
        t_path = "/channel_groups/{0:d}/spikes/time_samples".format(self.group)
        r_path = "/channel_groups/{0:d}/spikes/recording".format(self.group)

        dtype = self.kwik_file[t_path].dtype
        # time samples are relative to the beginning of the corresponding rec
        time_samples = np.array(self.kwik_file[t_path][self.kwik_file[clu_path][:] == self.clu],
                                dtype=np.dtype(dtype))

        dtype = self.kwik_file[r_path].dtype
        # recordings ids (as in the key)
        recordings = np.array(self.kwik_file[r_path][self.kwik_file[clu_path][:] == self.clu],
                              dtype=np.dtype(dtype))

        # patch for a random kilosort error that throws a random 0 for a time_stamp
        self.time_samples = time_samples[time_samples > 0]
        self.recordings = recordings[time_samples > 0]
        return self.time_samples, self.recordings

    def get_rec_offsets(self):
        self.recording_offsets = kf.rec_start_array(self.kwik_file)
        return self.recording_offsets

    # get the quality of the unit
    def get_qlt(self):
        assert (self.kwik_file is not None)

        path = "/channel_groups/{0:d}/clusters/main/{1:d}".format(self.group, self.clu)
        self.qlt = self.kwik_file[path].attrs.get('cluster_group')

    def get_sampling_rate(self):
        assert (self.kwik_file is not None)
        self.sampling_rate = h5f.get_record_sampling_frequency(self.kwik_file)
        return self.sampling_rate

    def get_sess_par(self):
        folder = self.get_folder()
        bird_folder, sess = os.path.split(folder)
        self.bird = str(os.path.split(bird_folder)[1])
        self.sess = str(sess)
        self.id = 'unit_{}_{}_{}_{}'.format(self.bird, self.sess, self.group, self.clu)

    def get_raster(self, starts, span, span_is_ms=True, return_ms=None, recs=None):
        """
        :param starts: start points of each event (in samples, absolute unles recs is provided)
        :param span: span of the raster (in samples or ms, depending on value of span_is_ms)
        :param span_is_ms: whether the span of the raster is given in samples or ms, default is ms
        :param return_ms: whether to return the raster in ms units, default is to do the units set by span_is_ms
        :param recs: if recs is provided, starts are referred to the beginning of each rec
                     otherwise, the method will identify which rec each start belongs to and offset accordingly
        :return: n x span
        """

        if recs is None:
            recs = kf.get_corresponding_rec(self.kwik_file, starts)
            start_rec_offsets = kf.rec_start_array(self.kwik_file)
            rec_list = kf.get_rec_list(self.kwik_file)
        else:
            assert (starts.size == recs.size)
            rec_list = kf.get_rec_list(self.kwik_file)
            start_rec_offsets = np.zeros_like(rec_list)

        return_ms = span_is_ms if return_ms is None else return_ms
        span_samples = np.int(span * self.sampling_rate * 0.001) if span_is_ms else span
        span_ms = span if span_is_ms else np.int(span * 1000. / self.sampling_rate)

        rows = starts.shape[0]
        cols = span_ms if return_ms else span_samples
        raster = np.empty((rows, cols), dtype=np.float64)
        raster[:] = np.nan

        # do the raster in samples
        i_trial = 0
        for rec in np.unique(recs):
            rec_time_samples = self.time_samples[self.recordings == rec]
            for start in starts[recs == rec]:
                start -= start_rec_offsets[rec_list == rec]
                end = np.int(start + span_samples)
                where = (rec_time_samples[:] >= start) & (rec_time_samples[:] <= end)
                n = np.sum(where)
                raster[i_trial, :n] = rec_time_samples[where] - start
                if return_ms:
                    raster[i_trial, :n] = np.round(raster[i_trial, :n] * 1000. / self.sampling_rate)
                i_trial += 1
        return raster

    def get_isi(self):
        if self.time_samples is None:
            self.get_time_stamps()
        all_isi_ms = np.round(np.diff(self.time_samples)/(self.sampling_rate * 0.001))
        return all_isi_ms


    def get_isi_dist(self, bin_size_ms=1, max_t=100):
        if self.time_samples is None:
            self.get_time_stamps()
        all_isi_ms = np.round(np.diff(self.time_samples)/(self.sampling_rate * 0.001))

        bins = np.arange(0, max_t, bin_size_ms)
        hist, bins = np.histogram(all_isi_ms, bins)

        bins = bins[1:]
        two_side_bins = np.concatenate([-bins[::-1], bins[1:]])
        two_side_hist = np.concatenate([hist[::-1], hist[1:]])
        return two_side_hist, two_side_bins

    def get_folder(self):
        return os.path.split(os.path.abspath(self.kwik_file.filename))[0]

    def get_kilo_folder(self):
        return os.path.join(self.get_folder(), 'kilo_{:02d}'.format(self.group))

    def get_kwd_path(self):
        return os.path.join(self.get_folder(), 'experiment.raw.kwd')

    def get_unit_chans(self):
        # load parameters
        sess_par = et.get_parameters(self.bird, self.sess, location='ss')
        # get the chan list fo this shank
        try:
            sess_chans = sess_par['channel_config']['neural_{}'.format(self.group)]
        except KeyError:
            if self.group == 0:
                sess_chans = sess_par['channel_config']['neural']
            else:
                raise
        return sess_chans

    def save_unit_spikes(self):
        unit_path = self.get_folder()
        file_folder = os.path.join(unit_path, 'unit_waveforms')
        et.mkdir_p(file_folder)
        file_path = os.path.join(file_folder,
                                 'unit_{}_{:03d}.npy'.format(self.group,
                                                             self.clu))
        logger.info('Saving unit {0} in file {1}'.format(self.clu, file_path))
        np.save(file_path, self.all_waveforms)
        par_path = os.path.join(file_folder,
                                'unit_{}_{:03d}.par.pickle'.format(self.group,
                                                            self.clu))
        pickle.dump(self.waveform_pars, open(par_path, 'wb'))

    def load_unit_spikes(self):
        logger.debug('will try to load previous unit files')
        # careful, loads the last saved
        folder = self.get_folder()
        f_name = 'unit_{}_{:03d}.npy'.format(self.group, self.clu)
        p_name = 'unit_{}_{:03d}.par.pickle'.format(self.group, self.clu)
        self.waveform_pars = pickle.load(open(os.path.join(folder, 'unit_waveforms', p_name),
                                              'rb'))
        self.all_waveforms = np.load(os.path.join(folder, 'unit_waveforms', f_name), mmap_mode='r')

        return self.all_waveforms

    def get_principal_channels(self, projectors=4):
        kilo_path = self.get_kilo_folder()
        all_features = np.load(os.path.join(kilo_path, 'pc_features.npy'))
        all_clusters = np.load(os.path.join(kilo_path, 'spike_templates.npy'))
        pc_ind = np.load(os.path.join(kilo_path, 'pc_feature_ind.npy'))

        this_clu_n = self.clu
        # average across all the pc values of this cluster
        clu_feat = all_features[np.where(all_clusters.flatten() == self.clu)[0], :, :]
        clu_mean_feat = np.abs(np.mean(clu_feat, axis=0))

        # larger projections
        main_feat = (np.fliplr(np.argsort(clu_mean_feat, axis=1))[:, :projectors])
        p_p = main_feat.reshape([1, -1], order='F').flatten()
        indexes = np.unique(p_p, return_index=True)[1]
        indexes.sort()
        principal_projections = p_p[indexes]
        logger.debug('Projections : {}'.format(principal_projections))
        logger.debug('PcOind(clu) {}'.format(pc_ind[self.clu]))
        # channels projecting onto larger features
        principal_channels = pc_ind[self.clu][principal_projections]
        return principal_channels, principal_projections

    def get_unit_spikes(self, before=20, after=20, only_principal=False):
        s_f = self.sampling_rate
        valid_times = self.time_samples[self.time_samples > before]
        valid_recs = self.recordings[self.time_samples > before]

        if valid_times.size < self.time_samples.size:
            logger.warn('Some frames were out of left bounds and will be discarded')
            logger.warn('will collect only {0} events...'.format(valid_times.size))

        if only_principal:
            chan_list = self.get_principal_channels()
        else:
            chan_list = self.get_unit_chans()

        self.waveform_pars = {'before': before,
                              'after': after,
                              'chan_list': np.array(chan_list)}

        self.all_waveforms = collect_frames_fast(valid_times - before,
                                              before + after,
                                              s_f,
                                              self.get_kwd_path(),
                                              valid_recs,
                                              np.array(chan_list))
        return self.waveforms

    def load_all_waveforms(self):
        folder = self.get_folder()
        f_name = 'unit_{:03d}.npy'.format(self.clu)
        return np.load(os.path.join(folder, 'unit_waveforms', f_name), mmap_mode='r')

    def set_n_waveforms(self, n_waveforms):
        self.n_waveforms = n_waveforms

    def get_waveforms(self, before=20, after=20, only_principal=False, force=False):

        try:
            logger.info('Trying to load waveforms file')
            assert force is False
            self.load_unit_spikes()
        except:
            logger.info('Could not load, wil try to gather wavefomrs')
            self.get_unit_spikes(before=before, after=after,
                                 only_principal=only_principal)
            logger.info('will save the spikes for the nest time around')
            self.save_unit_spikes()
        # all waveforms were loaded into self.all_waveforms.
        # now we want to make a sample fo them in self.waveforms, to show and compute metrics
        self.n_waveforms = min(self.n_waveforms, self.all_waveforms.shape[0])
        waveform_samples = np.random.choice(self.all_waveforms.shape[0], self.n_waveforms,
        replace=False)
        self.waveforms = self.all_waveforms[waveform_samples, :, :]
        return self.waveforms

    def get_avg_wave(self):
        if self.waveforms is None:
            self.get_waveforms()
        return np.mean(self.waveforms, axis=0)

    def get_unit_main_chan(self):
        a_w_f = self.get_avg_wave()
        main_chan = np.argmax(np.ptp(a_w_f, axis=0))
        main_chan_absolute = self.waveform_pars['chan_list'][main_chan]
        return main_chan, main_chan_absolute

    def get_unit_main_chans(self, n_chans=1):
        a_w_f = self.get_avg_wave()
        main_chans = np.argsort(np.ptp(a_w_f, axis=0))[::-1][:n_chans]
        #logger.info('main chans {}'.format(main_chans))
        main_chan_absolute = np.array(self.waveform_pars['chan_list'])[main_chans]
        return main_chans.astype(np.int), main_chan_absolute

    def get_unit_main_wave(self, n_chans=1):
        ch = self.get_unit_main_chans(n_chans=n_chans)[0]
        return self.waveforms[:, :, ch]

    def get_unit_ptp(self):
        wf_main = self.get_unit_main_wave()
        all_ptp = wf_main.ptp(axis=1)
        return np.median(all_ptp), np.std(all_ptp)

    def get_all_unit_widths(self):
        logger.info('Getting width of all spikes from clu {}'.format(self.clu))
        wf_main = self.get_unit_main_wave()
        wf_samples = wf_main.shape[1]
        mid_points = np.min(wf_main, axis=1) + np.ptp(wf_main, axis=1) / 2.
        mid_points_array = np.reshape(np.repeat(mid_points, wf_samples), [-1, wf_samples])
        x, y = np.where(np.diff((wf_main > mid_points_array), 1))
        widths = []
        for i in np.unique(x):
            zero_xings = y[x == i]
            if zero_xings.size > 1:
                widths.append(np.max(np.diff(y[x == i])))
        return np.array(widths)

    def get_unit_widths(self):
        widths = self.get_all_unit_widths()
        return np.median(widths), np.std(widths)


def support_vector_ms(starts, len_samples, all_units,
                   win_size=10, s_f=30000, history_steps=1, step_size=1,
                   no_silent=False):
    """
    :param starts: list or np array of starting points (absolute)
    :param len_samples: length in samples of the 'trial'
    :param all_units: list of threshUnit or Unit objects (as in units.py)
    :param win_size: size of the bin for the spike count
    :param history_steps: step size (ms)
    :param history_bins: number of steps previous to starting points to include
    :param no_silent: exclude units that don't spike (to prevent singular support arrays)
    :return: np array [n_bins, n_units, n_trials] (compatible with other features sup vecs)
    """
    logger.debug('Getting spike vectors')
    win_size_samples = int(win_size * s_f / 1000.)
    step_size_samples = int(step_size * s_f / 1000.) #samples per step

    len_steps = int(len_samples /step_size_samples) # len of the trial in steps
    #len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_samples * step_size)

    history_samples = history_steps * step_size_samples + win_size_samples

    span_ms = len_ms + step_size * history_steps + win_size
    #span_samples = int(span_ms * s_f / 1000.)
    # logger.info('span_ms = {}'.format(span_ms))
    sup_vec = []
    sup_vec_units = []
    # logger.info('{} units'.format(len(all_units)))

    unit_type = all_units[0].type
    # logger.info('Type of units is {}'.format(unit_type))
    # logger.info('unit type is {}'.format(unit_type))

    if unit_type is 'cluster':
        for i, a_unit in enumerate(all_units):
            raster = a_unit.get_raster(starts - history_samples,
                                       span_ms,
                                       span_is_ms=True,
                                       return_ms=True)
            
            # Instead of binning everything, 
            #sparse_raster = bp.col_binned(bp.sparse_raster(raster), bin_size)
            sparse_raster_ms = bp.sparse_raster(raster)
            sparse_raster = np.stack([np.convolve(x, np.ones(win_size), mode='valid') for x in sparse_raster_ms])/win_size
            sparse_raster_stepped = sparse_raster[:, ::step_size]

            if no_silent and not sparse_raster.any():
                logger.warn('Watch out, found lazy unit')
                pass
            else:
                sup_vec.append(sparse_raster_stepped.T)
                sup_vec_units.append(a_unit)
            # logger.info('sparse raster shape = {}'.format(sparse_raster.shape))
            # return sup_vec
            feature_vector = np.stack(sup_vec, axis=0)
            used_units = sup_vec_units

    else:
        raise NotImplementedError('Dont know how to do with unit type {}'.format(unit_type))

    logger.debug('returning feature vector shape {}'.format(feature_vector.shape))
    return feature_vector, used_units
    

def support_vector(starts, len_samples, all_units,
                   bin_size=10, s_f=30000, history_bins=1,
                   no_silent=False):
    """
    :param starts: list or np array of starting points (absolute)
    :param len_samples: length in samples of the 'trial'
    :param all_units: list of threshUnit or Unit objects (as in units.py)
    :param bin_size: size of the bin for the spike count
    :param history_bins: number of bins previous to starting points to include
    :param no_silent: exclude units that don't spike (to prevent singular support arrays)
    :return: np array [n_bins, n_units, n_trials] (compatible with other features sup vecs)
    """
    logger.debug('Getting spike vectors')
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)

    history_samples = history_bins * bin_size_samples

    span_ms = len_ms + bin_size * history_bins
    span_samples = int(span_ms * s_f / 1000.)
    # logger.info('span_ms = {}'.format(span_ms))
    sup_vec = []
    sup_vec_units = []
    # logger.info('{} units'.format(len(all_units)))

    unit_type = all_units[0].type
    # logger.info('Type of units is {}'.format(unit_type))
    # logger.info('unit type is {}'.format(unit_type))
    if unit_type is 'lfp':
        # units are chan, lfp, get the whole set of lfp ch/band events at once
        all_channels = np.array([u.clu for u in all_units], dtype=np.int)
        first_unit = all_units[0]
        kwd_file = first_unit.h5_file
        s_f = first_unit.sampling_rate
        window_samples = first_unit.window_samples
        # logger.info('window samples {}'.format(window_samples))
        # collect_frames will go through all the recs
        all_frames = collect_frames(starts - history_samples - window_samples,
                                    span_samples + window_samples,
                                    s_f,
                                    kwd_file,
                                    all_channels,
                                    recs=None)

        # logger.info('Collecting the spectral features for the {} frames'.format(starts.size))
        # logger.info('{}'.format(first_unit))
        # logger.info('lfp pars {}'.format(first_unit.lfp_pars))
        all_spec_array = collect_all_spectra_arr(all_frames, first_unit.lfp_pars)
        # logger.info('all_spec_array shape {}'.format(all_spec_array.shape))

        # reshape it into a [n_frame, n_chans*n_bands, n_steps] array
        [n_frames, n_chans, n_bands, n_steps] = all_spec_array.shape

        all_spectra_fv = all_spec_array.reshape([n_frames, -1, n_steps])

        # logger.info('all_spectra_fv shape = {}'.format(all_spectra_fv.shape))

        all_spectra_sv = np.stack([bp.col_binned(all_spectra_fv[t, :, :span_ms], bin_size)
                                   for t in range(n_frames)], axis=2)
        # logger.info('all_spectra_sv shape = {}'.format(all_spectra_sv.shape))

        feature_vector = all_spectra_sv / np.max(all_spectra_sv)
        used_units = all_channels


    elif unit_type is 'threshold':
        # units are channel, get the whole set of supra-threshold events at once
        all_channels = np.array([u.clu for u in all_units], dtype=np.int)
        # they all come from the same file
        first_unit = all_units[0]
        kwd_file = first_unit.h5_file
        s_f = first_unit.sampling_rate
        filter_pars = first_unit.filter_pars
        filter_func = first_unit.filter_func

        # set the thresholds
        thresh_factors = np.array([u.thresh_factor for u in all_units])
        # rms as the mean rms across all the recs
        all_rms = np.stack([d['rms'] for d in first_unit.rec_dicts])
        mean_rms = all_rms.mean(axis=0)

        chans_thf = np.array([u.thresh_factor for u in all_units])
        thresholds = mean_rms * chans_thf

        # collect_frames will go through all the recs
        all_frames = collect_frames(starts - history_samples,
                                    span_samples,
                                    s_f,
                                    kwd_file,
                                    all_channels,
                                    recs=None)

        [fr.apply_filter(filter_func, filter_pars) for fr in all_frames]
        all_spk_arr = collect_all_spk_arr(all_frames, thresholds)
        rst_sv = np.stack([bp.col_binned(all_spk_arr[t, :, :].T, bin_size_samples)
                           for t in range(all_spk_arr.shape[0])],
                          axis=2)
        if no_silent:
            good_chans = ~find_silent(rst_sv)
        else:
            good_chans = all_channels

        feature_vector = rst_sv[good_chans, :, :]
        used_units = good_chans

    elif unit_type is 'cluster':
        for i, a_unit in enumerate(all_units):
            raster = a_unit.get_raster(starts - history_samples,
                                       span_ms,
                                       span_is_ms=True,
                                       return_ms=True)
            sparse_raster = bp.col_binned(bp.sparse_raster(raster), bin_size)

            if no_silent and not sparse_raster.any():
                logger.warn('Watch out, found lazy unit')
                pass
            else:
                sup_vec.append(sparse_raster.T)
                sup_vec_units.append(a_unit)
            # logger.info('sparse raster shape = {}'.format(sparse_raster.shape))
            # return sup_vec
            feature_vector = np.stack(sup_vec, axis=0)
            used_units = sup_vec_units

    else:
        raise NotImplementedError('Dont know how to do with unit type {}'.format(unit_type))

    logger.debug('returning feature vector shape {}'.format(feature_vector.shape))
    return feature_vector, used_units


def support_vector_units(starts, len_samples, all_units,
                         bin_size=10, s_f=30000, history_bins=1,
                         no_silent=True):
    """
    :param starts: list or np array of starting points
    :param len_samples: length in samples of the 'trial'
    :param all_units: list of Unit objects (as in units.py)
    :param bin_size: size of the bin for the spike count
    :param history_bins: number of bins previous to starting points to include
    :param no_silent: exclude units that don't spike (to prevent singular support arrays)
    :return: np array [n_bins, n_units, n_trials] (compatible with other features sup vecs)
    """
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)

    history_samples = history_bins * bin_size_samples

    span_ms = len_ms + bin_size * history_bins
    # logger.info('span_ms = {}'.format(span_ms))
    sup_vec = []
    sup_vec_units = []
    # logger.info('{} units'.format(len(all_units)))

    for i, a_unit in enumerate(all_units):
        raster = a_unit.get_raster(starts - history_samples,
                                   span_ms,
                                   span_is_ms=True,
                                   return_ms=True)
        sparse_raster = bp.col_binned(bp.sparse_raster(raster), bin_size)

        if no_silent and not sparse_raster.any():
            logger.warn('Watch out, found lazy unit')
            pass
        else:
            sup_vec.append(sparse_raster.T)
            sup_vec_units.append(a_unit)
    # logger.info('sparse raster shape = {}'.format(sparse_raster.shape))
    # return sup_vec
    return np.stack(sup_vec, axis=0), sup_vec_units


def support_vector_thresh(starts, len_samples, all_units,
                          bin_size=10, s_f=30000, history_bins=1,
                          no_silent=False):
    """
    :param starts: list or np array of starting points (absolute)
    :param len_samples: length in samples of the 'trial'
    :param all_units: list of threshUnit objects (as in units.py)
    :param bin_size: size of the bin for the spike count
    :param history_bins: number of bins previous to starting points to include
    :param no_silent: exclude units that don't spike (to prevent singular support arrays)
    :return: np array [n_bins, n_units, n_trials] (compatible with other features sup vecs)
    """
    logger.debug('Getting spike vectors')
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)

    history_samples = history_bins * bin_size_samples

    span_ms = len_ms + bin_size * history_bins
    span_samples = int(span_ms * s_f / 1000.)
    # logger.info('span_ms = {}'.format(span_ms))
    sup_vec = []
    sup_vec_units = []
    # logger.info('{} units'.format(len(all_units)))

    # units are channel, get the whole set of supra-threshold events at once
    all_channels = np.array([u.clu for u in all_units], dtype=np.int)
    # they all come from the same file
    fist_unit = all_units[0]
    kwd_file = fist_unit.h5_file
    s_f = fist_unit.sampling_rate
    filter_pars = fist_unit.filter_pars
    filter_func = fist_unit.filter_func

    # set the thresholds
    thresh_factors = np.array([u.thresh_factor for u in all_units])
    # rms as the mean rms across all the recs
    all_rms = np.stack([d['rms'] for d in fist_unit.rec_dicts])
    mean_rms = all_rms.mean(axis=0)

    chans_thf = np.array([u.thresh_factor for u in all_units])
    thresholds = mean_rms * chans_thf

    # collect_frames will go through all the recs
    all_frames = collect_frames(starts - history_samples,
                                span_samples,
                                s_f,
                                kwd_file,
                                all_channels,
                                recs=None)

    [fr.apply_filter(filter_func, filter_pars) for fr in all_frames]
    all_spk_arr = collect_all_spk_arr(all_frames, thresholds)
    rst_sv = np.stack([bp.col_binned(all_spk_arr[t, :, :].T, bin_size_samples)
                       for t in range(all_spk_arr.shape[0])],
                      axis=2)
    if no_silent:
        good_chans = ~find_silent(rst_sv)
    else:
        good_chans = all_channels

    return rst_sv[good_chans, :, :], good_chans


def filter_unit_list(in_list, filter_func, *args, **kwargs):
    return [unit for unit in in_list if filter_func(unit, *args, **kwargs)]


def no_silent_filter(a_unit, starts, len_samples, bin_size=10, s_f=30000, history_bins=1):
    """
    :param starts: list or np array of starting points
    :param len_samples: length in samples of the 'trial'
    :param a_unit: one Unit objects (as in units.py)
    :param bin_size: size of the bin for the spike count
    :param history_bins: number of bins previous to starting points to include
    :return: True if the unit has at leas one spike in the raster
    """
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)
    history_samples = history_bins * bin_size_samples
    span_ms = len_ms + bin_size * history_bins

    raster = a_unit.get_raster(starts - history_samples,
                               span_ms,
                               span_is_ms=True,
                               return_ms=True)

    sparse_raster = bp.col_binned(bp.sparse_raster(raster), bin_size)

    return not any(~sparse_raster.any(axis=1))


def no_singularity_filter(a_unit, starts, len_samples, bin_size=10, s_f=30000, history_bins=1):
    """
    :param starts: list or np array of starting points
    :param len_samples: length in samples of the 'trial'
    :param a_unit: one Unit objects (as in units.py)
    :param bin_size: size of the bin for the spike count
    :param history_bins: number of bins previous to starting points to include
    :return: True if the unit has at leas one spike in the raster
    """
    bin_size_samples = int(bin_size * s_f / 1000.)
    len_bin = int(len_samples / bin_size_samples)
    len_ms = int(len_bin * bin_size)
    history_samples = history_bins * bin_size_samples
    span_ms = len_ms + bin_size * history_bins

    raster = a_unit.get_raster(starts - history_samples,
                               span_ms,
                               span_is_ms=True,
                               return_ms=True)

    sparse_raster = bp.col_binned(bp.sparse_raster(raster), bin_size)

    return (sparse_raster.any())


def filter_rms(x, filter_pars):
    return st.rms(sp.apply_butter_bandpass(x, filter_pars))


def collect_all_spk_arr(frames_list, thresholds, min_dist=10):
    logger.info('Will get spikes form {} frams'.format(len(frames_list)))
    sp_stack = np.stack([tp.spikes_array(fr.data, thresholds, min_dist=min_dist)
                         for fr in frames_list],
                        axis=0)
    logger.info('done getting spikes')
    return sp_stack


def collect_all_spectra_arr(all_frames, lfp_pars):
    all_spectra = []
    for i, fr in tqdm(enumerate(all_frames)):
        one_s = sp.array_spectrogram(fr.data, lfp_pars, axis=-1)
        all_spectra.append(one_s)

    all_spectra_array = np.stack(all_spectra, axis=0)
    return all_spectra_array


def find_silent(sup_vec):
    silent_list = np.array([any(~(sup_vec[i, :, :].any(axis=0))) for i in range(sup_vec.shape[0])])
    return silent_list


def collect_frames(starts, span, s_f, kwd_file, chan_list, recs=None):
    # starts is absolute if recs is none
    frames = []

    if recs is None:
        # starts come in abs values
        recs = kf.get_corresponding_rec(kwd_file, starts)
        start_rec_offsets = kf.rec_start_array(kwd_file)
    else:
        assert (starts.size == recs.size)
        rec_list = kf.get_rec_list(kwd_file)
        start_rec_offsets = np.zeros_like(rec_list)

    logger.info('Collecting {} frames...'.format(starts.size))
    for i_start, start in enumerate(starts):
        if i_start % 10 == 0:
            logger.info("Frame {} ...".format(i_start))
        rec = recs[i_start]
        start = start - start_rec_offsets[rec]
        one_frame = st.Chunk(st.H5Data(h5f.get_data_set(kwd_file, rec),
                                       s_f,
                                       dtype=np.float),
                             np.array(chan_list),
                             [start, start + span])
        frames.append(one_frame)
    return frames


def collect_frames_array(starts, span, s_f, kwd_file, recs_list, chan_list):
    recs = np.unique(recs_list)
    logger.info('Collecting {} recs...'.format(recs.size))
    all_frames_array = []
    for i_rec, rec in tqdm(enumerate(recs)):
        starts_from_rec = starts[recs_list == rec]
        logger.info("Rec {0}, {1} events ...".format(rec, starts_from_rec.size))
        stream_obj = st.H5Data(h5f.get_data_set(kwd_file, rec),
                               s_f,
                               chan_list=chan_list,
                               dtype=np.float)
        valid_starts = starts_from_rec[(starts_from_rec > 0)
                                       & (starts_from_rec + span < stream_obj.n_samples)]
        if valid_starts.size < starts_from_rec.size:
            logger.warn('Some frames were out of bounds and will be discarded')
            logger.warn('will collect only {0} events...'.format(valid_starts.size))
        rec_frames = stream_obj.apply_repeated(valid_starts, span, lambda x: x)
        all_frames_array.append(rec_frames)
    logger.info('Done collecting')
    return np.concatenate(all_frames_array, axis=0)


def collect_frames_fast(starts, span, s_f, kwd_file, recs_list, chan_list):
    recs = np.unique(recs_list)
    logger.info('Collecting {} recs...'.format(recs.size))
    all_frames_list = []
    for i_rec, rec in tqdm(enumerate(recs)):
        starts_from_rec = starts[recs_list == rec]
        logger.info("Rec {0}, {1} events ...".format(rec, starts_from_rec.size))
                
        h5_dset = h5f.get_data_set(kwd_file, rec)
        n_samples = h5_dset.shape[0]
                             
        valid_starts = starts_from_rec[(starts_from_rec > 0)
                                       & (starts_from_rec + span < n_samples)]
        if valid_starts.size < starts_from_rec.size:
            logger.warn('Some frames were out of bounds and will be discarded')
            logger.warn('will collect only {0} events...'.format(valid_starts.size))
        
        this_rec_spikes = st.repeated_slice(h5_dset, valid_starts, span, chan_list)
        all_frames_list.append(this_rec_spikes)
    
    logger.info('Done collecting')
    return np.concatenate(all_frames_list, axis=0)

