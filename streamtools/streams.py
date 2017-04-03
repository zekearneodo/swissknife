# Objects and helper functions to do stuff with sound
# The logic of using these is to be able to page through long structure and apply the same whether they are wavs, h5, bin.
from __future__ import division
import numpy as np
import wave
import struct
import copy

import logging

from h5tools import tables as h5t
from matplotlib import pyplot as plt

# TODO:  Make DatSound a subclass of WavData
#        Save chunk as wav file (using wave)

logger = logging.getLogger("streamtools.streams")


def rms(x):
    return np.std(x, axis=0)


def car(x, chans=None):
    ch = np.arange(x.shape[1]) if chans is None else chans
    return x[:, ch] - x[:, ch].mean(axis=1, keepdims=True)

def plot_array(array, ax=None):
    # Plot all channels into one plot
    # Offset them
    amps = np.ptp(array, axis=0)
    plot_data = np.zeros_like(array)
    for i in np.arange(array.shape[1]):
        plot_data[:, i] = array[:, i] / amps[i] + i
    if ax is None:
        waveforms_fig = plt.figure()
        ax = waveforms_fig.add_axes([0, 0, 1, 1])
    else:
        waveforms_fig = ax.figure

    lines = ax.plot(plot_data);
    return waveforms_fig, ax, lines


def sum_frames(frames_list):
    all_frames_array = np.array([f.data for f in frames_list])
    all_avg = all_frames_array.mean(axis=0)
    return all_avg

class WavData2:
    # same as wavdata, but streams are read in columns into an N_samp X N_ch array (one channel = one column)
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = wave.open(file_path, 'rb')
        self.n_samples = self.raw.getparams()[3]
        self.s_f = self.raw.getparams()[2]
        self.frame_size = self.raw.getparams()[1]
        self.n_chans = self.raw.getparams()[0]

    def get_chunk(self, start, end, chan_list=[0]):
        frame_to_type = {'2': 'h', '4': 'i'}
        # returns a vector with a channel
        assert (start >= 0)
        assert (end <= self.n_samples)
        assert (end > start)

        n_chans = self.n_chans
        self.raw.setpos(start)
        chunk_bit = self.raw.readframes(end - start)
        data_type = frame_to_type[str(self.frame_size)]

        # TODO: this is dirty, change by one read, unpack and reshape
        data = np.zeros((end - start, len(chan_list)), dtype=np.dtype(data_type))
        data_unpacked = struct.unpack('<' + str((end - start) * n_chans) + data_type, chunk_bit)
        for i, channel in enumerate(chan_list):
            data[:, i] = data_unpacked[channel::n_chans]

        data = np.array(data, dtype=np.float32)
        return data

    # applies a scalar function to many starting points
    def apply_repeated(self, starts, window, scalar_func, *args, **kwargs):
        # starts, window in sample units

        y = np.empty_like(starts)
        for i_s, start in enumerate(starts):
            a_chunk = Chunk(self, segment=[start, start + window])
            y[i_s] = scalar_func(a_chunk.data, *args, **kwargs)

        return y

    def get_rms(self, t_ms):
        pass


class WavData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw = wave.open(file_path, 'rb')
        self.n_samples = self.raw.getparams()[3]
        self.s_f = self.raw.getparams()[2]
        self.frame_size = self.raw.getparams()[1]
        self.n_chans = self.raw.getparams()[0]

    def get_chunk(self, start, end, chan_list=[0]):
        frame_to_type = {'2': 'h', '4': 'i'}
        # returns a vector with a channel
        assert (start >= 0)
        assert (end <= self.n_samples)
        assert (end > start)

        n_chans = self.n_chans
        self.raw.setpos(start)
        chunk_bit = self.raw.readframes(end - start)
        data_type = frame_to_type[str(self.frame_size)]

        # TODO: this is dirty, change by one read, unpack and reshape
        data = np.zeros((len(chan_list), end - start), dtype=np.dtype(data_type))
        data_unpacked = struct.unpack('<' + str((end - start) * n_chans) + data_type, chunk_bit)
        for i, channel in enumerate(chan_list):
            data[i, :] = data_unpacked[channel::n_chans]

        data = np.array(data, dtype=np.float32)
        return data

    # applies a scalar function to many starting points
    def apply_repeated(self, starts, window, scalar_func, *args, **kwargs):
        # starts, window in sample units

        y = np.empty_like(starts)
        for i_s, start in enumerate(starts):
            a_chunk = Chunk(self, segment=[start, start + window])
            y[i_s] = scalar_func(a_chunk.data, *args, **kwargs)

        return y

    def get_rms(self, t_ms):
        pass


class DatSound:
    def __init__(self, data, s_f, data_type=None):

        self.data_type = data.dtype if data_type is None else data_type
        self.stream = np.array(data, self.data_type)
        self.s_f = s_f
        self.n_samples = data.shape[0]
        self.n_chans = data.size / self.n_samples

    def get_chunk(self, start, end, chan_list=[0]):
        assert (start >= 0)
        assert (end <= self.n_samples)
        assert (end > start)

        if self.stream.shape[0] == self.stream.size:
            data = self.stream[start:end]
        else:
            data = self.stream[start:end, chan_list]

        return np.array(data, dtype=np.dtype(self.data_type)).reshape([data.shape[0], len(chan_list)])

    # applies a scalar function to many starting points
    def apply_repeated(self, starts, window, scalar_func, *args, **kwargs):
        # starts, window in sample units
        a_chunk = Chunk(self, segment=[starts[0], starts[0] + window])
        y_0 = scalar_func(a_chunk.data, *args, **kwargs)
        d_type = type(y_0)

        y = np.empty_like(starts, dtype=np.dtype(d_type))
        for i_s, start in enumerate(starts):
            a_chunk = Chunk(self, segment=[start, start + window])
            y[i_s] = scalar_func(a_chunk.data, *args, **kwargs)

        return y


class DatSoundCompat:
    def __init__(self, data, s_f, data_type=None):
        """
        :param data: a N_ch x N_samples numpy array or N_samples vector
        :param s_f: sampling rate
        :param data_type: data type of the array
        :return:
        """

        self.data_type = data.dtype if data_type is None else data_type
        self.s_f = s_f
        self.n_samples = data.shape[data.ndim - 1]
        self.n_chans = data.size / self.n_samples
        self.stream = np.array(data, self.data_type).reshape(self.n_samples, self.n_chans)

    def get_chunk(self, start, end, chan_list=[0]):
        assert (start >= 0)
        assert (end <= self.n_samples)
        assert (end > start)
        data = self.stream[start:end, chan_list].reshape(end - start, len(chan_list))
        return data

    # applies a scalar function to many starting points
    def apply_repeated(self, starts, window, scalar_func, *args, **kwargs):
        # starts, window in sample units
        a_chunk = Chunk(self, segment=[starts[0], starts[0] + window])
        y_0 = scalar_func(a_chunk.data, *args, **kwargs)
        d_type = type(y_0)

        y = np.empty_like(starts, dtype=np.dtype(d_type))
        for i_s, start in enumerate(starts):
            a_chunk = Chunk(self, segment=[start, start + window])
            y[i_s] = scalar_func(a_chunk.data, *args, **kwargs)
        return y


class H5Data:
    def __init__(self, h5_table, s_f, dtype=None, chan_list=None):
        self.data_type = np.dtype(h5_table.dtype) if dtype is None else dtype
        self.s_f = s_f
        if chan_list is None:
            self.n_chans = h5_table.shape[h5_table.ndim - 1]
            self.chan_list = np.arange(self.n_chans)
        else:
            self.n_chans = np.size(chan_list)
            self.chan_list = chan_list
        self.n_samples = int(h5_table.size / h5_table.shape[h5_table.ndim - 1])
        self.data = h5_table
        self.rms = None

    # applies a scalar function to many starting points
    def apply_repeated(self, starts, window, func, *args, **kwargs):
        # starts, window in sample units
        results = [func(Chunk(self,
                              chan_list=self.chan_list,
                              segment=[start, start + window]).data,
                        *args,
                        **kwargs) for start in starts]
        #return results
        return np.stack(results, axis=0)

    def collect_frames(self, starts, span, chan_list=None):
        frames = []
        bad_frames = []
        logger.info('Collecting {} frames...'.format(starts.size))
        for i_start, start in enumerate(starts):
            if i_start % 10 == 0:
                logger.info("Frame {} ...".format(i_start))
            try:
                one_frame = self.get_chunk(start, start + span, chan_list=chan_list)
                frames.append(one_frame)
            except AssertionError:
                logger.warning('Frame out of bounds [{0}:{1}]'.format(start, start+span))
                bad_frames.append(i_start)
        return frames, bad_frames

    def get_rms(self, window_size_samples=50000, n_windows=5000, rms_func=rms, rms_args=(), rms_kwargs={}):
        logger.debug('Computing rms over {0} windows for {1} channels'.format(n_windows, self.n_chans))
        window_size_samples = min(window_size_samples, self.n_samples)
        all_starts = np.random.randint(self.n_samples - window_size_samples, size=n_windows)
        self.rms = self.apply_repeated(all_starts, window_size_samples, rms_func, *rms_args, **rms_kwargs).mean(axis=0)
        return self.rms

    def get_chunk(self, start, end, chan_list=None):
        if chan_list is None:
            chan_list = self.chan_list
        assert (start >= 0)
        assert (end <= self.n_samples)
        assert (end > start)
        chunk_data = np.array(h5t.load_table_slice(self.data,
                                                   np.arange(start,
                                                             end,
                                                             dtype=int),
                                                   chan_list),
                              dtype=self.data_type)
        return chunk_data



# class of methods for chunks of a signal
# A chunk is a part of a signal and it is referenced to that signal.
class Chunk:
    def __init__(self, sound, chan_list=np.array([0]), segment=[0, None]):
        """
        :param sound: Sound where it comes from. Sound has to have methods that return
        n_samples (int, total number of samples)
        s_f (int, sampling frequency in KHZ)
        chunk: a slice of a set of samples across a list of channels.
        :type sound: object
        :param chan_list: list of channels to extract
        :type chan_list: int list
        :param segment: begin and end of segment to extract (in samples)
        :type segment: list of starting point, end point (in samples)
        :return:
        :rtype:
        """

        self.sound = sound
        self.start = segment[0]
        self.end = segment[1] if segment[1] is not None else sound.n_samples
        self.samples = self.end - self.start
        self.chan_list = chan_list

        # Read the data
        self.data = sound.get_chunk(self.start, self.end, chan_list=chan_list)

    def __add__(self, other):
        assert (self.data.shape == other.data.shape)
        new_chunk = copy.copy(self)
        new_chunk.data = np.mean(np.array([self.data, other.data]), axis=0)
        return new_chunk

    def __sub__(self, other):
        assert (self.data.shape == other.data.shape)
        new_chunk = copy.copy(self)
        new_chunk.data = self.data - other.data
        return new_chunk

    def apply_filter(self, filter_func, *args, **kwargs):
        # Apply some filter function to the chunk of data
        self.data = filter_func(self.data, *args, **kwargs)

    def plot(self, ax=None):
        # Plot all channels into one plot
        # Offset them
        amps = np.ptp(self.data, axis=0)
        plot_data = np.zeros_like(self.data)
        for i in np.arange(self.chan_list.size):
            plot_data[:, i] = self.data[:, i] / amps[i] + i
        if ax is None:
            waveforms_fig = plt.figure()
            ax = waveforms_fig.add_axes([0, 0, 1, 1])
            ax.plot(plot_data)
        return waveforms_fig, ax

    def export_wav(self, out_file_path):
        pass

    def get_f0(self):
        pass


class Frame:
    def __init__(self, data):
        self.data = data

    def __add__(self, other):
        assert (self.data.shape == other.data.shape)
        new_data = np.mean(np.array([self.data, other.data]), axis=0)
        return Frame(new_data)

    def __sub__(self, other):
        assert (self.data.shape == other.data.shape)

        new_data = self.data - other.data
        return Frame(new_data)

    def apply_filter(self, filter_func, *args, **kwargs):
        # Apply some filter function to the chunk of data
        self.data = filter_func(self.data, *args, **kwargs)

    def plot(self, ax=None):
        # Plot all channels into one plot
        # Offset them
        amps = np.ptp(self.data, axis=0)
        plot_data = np.zeros_like(self.data)
        for i in np.arange(self.chan_list.size):
            plot_data[:, i] = self.data[:, i] / amps[i] + i
        if ax is None:
            waveforms_fig = plt.figure()
            ax = waveforms_fig.add_axes([0, 0, 1, 1])
            ax.plot(plot_data)
        return waveforms_fig, ax


def list_apply_filter(chunk_list, filter_func, *filter_args, **filter_kwargs):
    return map(lambda x:
               x.apply_filter(filter_func, *filter_args, **filter_kwargs),
               chunk_list)


def get_rms_threshold(full_sound, window_size_samples, rms_threshold_factor):
    """
    :param full_sound: A whole object that can give Chunks
    :param window_size_samples: size of the running window (in samples)
    :param rms_threshold_factor: standard deviations to define the threshold
    :return:
    """
    all_starts = np.arange(0, full_sound.n_samples - window_size_samples, window_size_samples)
    all_rms = full_sound.apply_repeated(all_starts, window_size_samples, rms)
    thresh_rms = np.mean(all_rms) + rms_threshold_factor * np.std(all_rms)
    return thresh_rms



# wrappers for streams
def array_wrap(stream_function):
    """
    Decorator to shape vectors as 1-d arrays
    :param stream_function: a function that receives an h5file as first argument
    :return: decorated function that takes open('r' mode) or path as first argument
    """

    def array_checker(h5_file, *args, **kwargs):
        if type(h5_file) is not h5py._hl.files.File:
            h5_file = h5py.File(h5_file, 'r')
        logging.debug('H5 file: {}'.format(h5_file))
        return_value = h5_function(h5_file, *args, **kwargs)
        # h5_file.close()
        return return_value

    return array_checker

