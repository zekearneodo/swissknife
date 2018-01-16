#! /bin/env python
#
# Michael Gibson 23 April 2015

import sys, struct
from qstring import read_qstring


def read_header(fid):
    """Reads the Intan File Format header from the given file, decide whether
    should read rhs or rhd based on the magic_number"""

    # Check 'magic number' at beginning of file to make sure this is an Intan
    # Technologies RHD2000 data file.
    magic_number, = struct.unpack('<I', fid.read(4))
    fid.seek(0, 0)
    print('magic number read {}'.format(magic_number))
    if magic_number == int('c6912702', 16):
        return read_header_rhd(fid)
    elif magic_number == int('0xD69127AC', 16):
        return read_header_rhs(fid)
    else:
        raise Exception('Unrecognized file type.')


def read_header_rhs(fid):
    """Reads the Intan File Format header from the given file."""

    # Check 'magic number' at beginning of file to make sure this is an Intan
    # Technologies RHD2000 data file.
    magic_number, = struct.unpack('<I', fid.read(4))

    if magic_number != int('0xD69127AC', 16): raise Exception('Unrecognized file type.')

    header = {}
    # Read version number.
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4))
    header['version'] = version

    print('')
    print('Reading Intan Technologies RHS2000 Data File, Version {}.{}'.format(version['major'], version['minor']))
    print('')

    freq = {}

    # Read information of sampling rate and amplifier frequency settings.
    header['sample_rate'], = struct.unpack('<f', fid.read(4))
    (freq['dsp_enabled'],
     freq['actual_dsp_cutoff_frequency'],
     freq['actual_lower_bandwidth'],
     freq['actual_lower_settle_bandwidth'],
     freq['actual_upper_bandwidth'],
     freq['desired_dsp_cutoff_frequency'],
     freq['desired_lower_bandwidth'],
     freq['desired_lower_settle_bandwidth'],
     freq['desired_upper_bandwidth']) = struct.unpack('<hffffffff', fid.read(34))


    # This tells us if a software 50/60 Hz notch filter was enabled during
    # the data acquisition.
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    freq['notch_filter_frequency'] = header['notch_filter_frequency']

    (freq['desired_impedance_test_frequency'], freq['actual_impedance_test_frequency']) = struct.unpack('<ff', fid.read(8))
    (freq['amp_settle_mode'], freq['charge_recovery_mode']) = struct.unpack('<hh', fid.read(4))

    (header['stim_step_size'],
     header['recovery_current_limit'],
     header['recovery_target_voltage']) = struct.unpack('fff', fid.read(12))

    note1 = read_qstring(fid)
    note2 = read_qstring(fid)
    note3 = read_qstring(fid)
    header['notes'] = { 'note1' : note1, 'note2' : note2, 'note3' : note3}

    (header['dc_amplifier_data_saved'],
     header['eval_board_mode']) = struct.unpack('<hh', fid.read(4))

    header['ref_channel_name'] = read_qstring(fid)


    # If data file is from GUI v1.1 or later, see if temperature sensor data was saved.
    header['num_temp_sensor_channels'] = 0
    # if (version['major'] == 1 and version['minor'] >= 1) or (version['major'] > 1) :
    #     header['num_temp_sensor_channels'], = struct.unpack('<h', fid.read(2))


    # Place frequency-related information in data structure. (Note: much of this structure is set above)
    freq['amplifier_sample_rate'] = header['sample_rate']
    freq['board_adc_sample_rate'] = header['sample_rate']
    freq['board_dac_sample_rate'] = header['sample_rate']
    freq['board_dig_in_sample_rate'] = header['sample_rate']

    header['frequency_parameters'] = freq

    # Create structure arrays for each type of data channel.
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['board_adc_channels'] = []
    header['board_dac_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []

    # Read signal summary from data file header.

    number_of_signal_groups, = struct.unpack('<h', fid.read(2))
    #print('n signal groups {}'.format(number_of_signal_groups))

    for signal_group in range(0, number_of_signal_groups):
        signal_group_name = read_qstring(fid)
        signal_group_prefix = read_qstring(fid)
        (signal_group_enabled, signal_group_num_channels, signal_group_num_amp_channels) = struct.unpack('<hhh', fid.read(6))

        if (signal_group_num_channels > 0) and (signal_group_enabled > 0):
            for signal_channel in range(0, signal_group_num_channels):
                new_channel = {'port_name' : signal_group_name, 'port_prefix' : signal_group_prefix, 'port_number' : signal_group}
                new_channel['native_channel_name'] = read_qstring(fid)
                new_channel['custom_channel_name'] = read_qstring(fid)
                (new_channel['native_order'], new_channel['custom_order'],
                 signal_type, channel_enabled, new_channel['chip_channel'],
                 new_channel['command_stream'], new_channel['board_stream']) = struct.unpack('<hhhhhhh', fid.read(14))
                new_trigger_channel = {}
                (new_trigger_channel['voltage_trigger_mode'],
                 new_trigger_channel['voltage_threshold'],
                 new_trigger_channel['digital_trigger_channel'],
                 new_trigger_channel['digital_edge_polarity']) = struct.unpack('<hhhh', fid.read(8))
                (new_channel['electrode_impedance_magnitude'],
                 new_channel['electrode_impedance_phase']) = struct.unpack('<ff', fid.read(8))

                if channel_enabled:
                    if signal_type == 0:
                        header['amplifier_channels'].append(new_channel)
                        header['spike_triggers'].append(new_trigger_channel)
                    elif signal_type == 1:
                        raise Exception('Wrong signal type for the rhs format')
                        #header['aux_input_channels'].append(new_channel)
                    elif signal_type == 2:
                        raise Exception('Wrong signal type for the rhs format')
                        #header['supply_voltage_channels'].append(new_channel)
                    elif signal_type == 3:
                        header['board_adc_channels'].append(new_channel)
                    elif signal_type == 4:
                        header['board_dac_channels'].append(new_channel)
                    elif signal_type == 5:
                        header['board_dig_in_channels'].append(new_channel)
                    elif signal_type == 6:
                        header['board_dig_out_channels'].append(new_channel)
                    else:
                        raise Exception('Unknown channel type.')

    # Summarize contents of data file.
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dac_channels'] = len(header['board_dac_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(header['board_dig_out_channels'])

    return header

if __name__ == '__main__':
    h=read_header(open(sys.argv[1], 'rb'))
    print(h)
