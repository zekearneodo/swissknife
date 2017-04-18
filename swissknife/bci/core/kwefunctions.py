import pandas as pd


def get_messages(h5file, rec_id=None, node=None):
    rec_table = h5file['event_types']['Messages']['events']['recording'][:]
    t_table = h5file['event_types']['Messages']['events']['time_samples'][:]
    text_table = h5file['event_types']['Messages']['events']['user_data']['Text'][:]
    node_table = h5file['event_types']['Messages']['events']['user_data']['nodeID'][:]

    return pd.DataFrame.from_items([('t', t_table),
                                    ('text', text_table),
                                    ('rec', rec_table),
                                    ('node', node_table)])
