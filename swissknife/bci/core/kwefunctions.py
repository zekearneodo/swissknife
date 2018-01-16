import pandas as pd
import numpy as np


def get_messages(h5file, rec_id=None, node=None):
    rec_table = h5file['event_types']['Messages']['events']['recording'][:]
    t_table = h5file['event_types']['Messages']['events']['time_samples'][:]
    text_table = h5file['event_types']['Messages']['events']['user_data']['Text'][:]
    node_table = h5file['event_types']['Messages']['events']['user_data']['nodeID'][:]

    decoder = np.vectorize(lambda x: x.decode('UTF-8'))

    return pd.DataFrame.from_items([('t', t_table),
                                    ('text', decoder(text_table)),
                                    ('rec', rec_table),
                                    ('node', node_table)])
