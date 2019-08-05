import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_line_ci(x, x_var, color='r', label='arr', ls='-', lw=1, ax=None, 
    alpha=0.04, t=np.empty(0)):
    
    if ax is None:
        f, ax = plt.subplots()
    
    if t.size==0:
        t = np.arange(x.size)
    ax.plot(t, x, lw=lw, ls=ls, color=color, alpha=1, label=label)
    ax.fill_between(t, x + x_var, x - x_var, color=color, 
    alpha=0.07)
    return ax


# reads kai 
# /mnt/cube/kai/results/spectrogram prediction model/mel/Resub_2018/lstm/ffnn_lstm.p' 
# type of file
# adds two indices for easily grouping by trial and averagin by time
def load_kai_pd(pickle_path):
    kai_pd = pd.read_pickle(pickle_path)
    kai_pd['t'] = kai_pd['time']*30000
    kai_pd['t'] = kai_pd['t'].apply(np.int)
    kai_pd['idx'] = kai_pd.index
    kai_pd.set_index(['t', 'idx'],inplace=True)
    kai_pd.sort_index(inplace=True)
    kai_pd.sort_values(['t', 'idx'])
    return kai_pd

def pd_to_arrays(k_pd, measure='correlation', model='LSTM'):
    # return an array with mean, std values
    # t in ms
    sel_filter = k_pd['model']==model
    
    pd_grouped = k_pd.loc[sel_filter, :].groupby('t')
    
    t = pd_grouped['time'].mean().values*1000
    avg = pd_grouped[measure].mean()
    err = pd_grouped[measure].std()
    
    return t.astype(np.int), avg, err