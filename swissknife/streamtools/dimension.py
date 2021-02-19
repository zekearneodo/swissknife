import numpy as np
import umap

from umap.parametric_umap import ParametricUMAP

import logging
from swissknife.streamtools.core import data as dt

logger = logging.getLogger('swissknife.streamtools.dimension')


def rolling_umap(x:np.ndarray, win_size:int, reducer=None, parametric=False, **umap_kwargs) -> tuple:
    # roll along last dimension with step 1, window size window
    rolled_x = dt.rolling_window(x, win_size)
    #rolled_x with this transposition will be [x.shape[0], x.shape[2]//win_size, win_size] = [n_feat, n_samples, win_size]
    n_feat, n_samp, win_size = rolled_x.shape
    # want y to be [x.shape[1], x_shape[2] * x.shape[0]]
    rolled_x_featflat = rolled_x.transpose(0, 2, 1).reshape(-1, n_samp)
    
    if reducer is None:
        if parametric:
            reducer = ParametricUMAP(**umap_kwargs)
        else:
            reducer = umap.UMAP(**umap_kwargs)
        embedding = reducer.fit_transform(rolled_x_featflat.T)
    else:
        embedding = reducer.transform(rolled_x_featflat.T)
    
    return reducer, embedding, rolled_x_featflat
