from __future__ import division
from __future__ import print_function

import numpy as np


def cusp_points(f_p):
    """
    Computes two cusp line points for a given value of beta (b)
    :param f_p: dictionary with keys  'b', 'a_plus', 'a_minus', 'x_plus', 'x_minus'
                'b' needs a value larger than -1/3
    :return: dictionary with the fixed points and values of the parameters at both bifurcation lines.
    """
    assert (f_p['b'] >= -1. / 3.)
    f_p['x_plus'] = (1. + np.sqrt(1. + 3. * f_p['b'])) / 3.
    f_p['x_minus'] = (1. - np.sqrt(1. + 3. * f_p['b'])) / 3.
    for bound in ['plus', 'minus']:
        f_p['a_' + bound] = f_p['x_' + bound] ** 3 - (f_p['x_' + bound]) ** 2 - f_p['b'] * f_p['x_' + bound]
    return f_p


def cusp_lines(b_max=0.1, b_min=-1 / 3, step=1E-4):
    b_values = np.linspace(b_min, b_max, (b_max - b_min) / step)
    lines = np.zeros([b_values.size, 3])
    for i, b in enumerate(b_values):
        fp = cusp_points({'b': b})
        lines[i, :] = np.array([b, fp['a_minus'], fp['a_plus']])
    return lines


def cusp_grid(b_max=0.07, b_min=-1 / 3, a_step=1E-3, b_steps=100, b_log_step_exp=5, f_max=None):
    # get the cusp borders
    border = cusp_lines(b_max=b_max, b_min=b_min, step=a_step)
    # keep only the values that are left of the Hopf line (a=0 line)
    # also, keep only the lower branch of the line
    cusp_line = border[border[:, 2] < 0, :][:, (0, 2)]
    b_sweep = np.logspace(0, b_log_step_exp, b_steps) * b_max * 2 / (10 ** (b_log_step_exp))
    b_strip_size = cusp_line.shape[0]
    ab_grid = np.zeros([b_strip_size * b_steps, 2])
    print(b_strip_size)
    for i, a in enumerate(cusp_line[:, 1]):
        ab_grid[i * b_steps:(i + 1) * b_steps, 0] = a * np.ones(b_steps)
        ab_grid[i * b_steps:(i + 1) * b_steps, 1] = cusp_line[i, 0] - b_sweep
    
    return border, ab_grid