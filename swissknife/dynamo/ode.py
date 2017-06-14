from __future__ import division
import numpy as np

# ode functions:
def rk4(f, x, h, *args):
    k_1 = f(x, *args) * h
    k_2 = f(x + 0.5 * k_1, *args) * h
    k_3 = f(x + 0.5 * k_2, *args) * h
    k_4 = f(x + k_3, *args) * h
    return x + (k_1 + 2. * (k_2 + k_3) + k_4) / 6.


# vector fields:
def harmonic(x, w, w_0):
    return np.array([x[1], - w * w * x[0] / (w_0 * w_0)])


def takens(x, a, b, g):
    gg = g * g
    return np.array([
        x[1],
        a * gg + b * gg * x[0] - gg * x[0] * x[0] * x[0] - g * x[0] * x[0] * x[1] + gg * x[0] * x[0] - g * x[0] * x[1]
    ])


def takens_fast(x, a, b, g):
    gg = g * g
    return np.array([
        x[1],
        gg * (a + x[0] * (b + x[0] * (1. - x[0]))) - g * x[0] * x[1] * (x[0] + 1.)
    ])


def takens_dict(x, pars):
    g = pars['gamma']
    gg = g * g

    return np.array([
        x[1],
        gg * (pars['alpha'] + x[0] * (pars['beta'] + x[0] * (1. - x[0]))) - g * x[0] * x[1] * (x[0] + 1.)
    ])


def takens_finch(v, pars):
    g = pars['gamma']
    gg = g * g
    [x, y, i_1, i_2, i_3] = v

    return np.array([
        y,
        pars['alpha_1'] * gg + pars['beta_1'] * gg * x - gg * x * x * x - g * x * x * y + g * x * x - g * x * y,
        i_2,
        -pars['Lg_inv'] * pars['Ch_inv'] * i_1 - pars['Rh'] * (pars['Lb_inv'] + pars['Lg_inv']) * i_2 + (
        pars['Lg_inv'] * pars['Ch_inv'] - pars['Rb'] * pars['Rh'] * pars['Lb_inv'] * pars['Lg_inv']) * i_3
        + pars['Lg_inv'] * pars['dV_ext'] + pars['Rh'] * pars['Lg_inv'] * pars['Lb_inv'] * pars['V_ext'],
        -pars['Lb_inv'] / pars['Lg_inv'] * i_2 - pars['Rb'] * pars['Lb_inv'] * i_3 + pars['Lb_inv'].pars['V_ext']
    ])


class System(object):
    """
    A class to represent a dynamical system
    """

    def __init__(self, vector_field, initial_cond,
                 t_0=0, field_pars=None, dt=1e-5, ode_func=rk4, sampling_rate=None):

        self.vector_field = vector_field
        self.field_pars = field_pars
        self.x = initial_cond
        self.t = t_0

        self.ode_func = ode_func
        self.dt = dt
        self.sampling_rate = np.int(0.1 / dt) if sampling_rate is None else sampling_rate
        self.steps_per_sample = np.int(1 / (self.sampling_rate * self.dt))

    def __iter__(self):
        return (self)

    def update_pars(self, new_pars):
        self.field_pars = new_pars

    def next(self):
        self.x = self.ode_func(self.vector_field, self.x, self.dt, self.field_pars)
        self.t += self.dt
        return self.x

    def integrate(self, t_f, t_0=None, x_0=None):
        # integrate over a period of time
        self.t = self.t if t_0 is None else t_0
        self.x = self.x if x_0 is None else x_0

        n_steps = np.int(np.round((t_f - self.t) / self.dt))
        #print(n_steps)
        n_samples = np.int(np.floor(n_steps / self.steps_per_sample))

        x = np.zeros([n_samples, self.x.shape[0]])
        x[:] = np.nan
        for i in range(n_steps):
            self.next()
            if not i % self.steps_per_sample:
                x[i / self.steps_per_sample, :] = self.x
        return x


