import numpy as np
import scipy
from . import matrix

def get_ephem_cw(lvlh:np.ndarray, a:float|np.ndarray, dt:float, step:int):
    '''
        args:
            `lvlh`: initial states of the objects, shape (batch_size, n, 6), unit: m.
            `a`: radius of the target's circular orbit, unit: m.
            `dt`: time step, unit: s.
            `step`: number of time steps.
        return:
            `ephem`: states of the objects at each time step, shape (step, batch_size, n, 6).
    '''
    batch_size, n, _ = lvlh.shape
    Phi = matrix.CW_TransMat_batch(0, dt, a) # shape: (batch_size, 6, 6) or (1, 6, 6).
    ephem = np.zeros((step, batch_size, n, 6))
    states = lvlh.copy()
    for i in range(step):
        ephem[i,...] = states[...]
        states = states@Phi.swapaxes(-1, -2)
    return ephem

def get_closest_approach_cw(primal_lvlh:np.ndarray, debris_lvlh:np.ndarray,
                            a:float|np.ndarray, dt=1., step=1200,):
    '''
        args:
            `primal_lvlh`: shape: (batch_size, 6), unit: m.
            `debris_lvlh`: shape: (batch_size, n_debris, 6), unit: m.
            `a`: radius of the target's circular orbit, unit: m.
            `dt`: time step, unit: s.
            `step`: number of time steps.
        returns:
            `closest_dist`, `(primal_ephem, debris_ephem, closest_approach_step)`.
    '''
    primal_lvlh = primal_lvlh.reshape((-1, 1, 6))
    batch_size = primal_lvlh.shape[0]
    debris_lvlh = debris_lvlh.reshape((batch_size, -1, 6))
    primal_ephem = get_ephem_cw(primal_lvlh, a=a, dt=dt, step=step)
    debris_ephem = get_ephem_cw(debris_lvlh, a=a, dt=dt, step=step)
    rel_ephem = debris_ephem - primal_ephem # shape (step, batch_size, n_debris, 6)
    rel_pos = rel_ephem[:, :, :, :3] # shape (step, batch_size, n_debris, 3)
    dist = np.linalg.norm(rel_pos, axis=-1) # shape (step, batch_size, n_debris)
    closest_approach_step = np.argmin(dist, axis=0) # shape (batch_size, n_debris)
    closest_dist = np.min(dist, axis=0) # shape (batch_size, n_debris)
    return closest_dist, (primal_ephem, debris_ephem, closest_approach_step)


def get_ephem_elliptical(lvlh:np.ndarray, 
                         a:np.ndarray, 
                         ecc:np.ndarray, 
                         phi:np.ndarray, 
                         dt:float, 
                         step:int):
    '''
        args:
            `lvlh`: initial states of the objects, shape (batch_size, n, 6), unit: m.
            `a`: radius of the target's circular orbit, unit: m.
            `dt`: time step, unit: s.
            `step`: number of time steps.
        return:
            `ephem`: states of the objects at each time step, shape (step, batch_size, n, 6).
    '''
    batch_size, n, _ = lvlh.shape
    State_mats = matrix.elliptical_StateMat_batch(a, ecc, phi)
    Phi = scipy.linalg.expm(State_mats*dt) # shape: (batch_size, 6, 6) or (1, 6, 6).
    ephem = np.zeros((step, batch_size, n, 6))
    states = lvlh.copy()
    for i in range(step):
        ephem[i,...] = states[...]
        states = states@Phi.swapaxes(-1, -2)
    return ephem

def get_closest_approach_elliptical(primal_lvlh:np.ndarray, 
                                    debris_lvlh:np.ndarray,
                                    a:np.ndarray, 
                                    ecc:np.ndarray, 
                                    phi:np.ndarray, 
                                    dt=1., 
                                    step=1200,):
    '''
        args:
            `primal_lvlh`: shape: (batch_size, 6), unit: m.
            `debris_lvlh`: shape: (batch_size, n_debris, 6), unit: m.
            `a`: radius of the target's circular orbit, unit: m.
            `dt`: time step, unit: s.
            `step`: number of time steps.
        returns:
            `closest_dist`, `(primal_ephem, debris_ephem, closest_approach_step)`.
    '''
    primal_lvlh = primal_lvlh.reshape((-1, 1, 6))
    batch_size = primal_lvlh.shape[0]
    debris_lvlh = debris_lvlh.reshape((batch_size, -1, 6))
    primal_ephem = get_ephem_elliptical(primal_lvlh, a=a, ecc=ecc, phi=phi, dt=dt, step=step)
    debris_ephem = get_ephem_elliptical(debris_lvlh, a=a, ecc=ecc, phi=phi, dt=dt, step=step)
    rel_ephem = debris_ephem - primal_ephem # shape (step, batch_size, n_debris, 6)
    rel_pos = rel_ephem[:, :, :, :3] # shape (step, batch_size, n_debris, 3)
    dist = np.linalg.norm(rel_pos, axis=-1) # shape (step, batch_size, n_debris)
    closest_approach_step = np.argmin(dist, axis=0) # shape (batch_size, n_debris)
    closest_dist = np.min(dist, axis=0) # shape (batch_size, n_debris)
    return closest_dist, (primal_ephem, debris_ephem, closest_approach_step)