import astropy.units as U
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import numpy as np

from env.dynamic.orbit.J2 import cowell_j2

def get_ephem_orbit(eci:np.ndarray, dt:float, step:int):
    '''
        args:
            `eci`: shape: (..., 6), unit: km.
            `dt`: float, unit: second.
            `step`: int.
        returns:
            `ephem`: shape: (step, ..., 6), unit: km.
    '''
    shape = eci.shape
    eci = eci.reshape((-1, 6)).astype(np.float64)
    batch_size = eci.shape[0]
    ephem = np.zeros((step, batch_size, 6))
    for i in range(batch_size):
        orbit = Orbit.from_vectors(Earth, eci[i, :3]*U.km, eci[i, 3:]*U.km/U.s)
        for j in range(step):
            r, v = orbit.rv()
            r, v = r.value, v.value
            state = np.hstack((r, v))
            ephem[j, i, :] = state
            orbit = orbit.propagate(dt*U.s)
    ephem = ephem.reshape((step, *shape))
    return ephem

def get_closest_approach_orbit(primal_eci:np.ndarray, debris_eci:np.ndarray,
                               dt=1., step=1200):
    '''
        args:
            `primal_eci`: shape: (batch_size, 6), unit: km.
            `debris_eci`: shape: (batch_size, n_debris, 6), unit: km.
        returns:
            `closest_dist`, `(primal_ephem, debris_ephem, closest_approach_step)`.
    '''
    primal_eci = primal_eci.reshape((-1, 1, 6))
    batch_size = primal_eci.shape[0]
    debris_eci = debris_eci.reshape((batch_size, -1, 6))
    primal_ephem = get_ephem_orbit(primal_eci, dt=dt, step=step)
    debris_ephem = get_ephem_orbit(debris_eci, dt=dt, step=step)
    rel_ephem = debris_ephem - primal_ephem # shape (step, batch_size, n_debris, 6)
    rel_pos = rel_ephem[:, :, :, :3] # shape (step, batch_size, n_debris, 3)
    dist = np.linalg.norm(rel_pos, axis=-1) # shape (step, batch_size, n_debris)
    closest_approach_step = np.argmin(dist, axis=0) # shape (batch_size, n_debris)
    closest_dist = np.min(dist, axis=0) # shape (batch_size, n_debris)
    return closest_dist, (primal_ephem, debris_ephem, closest_approach_step)

def get_ephem_J2(eci:np.ndarray, dt:float, step:int):
    '''
        args:
            `eci`: shape: (..., 6), unit: km.
            `dt`: float, unit: second.
            `step`: int.
        returns:
            `ephem`: shape: (step, ..., 6), unit: km.
    '''
    shape = eci.shape
    eci = eci.reshape((-1, 6)).astype(np.float64)
    batch_size = eci.shape[0]
    ephem = np.zeros((step, batch_size, 6))
    for i in range(batch_size):
        orbit = Orbit.from_vectors(Earth, eci[i, :3]*U.km, eci[i, 3:]*U.km/U.s)
        for j in range(step):
            r, v = orbit.rv()
            r, v = r.value, v.value
            state = np.hstack((r, v))
            ephem[j, i, :] = state
            orbit = orbit.propagate(dt*U.s, method=cowell_j2())
    ephem = ephem.reshape((step, *shape))
    return ephem

def get_closest_approach_J2(primal_eci:np.ndarray, debris_eci:np.ndarray,
                               dt=1., step=1200):
    '''
        args:
            `primal_eci`: shape: (batch_size, 6), unit: km.
            `debris_eci`: shape: (batch_size, n_debris, 6), unit: km.
        returns:
            `closest_dist`, `(primal_ephem, debris_ephem, closest_approach_step)`.
    '''
    primal_eci = primal_eci.reshape((-1, 1, 6))
    batch_size = primal_eci.shape[0]
    debris_eci = debris_eci.reshape((batch_size, -1, 6))
    primal_ephem = get_ephem_J2(primal_eci, dt=dt, step=step)
    debris_ephem = get_ephem_J2(debris_eci, dt=dt, step=step)
    rel_ephem = debris_ephem - primal_ephem # shape (step, batch_size, n_debris, 6)
    rel_pos = rel_ephem[:, :, :, :3] # shape (step, batch_size, n_debris, 3)
    dist = np.linalg.norm(rel_pos, axis=-1) # shape (step, batch_size, n_debris)
    closest_approach_step = np.argmin(dist, axis=0) # shape (batch_size, n_debris)
    closest_dist = np.min(dist, axis=0) # shape (batch_size, n_debris)
    return closest_dist, (primal_ephem, debris_ephem, closest_approach_step)
