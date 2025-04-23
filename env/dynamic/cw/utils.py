from . import matrix
from ..orbit.coord import lvlh_to_eci

from astropy import units as U
from poliastro.twobody import Orbit
from poliastro.bodies import Earth

import numpy as np

def CW_tInv(a:np.ndarray, 
            forecast_states:np.ndarray, 
            t2c:np.ndarray):
    '''
        TODO: unify the interface of tInv and rInv. Currently `CW_tInv` works for batched `t2c`, while `CW_rInv` works for single `d2c`. 
    '''
    if not isinstance(a, np.ndarray):
        a = np.array(a)
    a = a.flatten()
    if not isinstance(t2c, np.ndarray):
        t2c = np.array(t2c)
    t2c = t2c.flatten()
    if not isinstance(forecast_states, np.ndarray):
        forecast_states = np.array(forecast_states)
    forecast_states = forecast_states.reshape((-1, 6))
    Phi = matrix.CW_TransMat_batch(t2c, np.zeros_like(t2c), a) # (batch_size, 6, 6)
    states_0 = Phi@np.expand_dims(forecast_states, axis=-1)
    states_0 = np.squeeze(states_0, axis=-1)
    return states_0

def CW_rInv(a:np.ndarray,
            forecast_states:np.ndarray,
            d2c:np.ndarray,
            dt = 1.,
            max_loop = 10000
            ):
    Phi = matrix.CW_TransMat_batch(dt, 0, a)
    batch_size = forecast_states.shape[0]
    n_d = forecast_states.shape[1]
    forecast_states = forecast_states.reshape((batch_size, -1, 6))
    states_0 = np.zeros((batch_size*n_d, 6))
    t2c = dt*max_loop*np.ones((batch_size*n_d,))
    flags = np.zeros(forecast_states.shape[0], dtype=np.bool_)
    states = forecast_states
    for k in range(max_loop):
        _shape = states.shape
        _states = states.reshape((-1, 6))
        r = np.linalg.norm(_states[:,:3], axis=-1)
        out = r>d2c
        _new = out & ~flags
        states_0[_new] = _states[_new]
        t2c[_new] = k*dt

        flags = (r>d2c) | flags
        if flags.all():
            break
        states = states@(Phi.swapaxes(-1, -2))
    
    _states = states.reshape((-1, 6))
    states_0[~flags] = _states[~flags]
    states_0 = states_0.reshape((batch_size, n_d, 6))
    t2c = t2c.reshape((batch_size, -1))
    return states_0, t2c

def get_orbit_rad(primal_eci:np.ndarray):
    '''
        Notice: `primal_eci` should be in km, but return `orbit_rads` in m.
        args:
            `primal_eci`: shape (batch_size, 6), unit km.
        returns:
            `orbit_rads`: shape (batch_size,), unit m.
    '''
    primal_eci = primal_eci.reshape((-1, 6))
    primal_eci = primal_eci.astype(np.float64)
    batch_size = primal_eci.shape[0]
    primal_orbits = []
    for i in range(batch_size):
        orbit = Orbit.from_vectors(Earth, primal_eci[i, :3]*U.km, primal_eci[i, 3:]*U.km/U.s)
        primal_orbits.append(orbit)
    orbit_rads = np.array([orbit.a.to(U.m).value for orbit in primal_orbits])
    return orbit_rads

def collision_eci_cw(primal_eci:np.ndarray, collision_vel:np.ndarray, t2c:np.ndarray|float):
    '''
        Notice: `primal_eci` and `collision_vel` should be in km. `collision_vel` should be in LVLH coordinate.
    '''
    primal_eci = primal_eci.reshape((-1, 6)).astype(np.float64)
    batch_size = primal_eci.shape[0]
    orbit_rads = get_orbit_rad(primal_eci)
    collision_vel = collision_vel.reshape((batch_size, -1, 3)) # km/s
    collision_pos = np.zeros_like(collision_vel) # km
    collision_lvlh = np.concatenate((collision_pos, collision_vel), axis=-1)
    debris_lvlh = CW_tInv(orbit_rads, collision_lvlh, t2c).astype(np.float64)
    debris_eci = lvlh_to_eci(primal_eci, debris_lvlh)
    return debris_eci