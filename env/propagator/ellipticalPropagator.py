import scipy.linalg
from .cwPropagator import cwPropagator
from .orbitPropagator import orbitPropagator, J2Propagator
from env.dynamic.orbit.utils import get_orbit_elements

import env.dynamic.cw.matrix as matrix
from env.dynamic.cw.matrix import MU_EARTH

from astropy import units as U
from poliastro.twobody import Orbit
from poliastro.bodies import Earth

import typing
import numpy as np
import scipy

class ellipticalPropagator(J2Propagator):
    def __init__(self, 
                 max_thrust, 
                 dt=0.1, 
                 n_substep=10,
                 max_dist=5e4, 
                 safe_dist=1e3, 
                 view_dist=5e4, 
                ):
        max_n_debris=None 
        orbit_rad=7e6
        p_new_debris=None
        cwPropagator.__init__(self, max_n_debris, max_thrust, dt, n_substep, orbit_rad, max_dist, safe_dist, view_dist, p_new_debris)
        self.main_obs_dim = 12
        '''
            real state relative to nominal state in LVLH coordinate, and nominal state in ECI coordinate.
        '''
        '''
            `_obs_zoom` deprived!
        '''

    def get_orbital_elements(self, nominal_states:np.ndarray):
        return get_orbit_elements(nominal_states.reshape((-1,6)))
    
    def get_state_matrices(self, orbital_elements:np.ndarray):
        orbital_elements = orbital_elements.astype(np.float64)
        a = orbital_elements[:, 0] * 1e3 # km -> m
        ecc = orbital_elements[:, 1]
        phi = orbital_elements[:, 5]
        return matrix.elliptical_StateMat_batch(a, ecc, phi)
    
    def get_transfer_matrices(self, nominal_states:np.ndarray, n_step:int=10):
        nominal_states = nominal_states.reshape((-1, 6)).astype(np.float64)
        batch_size = nominal_states.shape[0]
        elements = np.zeros((n_step, batch_size, 6))
        orbits = [Orbit.from_vectors(Earth, nominal_states[i, :3]*U.km, nominal_states[i, 3:]*U.km/U.s) for i in range(batch_size)]
        state_matrices = np.zeros((n_step, batch_size, 6, 6))
        exp_state_matrices = np.zeros((n_step, batch_size, 6, 6))
        for step in range(n_step):
            for i in range(batch_size):
                orbit = orbits[i]
                a, ecc, inc, raan, argp, nu = orbit.a, orbit.ecc, orbit.inc, orbit.raan, orbit.argp, orbit.nu
                elements[step, i, :] = np.array([a.value, ecc.value, inc.value, raan.value, argp.value, nu.value])
                orbits[i] = orbit.propagate(self.dt*U.s, method=self._new_propagator())
            state_matrices[step, ...] = self.get_state_matrices(elements[step])
            exp_state_matrices[step, ...] = scipy.linalg.expm(state_matrices[step]*self.dt)
        transfer_matrices = exp_state_matrices[0, ...]
        for step in range(1, n_step):
            transfer_matrices = exp_state_matrices[step, ...] @ transfer_matrices
        return transfer_matrices

    def error_test(self, nominal_states:np.ndarray, debris_states:np.ndarray, n_step:int):
        batch_size = nominal_states.shape[0]
        n_d = debris_states.shape[-2]
        actions = np.zeros((batch_size, 3))
        linear_states = self.eci_to_lvlh(nominal_states, debris_states)
        Linear_states = np.zeros((n_step, batch_size, n_d, 6))
        LVLH_states = np.zeros((n_step, batch_size, n_d, 6))
        for step in range(n_step):
            Linear_states[step, ...] = linear_states
            LVLH_states[step, ...] = self.eci_to_lvlh(nominal_states, debris_states)
            transfer_mats = self.get_transfer_matrices(nominal_states, n_step=1)
            linear_states = linear_states @ transfer_mats.swapaxes(-1, -2)
            (next_nominal_states, next_debris_states, _), _, dones, (next_primal_obss, next_debris_obss) = self._propagate(nominal_states, debris_states, actions)
            nominal_states, debris_states = next_nominal_states, next_debris_states
        return Linear_states, LVLH_states