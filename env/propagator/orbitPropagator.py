from .cwPropagator import cwPropagator
from env.dynamic.orbit.coord import eci_to_lvlh, eci_to_lvlh_mat
from env.dynamic.orbit.utils import get_orbit_elements
from utils import norm

from ..dynamic.orbit.J2 import cowell_j2

from astropy import units as U
from poliastro.twobody.propagation import FarnocchiaPropagator
from poliastro.twobody import Orbit
from poliastro.bodies import Earth

import typing
import numpy as np

class orbitPropagator(cwPropagator):
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
        super().__init__(max_n_debris, max_thrust, dt, n_substep, orbit_rad, max_dist, safe_dist, view_dist, p_new_debris)
        self.main_obs_dim = 12
        '''
            real state relative to nominal state in LVLH coordinate, and nominal state in ECI coordinate.
        '''

        self._nominal_orbit = Orbit.from_classical(Earth, 7155.459*U.km, 1.174*1e-3*U.one, 1.292*U.rad, 0.301*U.rad, 1.468*U.rad, 0.234*U.rad)

    def _new_propagator(self):
        return FarnocchiaPropagator()

    def randomDebrisStates(self,  *args, **kwargs):
        raise NotImplementedError

    def randomPrimalStates(self, *args, **kwargs):
        raise NotImplementedError

    def initPrimalStates(self):
        r = self._nominal_orbit.r.to(U.km).value
        v = self._nominal_orbit.v.to(U.km/U.s).value
        x = np.hstack((r, v)).reshape((1, 6))
        return x
    
    def discard_leaving(self, *args, **kwargs):
        raise NotImplementedError
    
    def _propagate(self, 
                   primal_states:np.ndarray,
                   debris_states:np.ndarray,
                   actions:np.ndarray,
                   nominal_primal_states:typing.Optional[np.ndarray]=None
                   ):
        '''
            returns: `(next_primal_states, next_debris_states, next_nominal_primal_states)`, `rewards`, `dones`, `(next_primal_obss, next_debris_obss)`
        '''
        next_primal_states, next_debris_states = self.getNextStates(primal_states, debris_states, actions, nominal_primal_states=nominal_primal_states)
        next_nominal_primal_states = self.getNextStatesNominal(nominal_primal_states) if nominal_primal_states is not None else None
        rewards = self.getRewards(next_primal_states, next_debris_states, actions, nominal_primal_states=next_nominal_primal_states)
        dones = self.getDones(next_primal_states, next_debris_states, nominal_primal_states=next_nominal_primal_states)
        next_primal_obss, next_debris_obss = self.getObss(next_primal_states, next_debris_states, nominal_primal_states=next_nominal_primal_states)
        return (next_primal_states, next_debris_states, next_nominal_primal_states), rewards, dones, (next_primal_obss, next_debris_obss)
    
    def _propagate_with_ephem(self,
                              primal_states:np.ndarray,
                              next_debris_states:np.ndarray,
                              actions:np.ndarray,
                              nominal_primal_states:typing.Optional[np.ndarray]=None,
                              next_nominal_primal_states:typing.Optional[np.ndarray]=None
                              ):
        '''
            returns: `(next_primal_states, next_debris_states, next_nominal_primal_states)`, `rewards`, `dones`, `(next_primal_obss, next_debris_obss)`
        '''
        next_primal_states, _ = self.getNextStates(primal_states, None, actions, nominal_primal_states=nominal_primal_states)
        rewards = self.getRewards(next_primal_states, next_debris_states, actions, nominal_primal_states=next_nominal_primal_states)
        dones = self.getDones(next_primal_states, next_debris_states, nominal_primal_states=next_nominal_primal_states)
        next_primal_obss, next_debris_obss = self.getObss(next_primal_states, next_debris_states, nominal_primal_states=next_nominal_primal_states)
        return (next_primal_states, next_debris_states, next_nominal_primal_states), rewards, dones, (next_primal_obss, next_debris_obss)

    def propagate(self, *args, **kwargs):
        raise NotImplementedError
    
    def eci_to_lvlh(self, main_eci_states:np.ndarray, sub_eci_states:np.ndarray):
        '''
                Convert eci coordinate (used for state) to lvlh coordinate (uesd for observation).
            Notice: eci states are in kilometer, while lvlh states are in meter.
            args:
                `main_eci_states`: shape: (batch_size, 6), unit: km.
                `sub_eci_states`: shape: (batch_size, n, 6), unit: km.
            returns:
                `lvlh_states`: `sub_eci_states`'s coordinate in corresponding LVLH. unit: m.

        '''
        lvlh_states = eci_to_lvlh(main_eci_states, sub_eci_states)
        # lvlh_states *= 1000 # km -> m
        return lvlh_states
    
    def eci_to_lvlh_both(self,
                         primal_eci_states:np.ndarray, 
                         debris_eci_states:np.ndarray, 
                         nominal_primal_states:typing.Optional[np.ndarray]=None):
        '''
            args: `primal_eci_states`, `debris_eci_states`, `nominal_primal_states`. unit: km.
            returns: `primal_lvlh`, `debris_lvlh`. unit: m.
        '''
        if debris_eci_states.ndim==2:
            debris_eci_states = np.expand_dims(debris_eci_states, axis=0)
        if nominal_primal_states is None:
            nominal_primal_states = primal_eci_states
            primal_lvlh = np.zeros_like(primal_eci_states)
        else:
            primal_lvlh = self.eci_to_lvlh(nominal_primal_states, np.expand_dims(primal_eci_states, axis=1))
            primal_lvlh = primal_lvlh.squeeze(axis=1)
        debris_lvlh = self.eci_to_lvlh(nominal_primal_states, debris_eci_states)
        return primal_lvlh, debris_lvlh

    def eci_thrusts(self, actions:np.ndarray, main_eci_states:np.ndarray):
        '''
            args:
                `actions`: in lvlh coord.
                `main_eci_states`: corresponding eci states.
            returns:
                `thrusts`: in eci states.
        '''
        R_inv = eci_to_lvlh_mat(main_eci_states)
        R = np.linalg.inv(R_inv)
        actions = self.clip_action(actions)
        actions = R@np.expand_dims(actions, axis=-1)
        actions = actions.squeeze(axis=-1)
        thrusts = actions*self.max_thrust
        return thrusts
    
    def getNextStatesNominal(self, states):
        shape = states.shape
        states = states.astype(np.float64).reshape((-1, 6))
        next_states = []
        for s in states:
            r, v = s[:3], s[3:]
            orbit = Orbit.from_vectors(Earth, r*U.km, v*U.km/U.s)
            r1, v1 = orbit.propagate(self.dt*U.s, method=self._new_propagator()).rv()
            next_states.append(np.concatenate((r1.value, v1.value)))
        next_states = np.vstack(next_states)
        next_states = next_states.reshape(shape)
        return next_states

    def getNextStates(self, 
                      primal_states:np.ndarray,
                      debris_states:typing.Optional[np.ndarray],
                      actions:np.ndarray,
                      nominal_primal_states:typing.Optional[np.ndarray]=None, ):
        '''
            Notice: actions are in lvlh coordinate, while states are in eci coordinate.
            Notice: for-loop is uesd for propagating orbit, which is extremely slow with a large batchsize.
        '''
        batch_size = primal_states.shape[0]
        if batch_size>1:
            print("Notice: for-loop is uesd for propagating orbit, which is extremely slow with a large batchsize.")
        nominal_primal_states = primal_states if nominal_primal_states is None else nominal_primal_states

        primal_states = primal_states.astype(np.float64)
        thrusts = self.eci_thrusts(actions, main_eci_states=nominal_primal_states)
        thrusts = thrusts.astype(np.float64)
        next_primal_states = []
        for i in range(batch_size):
            s = primal_states[i]
            u = thrusts[i]
            r, v = s[:3], s[3:]
            orbit = Orbit.from_vectors(Earth, r*U.km, v*U.km/U.s)
            r1, v1 = orbit.propagate(self.dt*U.s, method=self._new_propagator()).rv()
            v1 += (u*self.dt)*U.m/U.s
            next_primal_states.append(np.concatenate((r1.value, v1.value)))
        next_primal_states = np.vstack(next_primal_states)

        if debris_states is None or len(debris_states)==0:
            next_debris_states = []
        else:
            debris_states = debris_states.astype(np.float64)
            debris_states = debris_states.reshape((-1, 6))
            next_debris_states = []
            for s in debris_states:
                r, v = s[:3], s[3:]
                orbit = Orbit.from_vectors(Earth, r*U.km, v*U.km/U.s)
                r1, v1 = orbit.propagate(self.dt*U.s, method=self._new_propagator()).rv()
                next_debris_states.append(np.concatenate((r1.value, v1.value)))
            next_debris_states = np.vstack(next_debris_states).reshape((batch_size, -1, 6))

        return next_primal_states, next_debris_states
    
    def getObss(self, 
                primal_states:np.ndarray, 
                debris_states:np.ndarray, 
                nominal_primal_states:typing.Optional[np.ndarray]=None,
                viewed_only=False):
        '''
            `primal_obss`: LVLH coord wrt to the nominal satellite and ECI coord of the nominal satellite. Unit: km.
            `debris_obss`: LVLH coord wrt to the nominal satellite. Unit: km.
        '''
        primal_lvlh, debris_lvlh = self.eci_to_lvlh_both(primal_states, debris_states, nominal_primal_states)
        primal_obss, debris_obss = super().getObss(primal_lvlh, debris_lvlh)
        debris_obss = debris_obss[:, :, :6] # LVLH coordinate only

        nominal_eci = primal_states if nominal_primal_states is None else nominal_primal_states
        primal_obss = np.concatenate((primal_obss, nominal_eci), axis=-1)

        if viewed_only:
            batch_size = primal_states.shape[0]
            debris_obss_viewed = []
            for i in range(batch_size):
                rel_states = debris_lvlh[i] - primal_lvlh[i].reshape((1,6))
                dist = norm(rel_states[:,:3])
                viewed = dist<=self.view_dist
                debris_obss_viewed.append(debris_obss[i, viewed])
            return primal_obss, debris_obss_viewed
        else:
            return primal_obss, debris_obss

    def getDones(self, primal_states:np.ndarray, debris_states:np.ndarray, nominal_primal_states:typing.Optional[np.ndarray]=None):
        primal_lvlh, debris_lvlh = self.eci_to_lvlh_both(primal_states, debris_states, nominal_primal_states)
        return super().getDones(primal_lvlh, debris_lvlh)
    
    def getRewards(self, 
                   primal_states:np.ndarray,
                   debris_states:np.ndarray,
                   actions:np.ndarray,
                   nominal_primal_states:typing.Optional[np.ndarray]=None,
                   ):
        primal_lvlh, debris_lvlh = self.eci_to_lvlh_both(primal_states, debris_states, nominal_primal_states)
        return super().getRewards(primal_lvlh, debris_lvlh, actions)
    
class J2Propagator(orbitPropagator):
    def __init__(self, max_thrust, dt=0.1, n_substep=10, max_dist=50000, safe_dist=1000, view_dist=50000):
        super().__init__(max_thrust, dt, n_substep, max_dist, safe_dist, view_dist)

    def _new_propagator(self):
        return cowell_j2()