from .basePropagator import basePropagator
from ..dynamic.cw import matrix
from ..dynamic.cw import utils as cwutils
from utils import *

import torch
import numpy as np
import typing

class cwPropagator0(basePropagator):
    def __init__(self,
                 max_thrust:float,
                 dt=.1, 
                 orbit_rad=7e6, 
                 max_dist=1e3) -> None:
        super().__init__(6, 6, 3, dt)

        self.max_dist = max_dist
        
        self.set_orbit_rad(orbit_rad)

        self._ps0_scale = torch.zeros(6)
        self._ps0_scale[:3] = self.max_dist/10
        self._ps0_scale[3:] = self.max_dist/10000
        self.primal_state0_dist = torch.distributions.Normal(
            loc=torch.zeros_like(self._ps0_scale),
            scale=self._ps0_scale)
        
        self.max_thrust = max_thrust
        self.beta_action = 1/self.max_dist
        self.beta_vel = 1/self.max_dist


    def set_orbit_rad(self, orbit_rad: float):
        '''
            args:
                `orbit_rad`: float, with unit m.
        '''
        self.orbit_rad = orbit_rad
        trans_mat = matrix.CW_TransMat_batch(0, self.dt, orbit_rad)
        self.trans_mat = trans_mat.squeeze(axis=0)

        state_mat = matrix.CW_StateMat_batch(orbit_rad)
        self.state_mat = state_mat.squeeze(axis=0)

        self._n = np.sqrt(matrix.MU_EARTH/orbit_rad**3)

    def randomPrimalStates(self, n:int, zero_init=True) -> np.ndarray:
        if zero_init:
            return np.zeros((n, 6))
        else:
            return self.primal_state0_dist.sample((n,)).numpy()
    
    def getNextStates(self, states:np.ndarray, actions:torch.Tensor|np.ndarray):
        '''
            args:
                `states`: shape: (n_primal, 6)
                `actions`: shape: (n_primal, 3)
        '''
        next_primal_states = states@self.trans_mat.T
        con_vec = self._conVecs(actions) # shape: (n_primal, 6)
        next_primal_states = next_primal_states + con_vec
        return next_primal_states
        
    def _conVecs(self, actions:torch.Tensor|np.ndarray):
        actions = self.clip_action(actions)
        thrust = actions*self.max_thrust
        return matrix.CW_constConVecs(0, self.dt, thrust, self.orbit_rad) # shape: (n_primal, 6)
        
    def getNextStatesNominal(self, states:np.ndarray):
        shape = states.shape
        states = states.view((-1, 6))
        next_states = states@self.trans_mat.T
        next_states = next_states.view(shape)
        return next_states
        
    def getTrajNominal(self, states0:np.ndarray, step:int):
        traj = [states0]
        for i in range(step):
            states0 = self.getNextStatesNominal(states0)
            traj.append(states0)
        traj = np.stack(traj, axis=0)
        return traj
        
    def getObss(self, states:torch.Tensor):
        '''
            args:
                `states`: shape: (n_primal, 6)
            returns:
                `obss`: shape: (n_primal, 6)
        '''
        obss = states*1. # shape: (n_primal, 6)
        return obss
        
    def getRewards(self, states:np.ndarray, actions:np.ndarray):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `actions`: shape: (n_primal, 3)
            returns:
                `rewards`: shape: (n_primal,)
        '''
        fuel_rewards = (1-norm(actions)/self.max_action_norm)*self.beta_action

        d2o = norm(states[:,:3])
        area_rewards = 1 - d2o/self.max_dist

        vel = norm(states[:,3:])
        vel_rewards = -vel*self.beta_vel

        rewards = fuel_rewards + area_rewards + vel_rewards
        return rewards
        
    def getDones(self, states:np.ndarray):
        '''
            args:
                `states`: shape: (n_primal, 6)
            returns:
                `dones`: shape: (n_primal,)
        '''
        dones = np.zeros(states.shape[0], dtype=np.bool_)
        return dones

class cwPropagator(cwPropagator0):
    def __init__(self,
                 max_n_debris:int,
                 max_thrust:float,
                 dt=.1, 
                 n_substep=10,
                 orbit_rad=7e6, 
                 max_dist=1e3, 
                 safe_dist=5e1,
                 view_dist=1e4,
                 p_new_debris=0.001,
                ) -> None:
        super().__init__(max_thrust=max_thrust, dt=dt, orbit_rad=orbit_rad, max_dist=max_dist)
        self.main_state_dim = 6
        self.main_obs_dim = 6
        self.sub_state_dim = 6
        self.sub_obs_dim = 2*6 + 6

        self.max_n_debris = max_n_debris
        self.n_substep = n_substep
        self.safe_dist = safe_dist
        self.view_dist = view_dist
        self.p_new_debris = p_new_debris

        self._with_obs_noise = False
        self._obs_noise = torch.zeros(6)
        self._obs_noise[:3] = 1e1
        self._obs_noise[3:] = 1e-1
        self.obs_noise_dist = torch.distributions.Normal(
            loc=torch.zeros_like(self._obs_noise),
            scale=self._obs_noise)
        self._obs_noise = self._obs_noise.reshape((1, 1, 6))
        
        self._dsC_scale = torch.zeros(6)
        self._dsC_scale[:3] = self.safe_dist/2
        # self._dsC_scale[3:] = self.max_dist/10
        self._dsC_scale[3:] = 100./3
        self.debris_stateC_dist = torch.distributions.Normal(
            loc=torch.zeros_like(self._dsC_scale),
            scale=self._dsC_scale)
        
        self.collision_reward = -1.
        self.area_reward = 1.
        self.beta_action = 0.1
        self.beta_vel = 1/self.max_dist
        self.max_action_norm = self.action_box.max_norm
    
    def randomDebrisStates(self, 
                           n:int|torch.Size, 
                           pinpoint:np.ndarray|None=None, 
                           old:np.ndarray|None=None,
                           t2c:np.ndarray|None=None) -> np.ndarray:
        '''
            args:
                `n`: n_debris or (n_primal, n_debris).
                `pinpoint`: Optional, position to be collided, shape: (..., 3).
                `old`: Optional, old debris states.
                `t2c`: Optional, time to collision.
            returns:
                `states`: debris states at t0.
                (`t2c`, `stateC`): time to collision; state of closest approach.
        '''
        if isinstance(n, int):
            stateC = self.debris_stateC_dist.sample((n,)).numpy()
            if pinpoint is not None:
                stateC[:, :3] = pinpoint.reshape((-1, 3))
        else:
            stateC = self.debris_stateC_dist.sample(n).numpy()
            if pinpoint is not None:
                stateC[:, :, :3] = pinpoint.reshape((-1, 1, 3))
        if t2c is None:
            states, t2c = cwutils.CW_rInv(self.orbit_rad, stateC, self.view_dist)
        else:
            states = cwutils.CW_tInv(self.orbit_rad, stateC, t2c)
        if old is not None:
            states = np.concatenate((old, states), axis=-2)
        return states, (t2c, stateC)
    
    def _propagate(self, 
                   primal_states:np.ndarray, 
                   debris_states:np.ndarray, 
                   actions:np.ndarray|torch.Tensor, 
                   discard_leaving=False,
                   new_debris=False):
        '''
            returns: `(next_primal_states, next_debris_states)`, `rewards`, `dones`, `(next_primal_obss, next_debris_obss)`
        '''
        if debris_states.ndim>2:
            if discard_leaving:
                raise ValueError("`discard_leaving` not support for batched `debris_states`")
            if new_debris:
                raise ValueError("`new_debris` not support for batched `debris_states`")
        next_primal_states, next_debris_states = self.getNextStates(primal_states, debris_states, actions)
        rewards = self.getRewards(next_primal_states, next_debris_states, actions)
        dones = self.getDones(next_primal_states, next_debris_states)
        if discard_leaving:
            next_debris_states = self.discard_leaving(next_debris_states)
            if next_debris_states.shape[-2]==0:
                next_debris_states, _ = self.randomDebrisStates(1, old=next_debris_states)
        if new_debris:
            while next_debris_states.shape[0]<self.max_n_debris and np.random.rand()<self.p_new_debris:
                next_debris_states, _ = self.randomDebrisStates(1, old=next_debris_states)
        next_primal_obss, next_debris_obss = self.getObss(next_primal_states, next_debris_states)
        return (next_primal_states, next_debris_states), rewards, dones, (next_primal_obss, next_debris_obss)

    def getNextStates(self, primal_states:np.ndarray, debris_states:np.ndarray, actions:np.ndarray):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6) or (n_primal, n_debris, 6)
                `actions`: shape: (n_primal, 3)
        '''
        next_primal_states = primal_states@self.trans_mat.T
        shape = debris_states.shape
        debris_states = debris_states.reshape((-1, 6))
        next_debris_states = debris_states@self.trans_mat.T
        next_debris_states = next_debris_states.reshape(shape)
        con_vec = self._conVecs(actions) # shape: (n_primal, 6)
        next_primal_states = next_primal_states + con_vec
        return next_primal_states, next_debris_states

    def getObss(self, primal_states:np.ndarray, debris_states:np.ndarray):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6) or (n_primal, n_debris, 6)
            returns:
                `primal_obss`: shape: (n_primal, 6)
                `debris_obss`: shape: (n_primal, n_debris, 9)
        '''
        if self._with_obs_noise:
            obs_noise = self.obs_noise_dist.sample(debris_states.shape[:-1])
            debris_states = debris_states + obs_noise
        if debris_states.ndim==2:
            debris_states = np.expand_dims(debris_states, axis=0)
        rel_states = debris_states - np.expand_dims(primal_states, axis=1) # shape: (n_primal, n_debris, 6)
        rel_obss = rel_states
        rel_pos, rel_vel = rel_obss[:,:,:3], rel_obss[:,:,3:]
        distance = dotEachRow(rel_pos, rel_pos, keepdim=True) # shape: (n_primal, n_debris, 1)
        speed = dotEachRow(rel_vel, rel_vel, keepdim=True)
        dot_rv = dotEachRow(rel_pos, rel_vel, keepdim=True)
        cos_rv = dot_rv/np.sqrt(distance*speed)
        sin_cone = np.clip(self.safe_dist/distance, a_min=None, a_max=1-1e-8)
        cos_cone = np.sqrt(1-sin_cone**2)
        base_primal_obss = primal_states # shape: (n_primal, 6)
        base_debris_obss = debris_states # shape: (n_primal, n_debris, 6)
        primal_obss = np.concatenate((base_primal_obss,), axis=-1) # shape: (n_primal, 6)
        debris_obss = np.concatenate((base_debris_obss, rel_obss, distance, speed, dot_rv, cos_rv, sin_cone, cos_cone), axis=-1) # shape (n_primal, n_debris, 18)

        return primal_obss, debris_obss
        
    def getRewards(self, primal_states:np.ndarray, debris_states:np.ndarray, actions:np.ndarray|torch.Tensor):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6) or (n_primal, n_debris, 6)
                `actions`: shape: (n_primal, 3)
            returns:
                `rewards`: shape: (n_primal,)
        '''
        if debris_states.ndim==2:
            debris_states = np.expand_dims(debris_states, axis=0)

        actions = self.clip_action(actions)
        fuel_rewards = (self.max_action_norm-norm(actions))/self.max_action_norm*self.beta_action

        vo_rewards, in_danger_each = self._getVoRewards(primal_states, debris_states, actions=actions)
        in_danger = in_danger_each.any(axis=-1)

        d2o = norm(primal_states[:,:3])
        area_rewards = 1 - d2o/self.max_dist
        area_rewards = np.where(in_danger, area_rewards/2, area_rewards)

        vel = norm(primal_states[:,3:])
        vel_rewards = -vel*self.beta_vel

        rewards = fuel_rewards + area_rewards # + vo_rewards
        return rewards
        
    def _getVoRewards(self, primal_states:np.ndarray, debris_states:np.ndarray, actions:np.ndarray|torch.Tensor) -> typing.Tuple[np.ndarray, np.ndarray]:
        if debris_states.ndim==2:
            debris_states = np.expand_dims(debris_states, axis=0)
        rel_states = debris_states - np.expand_dims(primal_states, axis=1) # shape: (n_primal, n_debris, 6)
        rel_pos, rel_vel = rel_states[:,:,:3], rel_states[:,:,3:]
        distance = dotEachRow(rel_pos, rel_pos) # shape: (n_primal, n_debris)
        speed = dotEachRow(rel_vel, rel_vel)
        dot_rv = dotEachRow(rel_pos, rel_vel)
        cos_rv = dot_rv/np.sqrt(distance*speed)
        sin_cone = np.clip(self.safe_dist/distance, a_min=None, a_max=1-1e-8)
        cos_cone = np.sqrt(1-sin_cone**2)
        in_danger_each = (-cos_rv)>cos_cone
        in_danger = (in_danger_each).any(axis=-1)
        vo_rewards = np.where(in_danger, 0., 1.)
        return vo_rewards, in_danger_each

    def getDones(self, primal_states:np.ndarray, debris_states:np.ndarray):
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6) or (n_primal, n_debris, 6)
            returns:
                `dones`: shape: (n_primal,)
        '''
        distances = self.distances(primal_states, debris_states)
        dones_collide = (distances<self.safe_dist).any(axis=-1)
        d2o = norm(primal_states[:,:3])
        dones_out = d2o>self.max_dist
        # dones = dones_collide | dones_out
        dones = dones_collide
        return dones
        
    def discard_leaving(self, debris_states:np.ndarray) -> np.ndarray:
        '''
            args:
                `debris_states`: shape: (n, 6)
        '''
        if debris_states.ndim>2:
            raise NotImplementedError("not support batched `debris_states` now.")
        leaving = self.is_leaving(debris_states)
        debris_states = debris_states[~leaving]
        return debris_states
        
    def is_leaving(self, debris_states:np.ndarray) -> np.ndarray:
        '''
            args:
                `states`: shape: (n, 6)
        '''
        if debris_states.ndim>2:
            raise NotImplementedError("not support batched `debris_states` now.")
        pos = debris_states[:, :3]
        vel = debris_states[:, 3:]
        dist = norm(pos)
        dot = np.sum(pos*vel, dim=-1)
        leaving = (dot>0.) & (dist>self.max_dist)
        return leaving
        
    def distances(self, primal_states:np.ndarray, debris_states:np.ndarray) -> np.ndarray:
        '''
            args:
                `primal_states`: shape: (n_primal, 6)
                `debris_states`: shape: (n_debris, 6) or (n_primal, n_debris, 6)
            returns:
                `distances`: shape: (n_primal, n_debris)
        '''
        primal_pos = primal_states[:, :3]
        primal_pos = np.expand_dims(primal_pos, axis=1)
        if debris_states.ndim==2:
            debris_states = np.expand_dims(debris_states, axis=0)
        debris_pos = debris_states[:, :, :3]
        distances = norm(primal_pos-debris_pos) # shape: (n_primal, n_debris)
        return distances