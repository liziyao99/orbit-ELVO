from ..basePolicy import basePolicy
from .voConstraint import voConstraint, inPlaneVoConstraint
from utils import obs_padding, norm
from env.dynamic.orbit.J2 import cowell_j2
from env.dynamic.orbit.coord import lvlh_to_eci, eci_to_lvlh
from env.dynamic.orbit.utils import eci_to_orbit, eci_propagate, eci_propagate_batch, find_tca, get_orbit_elements
from env.dynamic.cw import matrix

import numpy as np
import scipy
import collections
import typing
import time


class safeOrbitVoPolicy(basePolicy):
    def __init__(self, 
                 dt, # s
                 max_thrust, # m/s^2
                 base_sep_dist, 
                 plan_horizon:int=20, # int
                 pred_window:float=2000, # s
                 obs_sigma=1e-2, # km
                 obs_lambda=1e-3, # km/s
                 ):
        
        self.space_dim = 3
        obs_dim = self.space_dim * 2
        action_dim = self.space_dim
        super().__init__(obs_dim, action_dim)

        self.dt = dt
        self.max_thrust = max_thrust
        '''
            Notice: unit is m/s^2.
        '''
        self.base_sep_dist = base_sep_dist
        self.plan_horizon = plan_horizon
        self.pred_window = pred_window

        self.obs_sigma = obs_sigma
        self.obs_lambda = obs_lambda
        self.D = 25.9
        self.max_delta_tca = 5 # s

        self.with_noise = True
        self.elliptical_pred = False

        self.max_safe_ball_radius = None
        self.max_safe_dist = None

        self._voc = []
        '''
            list of `voConstraint`.
        '''
        self._opt_res = []
        '''
            list of optimization results.
        '''
        self._action_seq = collections.deque(maxlen=plan_horizon)
        '''
            deque of actions to be applied.
        '''
        # self.defualt_initial_guess = np.zeros((1, space_dim))
        self.defualt_initial_guess = 3. * np.random.randn(6, self.space_dim)
        '''
            default initial guess for VO.
        '''
        self.delta_v_not_completed = np.zeros((0, self.space_dim))
        '''
            delta_v not completed in last plan.
        '''

        self.init_log()
        
        self.EL_states = np.zeros((0, 0, self.space_dim*2))
        '''
            Equivalent linear state of the debris pieces. Shape: (n_p, n_o, 6). Unit: km.
        '''

        self.in_plane_mode = ""
        '''
            'in_plane', 'along_track', others.
        '''

    
    def start(self, 
              primal_obss:np.ndarray, 
              debris_obss:np.ndarray|typing.List[np.ndarray], 
              pred_horizon:typing.Optional[int]=None,
              sep_dists: typing.Optional[np.ndarray|float]=None,
              *args, **kwargs):

        self._voc = []
        self._opt_res = []
        self._action_seq.clear()
        self.init_log()

        _t0 = time.time()

        if pred_horizon is not None:
            self.pred_horizon = pred_horizon # TODO: check

        _t1 = time.time()
        self.log["start_time"].append(_t1-_t0)

        # self._plan(primal_lvlh*1000, primal_eci, sep_dists=sep_dists) # m, km


    def init_log(self):
        self.log = {
            "opt_res": [],
            "start_time": [],
            "propagate_time": [],
            "solve_time": [],
            "inevitable": [],
            "closest_dists": [],
            "closest_time": [],
        }

    def orbit_prop(self):
        return cowell_j2()

    def act(self, 
            primal_obss:np.ndarray, 
            debris_obss:np.ndarray|typing.List[np.ndarray], 
            sep_dists:typing.Optional[np.ndarray|float]=None,
            pref_delta_v: typing.Optional[np.ndarray]=None, 
            initial_guess: typing.Optional[np.ndarray]=None
            ):
        '''
            args:
                `primal_obss`: shape: (n_p, ...)
                `debris_obss`: shape: (n_p, n_o, ...)
                `sep_dists`: separating distance. shape: (n_p, n_o, ) or float
        '''
        (primal_eci, debris_eci), (primal_lvlh, debris_lvlh), nominal_eci = self.obs_to_state(primal_obss, debris_obss)
        if len(self._action_seq)==0:
            self._plan(primal_eci, debris_eci, nominal_eci, sep_dists=sep_dists, pref_delta_v=pref_delta_v, initial_guess=initial_guess)
        vo_actions = self._action_seq.pop()

        self._propagate_EL_states()

        return vo_actions, vo_actions
    

    def _propagate_EL_states(self):
        self.EL_states[:, :, :3] = self.EL_states[:, :, :3] + self.EL_states[:, :, 3:]*self.dt


    def obs_to_state(self, primal_obss:np.ndarray, debris_obss:typing.Optional[np.ndarray|typing.Iterable[np.ndarray]]=None):
        '''
            return: `(primal_eci, debris_eci)`, `(primal_lvlh, debris_lvlh)` and `nomimal_eci`, all in unit of km.
        '''
        n_p = primal_obss.shape[0]
        primal_lvlh = primal_obss[:, :2*self.space_dim] # km
        nominal_eci = primal_obss[:, 2*self.space_dim:] # km
        primal_eci = lvlh_to_eci(nominal_eci, primal_lvlh).reshape((n_p, 2*self.space_dim))
        if debris_obss is not None:
            if not isinstance(debris_obss, np.ndarray): # list
                # length of obs is varying, need padding. Pad_value=1e6 represents far and leaving debris.
                debris_obss = obs_padding(debris_obss, pad_value=1e6)
            debris_lvlh = debris_obss[:, :, :2*self.space_dim] # km
            debris_eci = lvlh_to_eci(nominal_eci, debris_lvlh) # km
        else:
            debris_eci, debris_lvlh = None, None
        return (primal_eci, debris_eci), (primal_lvlh, debris_lvlh), nominal_eci

    @property
    def safe_ball_radius(self):
        R = self.base_sep_dist + self.D * self.obs_sigma
        if self.max_safe_ball_radius is not None:
            R = np.minimum(R, self.max_safe_ball_radius)
        return R
    
    def delta_1_rc(self):
        delta = self.D * self.obs_sigma
        return delta
    
    def delta_2_rc(self, 
                   nominal_eci:np.ndarray,
                   max_delta_tca:float=None):
        n_p = self.EL_states.shape[0]
        n_o = self.EL_states.shape[1]

        max_delta_tca = self.max_delta_tca if max_delta_tca is None else max_delta_tca
        vc = self.EL_states[:, :, 3:]
        vc = np.linalg.norm(vc, axis=-1)

        state_mats = matrix.elliptical_StateMat_batch_fromECI(nominal_eci) # TODO: propagate nominal eci to TCA one by one
        state_mats = state_mats.reshape((n_p, 1, 6, 6))
        trans_mats = scipy.linalg.expm(state_mats*max_delta_tca)
        trans_mats_rv = trans_mats[:, :, :3, 3:]
        unifrom_motion_rv = np.eye(3).reshape((1, 1, 3, 3))
        unifrom_motion_rv = max_delta_tca * unifrom_motion_rv
        Delta_Phi_rv = trans_mats_rv - unifrom_motion_rv
        norm_Delta_Phi_rv = np.zeros((n_p, n_o))
        for i in range(n_p):
                norm_Delta_Phi_rv[i, ...] = np.linalg.norm(Delta_Phi_rv[i, 0], ord=2)

        delta = norm_Delta_Phi_rv*vc + np.abs(max_delta_tca)*self.D*self.obs_lambda
        return delta

    def delta_3_rc(self,
                   nominal_eci:np.ndarray):
        n_p = self.EL_states.shape[0]
        n_o = self.EL_states.shape[1]

        T = self.plan_horizon*self.dt
        TCA = self.log["closest_time"][-1]
        TCA = np.maximum(TCA, 0.)
        TCA = np.expand_dims(TCA, axis=(-2,-1))
        
        state_mats = matrix.elliptical_StateMat_batch_fromECI(nominal_eci)
        state_mats = state_mats.reshape((n_p, 1, 6, 6))
        trans_mats = scipy.linalg.expm(state_mats*TCA)
        trans_mats_rv = trans_mats[:, :, :3, 3:]
        unifrom_motion_rv = np.eye(3).reshape((1, 1, 3, 3))
        unifrom_motion_rv = TCA * unifrom_motion_rv
        Delta_Phi_rv = trans_mats_rv - unifrom_motion_rv
        norm_Delta_Phi_rv = np.zeros((n_p, n_o))
        for i in range(n_p):
            for j in range(n_o):
                norm_Delta_Phi_rv[i, j] = np.linalg.norm(Delta_Phi_rv[i, j], ord=2)

        nominal_eci_2 = eci_propagate_batch(nominal_eci, T)
        state_mats_2 = matrix.elliptical_StateMat_batch_fromECI(nominal_eci_2)
        trans_mats_2 = scipy.linalg.expm(state_mats_2*(TCA-T))
        trans_mats_rv_2 = trans_mats_2[:, :, :3, 3:]
        Diff_Phi_rv = trans_mats_rv_2 - trans_mats_rv
        norm_Diff_Phi_rv = np.zeros((n_p, n_o))
        for i in range(n_p):
            for j in range(n_o):
                norm_Diff_Phi_rv[i, j] = np.linalg.norm(Diff_Phi_rv[i, j], ord=2)

        max_thrust = self.max_thrust/1e3 # m -> km
        delta = (norm_Delta_Phi_rv+norm_Diff_Phi_rv)*max_thrust*T
        return delta

    def get_compensating_term(self,
                              nominal_eci:np.ndarray,
                              max_delta_tca:float=None):

        delta_1 = self.delta_1_rc()
        delta_2 = self.delta_2_rc(nominal_eci, max_delta_tca=max_delta_tca)
        delta_3 = self.delta_3_rc(nominal_eci)
        r = delta_1 + delta_2 + delta_3
        if self.max_safe_dist is not None:
            r = np.minimum(r, self.max_safe_dist)
        return r

    
    def _update_EL_states(self, primal_eci:np.ndarray, debris_eci:np.ndarray, nominal_eci:np.ndarray):
        '''
            Get closest approach of debris pieces. Update `EL_states`.
            `log[closest_time]` and `log[closest_dists]` will be updated as well.
        '''
        n_p = primal_eci.shape[0]
        n_o = debris_eci.shape[1]
        TCA = np.zeros((n_p, n_o))
        TCA_LVLH = np.zeros((n_p, n_o, 2*self.space_dim))
        CLOSEST_DISTS = np.zeros((n_p, n_o))
        for i in range(n_p): # equivalent linear velocity
            sat_orbit = eci_to_orbit(primal_eci[i])[0]
            debris_orbits = eci_to_orbit(debris_eci[i])
            for j in range(n_o):
                tca, min_dist, (rc_sat, vc_sat), (rc_deb, vc_deb) = find_tca(sat_orbit, debris_orbits[j], time_bounds=(0, self.pred_window), prop_method=self.orbit_prop())
                tca = tca.value
                tca_eci_deb = np.hstack((rc_deb.value, vc_deb.value))
                tca_eci_nom = eci_propagate(nominal_eci[i], tca, method=self.orbit_prop())
                TCA[i, j] = tca
                TCA_LVLH[i, j] = eci_to_lvlh(tca_eci_nom, tca_eci_deb) # km
                CLOSEST_DISTS[i, j] = min_dist.value
        EL_debris_lvlh = TCA_LVLH.copy()
        EL_debris_lvlh[:, :, :3] = EL_debris_lvlh[:, :, :3] - EL_debris_lvlh[:, :, 3:]*TCA.reshape((n_p, n_o, 1))
        self.EL_states = EL_debris_lvlh

        self.log["closest_time"].append(TCA)
        self.log["closest_dists"].append(CLOSEST_DISTS)

    
    def _plan(self, 
              primal_eci:np.ndarray,
              debris_eci:np.ndarray, 
              nominal_eci:np.ndarray,
              sep_dists: typing.Optional[np.ndarray|float]=None,
              pref_delta_v: typing.Optional[np.ndarray]=None, 
              initial_guess: typing.Optional[np.ndarray]=None):
        '''
            `_solve_vo` by using `primal_lvlh` and `EL_states`. 
            Notice: lvlh states should be of unit m, while eci states in km.
        '''
        n_p = primal_eci.shape[0]

        _t0 = time.time()
        self._update_EL_states(primal_eci, debris_eci, nominal_eci)
        _t1 = time.time()
        self.log["propagate_time"].append(_t1-_t0)

        _t0 = time.time()

        # safe_ball_radius = self.safe_ball_radius if self.with_noise else self.base_sep_dist
        safe_ball_radius = self.base_sep_dist
        if pref_delta_v is not None or (self.log["closest_dists"][-1] < safe_ball_radius).any():
            sep_dists = self.base_sep_dist + self.get_compensating_term(nominal_eci)
            EL_debris_states = self.EL_states.copy()
            if self.with_noise:
                EL_debris_states[:, :, :3] += np.random.randn(*EL_debris_states[:, :, :3].shape) * self.obs_sigma
                EL_debris_states[:, :, 3:] += np.random.randn(*EL_debris_states[:, :, 3:].shape) * self.obs_lambda
            primal_lvlh = eci_to_lvlh(nominal_eci, np.expand_dims(primal_eci, axis=1))
            self._solve_vo(primal_lvlh, EL_debris_states, sep_dists, pref_delta_v, initial_guess)
        else:
            self._action_seq.extend([np.zeros((n_p, self.action_dim))] * self.plan_horizon)

        _t1 = time.time()
        self.log["solve_time"].append(_t1-_t0)


    def _solve_vo(self,
                  primal_states:np.ndarray, # lvlh, unit: m.
                  debris_states:np.ndarray, # equivalent linear of lvlh, unit: m.
                  sep_dists: typing.Optional[np.ndarray|float],
                  pref_delta_v: typing.Optional[np.ndarray]=None, 
                  initial_guess: typing.Optional[np.ndarray]=None):
        '''
            Solve VO and fill `_action_seq` with planed actions.
            Notice: `primal_states`, `debris_states` and `pref_delta_v` should be in LVLH coordinate.
        '''
        n_p = primal_states.shape[0]

        if self.in_plane_mode=="in_plane":
            self._voc = [inPlaneVoConstraint(primal_states[i], debris_states[i], sep_dists) for i in range(n_p)]
        elif self.in_plane_mode=="along_track":
            self._voc = [inPlaneVoConstraint(primal_states[i], debris_states[i], sep_dists, along_track=True) for i in range(n_p)]
        else:
            self._voc = [voConstraint(primal_states[i], debris_states[i], sep_dists) for i in range(n_p)]
        
        self._opt_res = []
        for i in range(n_p):
            _pref_delta_v = None if pref_delta_v is None else pref_delta_v[i]
            if initial_guess is None:
                if len(self.log["opt_res"])>0 and self.log["opt_res"][-1][i].success:
                    opt_res, _ = self._voc[i].solve(initial_guess=self.delta_v_not_completed[i], pref_delta_v=_pref_delta_v)
                else:
                    opt_res, _ = self._voc[i].solve_batch(self.defualt_initial_guess, pref_delta_v=_pref_delta_v)
            else:
                opt_res, _ = self._voc[i].solve(initial_guess=initial_guess[i], pref_delta_v=_pref_delta_v)
            self._opt_res.append(opt_res)
        self.log["opt_res"].append(self._opt_res)

        delta_v = [res.x*1e3 for res in self._opt_res] # km -> m
        delta_v = np.concatenate(delta_v, axis=0).reshape(n_p, self.space_dim) # shape: (n_p, space_dim)
        total_actions = delta_v/(self.dt*self.max_thrust) # normalize
        self._action_seq.clear()
        for step in range(self.plan_horizon): # fill _action_seq
            total_actions_norm = norm(total_actions, keepdim=True)
            total_actions_norm = np.maximum(total_actions_norm, 1e-8) # in case divide zero
            actions = np.where(total_actions_norm>1, total_actions/total_actions_norm, total_actions)
            self._action_seq.appendleft(actions)
            total_actions = total_actions - actions
        self.delta_v_not_completed = total_actions*(self.dt*self.max_thrust)

        # record "inevitable"
        maneuver_increment = np.linalg.norm(delta_v, axis=-1)
        tca = np.stack([_voc.TCA for _voc in self._voc])
        TIME_BUFFER = 5.
        tca = np.where(tca>TIME_BUFFER, tca, 1e6)
        min_tca = np.min(tca, axis=-1)
        inevitable = (maneuver_increment > self.max_thrust*min_tca)
        self.log["inevitable"].append(inevitable)