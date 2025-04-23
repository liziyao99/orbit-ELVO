import torch
import numpy as np
import scipy.optimize
import typing

from utils import dotEachRow

class voConstraint:
    def __init__(self, primal_state:torch.Tensor|np.ndarray, obstacle_states:torch.Tensor|np.ndarray, sep_dists:torch.Tensor|np.ndarray|float):
        if isinstance(primal_state, torch.Tensor):
            primal_state = primal_state.detach().cpu().numpy()
        elif not isinstance(primal_state, np.ndarray):
            primal_state = np.array(primal_state)
        if isinstance(obstacle_states, torch.Tensor):
            obstacle_states = obstacle_states.detach().cpu().numpy()
        elif not isinstance(obstacle_states, np.ndarray):
            obstacle_states = np.array(obstacle_states)
        primal_state = primal_state.reshape((1,-1)).astype(np.float64)
        if primal_state.shape[-1]%2 != 0:
            raise ValueError("state dim must be even.")
        space_dim = int(primal_state.shape[-1]/2)
        obstacle_states = obstacle_states.reshape((-1,space_dim*2)).astype(np.float64)
        n_o = obstacle_states.shape[0]
        if isinstance(sep_dists, torch.Tensor):
            sep_dists = sep_dists.detach().cpu().numpy()
        elif not isinstance(sep_dists, typing.Iterable):
            sep_dists = np.array([sep_dists]*n_o, dtype=np.float64)
        elif not isinstance(sep_dists, np.ndarray):
            sep_dists = np.array(sep_dists)
        sep_dists = sep_dists.flatten().astype(np.float64)

        self.space_dim = space_dim
        self.n_o = n_o
        self.primal_state = primal_state
        self.obstacle_states = obstacle_states
        self.sep_dists = sep_dists
        
        self.r_p, self.v_p = primal_state[:, :space_dim], primal_state[:, space_dim:]
        self.r_o, self.v_o = obstacle_states[:, :space_dim], obstacle_states[:, space_dim:]
        self.r_rel, self.v_rel = self.r_p-self.r_o, self.v_p-self.v_o

    @property
    def TCA(self) -> np.ndarray:
        '''
            Time of Closest Approach.
        '''
        tca = -dotEachRow(self.r_rel, self.v_rel)/dotEachRow(self.v_rel, self.v_rel)
        tca = np.maximum(tca, 0.)
        return tca

    @property
    def TCA_infos(self) -> typing.Dict[str, np.ndarray]:
        '''
            Time of Closest Approach infomations.
        '''
        tca = self.TCA
        ca = self.r_rel + self.TCA[:, None]*self.v_rel
        cd = np.linalg.norm(ca[:, :3], axis=-1)
        infos = {
            "TCA": tca,
            "closest_approaches": ca,
            "closest_distances": cd
        }
        return infos
    
    @property
    def VO_infos(self) -> typing.Dict[str, np.ndarray]:
        '''
            VO infomations.
        '''
        r = np.linalg.norm(self.r_rel, axis=-1)
        v = np.linalg.norm(self.v_rel, axis=-1)
        sin_phi = np.clip(self.sep_dists/r, a_min=None, a_max=1-1e-11) # angle of cone
        cos_phi = np.sqrt(1.-sin_phi**2)
        cos_theta = dotEachRow(self.r_rel, self.v_rel)/(r*v) # angle between r and v
        infos = {
            "cos_phi": cos_phi,
            "cos_theta": cos_theta,
            "in_VO": cos_phi+cos_theta<=0
        }
        return infos


    @staticmethod
    def get_constraint_fun(_r_rel:np.ndarray, _v_rel:np.ndarray, sep_dist:float|np.ndarray):
        '''
        staticmethod
            args:
                `r_rel`, `v_rel`: shape: (space_dim, ) or (n_obstacle, space_dim) or (batch_size, n_obstacle, space_dim)
                `sep_dist`: float or array of shape (n_obstacle, ) or (batch_size, n_obstacle)
            returns:
                `func`: a function that takes `delta_v` and returns the constraint value.
        '''
        def func(delta_v:np.ndarray) -> float|np.ndarray:
            '''
                args:
                    `delta_v`: shape: (space_dim, ) or (batch_size, 1, space_dim)
                returns:
                    `cos_phi+cos_theta`: where `phi` is the angle of relative collision cone (CC),
                    and `theta` is the angle between `r_rel` and `v_rel`.
                    Constraint should be `pi-theta>phi`, which is identical to `cos_phi+cos_theta>0`.
            '''
            r_rel = _r_rel
            v_rel = _v_rel+delta_v
            r = np.linalg.norm(r_rel, axis=-1)
            v = np.linalg.norm(v_rel, axis=-1)
            sin_phi = np.clip(sep_dist/r, a_min=None, a_max=1-1e-11) # angle of cone
            cos_phi = np.sqrt(1.-sin_phi**2)
            cos_theta = np.sum(r_rel*v_rel, axis=-1)/(r*v) # angle between r and v
            return cos_phi+cos_theta
        return func
    
    def get_constraint(self, idx:int):
        func = self.get_constraint_fun(self.r_rel[idx], self.v_rel[idx], self.sep_dists[idx])
        constraint = scipy.optimize.NonlinearConstraint(func, lb=0., ub=np.inf)
        return constraint
    
    def get_additional_constraint(self, *args, **kwargs):
        return []
    
    @staticmethod
    def get_opt_fun(pref_delta_v:np.ndarray):
        return lambda x: np.linalg.norm(x-pref_delta_v)
    
    def check(self, x:np.ndarray|torch.Tensor, indices:typing.List[int]|None=None, pref_delta_v:np.ndarray|torch.Tensor|None=None):
        '''
            args:
                `x`: shape: (space_dim,)
                `indices`: indices of obstacles to check. If None, all obstacles will be checked.
            returns:
                `passed`: True if all constraints are passed
                `guess_reward`: float, -tanh(opt_res.fun) if passed else -1
                `constraints_passed`: list of bool, True if the corresponding constraint is passed
        '''
        x = x.flatten()
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy().astype(np.float64)
        if indices is None:
            indices = list(range(self.n_o))
        pref_delta_v = np.zeros(self.space_dim) if pref_delta_v is None else pref_delta_v.flatten()
        if isinstance(pref_delta_v, torch.Tensor):
            pref_delta_v = pref_delta_v.detach().cpu().numpy().astype(np.float64)
        func = self.get_constraint_fun(self.r_rel[indices], self.v_rel[indices], self.sep_dists[indices])
        constraints_passed = func(x)>0
        passed = constraints_passed.all()
        guess_reward = -np.tanh(self.get_opt_fun(pref_delta_v)(x)) if passed else -1.
        return passed, guess_reward, constraints_passed
    
    def solve(self, initial_guess:np.ndarray|torch.Tensor|None=None, pref_delta_v:np.ndarray|torch.Tensor|None=None):
        '''
            args:
                `initial_guess`: shape: (space_dim,)
                `pref_delta_v`: preferable delta velocity. Optimized func will be `norm(x-pref_delta_v)`. shape: (space_dim,)
            returns:
                `opt_res`: scipy.optimize.OptimizeResult
                `guess_reward`: float, -tanh(opt_res.fun) if success else -1
        '''
        constraints = [self.get_constraint(i) for i in range(self.n_o)]
        constraints += self.get_additional_constraint()
        pref_delta_v = np.zeros(self.space_dim) if pref_delta_v is None else pref_delta_v.flatten()
        if isinstance(pref_delta_v, torch.Tensor):
            pref_delta_v = pref_delta_v.detach().cpu().numpy().astype(np.float64)
        initial_guess = pref_delta_v if initial_guess is None else initial_guess.flatten().astype(np.float64)
        if isinstance(initial_guess, torch.Tensor):
            initial_guess = initial_guess.detach().cpu().numpy().astype(np.float64)
        opt_res = scipy.optimize.minimize(self.get_opt_fun(pref_delta_v), initial_guess, constraints=constraints)
        guess_reward = -np.tanh(opt_res.fun) if opt_res.success else -1.
        return opt_res, guess_reward
    
    def solve_batch(self, initial_guesses:typing.Iterable, pref_delta_v:np.ndarray|torch.Tensor|None=None):
        '''
        solve with a batch of initial guesses.
            args:
                `initial_guesses`: List of initial guess
                `pref_delta_v`: preferable delta velocity. Optimized func will be `norm(x-pref_delta_v)`. shape: (space_dim,)
            returns:
                `opt_res`: best scipy.optimize.OptimizeResult
                `(Opt_res, Guess_reward, best_ig, best_idx, best_reward)`: all opt res and rewards, best initial guess, best's index and its reward.
        '''
        constraints = [self.get_constraint(i) for i in range(self.n_o)]
        constraints += self.get_additional_constraint()
        pref_delta_v = np.zeros(self.space_dim) if pref_delta_v is None else pref_delta_v.flatten()
        if isinstance(pref_delta_v, torch.Tensor):
            pref_delta_v = pref_delta_v.detach().cpu().numpy().astype(np.float64)
        if isinstance(initial_guesses, torch.Tensor):
            initial_guesses = initial_guesses.detach().cpu().numpy().astype(np.float64)
        elif isinstance(initial_guesses, np.ndarray):
            initial_guesses = initial_guesses.astype(np.float64)
        elif isinstance(initial_guesses[0], torch.Tensor):
            initial_guesses = [ig.detach().cpu().numpy().astype(np.float64) for ig in initial_guesses]
        Opt_res = []
        Guess_reward = []
        best_ig = None
        best_idx = None
        best_reward = -1.
        for ig in initial_guesses:
            opt_res = scipy.optimize.minimize(self.get_opt_fun(pref_delta_v), ig, constraints=constraints)
            guess_reward = -np.tanh(opt_res.fun) if opt_res.success else -1.
            Opt_res.append(opt_res)
            Guess_reward.append(Guess_reward)
            if guess_reward >= best_reward:
                best_reward = guess_reward
                best_ig = ig
                best_idx = len(Opt_res)-1
        return Opt_res[best_idx], (Opt_res, Guess_reward, best_ig, best_idx, best_reward)
    
class inPlaneVoConstraint(voConstraint):
    def __init__(self, primal_state, obstacle_states, sep_dists, along_track=False):
        super().__init__(primal_state, obstacle_states, sep_dists)
        self.along_track = along_track
        self.posi_along_track = True

    def get_additional_constraint(self, *args, **kwargs):
        cons = [scipy.optimize.LinearConstraint(np.array([0., 0., 1.]), lb=0., ub=0.)]
        if self.along_track:
            cons.append(scipy.optimize.LinearConstraint(np.array([1., 0., 0.]), lb=0., ub=0.))
        if self.posi_along_track:
            cons.append(scipy.optimize.LinearConstraint(np.array([0., 1., 0.]), lb=0.))
        return cons