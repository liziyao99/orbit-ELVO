import torch
import numpy as np
from policy.obc import outputBoundConfig

class basePropagator:
    def __init__(self, state_dim:int, obs_dim:int, action_dim:int, dt:float) -> None:
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dt = dt
        self.action_box = outputBoundConfig([ 1]*action_dim, [-1]*self.action_dim)
        self.max_action_norm = self.action_box.max_norm

    def getNextStates(self, states:np.ndarray, actions:torch.Tensor|np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getObss(self, states:np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def getRewards(self, states:np.ndarray, actions:torch.Tensor|np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def getDones(self, states:np.ndarray) -> torch.Tensor:
        raise NotImplementedError
    
    def _propagate(self, 
                   states:np.ndarray, 
                   actions:torch.Tensor|np.ndarray):
        '''
            returns: `next_states,`, `rewards`, `dones`, `next_obss`
        '''
        next_states = self.getNextStates(states, actions)
        rewards = self.getRewards(next_states, actions)
        dones = self.getDones(next_states)
        next_obss = self.getObss(next_states)
        return next_states, rewards, dones, next_obss
    
    def clip_action(self, actions:torch.Tensor|np.ndarray) -> np.ndarray:
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)
        actions = self.action_box.clip(actions.cpu())
        actions = actions.numpy()
        return actions