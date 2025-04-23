import numpy as np

class basePolicy:
    def __init__(self, obs_dim:int, action_dim:int) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def act(self, obs):
        raise NotImplementedError
    
    def criticize(self, obs, action):
        raise NotImplementedError
    
    def act_criticize(self, obs, *args, **kwargs):
        raise NotImplementedError
    
    def update(self, *args, **kwargs):
        '''
            `basePolicy` need no update
        '''
        pass

    def start(self, *args, **kwargs):
        '''
            call `start` before running a simulation.
        '''
        pass

class nullPolicy(basePolicy):
    def __init__(self, obs_dim: int, action_dim: int, *args, **kwargs) -> None:
        super().__init__(obs_dim, action_dim)

    def act(self, obs:np.ndarray, *args, **kwargs):
        batch_size = obs.shape[0]
        actions = np.zeros((batch_size, self.action_dim))
        return actions, actions
    
    def criticize(self, obs:np.ndarray, *args, **kwargs):
        batch_size = obs.shape[0]
        values = np.zeros((batch_size, 1))
        return values
    
    def act_criticize(self, obs, *args, **kwargs):
        _action = self.act(obs, *args, **kwargs)
        _critic = self.criticize(obs, *args, **kwargs)
        return _action, _critic
    
class OUNoise(basePolicy):
    def __init__(self, obs_dim: int, action_dim: int, theta=1e-2, sigma=2e-2) -> None:
        super().__init__(obs_dim, action_dim)
        self.OU_noise = np.random.randn((0, action_dim))
        self.OU_mu = np.zeros((1, action_dim))
        self.OU_theta = theta*np.ones((1, action_dim))
        self.OU_sigma = sigma*np.ones((1, action_dim))

    def init_OU_noise(self, size, scale=1.):
        self.OU_noise = np.random.randn((size, self.action_dim))*scale

    def propagate_OU_noise(self):
        delta_noise = self.OU_theta*(self.OU_mu-self.OU_noise)+self.OU_sigma*np.random.randn(self.OU_noise.shape)
        self.OU_noise = self.OU_noise+delta_noise
        return self.OU_noise
    
    def test_OU_noise(self, size=1, init_scale=1., horizon=1000):
        noise = np.zeros((horizon, self.action_dim))
        self.init_OU_noise(size=size, scale=init_scale)
        for i in range(horizon):
            noise[i] = self.propagate_OU_noise()
        return noise
    
    def act(self, obs:np.ndarray, *args, **kwargs):
        assert obs.shape[0]==self.OU_noise.shape[0]
        noise = self.propagate_OU_noise()
        return noise, noise
    
    def start(self, obs0:np.ndarray, *args, **kwargs):
        size = obs0.shape[0]
        self.init_OU_noise(size=size, scale=1.)
        