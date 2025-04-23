import torch
from utils import affine

class outputBoundConfig:
    def __init__(self, upper_bounds: torch.Tensor, lower_bounds: torch.Tensor) -> None:
        if type(upper_bounds) is not torch.Tensor:
            upper_bounds = torch.tensor(upper_bounds)
        if type(lower_bounds) is not torch.Tensor:
            lower_bounds = torch.tensor(lower_bounds)
        upper_bounds = upper_bounds.flatten().to(torch.float32)
        lower_bounds = lower_bounds.flatten().to(torch.float32)
        if upper_bounds.shape[0]!=lower_bounds.shape[0]:
            raise(ValueError("upper_bounds' and lower_bounds' shape incompatible."))
        self.n_output = upper_bounds.shape[0]
        self.upper_bounds = upper_bounds
        self.lower_bounds = lower_bounds
        self.activation_types = []
        for i in range(self.n_output):
            if torch.isfinite(self.upper_bounds[i]) and torch.isfinite(self.lower_bounds[i]):
                self.activation_types.append(0)
            elif torch.isfinite(self.upper_bounds[i]) and (not torch.isfinite(self.lower_bounds[i])):
                self.activation_types.append(1)
            elif (not torch.isfinite(self.upper_bounds[i])) and torch.isfinite(self.lower_bounds[i]):
                self.activation_types.append(2)
            else:
                self.activation_types.append(3)

    def activate(self, x:torch.Tensor, type:int, idx:int):
        if type==0: # both side
            # span = self.upper_bounds[idx].item()-self.lower_bounds[idx].item()
            return affine(torch.tanh(x), -1, 1, self.lower_bounds[idx].item(), self.upper_bounds[idx].item())
        elif type==1: # only upper
            return (-torch.relu(x)+self.upper_bounds[idx].item())
        elif type==2: # only lower
            return ( torch.relu(x)+self.lower_bounds[idx].item())
        else: # no bound
            return x
        
    def __call__(self, x:torch.Tensor):
        shape = x.shape
        x = x.view((-1, self.n_output))
        y = torch.zeros_like(x)
        for i in range(self.n_output):
            y[:,i] = self.activate(x[:,i], type=self.activation_types[i], idx=i)
        y = y.reshape(shape)
        return y
    
    def uniSample(self, size:int, indices=None):
        '''
            sample uniformally between output bounds.
        '''
        upper_bounds = self.upper_bounds
        lower_bounds = self.lower_bounds
        if indices is not None:
            upper_bounds = upper_bounds[indices]
            lower_bounds = lower_bounds[indices]
        if upper_bounds.isinf().any() or lower_bounds.isinf().any():
            raise(ValueError("output bounds are not finite."))
        return torch.rand(size, self.n_output).to(self.device)*torch.abs(upper_bounds-lower_bounds)+lower_bounds

    def clip(self, x):
        return torch.clamp(x, min=self.lower_bounds, max=self.upper_bounds)

    def to(self, device:str):
        self.upper_bounds = self.upper_bounds.to(device)
        self.lower_bounds = self.lower_bounds.to(device)
        return self
    
    @property
    def max_norm(self):
        bounds = torch.vstack((self.upper_bounds, self.lower_bounds))
        bounds = torch.abs(bounds)
        bigger = torch.max(bounds, dim=0)[0]
        norm = torch.norm(bigger)
        return norm.item()

    @property
    def device(self):
        return self.upper_bounds.device