import torch
import numpy as np
import typing

def affine(x, s0, t0, s1, t1):
    '''
        return x linearly transformed from [s0, t0] to [s1, t1].
    '''
    return ((x-s0)*t1 + (t0-x)*s1)/(t0-s0)

def dotEachRow(a:np.ndarray|torch.Tensor, b:np.ndarray|torch.Tensor, keepdim=False):
    prod = a*b
    if isinstance(a, torch.Tensor):
        dot = torch.sum(prod, dim=-1, keepdim=keepdim)
    else:
        dot = np.sum(prod, axis=-1, keepdims=keepdim)
    return dot

def norm(a:np.ndarray|torch.Tensor, dim=-1, keepdim=False):
    if isinstance(a, torch.Tensor):
        norm = torch.norm(a, dim=dim, keepdim=keepdim)
    else:
        norm = np.linalg.norm(a, axis=dim, keepdims=keepdim)
    return norm

def obs_padding(obss:typing.List[torch.Tensor|np.ndarray], max_len:typing.Optional[int]=None, pad_value=0., stack=True):
    batch_size = len(obss)
    _, n_feature = obss[0].shape
    seq_lens = [obs.shape[0] for obs in obss]
    max_seq_len = max(seq_lens)
    if max_len is None:
        max_len = max_seq_len
    else:
        max_len = max(max_len, max_seq_len)
    padded = []
    obs = obss[0]
    if isinstance(obs, torch.Tensor):
        for obs in obss:
            padded.append(torch.cat((obs, torch.full((max_len-obs.shape[0], *obs.shape[1:]), pad_value, device=obs.device)), dim=0))
        if stack:
            padded = torch.stack(padded, dim=0)
    elif isinstance(obs, np.ndarray):
        for obs in obss:
            padded.append(np.concatenate((obs, np.full((max_len-obs.shape[0], *obs.shape[1:]), pad_value, dtype=obs.dtype)), axis=0))
        if stack:
            padded = np.stack(padded, axis=0)
    else:
        raise TypeError("Elements of `obss` must be torch.Tensor or np.ndarray")
    return padded

def toTorch(x:np.ndarray|typing.Iterable[np.ndarray], device=None):
        if isinstance(x, torch.Tensor):
            return x.to(device=device)
        elif isinstance(x, np.ndarray):
            if x.ndim == 1:
                x = x.reshape(1,-1)
            return torch.from_numpy(x).float().to(device=device)
        else:
            return [toTorch(_x) for _x in x]
    
def toNumpy(x:torch.Tensor|typing.Iterable[torch.Tensor]):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.squeeze(dim=-1).detach().cpu().numpy()
    else:
        return [toNumpy(_x) for _x in x]
    
def traj_refine(x:np.ndarray, num=10):
    horizon1 = x.shape[0]
    shape = [(horizon1-1)*num+1] + list(x.shape[1:])
    x_ = np.zeros(shape)
    for i in range(horizon1-1):
        interpolated = np.linspace(x[i], x[i+1], num, endpoint=False)
        x_[i*num:(i+1)*num, ...] = interpolated
    x_[-1, ...] = x[-1, ...]
    return x_