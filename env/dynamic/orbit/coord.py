import numpy as np

def eci_to_lvlh_mat(eci_states:np.ndarray):
    '''
        args:
            `eci_states`: shape: (batch_size, 6)
        returns:
            `R`: rotation matrix, shape: (batch_size, 3, 3)
    '''
    if eci_states.ndim == 1:
        eci_states = eci_states.reshape((1, 6))
    r = eci_states[:, :3]
    v = eci_states[:, 3:]
    rxv = np.cross(r, v)

    # Calculate LVLH unit vectors. Shape: (batch_size, 3)
    x_lvlh = r / np.linalg.norm(r, axis=-1, keepdims=True)
    z_lvlh = rxv / np.linalg.norm(rxv, axis=-1, keepdims=True)
    y_lvlh = np.cross(z_lvlh, x_lvlh)

    # Rotation matrix
    R = np.stack([x_lvlh, y_lvlh, z_lvlh], axis=-1)
    return R.swapaxes(-1, -2) # transpose

def eci_to_lvlh(main_eci_states:np.ndarray, sub_eci_states:np.ndarray):
    '''
        args:
            `main_eci_states`: shape: (batch_size, 6)
            `sub_eci_states`: shape: (batch_size, n, 6)
        returns:
            `lvlh_states`: `sub_eci_states`'s coordinate in corresponding LVLH.
    '''
    if main_eci_states.ndim == 1:
        main_eci_states = main_eci_states.reshape((1, 6))
    if sub_eci_states.ndim == 1:
        sub_eci_states = sub_eci_states.reshape((1, 1, 6))
    elif sub_eci_states.ndim == 2:
        sub_eci_states = sub_eci_states.reshape((1, -1, 6))
    batch_size, n, _ = sub_eci_states.shape
    rel_states = sub_eci_states - main_eci_states.reshape((batch_size, 1, 6)) # relative states
    R = eci_to_lvlh_mat(main_eci_states).reshape((batch_size, 1, 3, 3))
    r = rel_states[:, :, :3].reshape((batch_size, n, 3, 1))
    v = rel_states[:, :, 3:].reshape((batch_size, n, 3, 1))
    r_lvlh = np.matmul(R, r).reshape((batch_size, n, 3))
    v_lvlh = np.matmul(R, v).reshape((batch_size, n, 3))
    lvlh_states = np.concatenate([r_lvlh, v_lvlh], axis=-1)
    return lvlh_states

def lvlh_to_eci(main_eci_states:np.ndarray, lvlh_states:np.ndarray):
    '''
        Notice: input should be of same unit.
        args:
            `main_eci_states`: shape: (batch_size, 6)
            `lvlh_states`: shape: (batch_size, n, 6)
        returns:
            `eci_states`: `lvlh_states`'s coordinate in ECI.
    '''
    if main_eci_states.ndim == 1:
        main_eci_states = main_eci_states.reshape((1, 6))
    if lvlh_states.ndim == 1:
        lvlh_states = lvlh_states.reshape((1, 1, 6))
    elif lvlh_states.ndim == 2:
        lvlh_states = lvlh_states.reshape((1, -1, 6))
    batch_size, n, _ = lvlh_states.shape
    R_inv = eci_to_lvlh_mat(main_eci_states)
    # R = np.linalg.inv(R_inv).reshape((batch_size, 1, 3, 3))
    R = R_inv.transpose((0,2,1)).reshape((batch_size, 1, 3, 3))
    r = lvlh_states[:, :, :3].reshape((batch_size, n, 3, 1))
    v = lvlh_states[:, :, 3:].reshape((batch_size, n, 3, 1))
    rel_eci_r = np.matmul(R, r).reshape((batch_size, n, 3))
    rel_eci_v = np.matmul(R, v).reshape((batch_size, n, 3))
    rel_eci_states = np.concatenate([rel_eci_r, rel_eci_v], axis=-1)
    eci_states = main_eci_states.reshape((batch_size, 1, 6)) + rel_eci_states
    return eci_states