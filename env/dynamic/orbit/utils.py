import astropy.units as U
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import CowellPropagator
import numpy as np
import typing
import scipy
import scipy.optimize

from .coord import lvlh_to_eci
from .J2 import cowell_j2

def orbit_to_eci(orbit:Orbit) -> np.ndarray:
    '''
        unit: km.
    '''
    r = orbit.r.value
    v = orbit.v.value
    eci_state = np.hstack((r, v))
    return eci_state

def eci_to_orbit(eci_states:np.ndarray) -> typing.List[Orbit]:
    '''
        unit: km.
    '''
    eci_states = eci_states.reshape((-1, 6))
    orbits = []
    for x in eci_states:
        r = x[:3] * U.km
        v = x[3:] * U.km / U.s
        orbit = Orbit.from_vectors(Earth, r, v)
        orbits.append(orbit)
    return orbits


def eci_propagate(eci_state, t, method=None):
    '''
        unit: km. NOT support batch.
    '''
    if method is None:
        method = CowellPropagator()
    orbit = eci_to_orbit(eci_state)[0]
    orbit = orbit.propagate(t*U.s, method=method)
    r = orbit.r.value
    v = orbit.v.value
    new_eci_state = np.hstack((r, v))
    return new_eci_state

def eci_propagate_batch(eci_state:np.ndarray, t:float|np.ndarray, method=None):
    '''
        unit: km. eci_state shape: (batch_size, 6).
    '''
    batch_size = eci_state.shape[0]
    if isinstance(t, np.ScalarType):
        t = np.array([t]*batch_size)
    new_eci_state = np.zeros_like(eci_state)
    for i in range(batch_size):
        new_eci_state[i] = eci_propagate(eci_state[i], t[i], method=method)
    return new_eci_state


def find_tca(orbit1:Orbit, orbit2:Orbit, time_bounds:tuple=None, prop_method=None):
    """
    计算两颗卫星轨道的最近接近时间(TCA)、距离及各自的状态矢量。

    参数
    ----------
    orbit1 : Orbit
        第一个卫星的轨道对象。
    orbit2 : Orbit
        第二个卫星的轨道对象。
    time_bounds : tuple (可选)
        时间搜索范围的上下界(秒)。默认为卫星最大轨道周期。
    prop_method: Propagator, 可选
        轨道使用的传播器。

    返回
    -------
    tca_time : astropy.Quantity
        最近接近时间(相对于输入的轨道历元)。
    min_distance : astropy.Quantity
        最近接近距离。
    r1, v1 : astropy.Quantity, astropy.Quantity
        第一个卫星在 TCA 时刻的位置和速度矢量(ECI)。
    r2, v2 : astropy.Quantity, astropy.Quantity
        第二个卫星在 TCA 时刻的位置和速度矢量(ECI)。
    """
    # 自动确定时间范围（若未提供）
    if time_bounds is None:
        T1 = orbit1.period.to(U.s).value
        T2 = orbit2.period.to(U.s).value
        max_period = max(T1, T2)
        t_lower = -max_period
        t_upper = max_period
    else:
        t_lower, t_upper = time_bounds

    if prop_method is None:
        prop_method = CowellPropagator()

    # 定义距离计算函数（无单位标量）
    def distance(t_sec):
        try:
            delta_t = t_sec * U.s
            new_o1 = orbit1.propagate(delta_t, method=prop_method)
            new_o2 = orbit2.propagate(delta_t, method=prop_method)
            r1 = new_o1.r.to(U.km).value
            r2 = new_o2.r.to(U.km).value
            return np.linalg.norm(r1 - r2)
        except Exception as e:
            return np.inf

    # 有界优化寻找最小距离
    result = scipy.optimize.minimize_scalar(
        distance,
        bounds=(t_lower, t_upper),
        method='bounded',
        options={'xatol': 1e-8}
    )

    if not result.success:
        raise RuntimeError(f"优化失败: {result.message}")

    # 提取 TCA 时刻的状态
    tca_delta = result.x * U.s
    orb1_at_tca = orbit1.propagate(tca_delta, method=prop_method)
    orb2_at_tca = orbit2.propagate(tca_delta, method=prop_method)

    # 获取位置和速度矢量
    r1 = orb1_at_tca.r
    v1 = orb1_at_tca.v
    r2 = orb2_at_tca.r
    v2 = orb2_at_tca.v

    return (
        tca_delta,
        result.fun * U.km,
        (r1, v1),
        (r2, v2)
    )


def get_orbit_elements(states:np.ndarray):
    '''
        args:
            `states`: shape: (batch_size, 6), unit: km, km/s.
        returns:
            `elements`: a, ecc, inc, raan, argp, nu. shape: (batch_size, 6), unit: km, km/s.
    '''
    states = states.reshape((-1, 6)).astype(np.float64)
    batch_size = states.shape[0]
    elements = np.zeros((batch_size, 6))
    for i in range(batch_size):
        orbit = Orbit.from_vectors(Earth, states[i, :3]*U.km, states[i, 3:]*U.km/U.s)
        a, ecc, inc, raan, argp, nu = orbit.a, orbit.ecc, orbit.inc, orbit.raan, orbit.argp, orbit.nu
        elements[i, :] = np.array([a.value, ecc.value, inc.value, raan.value, argp.value, nu.value])
    return elements

def collision_eci_orbit(primal_eci:np.ndarray, collision_vel:np.ndarray, t2c:np.ndarray|float,
                        collision_pos:typing.Optional[np.ndarray]=None):
    '''
        Notice: `primal_eci` and `collision_vel` should be in km. `collision_vel` should be in LVLH coordinate.
    '''
    if isinstance(t2c, np.ScalarType):
        t2c = np.array([t2c])
    t2c = t2c.astype(np.float64).flatten()
    primal_eci = primal_eci.reshape((-1, 6)).astype(np.float64)
    batch_size = primal_eci.shape[0]
    for i in range(batch_size):
        t = t2c[i] if isinstance(t2c, np.ndarray) else t2c
        orbit = Orbit.from_vectors(Earth, primal_eci[i, :3]*U.km, primal_eci[i, 3:]*U.km/U.s)
        r, v = orbit.propagate(t*U.s).rv()
        r, v = r.value, v.value
        primal_eci[i, :3] = r[...]
        primal_eci[i, 3:] = v[...]

    collision_vel = collision_vel.reshape((batch_size, -1, 3)) # km/s
    collision_pos = np.zeros_like(collision_vel) if collision_pos is None else collision_pos.reshape((batch_size, -1, 3)) # km
    collision_lvlh = np.concatenate((collision_pos, collision_vel), axis=-1)
    collision_eci = lvlh_to_eci(primal_eci, collision_lvlh)
    debris_eci = np.zeros_like(collision_eci)
    
    for i in range(batch_size):
        t = t2c[i] if isinstance(t2c, np.ndarray) else t2c
        for j in range(collision_eci.shape[1]):
            orbit = Orbit.from_vectors(Earth, collision_eci[i, j, :3]*U.km, collision_eci[i, j, 3:]*U.km/U.s)
            r, v = orbit.propagate(-t*U.s).rv()
            r, v = r.value, v.value
            debris_eci[i, j, :3] = r[...]
            debris_eci[i, j, 3:] = v[...]
    return debris_eci

def beseige(n_phi=15, n_theta=20, remove_size=20, vel=0.2):
    # Define the number of points in each dimension
    phi, theta = np.linspace(0, np.pi, n_phi), np.linspace(0, 2*np.pi, n_theta)
    phi, theta = np.meshgrid(phi, theta)
    # Convert spherical coordinates to Cartesian coordinates
    x = vel * np.sin(phi) * np.cos(theta)
    y = vel * np.sin(phi) * np.sin(theta)
    z = vel * np.cos(phi)
    collision_vel = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    del_row = np.random.choice(np.arange(1, n_phi*n_theta), remove_size, replace=False)
    collision_vel = np.delete(collision_vel, del_row, axis=0)
    return collision_vel

class randomScenarioGenerator:
    '''
        NOT support batch.
    '''
    def __init__(self, 
                 n_d=[3, 11], 
                 safe_distance=[1e3, 2e3], 
                 tca=[300., 600.], 
                 tca_pos=[0., 2e3], 
                 tca_vel_x=[0, 1e3],
                 tca_vel_y=[0, 8e3],
                 tca_vel_z=[0, 8e3]):
        self.n_d = n_d
        self.safe_distace = safe_distance
        self.tca = tca
        self.tca_pos = tca_pos
        self.tca_vel_x = tca_vel_x
        self.tca_vel_y = tca_vel_y
        self.tca_vel_z = tca_vel_z

        self.pin_point_sigma = 10.

    def gen_scenario(self, primal_eci:np.ndarray, n_d:typing.Optional[int]=None, pin_point=True):
        '''
            args: `primal_eci`, `n_d`.
            returns: `debris_eci`: km, `safe_distance`: m, `(tca, tca_pos_lvlh, tca_vel_lvlh)`: km.
        '''
        n_d = np.random.randint(*self.n_d) if n_d is None else n_d
        safe_distances = np.random.uniform(*self.safe_distace, size=(n_d,))
        tca = np.random.uniform(*self.tca, size=(n_d,))
        _shape = (n_d, 3)
        tca_pos_lvlh = np.random.uniform(*self.tca_pos, size=_shape) * np.where(np.random.rand(*_shape)>0.5, 1, -1) / 1e3
        if pin_point:
            tca_pos_lvlh[0, :] = np.random.normal(0, self.pin_point_sigma, size=(3,)) / 1e3
        tca_vel_x_lvlh = np.random.uniform(*self.tca_vel_x, size=(n_d, )) * np.where(np.random.rand(n_d)>0.5, 1, -1) / 1e3
        tca_vel_y_lvlh = np.random.uniform(*self.tca_vel_y, size=(n_d, )) * np.where(np.random.rand(n_d)>0.5, 1, -1) / 1e3
        tca_vel_z_lvlh = np.random.uniform(*self.tca_vel_z, size=(n_d, )) * np.where(np.random.rand(n_d)>0.5, 1, -1) / 1e3
        tca_vel_lvlh = np.stack((tca_vel_x_lvlh, tca_vel_y_lvlh, tca_vel_z_lvlh), axis=-1)
        debris_eci = self.get_collision_eci(primal_eci, collision_pos=tca_pos_lvlh, collision_vel=tca_vel_lvlh, t2c=tca)
        return debris_eci, safe_distances, (tca, tca_pos_lvlh, tca_vel_lvlh)

    def get_collision_eci(self, 
                          primal_eci:np.ndarray, 
                          collision_pos:np.ndarray, 
                          collision_vel:np.ndarray,
                          t2c:np.ndarray):
        '''
            Notice: `primal_eci` and `collision_vel` should be in km. `collision_vel` should be in LVLH coordinate.
        '''
        n_d = collision_pos.shape[0]
        t2c = t2c.astype(np.float64).flatten()
        primal_eci = primal_eci.reshape((6,)).astype(np.float64)
        pred_primal_eci = np.zeros((n_d, 6), dtype=np.float64)
        for j in range(n_d):
            t = t2c[j]
            orbit = Orbit.from_vectors(Earth, primal_eci[:3]*U.km, primal_eci[3:]*U.km/U.s)
            r, v = orbit.propagate(t*U.s, method=self._new_propagator()).rv()
            r, v = r.value, v.value
            pred_primal_eci[j, :3] = r[...]
            pred_primal_eci[j, 3:] = v[...]

        collision_pos = collision_pos.reshape((-1, 1, 3)) # km
        collision_vel = collision_vel.reshape((-1, 1, 3)) # km/s
        collision_lvlh = np.concatenate((collision_pos, collision_vel), axis=-1) # shape: (n_d, 1, 6)
        collision_eci = lvlh_to_eci(pred_primal_eci, collision_lvlh)
        collision_eci = collision_eci.reshape((n_d, 6))

        debris_eci = np.zeros((n_d, 6))
        for j in range(n_d):
            t = t2c[j]
            orbit = Orbit.from_vectors(Earth, collision_eci[j, :3]*U.km, collision_eci[j, 3:]*U.km/U.s)
            r, v = orbit.propagate(-t*U.s, method=self._new_propagator()).rv()
            r, v = r.value, v.value
            debris_eci[j, :3] = r[...]
            debris_eci[j, 3:] = v[...]
        return debris_eci
    
    def gen_flyby_scenario(self, primal_eci:np.ndarray, n_d:typing.Optional[int]=None):
        '''
            args: `primal_eci`, `n_d`.
            returns: `debris_eci`: km, `safe_distance`: m, `(tca, ca_pos, ca_vel)`: km.
        '''
        n_d = np.random.randint(*self.n_d) if n_d is None else n_d
        safe_distances = np.random.uniform(*self.safe_distace, size=(n_d,))
        tca = np.random.uniform(*self.tca, size=(n_d,))
        _shape = (n_d, 3)
        tca_pos_lvlh = np.zeros(_shape)
        tca_vel_x_lvlh = np.random.uniform(*self.tca_vel_x, size=(n_d, )) * np.where(np.random.rand(n_d)>0.5, 1, -1) / 1000.
        tca_vel_y_lvlh = np.random.uniform(*self.tca_vel_y, size=(n_d, )) * np.where(np.random.rand(n_d)>0.5, 1, -1) / 1000.
        tca_vel_z_lvlh = np.zeros(n_d, )
        tca_vel_lvlh = np.stack((tca_vel_x_lvlh, tca_vel_y_lvlh, tca_vel_z_lvlh), axis=-1)
        debris_eci = self.get_collision_eci(primal_eci, collision_pos=tca_pos_lvlh, collision_vel=tca_vel_lvlh, t2c=tca)
        return debris_eci, safe_distances, (tca, tca_pos_lvlh, tca_vel_lvlh)
    
    def gen_front_scenario(self,  primal_eci:np.ndarray, n_d:typing.Optional[int]=None, pin_point=True):
        n_d = np.random.randint(*self.n_d) if n_d is None else n_d
        safe_distances = np.random.uniform(*self.safe_distace, size=(n_d,))
        tca = np.linspace(200., 400., n_d)
        _shape = (n_d, 3)
        tca_pos_lvlh = np.zeros(_shape)
        tca_vel_x_lvlh = np.zeros(n_d, )
        tca_vel_y_lvlh = np.ones(n_d, ) * 0.1
        tca_vel_z_lvlh = np.zeros(n_d, )
        tca_vel_lvlh = np.stack((tca_vel_x_lvlh, tca_vel_y_lvlh, tca_vel_z_lvlh), axis=-1)
        debris_eci = self.get_collision_eci(primal_eci, collision_pos=tca_pos_lvlh, collision_vel=tca_vel_lvlh, t2c=tca)
        return debris_eci, safe_distances, (tca, tca_pos_lvlh, tca_vel_lvlh)
    
    
    def _new_propagator(self):
        return cowell_j2()