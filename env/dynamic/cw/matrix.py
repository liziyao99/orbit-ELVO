import numpy as np
from numpy import sin, cos
import scipy
import scipy.linalg

MU_EARTH = 3.986004418E14 # m3s-2
'''
    Earth's standard gravitational parameter: 
    `MU_EARTH` = 3.986004418E14 m^3 s^(-2).
'''

from ..orbit.utils import get_orbit_elements

def CW_StateMat_batch(a:np.ndarray, 
                      mu = None
                      ) -> np.ndarray:
    '''
        args:
            `a`: radius of the target's circular orbit, m
            `mu`: standard gravitational parameter, m3s-2
        return:
            `A`: state matrix of CW equation, np array of shape [6,6]
    '''
    if isinstance(a, np.ScalarType):
        a = np.array([a], dtype=np.float64)
    if mu is None:
        mu = np.ones_like(a, dtype=np.float64)*MU_EARTH
    elif isinstance(mu, np.ScalarType):
        mu = np.array([mu], dtype=np.float64)
    n = np.sqrt(mu/a**3)
    zero = np.zeros_like(n)
    A_rr = np.zeros((3, 3, *a.shape))
    A_rv = np.tile(np.eye(3), (*a.shape, 1, 1))
    A_rv = np.moveaxis(A_rv, [-2,-1], [0,1])
    A_vr = np.array([
            [3*n**2, zero, zero],
            [zero, zero, zero],
            [zero, zero, -n**2]
        ])
    A_vv = np.array([
            [zero, 2*n, zero],
            [-2*n, zero, zero],
            [zero, zero, zero]
        ])
    A = np.vstack([
            np.hstack([A_rr, A_rv]),
            np.hstack([A_vr, A_vv])
        ]).astype(np.float32) # shape (6, 6, *batch_shape)
    A = np.moveaxis(A, [0,1], [-2,-1]) # shape (*batch_shape, 6, 6)
    return A

def CW_TransMat_batch(t0:np.ndarray|float,
                      t1:np.ndarray|float,
                      a:np.ndarray|float,
                      mu:np.ndarray|None=None):
    '''
        all args are of shape (batch_size,). Return array of shape (batch_size, 6, 6), 
        corresponding to each entry.
    '''
    if isinstance(t0, np.ScalarType):
        t0 = np.array([t0], dtype=np.float64)
    if isinstance(t1, np.ScalarType):
        t1 = np.array([t1], dtype=np.float64)
    if isinstance(a, np.ScalarType):
        a = np.array([a], dtype=np.float64)
    if mu is None:
        mu = np.ones_like(t0, dtype=np.float64)*MU_EARTH
    elif isinstance(mu, np.ScalarType):
        mu = np.array([mu], dtype=np.float64)
    dt = (t1-t0)
    if isinstance(dt, np.ndarray):
        dt = dt.astype(np.float64)
    else:
        dt = np.array(dt, dtype=np.float64)
    if isinstance(a, np.ndarray):
        a = a.astype(np.float64)
    else:
        a = np.array(a, dtype=np.float64)
    # NOTE: below have small divide small, use float64 in case loss of accuracy.
    n = np.sqrt(mu/a**3)
    tt = n*dt
    c = cos(tt)
    s = sin(tt)
    zero = np.zeros_like(tt)
    one = np.ones_like(tt)
    Phi_rr = np.array([
            [4-3*c, zero, zero],
            [6*(s-tt), one, zero],
            [zero, zero, c]
        ]) # shape (3, 3, *batch_size)
    Phi_rv = np.array([
            [s/n, 2*(1-c)/n, zero],
            [2*(c-1)/n, (4*s-3*tt)/n, zero],
            [zero, zero, s/n]
        ])
    Phi_vr = np.array([
            [3*n*s, zero, zero],
            [6*n*(c-1), zero, zero],
            [zero, zero, -n*s]
        ])
    Phi_vv = np.array([
            [c, 2*s, zero],
            [-2*s, 4*c-3, zero],
            [zero, zero, c]
        ])
    Phi = np.vstack((
            np.hstack((Phi_rr, Phi_rv)),
            np.hstack((Phi_vr, Phi_vv))
        )).astype(np.float32) # shape (6, 6, *batch_shape)
    Phi = np.moveaxis(Phi, [0,1], [-2,-1]) # shape (*batch_shape, 6, 6)
    return Phi

def CW_constConVecs(t0:float,
                    t :float,
                    u :np.ndarray,
                    a :float,
                    mu = MU_EARTH
                    ):
    '''
        `u` is of shape (population, 3).\n
        return $\\int_{0}^{t} \Phi(t-\\tau) B u(\\tau) d\\tau$, shape (population, 6).
    '''
    dt = t-t0
    n = np.sqrt(mu/a**3)
    tt = n*dt
    c = cos(tt)
    s = sin(tt)
    trans_u = np.vstack((
        (u[:,0]*(1-c) + 2*u[:,1]*(tt-s))/n**2,
        (8*u[:,1]*(1-c) - 3*u[:,1]*tt**2 + 4*u[:,0]*(s-tt))/(2*n**2),
        (u[:,2]*(1-c))/n**2,
        (2*u[:,1]*(1-c) + u[:,0]*s)/n,
        (2*u[:,0]*(c-1) - 3*u[:,1]*tt + 4*u[:,1]*s)/n,
        (u[:,2]*s)/n
    )).T.astype(np.float32)
    return trans_u

def CW_discreteConMat(t0:np.ndarray|float,
                  t1 :np.ndarray|float,
                  a :np.ndarray|float,
                  mu = MU_EARTH
                  ):
    if isinstance(t0, np.ScalarType):
        t0 = np.array([t0], dtype=np.float64)
    if isinstance(t1, np.ScalarType):
        t1 = np.array([t1], dtype=np.float64)
    if isinstance(a, np.ScalarType):
        a = np.array([a], dtype=np.float64)
    if mu is None:
        mu = np.ones_like(t0, dtype=np.float64)*MU_EARTH
    elif isinstance(mu, np.ScalarType):
        mu = np.array([mu], dtype=np.float64)
    dt = (t1-t0)
    if isinstance(dt, np.ndarray):
        dt = dt.astype(np.float64)
    if isinstance(a, np.ndarray):
        a = a.astype(np.float64)
    # NOTE: below have small divide small, use float64 in case loss of accuracy.
    n = np.sqrt(mu/a**3)
    tt = n*dt
    c = cos(tt)
    s = sin(tt)
    zero = np.zeros_like(tt)
    Int_Phi = np.array([
        [(4*tt-3*s)/n, zero, zero, (1-c)/n**2, 2*(tt-s)/n**2, zero],
        [(6*(1-c)-3*tt**2)/n, dt, zero, 2*(s-tt)/n**2, (8*(1-c)-3*tt**2)/(2*n**2), zero],
        [zero, zero, s/n, zero, zero, (1-c)/n**2],
        [3-3*c, zero, zero, s/n, (2-2*c)/n, zero],
        [6*(s-tt), zero, zero, (2*c-2)/n, (4*s-3*tt)/n, zero],
        [zero, zero, c-1, zero, zero, s/n]
    ]) # Integrate of Phi(dt-tau) over [0, dt]. Shape: (6, 6, *batch_shape)
    Int_Phi = np.moveaxis(Int_Phi, [0,1], [-2,-1]) # shape (*batch_shape, 6, 6)
    B = np.vstack((np.zeros((3,3)), np.eye(3)))
    B = np.tile(B, ((*Int_Phi.shape[:-2], 1, 1)))
    return Int_Phi@B

def elliptical_StateMat_batch_fromECI(eci_states:np.ndarray):
    orbital_elements = get_orbit_elements(eci_states).astype(np.float64)
    a = orbital_elements[:, 0] * 1e3 # km -> m
    ecc = orbital_elements[:, 1]
    phi = orbital_elements[:, 5]
    return elliptical_StateMat_batch(a, ecc, phi)

def elliptical_StateMat_batch(a:np.ndarray, ecc:np.ndarray, phi:np.ndarray):
    '''
        args:
            `a`: semi-major axis, unit: m.
            `ecc`: eccentricity.
            `phi`: true anomaly.
    '''
    a = a.astype(np.float64)
    ecc = ecc.astype(np.float64)
    phi = phi.astype(np.float64)

    r = a*(1-ecc**2)/(1+ecc*np.cos(phi))
    n = np.sqrt( (MU_EARTH*(1+ecc*np.cos(phi))) / r**3 )
    epsilon = - (2*MU_EARTH*ecc*np.sin(phi)) / r**3
    zero = np.zeros_like(a)
    A_rr = np.zeros((3, 3, *a.shape))
    A_rv = np.tile(np.eye(3), (*a.shape, 1, 1)).transpose((1,2,0))
    A_vr = np.array([
            [n**2+2*MU_EARTH/r**3, -epsilon, zero],
            [epsilon, n**2-MU_EARTH/r**3, zero],
            [zero, zero, -MU_EARTH/r**3]
        ])
    A_vv = np.array([
            [zero, 2*n, zero],
            [-2*n, zero, zero],
            [zero, zero, zero]
        ])
    A = np.vstack([
            np.hstack([A_rr, A_rv]),
            np.hstack([A_vr, A_vv])
        ]).astype(np.float32) # shape (6, 6, *batch_shape)
    A = np.moveaxis(A, [0,1], [-2,-1]) # shape (*batch_shape, 6, 6)
    return A


def simple_continuousFeedback(space_dim:int, Q:np.ndarray|float=1., R:np.ndarray|float=1e3):
    '''
        args:
            `Q`: state weight matrix.
            `R`: control weight matrix.
        returns:
            `Kc`: Feedback matrix of continuous LQR. shape: (3, 6).
        '''
    A = np.vstack([
            np.hstack((np.zeros((space_dim, space_dim)), np.eye(space_dim))),
            np.hstack((np.zeros((space_dim, space_dim)), np.zeros((space_dim, space_dim))))
        ])
    B = np.vstack((np.zeros((space_dim,space_dim)), np.eye(space_dim)))
    if isinstance(Q, np.ScalarType):
        Q = Q * np.eye(2*space_dim) # state weight
    if isinstance(R, np.ScalarType):
        R = R * np.eye(space_dim) # control weight
    P = np.array(scipy.linalg.solve_continuous_are(A, B, Q, R))
    Kc = np.array(scipy.linalg.inv(R) @ (B.T @ P))
    return Kc

def simple_discreteFeedback(space_dim:int, dt:np.ndarray|float, Q:np.ndarray|float=1., R:np.ndarray|float=1e3):
    '''
        args:
            `dt`: time step of discrete system, unit: s.
            `orbit_rad`: radius of the target's circular orbit, unit: m.
            `Q`: state weight matrix.
            `R`: control weight matrix.
        returns:
            `Kd`: Feedback matrix of discrete LQR. shape: (3, 6).
    '''
    Phi = np.vstack([
            np.hstack(((np.eye(space_dim)), dt*np.eye(space_dim))),
            np.hstack((np.zeros((space_dim, space_dim)), np.eye(space_dim)))
        ])
    B = np.vstack((np.zeros((space_dim,space_dim)), dt*np.eye(space_dim)))
    if isinstance(Q, np.ScalarType):
        Q = Q * np.eye(2*space_dim) # state weight
    if isinstance(R, np.ScalarType):
        R = R * np.eye(space_dim) # control weight
    P = np.array(scipy.linalg.solve_discrete_are(Phi, B, Q, R))
    Kd = np.array(scipy.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ Phi))
    return Kd

def CW_continuousFeedback(orbit_rad:np.ndarray|float, Q:np.ndarray|float=1., R:np.ndarray|float=1e3):
    '''
        args:
            `orbit_rad`: radius of the target's circular orbit, unit: m.
            `Q`: state weight matrix.
            `R`: control weight matrix.
        returns:
            `Kc`: Feedback matrix of continuous LQR. shape: (..., 3, 6).
        '''
    if isinstance(orbit_rad, np.ScalarType):
        orbit_rad = np.array([orbit_rad], dtype=np.float64)
    shape = orbit_rad.shape
    orbit_rad = orbit_rad.flatten()
    A = CW_StateMat_batch(orbit_rad)
    n_p = A.shape[0]
    B = np.vstack((np.zeros((3,3)), np.eye(3)))
    if isinstance(Q, np.ScalarType):
        Q = Q * np.eye(6) # state weight
    if isinstance(R, np.ScalarType):
        R = R * np.eye(3) # control weight
    P = np.array([scipy.linalg.solve_continuous_are(A[i], B, Q, R) for i in range(n_p)])
    Kc = np.array([scipy.linalg.inv(R) @ (B.T @ P[i]) for i in range(n_p)])
    Kc = Kc.reshape((*shape, 3, 6))
    return Kc

def CW_discreteFeedback(dt:np.ndarray|float, orbit_rad:np.ndarray|float, Q:np.ndarray|float=1., R:np.ndarray|float=1e3):
    '''
        args:
            `dt`: time step of discrete system, unit: s.
            `orbit_rad`: radius of the target's circular orbit, unit: m.
            `Q`: state weight matrix.
            `R`: control weight matrix.
        returns:
            `Kd`: Feedback matrix of discrete LQR. shape: (..., 3, 6).
    '''
    if isinstance(orbit_rad, np.ScalarType):
        orbit_rad = np.array([orbit_rad], dtype=np.float64)
    shape = orbit_rad.shape
    orbit_rad = orbit_rad.flatten()
    Phi = CW_TransMat_batch(0, dt, orbit_rad)
    n_p = Phi.shape[0]
    B = CW_discreteConMat(0, dt, orbit_rad)
    if isinstance(Q, np.ScalarType):
        Q = Q * np.eye(6) # state weight
    if isinstance(R, np.ScalarType):
        R = R * np.eye(3) # control weight
    P = np.array([scipy.linalg.solve_discrete_are(Phi[i], B[i], Q, R) for i in range(n_p)])
    Kd = np.array([scipy.linalg.inv(R + B[i].T @ P[i] @ B[i]) @ (B[i].T @ P[i] @ Phi[i]) for i in range(n_p)])
    Kd = Kd.reshape((*shape, 3, 6))
    return Kd