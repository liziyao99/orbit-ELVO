import numpy as np

from poliastro.core.propagation import func_twobody
from poliastro.twobody.propagation import CowellPropagator

# J2 perturbation constants
J2 = 1.08263e-3  # Earth's second zonal harmonic
R_Earth = 6378.1  # Earth's radius in km

def j2_accel(t0, state, k):
    x, y, z, vx, vy, vz = state
    r = np.sqrt(x**2 + y**2 + z**2)
    factor = 1.5 * J2 * (R_Earth**2) * k / (r**5)
    ax_j2 = factor * x * (5 * (z**2 / r**2) - 1)
    ay_j2 = factor * y * (5 * (z**2 / r**2) - 1)
    az_j2 = factor * z * (5 * (z**2 / r**2) - 3)
    return np.array([ax_j2, ay_j2, az_j2])

def j2_f(t0, state, k):
    du_kep = func_twobody(t0, state, k)
    ax_j2, ay_j2, az_j2 = j2_accel(t0, state, k)
    du_j2 = np.array([0, 0, 0, ax_j2, ay_j2, az_j2])
    return du_kep + du_j2

def cowell_j2(rtol=1e-11):
    return CowellPropagator(rtol=rtol, f=j2_f)