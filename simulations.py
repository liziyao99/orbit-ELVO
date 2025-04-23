import numpy as np
import matplotlib.pyplot as plt
import rich.progress
from scipy.spatial.transform import Rotation

from env.propagator.orbitPropagator import orbitPropagator
from policy.vo.voPolicy import safeOrbitVoPolicy
from env.dynamic.orbit.utils import *

import poliastro
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
import astropy.units as U

def simulate(episode:int, 
             n_d:int, 
             horizon:int,
             prop:orbitPropagator, 
             policy:safeOrbitVoPolicy, 
             rsg:randomScenarioGenerator,
             _nominal_eci:np.ndarray=None):
    with rich.progress.Progress() as progress:
        task = progress.add_task(f"collision avoidance n_d={n_d}", total=episode)
        fails = 0
        Maneuver_increment = []
        Miss_distances = []
        Nominal_miss_distances = []
        Propagate_times = []
        Solve0_times = []
        Solve_times = []
        TD = []
        for i in range(episode):
            primal_eci = prop.initPrimalStates() # unit: km
            debris_eci, _, _ = rsg.gen_scenario(primal_eci, n_d=n_d) # unit: km
            debris_eci = debris_eci.reshape((1, n_d, 6))
            nominal_eci = primal_eci.copy() if _nominal_eci is None else _nominal_eci.copy()
            primal_obs, debris_obs = prop.getObss(primal_states=primal_eci, 
                                                  debris_states=debris_eci,
                                                  nominal_primal_states=nominal_eci)
            trans_dict = {"primal_eci": [],
                          "debris_eci": [],
                          "nominal_eci": [],
                          "actions": []}
            for step in range(horizon):
                action, _ = policy.act(primal_obs, debris_obs)
                (next_primal_eci, next_debris_eci, next_nominal_eci), _, _, (next_primal_obs, next_debris_obs) = prop._propagate(primal_eci, debris_eci, action, nominal_eci)

                trans_dict["primal_eci"].append(primal_eci)
                trans_dict["debris_eci"].append(debris_eci)
                trans_dict["nominal_eci"].append(nominal_eci)
                trans_dict["actions"].append(action)

                primal_eci, debris_eci, nominal_eci, primal_obs, debris_obs = next_primal_eci, next_debris_eci, next_nominal_eci, next_primal_obs, next_debris_obs

            TD.append(trans_dict)
            
            maneuver_increment, miss_distances, nominal_miss_distances, fail = get_metrics(trans_dict, prop.safe_dist, prop.dt, prop.max_thrust)
            Maneuver_increment.append(maneuver_increment)
            Miss_distances.append(miss_distances)
            Nominal_miss_distances.append(nominal_miss_distances)
            fails += fail

            Propagate_times.append(policy.log["propagate_time"])
            Solve0_times.append(policy.log["solve_time"][0])
            Solve_times.append(policy.log["solve_time"][1:])

            progress.advance(task, 1)
        fail_rate = fails/episode
    return TD, fail_rate, Maneuver_increment, Miss_distances, Nominal_miss_distances, Propagate_times, Solve0_times, Solve_times


def simulate2(episode:int, 
              n_d:int, 
              horizon:int,
              prop:orbitPropagator, 
              policy:safeOrbitVoPolicy, 
              rsg:randomScenarioGenerator,
              _nominal_eci:np.ndarray=None,
              deb_orbit:Orbit=None,
              deb_safe_dist:float=None,):
    with rich.progress.Progress() as progress:
        task = progress.add_task(f"collision avoidance n_d={n_d}", total=episode)
        fails = 0
        Maneuver_increment = []
        Miss_distances = []
        Nominal_miss_distances = []
        Propagate_times = []
        Solve0_times = []
        Solve_times = []
        TD = []
        for i in range(episode):
            primal_eci = prop.initPrimalStates() # unit: km
            debris_eci, _, _ = rsg.gen_scenario(primal_eci, n_d=n_d) # unit: km
            debris_eci = debris_eci.reshape((1, n_d, 6))
            nominal_eci = primal_eci.copy() if _nominal_eci is None else _nominal_eci.copy()
            primal_obs, debris_obs = prop.getObss(primal_states=primal_eci, 
                                                  debris_states=debris_eci,
                                                  nominal_primal_states=nominal_eci)
            trans_dict = {"primal_eci": [],
                          "debris_eci": [],
                          "nominal_eci": [],
                          "actions": []}
            policy.start(primal_obs, debris_obs)
            for step in range(horizon):
                pref_delta_v = None
                pri_orbit = eci_to_orbit(primal_eci)[0]
                if len(policy._action_seq)==0 and deb_orbit is not None:
                    tca, md, _, _ = find_tca(pri_orbit, deb_orbit, [3000, 4000], prop._new_propagator())
                    if md.value < deb_safe_dist:
                        pref_delta_v = -primal_obs[:, 3:6] # adjust velocity to match nominal orbit
                action, _ = policy.act(primal_obs, debris_obs, pref_delta_v=pref_delta_v)
                (next_primal_eci, next_debris_eci, next_nominal_eci), _, _, (next_primal_obs, next_debris_obs) = prop._propagate(primal_eci, debris_eci, action, nominal_eci)

                trans_dict["primal_eci"].append(primal_eci)
                trans_dict["debris_eci"].append(debris_eci)
                trans_dict["nominal_eci"].append(nominal_eci)
                trans_dict["actions"].append(action)

                primal_eci, debris_eci, nominal_eci, primal_obs, debris_obs = next_primal_eci, next_debris_eci, next_nominal_eci, next_primal_obs, next_debris_obs
                if deb_orbit is not None:
                    deb_orbit = deb_orbit.propagate(prop.dt*U.s, prop._new_propagator())
            TD.append(trans_dict)
            
            maneuver_increment, miss_distances, nominal_miss_distances, fail = get_metrics(trans_dict, prop.safe_dist, prop.dt, prop.max_thrust)
            Maneuver_increment.append(maneuver_increment)
            Miss_distances.append(miss_distances)
            Nominal_miss_distances.append(nominal_miss_distances)
            fails += fail

            Propagate_times.append(policy.log["propagate_time"])
            Solve0_times.append(policy.log["solve_time"][0])
            Solve_times.append(policy.log["solve_time"][1:])

            progress.advance(task, 1)
        fail_rate = fails/episode
    return TD, fail_rate, Maneuver_increment, Miss_distances, Nominal_miss_distances, Propagate_times, Solve0_times, Solve_times


def simulate_front(episode:int, 
                   n_d:int, 
                   horizon:int,
                   prop:orbitPropagator, 
                   policy:safeOrbitVoPolicy, 
                   rsg:randomScenarioGenerator,
                   _nominal_eci:np.ndarray=None,
                   deb_orbit:Orbit=None,
                   deb_safe_dist:float=None,):
    with rich.progress.Progress() as progress:
        task = progress.add_task(f"collision avoidance n_d={n_d}", total=episode)
        fails = 0
        Maneuver_increment = []
        Miss_distances = []
        Nominal_miss_distances = []
        Propagate_times = []
        Solve0_times = []
        Solve_times = []
        TD = []
        for i in range(episode):
            primal_eci = prop.initPrimalStates() # unit: km
            debris_eci, _, _ = rsg.gen_front_scenario(primal_eci, n_d=n_d) # unit: km
            debris_eci = debris_eci.reshape((1, n_d, 6))
            nominal_eci = primal_eci.copy() if _nominal_eci is None else _nominal_eci.copy()
            primal_obs, debris_obs = prop.getObss(primal_states=primal_eci, 
                                                  debris_states=debris_eci,
                                                  nominal_primal_states=nominal_eci)
            trans_dict = {"primal_eci": [],
                          "debris_eci": [],
                          "nominal_eci": [],
                          "actions": []}
            policy.start(primal_obs, debris_obs)
            for step in range(horizon):
                pref_delta_v = None
                pri_orbit = eci_to_orbit(primal_eci)[0]
                if len(policy._action_seq)==0 and deb_orbit is not None:
                    tca, md, _, _ = find_tca(pri_orbit, deb_orbit, [3000, 4000], prop._new_propagator())
                    if md.value < deb_safe_dist:
                        pref_delta_v = -primal_obs[:, 3:6] # adjust velocity to match nominal orbit
                action, _ = policy.act(primal_obs, debris_obs, pref_delta_v=pref_delta_v)
                (next_primal_eci, next_debris_eci, next_nominal_eci), _, _, (next_primal_obs, next_debris_obs) = prop._propagate(primal_eci, debris_eci, action, nominal_eci)

                trans_dict["primal_eci"].append(primal_eci)
                trans_dict["debris_eci"].append(debris_eci)
                trans_dict["nominal_eci"].append(nominal_eci)
                trans_dict["actions"].append(action)

                primal_eci, debris_eci, nominal_eci, primal_obs, debris_obs = next_primal_eci, next_debris_eci, next_nominal_eci, next_primal_obs, next_debris_obs
                if deb_orbit is not None:
                    deb_orbit = deb_orbit.propagate(prop.dt*U.s, prop._new_propagator())
            TD.append(trans_dict)
            
            maneuver_increment, miss_distances, nominal_miss_distances, fail = get_metrics(trans_dict, prop.safe_dist, prop.dt, prop.max_thrust)
            Maneuver_increment.append(maneuver_increment)
            Miss_distances.append(miss_distances)
            Nominal_miss_distances.append(nominal_miss_distances)
            fails += fail

            Propagate_times.append(policy.log["propagate_time"])
            Solve0_times.append(policy.log["solve_time"][0])
            Solve_times.append(policy.log["solve_time"][1:])

            progress.advance(task, 1)
        fail_rate = fails/episode
    return TD, fail_rate, Maneuver_increment, Miss_distances, Nominal_miss_distances, Propagate_times, Solve0_times, Solve_times



# deprived

def simulate3(episode:int, 
              n_d:int, 
              horizon:int,
              prop:orbitPropagator, 
              policy:safeOrbitVoPolicy, 
              rsg:randomScenarioGenerator,
              _nominal_eci:np.ndarray=None,
              tar_orbit:Orbit=None,):
    with rich.progress.Progress() as progress:
        task = progress.add_task(f"collision avoidance n_d={n_d}", total=episode)
        fails = 0
        Maneuver_increment = []
        Miss_distances = []
        Nominal_miss_distances = []
        Propagate_times = []
        Solve0_times = []
        Solve_times = []
        TD = []
        for i in range(episode):
            primal_eci = prop.initPrimalStates() # unit: km
            debris_eci, _, _ = rsg.gen_scenario(primal_eci, n_d=n_d) # unit: km
            debris_eci = debris_eci.reshape((1, n_d, 6))
            nominal_eci = primal_eci.copy() if _nominal_eci is None else _nominal_eci.copy()
            primal_obs, debris_obs = prop.getObss(primal_states=primal_eci, 
                                                  debris_states=debris_eci,
                                                  nominal_primal_states=nominal_eci)
            trans_dict = {"primal_eci": [],
                          "debris_eci": [],
                          "nominal_eci": [],
                          "actions": []}
            for step in range(horizon):
                pref_delta_v = None
                pri_orbit = eci_to_orbit(primal_eci)[0]
                if len(policy._action_seq)==0 and tar_orbit is not None:
                    # delta_v1, delta_v2 = lambert_con(pri_orbit, tar_orbit, prop.dt*policy.plan_horizon*U.s, prop._new_propagator())
                    delta_v1, delta_v2 = lambert_con(pri_orbit, tar_orbit, 1*U.s, prop._new_propagator())
                    pref_delta_v = delta_v1.value.reshape((1,3)) # adjust velocity to match nominal orbit
                action, _ = policy.act(primal_obs, debris_obs, pref_delta_v=pref_delta_v)
                (next_primal_eci, next_debris_eci, next_nominal_eci), _, _, (next_primal_obs, next_debris_obs) = prop._propagate(primal_eci, debris_eci, action, nominal_eci)

                trans_dict["primal_eci"].append(primal_eci)
                trans_dict["debris_eci"].append(debris_eci)
                trans_dict["nominal_eci"].append(nominal_eci)
                trans_dict["actions"].append(action)

                primal_eci, debris_eci, nominal_eci, primal_obs, debris_obs = next_primal_eci, next_debris_eci, next_nominal_eci, next_primal_obs, next_debris_obs
                if tar_orbit is not None:
                    tar_orbit = tar_orbit.propagate(prop.dt*U.s, prop._new_propagator())
            TD.append(trans_dict)
            
            maneuver_increment, miss_distances, nominal_miss_distances, fail = get_metrics(trans_dict, prop.safe_dist, prop.dt, prop.max_thrust)
            Maneuver_increment.append(maneuver_increment)
            Miss_distances.append(miss_distances)
            Nominal_miss_distances.append(nominal_miss_distances)
            fails += fail

            Propagate_times.append(policy.log["propagate_time"])
            Solve0_times.append(policy.log["solve_time"][0])
            Solve_times.append(policy.log["solve_time"][1:])

            progress.advance(task, 1)
        fail_rate = fails/episode
    return TD, fail_rate, Maneuver_increment, Miss_distances, Nominal_miss_distances, Propagate_times, Solve0_times, Solve_times


from poliastro.iod import lambert
from poliastro.util import norm
def lambert_con(orbit_1:Orbit, orbit_2:Orbit, delta_t, method):
    # 在初始轨道的当前时刻获取位置矢量 r0
    r0 = orbit_1.r

    # 计算目标轨道在 t0 + Δt 时的位置 r1
    orbit_2_proped = orbit_2.propagate(delta_t, method=method)
    r1 = orbit_2_proped.r

    # 求解Lambert问题，得到转移轨道的初末速度 v0_lambert 和 v1_lambert
    v0_lambert, v1_lambert = lambert(Earth.k, r0, r1, delta_t)

    # 计算初始轨道在r0处的速度 v_i 和目标轨道在r1处的速度 v_t
    v_i = orbit_1.v
    v_t = orbit_2_proped.v

    # 计算两次脉冲
    delta_v1 = v0_lambert - v_i
    delta_v2 = v_t - v1_lambert

    # 总Δv
    total_delta_v = norm(delta_v1) + norm(delta_v2)
    # print(f"总Δv: {total_delta_v.to(U.m/U.s):.2f}")

    return delta_v1, delta_v2


from utils import traj_refine
def get_metrics(td:dict, safe_dist, dt, max_thrust, refine_num=20):
    maneuver_increment = np.linalg.norm(np.stack(td["actions"]), axis=-1).sum() * dt * max_thrust

    ds = np.stack(td['debris_eci']).squeeze(axis=1)
    ps = np.stack(td['primal_eci']).squeeze(axis=1)
    ns = np.stack(td['nominal_eci']).squeeze(axis=1)
    ds = traj_refine(ds, num=refine_num)
    ps = traj_refine(ps, num=refine_num)
    ns = traj_refine(ns, num=refine_num)
    rel_states = ds - ps.reshape((ps.shape[0], 1, 6))
    rel_pos = rel_states[:, :, :3]
    miss_distances = np.min(np.linalg.norm(rel_pos, axis=-1), axis=0) * 1e3
    nom_rel_states = ds - ns.reshape((ps.shape[0], 1, 6))
    nom_rel_pos = nom_rel_states[:, :, :3]
    nominal_miss_distances = np.min(np.linalg.norm(nom_rel_pos, axis=-1), axis=0) * 1e3
    fail = (miss_distances<safe_dist*1e3).any()

    return maneuver_increment, miss_distances, nominal_miss_distances, fail
    

def get_scenario2(sat_orbit:Orbit, tca=3600, prop_method=None):
    sat_orbit = sat_orbit.propagate(tca*U.s, method=prop_method) if prop_method is not None else sat_orbit.propagate(tca*U.s)
    r1, v1 = sat_orbit.r.value, sat_orbit.v.value
    v1 = v1 @ Rotation.from_rotvec(np.pi/2 * r1/np.linalg.norm(r1)).as_matrix()
    deb_orbit = Orbit.from_vectors(Earth, r1*U.km, v1*U.km/U.s)
    deb_orbit = deb_orbit.propagate(-tca*U.s, method=prop_method) if prop_method is not None else deb_orbit.propagate(-tca*U.s)
    return deb_orbit


def get_scenario3(sat_orbit:Orbit, n_slow=3, tca=4*3600, prop_method=None):
    deb_orbits = []
    sat_orbit = sat_orbit.propagate(tca*U.s, method=prop_method) if prop_method is not None else sat_orbit.propagate(tca*U.s)
    rs, vs = sat_orbit.r.value, sat_orbit.v.value
    main_eci = np.hstack([rs, vs])
    r1 = rs
    v1 = vs @ Rotation.from_rotvec(np.pi/2 * rs/np.linalg.norm(rs)).as_matrix()
    deb_orbit = Orbit.from_vectors(Earth, r1*U.km, v1*U.km/U.s)
    deb_orbit = deb_orbit.propagate(-tca*U.s, method=prop_method) if prop_method is not None else deb_orbit.propagate(-tca*U.s)
    deb_orbits.append(deb_orbit)
    rel_states = []
    for i in range(n_slow):
        rel_pos = 50. * np.random.uniform(-1, 1, (3,))
        rel_vel = 20. * np.random.uniform(-1, 1, (3,))
        deb_lvlh = np.hstack([rel_pos, rel_vel])
        deb_eci = lvlh_to_eci(main_eci, deb_lvlh).flatten()
        rd, vd = deb_eci[:3] * U.km, deb_eci[3:] * U.km/U.s
        deb_orbit = Orbit.from_vectors(Earth, rd, vd)
        deb_orbit = deb_orbit.propagate(-tca*U.s, method=prop_method) if prop_method is not None else deb_orbit.propagate(-tca*U.s)
        rel_states.append(deb_lvlh)
        deb_orbits.append(deb_orbit)
    return deb_orbits, rel_states

