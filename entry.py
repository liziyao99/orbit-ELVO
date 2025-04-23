from policy.vo.voPolicy import safeOrbitVoPolicy


def init_policy(dt: float,
                max_thrust: float,
                sep_dist: float,
                plan_horizon: int = 20,
                pred_window: float  = 3000,
                ):
    policy = safeOrbitVoPolicy(dt=dt, 
                           max_thrust=max_thrust, 
                           base_sep_dist=sep_dist,
                           plan_horizon=plan_horizon, # int
                           pred_window=pred_window, # unit: s
                           )
    return policy
