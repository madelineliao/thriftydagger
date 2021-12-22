import numpy as np
import random
# from run_thriftydagger import expert_pol
import torch

ACTION_MAGNITUDE = 0.05

def sample_reach(N_trajectories):
    range_x = 3.0
    range_y = 3.0
    demos = []

    for _ in range(N_trajectories):
        # Sample goal from 1st quadrant
        goal_ee_state = np.array([random.uniform(0, range_x), random.uniform(0, range_y)])
        curr_state = np.zeros(2)
        traj = goal_ee_state - curr_state
        states = []
        actions = []
        goals = []
        action = ACTION_MAGNITUDE * traj / np.linalg.norm(traj)
        while (curr_state[0] < goal_ee_state[0] and curr_state[1] < goal_ee_state[1]):
            states.append(curr_state)
            actions.append(action)
            goals.append(goal_ee_state)
            curr_state = curr_state + action
        demos.append([(state, action, goal) for state, action, goal in zip(states, actions, goals)])
    return demos


def sample_pi_r(N_trajectories, model_path, expert_pol, device, add_noise=False):
    state = torch.load(model_path, map_location=device)
    ac = state['model'].to(device)
    pi_optimizers = state['pi_optimizers']
    ac.device = device
    demos = []
    for j in range(N_trajectories):
        obs, act,rew = [], [], []
        range_x, range_y = 3., 3.
        goal_ee_state = np.array([random.uniform(0, range_x), random.uniform(0, range_y)])
        o, ep_len = np.zeros(4), 0
        o[2:] = goal_ee_state
        d = False
        obs.append(o.copy())
        while not d and ep_len < 100:
            a = ac.act(o)
            a_target = goal_ee_state - o[:2]
            a_target = ACTION_MAGNITUDE * a_target / np.linalg.norm(a_target)
            act.append(a_target)
            o[:2] = o[:2] + a
            if add_noise:
                max_variance = 0
                for _ in range(20):
                    noise = np.random.normal(np.zeros_like(o[:2]), scale=0.5)
                    candidate = o[:2] + noise
                    if ac.variance(np.concatenate([candidate, o[2:]])) >= max_variance:
                        o[:2] = candidate
            ep_len += 1
            d = ((o[:2][0] >= o[2:][0]) and (o[:2][1] >= o[2:][1]) or np.linalg.norm(o[:2] - o[2:]) <= 0.1)
            obs.append(o.copy())
        # rew.append(np.linalg.norm(o[:2] - o[2:]) <= 0.1)
        demo = [(o[:2], a, o[2:]) for o, a in zip(obs, act)]
        demos.append(demo)
    return demos

class HardcodedReach2DPolicy():
    def __init__(self, obj_loc, sim_style):
        self.obj_loc = obj_loc # (2,)
        self.sim_style = sim_style

    def act(self, curr_pos): 
       a = self.obj_loc - curr_pos
       a /= np.linalg.norm(a)
       a *= ACTION_MAGNITUDE
       return a