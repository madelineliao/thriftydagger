import random

import torch

from constants import REACH2D_ACT_DIM, REACH2D_PILLAR_OBS_DIM, REACH2D_SUCCESS_THRESH


class Pillar:
    def __init__(self, height, width, start, goal) -> None:
        # Set dimensions
        self.height = height
        self.width = width
        
        # Set location (self.x, self.y = upper left corner coordinates)
        # Pillar is centered at the midpoint of start and goal
        self.x, self.y = self._init_loc(start, goal)
        self.loc = torch.tensor([self.x, self.y])
        
    def _init_loc(self, start, goal):
        center = (start + goal) / 2
        x = center[0] - self.width / 2
        y = center[1] + self.height / 2
        
        return x, y
    
    def overlaps(self, state):
        state_x, state_y = state
        x_overlaps = state_x >= self.x and state_x <= (self.x + self.width)
        y_overlaps = state_y <= self.y and state_y >= (self.y - self.height)
        
        return x_overlaps and y_overlaps

class Reach2DPillar:
    def __init__(self, device, max_ep_len, pillar_height, pillar_width, 
                 random_start_state=False, range_x=3.0, range_y=3.0):
        self.device = device
        self.max_ep_len = max_ep_len
        self.random_start_state = random_start_state
        self.range_x = range_x
        self.range_y = range_y
        self.curr_state = self._init_start_state() 
        self.goal_state = torch.tensor(
            [random.uniform(0, self.range_x), random.uniform(0, self.range_y)], device=self.device
        )
        self.pillar = Pillar(pillar_height, pillar_width, self.curr_state, self.goal_state)
        self.curr_obs = torch.cat([self.curr_state, self.goal_state, self.pillar.loc])
        self.act_dim = REACH2D_ACT_DIM
        self.obs_dim = REACH2D_PILLAR_OBS_DIM
        self.ep_len = 0
        
    def _init_start_state(self):
        if self.random_start_state:
            start_state = torch.tensor(
                [random.uniform(0, self.range_x), random.uniform(0, self.range_y)], device=self.device
            )

        else:
            start_state = torch.zeros(2, device=self.device)

        return start_state

    def close(self):
        """ Empty function so callers don't break with use of this class. """
        pass

    def reset(self):
        self.curr_state = self._init_start_state() 
        self.goal_state = torch.tensor(
            [random.uniform(0, self.range_x), random.uniform(0, self.range_y)], device=self.device
        )
        self.pillar = Pillar(self.pillar.height, self.pillar.width, self.curr_state, self.goal_state)
        
        self.curr_obs = torch.cat([self.curr_state, self.goal_state, self.pillar.loc])
        self.ep_len = 0
        return self.curr_obs

    def step(self, action):
        self.curr_state += action
        self.curr_obs = torch.cat([self.curr_state, self.goal_state, self.pillar.loc])
        self.ep_len += 1
        return self.curr_obs, self._check_success(), self._check_done(), None

    def _check_success(self):
        return (torch.norm(self.curr_state - self.goal_state) <= REACH2D_SUCCESS_THRESH).item()

    def _check_done(self):
        return self.ep_len >= self.max_ep_len or self.pillar.overlaps(self.curr_state)