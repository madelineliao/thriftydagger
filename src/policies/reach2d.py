import torch

from src.constants import REACH2D_ACT_MAGNITUDE, REACH2D_ACT_DIM

def straight_line_policy(obs):
    curr_state, goal_state = obs
    act = goal_state - curr_state
    act = REACH2D_ACT_MAGNITUDE / torch.norm(act)
    
    return act

def up_right_policy(obs):
    curr_state = obs[:2]
    goal_state  = obs[2:]
    curr_x, curr_y = curr_state
    goal_x, goal_y = goal_state
    
    if not torch.isclose(curr_y, goal_y):
        # First, align y-coordinate
        if curr_y < goal_y:
            act = torch.tensor([0, REACH2D_ACT_MAGNITUDE])
        elif curr_y > goal_y:
            act = torch.tensor([0, -REACH2D_ACT_MAGNITUDE])
    elif not torch.isclose(curr_x, goal_x):
        # Then align x-coordinate (only if y-coordinate aligned already)
        if curr_x < goal_x:
            act = torch.tensor([REACH2D_ACT_MAGNITUDE, 0])
        elif curr_x > goal_x:
            act = torch.tensor([-REACH2D_ACT_MAGNITUDE, 0])
    else:
        act = torch.zeros(REACH2D_ACT_DIM)
    
    return act

def right_up_policy(obs):
    curr_state = obs[:2]
    goal_state  = obs[2:]
    curr_x, curr_y = curr_state
    goal_x, goal_y = goal_state
    
    
    if not torch.isclose(curr_x, goal_x):
        # First, align x-coordinate
        if curr_x < goal_x:
            act = torch.tensor([REACH2D_ACT_MAGNITUDE, 0])
        elif curr_x > goal_x:
            act = torch.tensor([-REACH2D_ACT_MAGNITUDE, 0])
    elif not torch.isclose(curr_y, goal_y):
        # Then align y-coordinate (only if x-coordinate aligned already)
        if curr_y < goal_y:
            act = torch.tensor([0, REACH2D_ACT_MAGNITUDE])
        elif curr_y > goal_y:
            act = torch.tensor([0, -REACH2D_ACT_MAGNITUDE])
    else:
        act = torch.zeros(REACH2D_ACT_DIM)
        
    return act