import numpy as np
import time
import torch

from robosuite.utils.input_utils import input2action

PAUSE_TIME = 0.001

class BaseRobosuitePolicy:
    def __init__(self, policy, env, robosuite_cfg) -> None:
        self.policy = policy
        self.env = env
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.input_device = robosuite_cfg["input_device"]
        self.arm = robosuite_cfg["arm"]
        self.env_config = robosuite_cfg["env_config"]
        self.active_robot = robosuite_cfg["active_robot"]
    
    def _user_policy(self, obs):
        a = np.zeros(self.act_dim)
        if self.env.gripper_closed:
            a[-1] = 1.
            self.input_device.grasp = True
        else:
            a[-1] = -1.
            self.input_device.grasp = False
        a_ref = a.copy()
        # pause simulation if there is no user input (instead of recording a no-op)
        while np.array_equal(a, a_ref):
            a, _ = input2action(
                device=self.input_device,
                robot=self.active_robot,
                active_arm=self.arm,
                env_configuration=self.env_config)
            self.env.render()
            time.sleep(PAUSE_TIME)
        a = torch.tensor(a)
        return a