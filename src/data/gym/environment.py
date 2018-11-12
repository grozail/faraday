import numpy as np
import gym
from .servo import ServoSimulation
from .config import ControlConfiguration


class SingleServoEnv(gym.GoalEnv):
    def __init__(self, servo: ServoSimulation, control_config: ControlConfiguration):
        self.servo = servo
        self.action_space = gym.spaces.Box(-servo.max_current,
                                           servo.max_current,
                                           shape=(control_config.control_horizon,),
                                           dtype='float32')
        self.observation_space = gym.spaces.Box(-np.inf,
                                                np.inf,
                                                shape=(control_config.prediction_horizon * servo.state().size,),
                                                dtype='float32')
        self.control_config = control_config

    def compute_reward(self, achieved_goal, desired_goal, info):
        pass

    def step(self, action):
        pass

    def render(self, mode='human'):
        pass

    def reset(self):
        return super().reset()

