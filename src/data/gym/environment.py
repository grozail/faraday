import numpy as np
import gym
from .servo import ServoSimulation
from .trajectory import RandomTrajectoryGenerationProcess
from .config import ControlConfiguration


class SingleServoEnv(gym.GoalEnv):
    def __init__(self,
                 process: RandomTrajectoryGenerationProcess,
                 control_config: ControlConfiguration):
        self.process = process
        self.servo = process.servo
        self.action_space = gym.spaces.Box(-self.servo.max_current,
                                           self.servo.max_current,
                                           shape=(control_config.control_horizon,),
                                           dtype='float32')
        self.observation_space = gym.spaces.Box(-np.inf,
                                                np.inf,
                                                shape=(control_config.prediction_horizon * self.servo.state().size,),
                                                dtype='float32')
        self.control_config = control_config
        self.trajectory = process.run_uniform()

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.exp(-np.sum(np.sqrt(achieved_goal - desired_goal)))

    def step(self, action):
        ac = float(action)


    def render(self, mode='human'):
        pass

    def reset(self):
        return super().reset()

