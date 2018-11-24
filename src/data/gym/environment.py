import numpy as np
import gym
from .trajectory import RandomTrajectoryGenerationProcess, trajectory_with_current_to_csv
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
                                                shape=((control_config.prediction_horizon + 1) * self.servo.state().size,),
                                                dtype='float32')
        self.control_config = control_config

        self.trajectory = process.run_uniform()
        trajectory_with_current_to_csv(self.servo, self.trajectory)
        self.trajectory = self.trajectory[:, 0:4]

        self.dynamic_error = 0
        self.max_dynamic_error = 0.1
        self.index = 1
        self.reset()

    def is_done_by_dynamic_error(self):
        return self.dynamic_error > self.max_dynamic_error

    def compute_dynamic_error(self, achieved_goal, desired_goal, info):
        self.dynamic_error += np.rad2deg(np.abs(achieved_goal[3] - desired_goal[3])) / 100
        return self.dynamic_error

    def compute_reward(self, achieved_goal, desired_goal, info):
        return np.exp(-np.sum(np.sqrt(achieved_goal - desired_goal)))

    def step(self, action):
        ac = float(action)
        achieved_state = self.servo.step(ac)
        desired_state = self.trajectory[self.index]
        info = {}
        reward = self.compute_reward(achieved_state, desired_state, info)
        self.compute_dynamic_error(achieved_state, desired_state, info)
        done = self.is_done_by_dynamic_error()
        self.index += 1
        try:
            observation = np.concatenate([self.servo.state(), self.get_prediction_horizon()])
        except IndexError:
            observation = np.concatenate([self.servo.state() for _ in range(self.control_config.prediction_horizon + 1)])
            done = True
            reward = 5
        return observation, reward, done, info

    def get_prediction_horizon(self):
        if self.index + self.control_config.prediction_horizon == self.trajectory.shape[0]:
            raise IndexError
        return np.concatenate(self.trajectory[self.index:self.index + self.control_config.prediction_horizon])

    def render(self, mode='human'):
        pass

    def reset(self):
        self.index = 1
        self.servo.set_state(*self.trajectory[0])
        self.dynamic_error = 0
        prediction_horizon = self.get_prediction_horizon()
        return np.concatenate([self.servo.state(), prediction_horizon])

