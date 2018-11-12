import math
import pandas as pd
import numpy as np
from .servo import ServoSimulation


class ServoTrajectoryGenerator(object):
    def __init__(self, servo: ServoSimulation):
        self.servo = servo
        self.trajectories = []

    def __check_traj_time_more_than_dt(self, time_s):
        if time_s < self.servo.dt:
            raise ValueError("time_s < servo.dt")

    def __calculate_number_of_steps(self, time_s):
        return math.ceil(time_s / self.servo.dt)

    @staticmethod
    def to_dataframe(data):
        return pd.DataFrame(data=data, columns=['mpa', 'mva', 'lpa', 'lva', 'applied_current'])

    def generate_with_const_current_no_load(self, current, time_s):
        self.__check_traj_time_more_than_dt(time_s)
        number_of_steps = self.__calculate_number_of_steps(time_s)
        states = [self.servo.step_with_applied_current(current) for _ in range(number_of_steps)]
        trajectory = np.array(states)
        self.trajectories.append(trajectory)
        return trajectory

    def generate_with_noisy_uniform_const_current_no_load(self, current, time_s, lower_bound, upper_bound):
        self.__check_traj_time_more_than_dt(time_s)
        number_of_steps = self.__calculate_number_of_steps(time_s)
        noise = np.random.uniform(lower_bound, upper_bound, number_of_steps)
        states = [self.servo.step_with_applied_current(current + n) for n in noise]
        trajectory = np.array(states)
        self.trajectories.append(trajectory)
        return trajectory

    def generate_with_noisy_gaussian_const_current_no_load(self, current, time_s, mean, std):
        self.__check_traj_time_more_than_dt(time_s)
        number_of_steps = self.__calculate_number_of_steps(time_s)
        noise = np.random.normal(mean, std, number_of_steps)
        states = [self.servo.step_with_applied_current(current + n) for n in noise]
        trajectory = np.array(states)
        self.trajectories.append(trajectory)
        return trajectory

    def clear_trajectories(self):
        self.trajectories.clear()

