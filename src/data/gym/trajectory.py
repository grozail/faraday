import pandas as pd
import numpy as np
import math
from pathlib import Path
import arrow
from src.data.gym.servo import ServoSimulation, create_old, create_rd50
import matplotlib.pyplot as plt

_TRAJECTORY_DUMP_PATH = Path(__file__).resolve().parents[3].joinpath('data/raw/')


def trajectory_with_current_to_csv(servo, trajectory):
    time = arrow.utcnow().format('[YYYY-MM-DD][HH:mm:ss]')
    fname = str(_TRAJECTORY_DUMP_PATH) + '/{!s}{!r}.csv'.format(time, servo)
    pd.DataFrame(trajectory, columns=['mpa', 'mva', 'lpa', 'lva', 'ac']).to_csv(fname, index=False)


class ServoTrajectoryGenerator(object):
    def __init__(self, servo: ServoSimulation):
        self.servo = servo
        self.trajectories = []

    def __check_traj_time_more_than_dt(self, time_s):
        if time_s < self.servo.dt:
            raise ValueError("time_s < servo.dt")

    def __calculate_number_of_steps(self, time_s):
        return math.ceil(time_s / self.servo.dt)

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
        states = [self.servo.step_with_applied_current(np.clip(current + n, -self.servo.max_current, self.servo.max_current)) for n in noise]
        trajectory = np.array(states)
        self.trajectories.append(trajectory)
        return trajectory

    def generate_with_noisy_gaussian_const_current_no_load(self, current, time_s, mean, std):
        self.__check_traj_time_more_than_dt(time_s)
        number_of_steps = self.__calculate_number_of_steps(time_s)
        noise = np.random.normal(mean, std, number_of_steps)
        states = [self.servo.step_with_applied_current(np.clip(current + n, -self.servo.max_current, self.servo.max_current)) for n in noise]
        trajectory = np.array(states)
        self.trajectories.append(trajectory)
        return trajectory

    def clear_trajectories(self):
        self.trajectories.clear()


class RandomTrajectoryGenerationProcess(object):
    def __init__(self, generator: ServoTrajectoryGenerator, n_pieces, max_piece_time):
        self.generator = generator
        self.servo = self.generator.servo
        self.pieces = n_pieces
        self.max_trajectory_time = max_piece_time

    def run_uniform(self):
        for _ in range(self.pieces):
            time = np.random.uniform(0.05, self.max_trajectory_time)
            current = np.random.uniform(-self.servo.max_current, self.servo.max_current)
            noize = abs(current) * 0.05
            self.generator.generate_with_noisy_uniform_const_current_no_load(current, time, -noize, noize)
        trajectory = np.concatenate(self.generator.trajectories)
        return trajectory


if __name__ == '__main__':
    np.random.seed(42)
    s = create_rd50(dt=0.0000625)
    traj = RandomTrajectoryGenerationProcess(
        ServoTrajectoryGenerator(s),
        10, 0.5
    ).run_uniform()
    trajectory_with_current_to_csv(s, traj)