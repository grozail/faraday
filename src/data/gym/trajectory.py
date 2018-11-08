import math
import pandas as pd
import numpy as np
from .servo import ServoSimulation


class ServoTrajectoryGenerator(object):
    def __init__(self, servo: ServoSimulation):
        self.servo = servo

    def __check_traj_time_more_than_dt(self, time_s):
        if time_s < self.servo.dt:
            raise ValueError("time_s < servo.dt")

    def __calculate_number_of_steps(self, time_s):
        return math.ceil(time_s / self.servo.dt)

    @staticmethod
    def to_dataframe(data):
        return pd.DataFrame(data=data, columns=['mpa', 'mva', 'lpa', 'lva', 'applied_current'])

    def generate_with_constant_current_no_load(self, current, time_s):
        self.__check_traj_time_more_than_dt(time_s)
        number_of_steps = self.__calculate_number_of_steps(time_s)
        states = [self.servo.step_with_current(current) for _ in range(number_of_steps)]
        return self.to_dataframe(np.array(states))



def generate_df_using_constant_current(servo: ServoSimulation, current, time_s):
    if time_s < servo.dt:
        raise ValueError("time_s < servo.dt")
    number_of_steps = math.ceil(time_s / servo.dt)

