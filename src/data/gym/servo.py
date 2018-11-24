import numpy as np
import math


class ServoSimulation(object):
    def __init__(self, kt, jm, jl, cs, cd, fsf, fvf, max_current, dt=0.0000625):
        self.mpa, self.mva, self.lpa, self.lva = 0.0, 0.0, 0.0, 0.0
        self.jm, self.jl, self.cs, self.cd, self.fsf, self.fvf = jm, jl, cs, cd, fsf, fvf
        self.dt = dt
        self.kt = kt
        self.applied_current = 0
        self.max_current = max_current

    def state(self):
        return np.array([self.mpa, self.mva, self.lpa, self.lva])

    def state_with_current(self):
        return np.array([self.mpa, self.mva, self.lpa, self.lva, self.applied_current])

    def step(self, current, tl=0):
        self._step(current, tl)
        return self.state()

    def step_with_applied_current(self, current, tl=0):
        self._step(current, tl)
        return self.state_with_current()

    def _step(self, current, tl):
        self.applied_current = current
        tm = self.kt * current - (self.fvf * self.mva + math.copysign(self.fsf, self.mva))
        mp_minus_lp, mv_minus_lv = self.mpa - self.lpa, self.mva - self.lva
        cs, cd, dt = self.cs, self.cd, self.dt
        self.mva += (dt / self.jm) * (tm - cs * mp_minus_lp - cd * mv_minus_lv)
        self.lva += (dt / self.jl) * (-tl + cs * mp_minus_lp + cd * mv_minus_lv)
        self.mpa += dt * self.mva
        self.lpa += dt * self.lva

    def set_state(self, mpa, mva, lpa, lva):
        self.mpa, self.mva, self.lpa, self.lva = mpa, mva, lpa, lva
        return self.state()

    def reset(self):
        self.mpa, self.mva, self.lpa, self.lva, self.applied_current = 0.0, 0.0, 0.0, 0.0, 0.0

    def __repr__(self):
        return '[{},{},{},{},{},{},{},{},{}]'.format(self.kt, self.jm, self.jl, self.cs, self.cd, self.fsf, self.fvf, self.max_current, self.dt)


def create_rd50(dt=0.001):
    return ServoSimulation(0.04, 8.2e-6, 0.6e-4, 0.47, 0.05, 0.1, 1e-6, 9, dt)


def create_rd60(dt=0.001):
    return ServoSimulation(0.05, 17.09e-6, 1e-4, 1, 0.05, 0.2, 1e-6, 9, dt)


def create_rd85(dt=0.001):
    return ServoSimulation(0.076, 1.08e-4, 2.5e-4, 3.1, 0.05, 0.4, 1e-6, 24, dt)


def create_old(dt=0.001):
    return ServoSimulation(0.04, 1.2e-4, 1.2e-4, 0.47, 5e-3, 0.042, 2e-5, 10, dt)