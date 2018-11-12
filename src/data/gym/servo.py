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
        tm = self.kt * current - (self.fsf * self.mva + math.copysign(self.fsf, self.mva))
        mp_minus_lp, mv_minus_lv = self.mpa - self.lpa, self.mva - self.lva
        cs, cd, dt = self.cs, self.cd, self.dt
        self.mva += dt / self.jm * (tm - cs * mp_minus_lp - cd * mv_minus_lv)
        self.lva += dt / self.jl * (-tl + cs * mp_minus_lp + cd * mv_minus_lv)
        self.mpa += dt * self.mva
        self.lpa += dt * self.lva

    def set_state(self, mpa, mva, lpa, lva):
        self.mpa, self.mva, self.lpa, self.lva = mpa, mva, lpa, lva
        return self.state()

    def reset(self):
        self.mpa, self.mva, self.lpa, self.lva, self.applied_current = 0.0, 0.0, 0.0, 0.0, 0.0


def create_rd50(dt=0.001):
    return ServoSimulation(0.04, 1.2e-4, 1.2e-4, 0.47, 5e-3, 0.042, 2e-5, 10, dt)
