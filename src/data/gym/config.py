

class ControlConfiguration(object):
    def __init__(self, control_horizon, prediction_horizon, min_diff=1e-3, max_diff=0.15, diff_decay=0.8):
        self.control_horizon = control_horizon
        self.prediction_horizon = prediction_horizon
        self.min_diff = min_diff
        self.max_diff = max_diff
        self.current_diff = max_diff
        self.diff_decay = diff_decay

    def apply_decay(self):
        if self.current_diff > self.min_diff:
            self.current_diff *= self.diff_decay
