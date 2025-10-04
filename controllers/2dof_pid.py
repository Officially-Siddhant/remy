from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
  # def __init__(self, alpha, p, i, d, b, c,dt):

    self.p = 0.20
    self.i = 0.05
    self.d = 0.20

    self.b = 1.5 # higher b = higher reference signal
    self.c = 0.10 # lower c values - higher the derivative operates on y (measurement)
    self.alpha = 0.98

    '''
    self.p = p
    self.i = i
    self.d = d

    self.b = b # higher b = higher reference signal
    self.c = c # lower c values - higher the derivative operates on y (measurement)
    self.alpha = alpha
    '''

    self.error_integral = 0
    self.derivative = 0.0
    self.prev_error_d = None

    self.k_roll = 0.20
    self.roll_N = 5

    # self.dt = getattr(self, "dt", 0.1)
    self.dt = 1.0
    self.steer_factor = 12.5
    self.minimum_v = 20.0
    self.N = 5

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    self.step = getattr(self, "step", 0) + 1
    fade = min(self.step / 100.0, 1.0)

    # gentlest on I, moderate on P/D
    Kp_eff = self.p * fade
    Ki_eff = self.i * (fade ** 3)
    Kd_eff = self.d * fade


    # Feed-forward terms
    rolls = [getattr(state, "roll_lataccel", 0.0)] + (getattr(future_plan, "roll_lataccel", []) or [])
    u_ff = -self.k_roll * np.average(rolls[:self.roll_N])

    future = future_plan.lataccel[:max(0, self.N - 1)]
    targets = [float(target_lataccel)] + [float(x) for x in future]
    weights = np.array([4, 3, 2, 2, 2, 1][:len(targets)], dtype=float)
    weights /= weights.sum()
    target_smooth = float(np.dot(weights, targets))

    # Proportional Error
    error_p = (self.b * target_smooth) - current_lataccel

    # Integral Error
    error_i = target_smooth - current_lataccel
    self.error_integral += Ki_eff*error_i

    # Derivative Error
    error_d = (self.c * target_smooth) - current_lataccel
    if self.prev_error_d is None:
      error_diff = 0.0
    else:
      error_diff = error_d - self.prev_error_d
    self.prev_error_d = error_d
    self.derivative = self.alpha * self.derivative + (1 - self.alpha) * error_diff


    steer_lataccel_target = (target_smooth - state.roll_lataccel)
    u_steer = steer_lataccel_target * (self.steer_factor / max(self.minimum_v, state.v_ego))
    u_steer = u_steer / (1.0 + np.abs(u_steer))

    u = Kp_eff * error_p + self.error_integral + Kd_eff * self.derivative


    u_tot = u + u_steer + u_ff

    return u_tot
