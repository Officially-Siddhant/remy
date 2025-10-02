from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):

    self.p = 0.15
    self.i = 0.01
    self.d = 0.1

    self.b = 2.0
    self.c = 0.11
    self.alpha = 0.9

    self.error_integral = 0
    self.derivative = 0.0
    self.prev_error_d = None

    self.k_roll = 0.3
    # self.alpha_roll = 0.90
    self.alpha_roll = 0.5
    self.roll_f = 0.0
    self.roll_inited = False
    self.roll_N = 5

    self.dt = getattr(self, "dt", 0.1)

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    # Feed-forward terms
    rolls = getattr(future_plan, "roll_lataccel", []) or []
    if rolls:
      # average first N points to reduce noise
      r0 = float(np.mean(rolls[:min(self.roll_N, len(rolls))]))
      if not self.roll_inited:
        # roll_f = mean(rolls)
        self.roll_f, self.roll_inited = r0, True
      else:
        # roll_f = α*roll_f + (1-α) * mean(rolls)
        self.roll_f = self.alpha_roll * self.roll_f + (1 - self.alpha_roll) * r0
    #
    # Feed-forward control term
    u_ff = -self.k_roll * self.roll_f


    # Proportional Error
    error_p = (self.b * target_lataccel) - current_lataccel

    # Integral Error
    error_i = target_lataccel - current_lataccel
    self.error_integral += self.i*error_i

    # Derivative Error
    error_d = (self.c * target_lataccel) - current_lataccel
    if self.prev_error_d is None:
      error_diff = 0.0
    else:
      error_diff = error_d - self.prev_error_d
    self.prev_error_d = error_d
    self.derivative = self.alpha * self.derivative + (1 - self.alpha) * error_diff

    u = self.p * error_p + self.error_integral + self.d * self.derivative

    return float(u + u_ff)
