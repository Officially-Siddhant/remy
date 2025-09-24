from . import BaseController
import numpy as np


class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    # Initial step dependent gains.
    # self.p = 0.195
    # self.i = 0.100
    # self.d = -0.053

    # Initial step dependent gains change 1
    self.p = 0.15
    self.i = 0.1
    self.d = 0.01

    self.b = 1.0
    self.c = 0.5
    self.alpha = 0.8

    self.error_integral = 0
    self.derivative = 0.0
    self.prev_error_d = 0.0
    self.dt = getattr(self, "dt", 0.1)

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    # Proportional Error
    error_p = (self.b * target_lataccel) - current_lataccel

    # Integral Error:  I[k] = I[k-1] + K_i*e_i[k]
    error_i = target_lataccel - current_lataccel
    self.error_integral += self.i * error_i

    # Derivative Error:
    error_d = (self.c * target_lataccel) - current_lataccel
    error_diff = error_d - self.prev_error_d
    self.prev_error_d = error_d
    self.derivative = self.alpha * self.derivative + (1 - self.alpha) * error_diff

    u = self.p * error_p + self.error_integral + self.d * self.derivative

    return float(u)
