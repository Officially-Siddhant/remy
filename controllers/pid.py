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

    self.error_integral = 0
    self.prev_error = 0
    self.dt = getattr(self, "dt", 0.1)

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    # Accounting for Road Roll
    if future_plan.roll_lataccel:
      print(future_plan.roll_lataccel[0])
      upcoming_roll = future_plan.roll_lataccel[0]
      roll_weight = 0.8
      possible_target_lataccel = target_lataccel - upcoming_roll*roll_weight
      error = possible_target_lataccel - current_lataccel
    else:
      error = (target_lataccel - current_lataccel)

    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error
    return self.p * error + self.i * self.error_integral + (self.d * error_diff)
