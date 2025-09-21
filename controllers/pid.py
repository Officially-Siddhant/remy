from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    self.d = -0.053
    self.error_integral = 0
    self.prev_error = 0
    self.window = 5 #I try and use a feedforward window which predicts the action in the next 5 steps.
    self.weight = [0.6,0.5,0.4,0.3,0.2]
    self.a_ff = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    print("These are the states: \n", state)
    print(type(future_plan))
    print("Future plan: \n", future_plan)
    sum_weight = 0.0

    for step in range(self.window):
      #do I try and calculate the gradient for each step, or simply adhere to the trajectory between the next 5 steps?
      # weighted average of the first N planned lateral-acceleration points
      if future_plan.lataccel[step+1]:
        self.a_ff += self.weight[step] * future_plan.lataccel[step+1]
        sum_weight += self.weight[step]
      else:
        break
    self.a_ff = self.a_ff / sum_weight

    error = (target_lataccel - current_lataccel)
    error = (self.a_ff - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error


    return self.p * error + self.i * self.error_integral + self.d * error_diff
