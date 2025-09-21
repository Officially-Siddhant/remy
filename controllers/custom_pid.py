from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.195
    self.i = 0.100
    #self.d = -0.053
    self.d = 0.02
    self.error_integral = 0
    self.prev_error = 0
    self.window = 5 #I try and use a feedforward window which predicts the action in the next 5 steps.
    self.weights = [0.6,0.5,0.4,0.3,0.2]
    #self.weights = [0.7, 0.5, 0.4, 0.1, 0.1]
    #self.weights = [1.0, 0.8, 0.5, 0.3, 0.2]
    self.dt = getattr(self, "dt", 0.1)


  def update(self, target_lataccel, current_lataccel, state, future_plan):
    #print("These are the states: \n", state)
    #print(type(future_plan))
    #print("Future plan: \n", future_plan)
    lat = future_plan.lataccel
    N = min(self.window, len(lat))  # how many points we actually have


    if N == 0:
      a_ff = 0.0
    else:
      # ensure weights match N and are normalized
      w = np.array(self.weights[:N], dtype=float)
      w /= w.sum()
      a_ff = float(np.dot(w, np.array(lat[:N], dtype=float)))

    #error = (target_lataccel - current_lataccel)
    error = (a_ff - current_lataccel)
    self.error_integral += error*self.dt
    error_diff = (error - self.prev_error) / self.dt
    self.prev_error = error


    return self.p * error + self.i * self.error_integral + self.d * error_diff
