from . import BaseController
import numpy as np
import scipy.linalg as l

class Controller(BaseController):
  # def __init__(self,qe,qj,r,tau,lamb,dt):
  def __init__(self,):
    self.qe = 0.22
    self.qj = 0.07
    self.r  = 0.07

    self.dt = 1.0
    self.tau = 0.3

    self.time_const = 0.14 # choose ~0.3–0.5 s
    self.lamb = self.time_const / self.tau

    A = np.array([[1.0 - self.lamb, 0.0],
                  [-self.lamb / self.dt, 0.0]])
    B = np.array([[self.lamb],
                  [self.lamb / self.dt]])

    Q = np.diag([self.qe, self.qj])
    R = np.array([[self.r]])

    P = l.solve_discrete_are(A, B, Q, R)
    self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    self.prev_a = None
    self.N = 5


    self.steer_factor = 12.6  # fixed
    self.minimum_v = 30 # fixed
    self.beta = 1.1 # fixed
    self.p = 1.9 # fixed



  def update(self, target_lataccel, current_lataccel, state, future_plan):

    L = 0 #lag
    H = min(self.N - 1 + L, len(future_plan.lataccel))
    future = [float(x) for x in future_plan.lataccel[:H]]

    targets = [float(target_lataccel)] + future
    weights = np.arange(1, len(targets) + 1, dtype=float) ** self.p
    weights /= weights.sum()

    target_smooth = float(np.dot(weights, targets))

    # states
    e = current_lataccel - self.beta*target_smooth
    j = 0.0 if self.prev_a is None else (current_lataccel - self.prev_a) / self.dt
    self.prev_a = current_lataccel

    # LQR control
    x = np.array([[e], [j]])
    self.step = getattr(self, "step",0) + 1
    fade = min(1.0, self.step/100)
    du = float(-(self.K @ x))
    du *= fade**3

    # Steer control
    steer_lataccel_target = (target_smooth - getattr(state, "roll_lataccel", 0.0))
    u_steer = steer_lataccel_target * (self.steer_factor / max(self.minimum_v, state.v_ego))
    u_steer = u_steer / (1.0 + np.abs(u_steer))

    return du + u_steer