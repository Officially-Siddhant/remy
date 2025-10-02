from . import BaseController
import numpy as np
import scipy.linalg as l

class Controller(BaseController):
  def __init__(self,):
    self.qe = 60.0
    self.qj = 5.0
    self.r  = 0.2

    self.dt = getattr(self, "dt", 0.1)
    self.tau = 0.4                    # choose ~0.3–0.5 s
    self.lamb = self.dt / self.tau

    A = np.array([[1.0 - self.lamb, 0.0],
                  [-self.lamb / self.dt, 0.0]])
    B = np.array([[self.lamb],
                  [self.lamb / self.dt]])

    Q = np.diag([self.qe, self.qj])
    R = np.array([[self.r]])

    P = l.solve_discrete_are(A, B, Q, R)
    self.K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    self.ff_gain = 0.2

    self.prev_a = None  # for jerk
    self.prev_ref = None
    self.alpha_roll = 0.45  # EMA on roll FF (tune 0.6–0.95)
    self.roll_ff = 0.0
    self.roll_N = 4


  def update(self, target_lataccel, current_lataccel, state, future_plan):

    rolls = getattr(future_plan, "roll_lataccel", []) or []
    if rolls:
      # average first N points to denoise the preview
      r0 = float(np.mean(rolls[:min(self.roll_N, len(rolls))]))
      # EMA smooth
      self.roll_ff = self.alpha_roll * self.roll_ff + (1.0 - self.alpha_roll) * r0
    else:
      # decay toward zero if no preview available
      self.roll_ff = self.alpha_roll * self.roll_ff

    ref = target_lataccel + self.roll_ff

    self.prev_ref = ref

    u_ff = ref*self.ff_gain  # add preview if desired

    # --- Feedback states: error & jerk ---
    e = current_lataccel - ref
    j = 0.0 if self.prev_a is None else (current_lataccel - self.prev_a) / self.dt
    self.prev_a = current_lataccel

    x = np.array([[e], [j]])
    du = float(-(self.K @ x))  # LQR correction

    u = u_ff + du  # final command (clip outside if needed)
    return float(u)