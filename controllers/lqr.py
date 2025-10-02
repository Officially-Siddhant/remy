from . import BaseController
import numpy as np
import scipy.linalg as l

class Controller(BaseController):
  def __init__(self,):
    self.qe = 100.0
    self.qj = 10.0
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

    self.prev_a = None  # for jerk
    self.prev_ref = None
    self.alpha_roll = 0.9  # EMA on roll FF (tune 0.6–0.95)
    self.roll_ff = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    # --- Feedforward reference: target + road-roll (EMA) ---
    rolls = getattr(future_plan, "roll_lataccel", []) or []
    roll0 = float(rolls[0]) if len(rolls) > 0 else 0.0

    # if alpha_roll = 1, then roll_ff is the exact roll
    self.roll_ff = self.alpha_roll * self.roll_ff + (1 - self.alpha_roll) * roll0

    ref = target_lataccel + self.roll_ff

    u_ff = ref * 0.25  # add preview if desired

    e = current_lataccel - ref
    j = 0.0 if self.prev_a is None else (current_lataccel - self.prev_a) / self.dt
    self.prev_a = current_lataccel

    x = np.array([[e], [j]])
    du = float(-(self.K @ x))  # LQR correction

    u = u_ff + du
    return float(u)