from . import BaseController
import numpy as np
import scipy.linalg as la

class Controller(BaseController):
  def __init__(self, q_a, q_j, q_i, r, dt, tau):
  # def __init__(self, ):
    self.qa, self.qj, self.qi = q_a, q_j, q_i
    self.r = r
    self.dt, self.tau = dt, tau
    '''
    self.alpha_roll = 1.00
    self.qa, self.qj, self.qi = 0.1,0.2,0.5
    self.r = 0.5
    self.dt = 1.0
    self.tau = 0.40 # because there is a clear lag between action and current lat accel
    # '''
    self.lamb = 1.0 - np.exp(-self.dt / self.tau)

    A = np.array([[1.0 - self.lamb, 0.0],
                  [-self.lamb / self.dt, 0.0]])
    B = np.array([[self.lamb],
                  [self.lamb / self.dt]])

    C_sel = np.array([[1.0, 0.0]])
    A_bar = np.block([[A,                 np.zeros((2,1))],
                      [-self.dt*C_sel,    np.ones((1,1))]])
    B_bar = np.block([[B],
                      [np.zeros((1,1))]])

    Q = np.diag([self.qa, self.qj, self.qi])
    Rm = np.array([[self.r]])
    S = la.solve_discrete_are(A_bar, B_bar, Q, Rm)
    self.K = np.linalg.inv(Rm + B_bar.T @ S @ B_bar) @ (B_bar.T @ S @ A_bar)

    self.prev_a = None
    self.roll_ff = 0.0
    self.xi = 0.0
    self.N = 5
    self.roll_N = 5

    self.steer_factor = 12.5
    self.minimum_v = 20.0
    self._u_prev = 0.0

    self.k_roll = 0.20
    self.gamma = 1.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):

      # --- 0) Fade-in like the 2DOF PID ---
      self.step = getattr(self, "step", 0) + 1
      fade = min(self.step / 100.0, 1.0)  # ramp over ~100 steps
      Kp_eff = fade  # for 'a' path
      Kd_eff = fade  # for 'j' path
      Ki_eff = fade ** 3  # gentlest on integrator

      # --- 1) Feed-forward roll (same pattern as 2DOF PID) ---
      rolls = [getattr(state, "roll_lataccel", 0.0)] + (getattr(future_plan, "roll_lataccel", []) or [])
      u_ff = -self.k_roll * np.average(rolls[:self.roll_N]) if getattr(self, "roll_N", 0) > 0 else 0.0

      # --- 2) Reference smoothing (look-ahead weighted average) ---
      future = future_plan.lataccel[:max(0, self.N - 1)]
      targets = [float(target_lataccel)] + [float(x) for x in future]
      weights = np.array([4, 3, 2, 2, 2, 1][:len(targets)], dtype=float)
      weights /= weights.sum()
      target_smooth = float(np.dot(weights, targets))

      # --- 3) Roll cancellation in reference (like 2DOF PID steering term) ---
      r = target_smooth - self.gamma * state.roll_lataccel

      # --- 4) Measurement + jerk estimate ---
      y = float(current_lataccel)
      if getattr(self, "prev_a", None) is None:
          j = 0.0
      else:
          j = (y - self.prev_a) / self.dt
      self.prev_a = y

      # LQI integrator
      self.xi = self.xi + Ki_eff * self.dt * (r - y)

      # --- 6) LQI control (apply fade on 'a' and 'j' channels, full k_i on xi) ---
      k_a, k_j, k_i = float(self.K[0, 0]), float(self.K[0, 1]), float(self.K[0, 2])
      u_lqi = - (Kp_eff * k_a * y + Kd_eff * k_j * j + k_i * self.xi)

      # --- 7) (Commented out) any accel/speed gain scheduling inside LQI ---
      # aego = float(getattr(state, "a_ego", 0.0))
      # v    = float(getattr(state, "v_ego", 20.0))
      # scale = min(1.0, 1.0 - max(0.0, abs(target_smooth) - 1.0) * 0.23)
      # u_lqi *= scale

      v = float(getattr(state, "v_ego", 20.0))
      steer_lataccel_target = (target_smooth - getattr(state, "roll_lataccel", 0.0))
      u_steer = steer_lataccel_target * (self.steer_factor / max(self.minimum_v, v))
      u_steer = u_steer / (1.0 + np.abs(u_steer))  # smooth saturator like PID

      # --- 9) Sum of parts ---
      return u_lqi + u_steer + u_ff