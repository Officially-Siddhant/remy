from . import BaseController
import numpy as np
import scipy.linalg as la

class Controller(BaseController):
  # def __init__(self, alpha_roll, q_a, q_j, q_i, r, dt, tau):
  def __init__(self, ):
    '''
    self.alpha_roll = alpha_roll
    self.qa, self.qj, self.qi = q_a, q_j, q_i
    self.r = r
    self.dt, self.tau = dt, tau
    '''
    self.alpha_roll = 1.00
    self.qa, self.qj, self.qi = 10.0,10.0,50.0
    self.r = 0.1
    self.dt, self.tau = 0.1,0.4
    # '''
    self.lamb = 1.0 - np.exp(-self.dt / self.tau)

    A = np.array([[1.0 - self.lamb, 0.0],
                  [-self.lamb / self.dt, 0.0]])
    B = np.array([[self.lamb],
                  [self.lamb / self.dt]])

    C = np.array([[1.0, 0.0],
                  [0.0, 1.0]])
    D = np.array([[0.0],
                  [0.0]])

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
  def update(self, target_lataccel, current_lataccel, state, future_plan):
    '''
    targets = getattr(target_lataccel, "target_lataccel", []) or []
    tars = float(targets[0]) if len(tars) > 0 else 0.0
    self.roll_ff = self.alpha_roll * self.roll_ff + (1.0 - self.alpha_roll) * tars
    '''

    future_targets = [float(target_lataccel)] + list(getattr(future_plan, "target_lataccel", []))[:self.N - 1]
    weights = np.array([5, 3, 3, 2, 1][:len(future_targets)], dtype=float)
    weights /= weights.sum()

    target_smooth = float(np.dot(weights, future_targets))
    roll_now = state.roll_lataccel # measured roll
    r = target_smooth - roll_now

    y = float(current_lataccel)  # measured lateral accel (a)

    # jerk estimate from measurement
    if getattr(self, "prev_a", None) is None:
        j = 0.0
    else:
        j = (y - self.prev_a) / self.dt
    self.prev_a = y

    # integral of accel error (LQI)
    e = r - y
    self.xi = self.xi + self.dt * e

    # state for feedback: z = [a, j, x_i]
    z = np.array([[y], [j], [self.xi]])

    # LQI control (K is from augmented design)
    u = float(-(self.K @ z))

    return u