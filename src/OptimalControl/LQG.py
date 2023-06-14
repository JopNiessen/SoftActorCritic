"""
Linear Quadratic Gaussian Regulator

i.e. Kalman filter + Linear Quadratic Regulator
"""

import jax.numpy as jnp

from src.OptimalControl.LQR import LQRSolver
from src.OptimalControl.KalmanFilter import KalmanFilter


class LQGSolver:
    def __init__(self, A, B, C, Wx, Wy, Q, R, y0):
        self.x_est = y0
        self.KalmanFilter = KalmanFilter(A, B, C, Wx, Wy)
        B = jnp.identity(len(B)) * B
        self.LQR = LQRSolver(A, B, Q, R, lqg=True)
    
    def step(self, observation, control=None, dt=1):
        if control == None:
            control = self.LQR(self.x_est)
            return control
        else:
            self.x_est += self.KalmanFilter(self.x_est, control, observation) * dt
            control = self.LQR(self.x_est)
            return control
    
    def reset(self, y0):
        self.x_est = y0
