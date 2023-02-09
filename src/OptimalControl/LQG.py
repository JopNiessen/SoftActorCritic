"""
Linear Quadratic Gaussian Regulator

i.e. Kalman filter + Linear Quadratic Regulator
"""

from LQR import LQRSolver
from KalmanFilter import KalmanFilter



class LQGSolver:
    def __init__(self, A, B, C, Wx, Wy, Q, R, y0):
        self.x_est = y0
        self.KalmanFilter = KalmanFilter(A, B, C, Wx, Wy)
        self.LQR = LQRSolver(A, B, Q, R)
    
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
