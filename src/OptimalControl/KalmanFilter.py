"""
Kalman filter
"""

import jax.numpy as jnp
from scipy import linalg


class KalmanFilter:
    def __init__(self, A, B, C, Wx, Wy):
        self.A = A
        self.B = B
        self.C = C
        self.Wx = Wx
        self.Wy = Wy
        self.solve_CARE()

    def update(self, **kwargs):
        self.A = kwargs.get('A', self.A)
        self.B = kwargs.get('B', self.B)
        self.C = kwargs.get('C', self.C)
        self.Wx = kwargs.get('Wx', self.Wx)
        self.Wy = kwargs.get('Wy', self.Wy)
        self.P = linalg.solve_continuous_are(self.A, self.C, self.Wx, self.Wy)
        self.L = self.P @ self.C @ jnp.linalg.inv(self.Wy)

    def solve_CARE(self, **kwargs):
        self.update(kwargs=kwargs)
        return self.L

    def __call__(self, x_est, u, y):
        return self.A @ x_est + self.B * u + self.L @ (y - self.C @ x_est)
