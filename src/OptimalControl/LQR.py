"""
Linear Quadratic Regulator
"""

from scipy import linalg


class LQRSolver:
    def __init__(self, A, B, Q, R):
        self.A = A
        self.B = B.reshape((2,1))
        self.Q = Q
        self.R = R
        self.solve_CARE()

    def update(self, **kwargs):
        self.A = kwargs.get('A', self.A)
        self.B = kwargs.get('B', self.B)
        self.Q = kwargs.get('Q', self.Q)
        self.R = kwargs.get('R', self.R)
        self.S = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)

    def solve_CARE(self, **kwargs):
        self.update(kwargs=kwargs)
        return self.S

    def __call__(self, x):
        return - 1/self.R * self.B.T @ self.S @ x
