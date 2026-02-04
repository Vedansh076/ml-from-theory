import numpy as np

def solve_normal_equation(X, y):
    """
    w = (X^T X)^(-1) X^T y
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    return XtX_inv @ X.T @ y
