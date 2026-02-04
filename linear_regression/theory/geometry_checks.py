import numpy as np

def check_orthogonality(X, y, y_hat, tol=1e-6):
    """
    Checks X^T (y - y_hat) ≈ 0
    """
    residual = y - y_hat
    return np.all(np.abs(X.T @ residual) < tol)


def matrix_rank(X):
    return np.linalg.matrix_rank(X)


def condition_number(X):
    return np.linalg.cond(X)
