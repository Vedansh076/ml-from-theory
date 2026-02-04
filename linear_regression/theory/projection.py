import numpy as np

def projection_matrix(X):
    """
    P = X (X^T X)^(-1) X^T
    Orthogonal projector onto Col(X)
    """
    XtX_inv = np.linalg.pinv(X.T @ X)
    return X @ XtX_inv @ X.T


def project_y(X, y):
    """
    y_hat = P y
    """
    P = projection_matrix(X)
    return P @ y
