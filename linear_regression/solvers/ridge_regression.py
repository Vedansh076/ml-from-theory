import numpy as np

def solve_ridge_regression(X, y, lam=1.0):
    """
    Ridge Regression closed-form solution:
    
    w = (X^T X + λI)^(-1) X^T y
    
    lam: regularization strength (lambda)
    """
    n_features = X.shape[1]
    
    I = np.eye(n_features)
    
    XtX = X.T @ X
    XtX_lam = XtX + lam * I
    
    XtX_lam_inv = np.linalg.pinv(XtX_lam)
    
    w = XtX_lam_inv @ X.T @ y
    
    return w
