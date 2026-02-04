import numpy as np

def design_matrix(x, degree=1):
    """
    Builds X = [1, x, x^2, ..., x^degree]
    This controls the dimension of Col(X)
    """
    x = np.asarray(x).reshape(-1, 1)
    X = np.ones((x.shape[0], 1))

    for d in range(1, degree + 1):
        X = np.hstack((X, x**d))

    return X
