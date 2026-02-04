import numpy as np

def linear_data(n=50, noise=0.0):
    """
    y = 2x + 1 + noise
    If noise = 0, then y ∈ Col(X) for degree=1
    """
    x = np.linspace(-5, 5, n)
    y = 2 * x + 1 + np.random.randn(n) * noise
    return x, y


def quadratic_data(n=50, noise=0.0):
    """
    y = x^2 + noise
    Requires degree >= 2 for perfect projection
    """
    x = np.linspace(-5, 5, n)
    y = x**2 + np.random.randn(n) * noise
    return x, y
