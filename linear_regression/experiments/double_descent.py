import numpy as np
import matplotlib.pyplot as plt
from linear_regression.solvers.normal_equation import solve_normal_equation
from linear_regression.solvers.ridge_regression import solve_ridge_regression


def true_function(x):
    return x**3 - 0.5*x**2 + 0.2*x


def design_matrix(x, degree):
    X = np.ones((len(x), 1))
    for d in range(1, degree + 1):
        X = np.hstack((X, x.reshape(-1, 1)**d))
    return X


def run_double_descent():

    np.random.seed(42)

    # Generate data
    n_train = 30
    n_test = 200

    x_train = np.linspace(-1, 1, n_train)
    x_test = np.linspace(-1, 1, n_test)

    noise = 0.1 * np.random.randn(n_train)

    y_train = true_function(x_train) + noise
    y_test = true_function(x_test)

    degrees = range(1, 40)

    train_errors = []
    test_errors = []
    condition_numbers = []

    for degree in degrees:

        X_train = design_matrix(x_train, degree)
        X_test = design_matrix(x_test, degree)

        # Solve using pseudoinverse
        w = np.linalg.pinv(X_train) @ y_train

        y_train_pred = X_train @ w
        y_test_pred = X_test @ w

        train_mse = np.mean((y_train - y_train_pred)**2)
        test_mse = np.mean((y_test - y_test_pred)**2)

        cond = np.linalg.cond(X_train)

        train_errors.append(train_mse)
        test_errors.append(test_mse)
        condition_numbers.append(cond)

    # Plot errors
    plt.figure()
    plt.plot(degrees, train_errors, label="Train MSE")
    plt.plot(degrees, test_errors, label="Test MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.title("Double Descent Curve")
    plt.legend()
    plt.show()

    # Plot condition number
    plt.figure()
    plt.plot(degrees, condition_numbers)
    plt.xlabel("Polynomial Degree")
    plt.ylabel("Condition Number")
    plt.title("Condition Number vs Degree")
    plt.show()
