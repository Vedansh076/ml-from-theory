import numpy as np
import matplotlib.pyplot as plt

from linear_regression.data.datasets import linear_data
from linear_regression.theory.subspace_analysis import design_matrix
from linear_regression.theory.geometry_checks import (
    check_orthogonality,
    matrix_rank,
    condition_number
)
from linear_regression.theory.projection import project_y
from linear_regression.solvers.normal_equation import solve_normal_equation
from linear_regression.solvers.sklearn_solver import solve_sklearn
from linear_regression.solvers.ridge_regression import solve_ridge_regression


from linear_regression.experiments.double_descent import run_double_descent

def main():
    # --- Choose Dataset ---
    x, y = linear_data(n=30, noise=2.0)

    # --- Choose Feature Space ---
    X = design_matrix(x, degree=1)

    # --- Theory: Projection ---
    y_hat_proj = project_y(X, y)

    # --- Algebra: Normal Equation ---
    w = solve_normal_equation(X, y)
    y_hat_ne = X @ w

    # --- Library: Sklearn ---
    w_sk, y_hat_sk = solve_sklearn(X, y)

    # --- Ridge Regression ---
    lam = 10.0
    w_ridge = solve_ridge_regression(X, y, lam=lam)
    y_hat_ridge = X @ w_ridge


    # --- Diagnostics ---
    print("Rank(X):", matrix_rank(X))
    print("Condition Number:", condition_number(X))
    print("Residual ⟂ Col(X):", check_orthogonality(X, y, y_hat_sk))
    print("Projection vs Normal Eq Difference:",
          np.linalg.norm(y_hat_proj - y_hat_ne))
    print("Projection vs Sklearn Difference:",
          np.linalg.norm(y_hat_proj - y_hat_sk))
    print("Ridge vs Normal Eq Difference:",
      np.linalg.norm(y_hat_ne - y_hat_ridge))


    # --- Visualization ---
    plt.scatter(x, y, label="Data")
    plt.plot(x, y_hat_sk, color="red", label="Projection Fit")
    plt.plot(x, y_hat_ridge, linestyle="--", label=f"Ridge (λ={lam})")
    plt.legend()
    plt.title("Linear Regression as Orthogonal Projection")
    plt.show()

EXPERIMENT = "double"  # or "linear"
if __name__ == "__main__":
    if EXPERIMENT == "linear":
        main()
    elif EXPERIMENT == "double":
        run_double_descent()

