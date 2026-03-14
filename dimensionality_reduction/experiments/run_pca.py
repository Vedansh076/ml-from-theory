import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from dimensionality_reduction.pca import MyPCA
from dimensionality_reduction.sklearn_pca import sklearn_pca

def main():
    data = load_iris()

    X = data.data
    y = data.target

    feature_names = data.feature_names
    print("Original dataset shape:", X.shape)

    n_components =2
    pca_model = MyPCA(n_components)
    X_my_pca = pca_model.fit_transform(X)

    X_pca_sklearn, sklearn_model = sklearn_pca(X, n_components)

    diff = np.linalg.norm(X_my_pca - X_pca_sklearn)
    print("Difference between implementations:", diff)

    print("\nExplained variance ratio (sklearn PCA):")
    print(sklearn_model.explained_variance_ratio_)


    plt.figure(figsize=(8, 6))

    plt.scatter(
        X_my_pca[:, 0],
        X_my_pca[:, 1],
        c=y,
        cmap="viridis"
    )

    plt.title("PCA from Scratch (Iris Dataset)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.show()


if __name__ == "__main__":
    main()
