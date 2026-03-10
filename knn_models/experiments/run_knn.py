import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from knn_models.knn import KNearestNeighbors
from knn_models.similarity import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance
)


def run_knn(distance_fn, name):

    print(f"\nRunning KNN with {name}")

    model = KNearestNeighbors(k=5, distance_fn=distance_fn)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)

    print("Accuracy:", acc)


if __name__ == "__main__":

    data = load_iris()

    X = data.data
    y = data.target

    global X_train, X_test, y_train, y_test

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    run_knn(euclidean_distance, "Euclidean Distance")

    run_knn(manhattan_distance, "Manhattan Distance")

    run_knn(cosine_distance, "Cosine Distance")