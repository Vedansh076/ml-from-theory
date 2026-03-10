import numpy as np


class KNearestNeighbors:

    def __init__(self, k=3, distance_fn=None):
        self.k = k
        self.distance_fn = distance_fn
        self.X_train = None
        self.y_train = None


    def fit(self, X, y):
        # KNN doesn't train anything
        # it just stores the dataset
        self.X_train = X
        self.y_train = y


    def predict(self, X):

        predictions = []

        for x in X:

            distances = []

            # compute distance to every training point
            for x_train in self.X_train:

                d = self.distance_fn(x, x_train)

                distances.append(d)

            # get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # get labels of those neighbors
            k_labels = self.y_train[k_indices]

            # majority vote
            label = np.bincount(k_labels).argmax()

            predictions.append(label)

        return np.array(predictions)