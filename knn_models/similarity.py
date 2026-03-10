import numpy as np


# Euclidean distance
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Manhattan distance
def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))


# Cosine distance (1 - similarity)
def cosine_distance(x1, x2):
    dot = np.dot(x1, x2)
    norm = np.linalg.norm(x1) * np.linalg.norm(x2)

    if norm == 0:
        return 0

    return 1 - (dot / norm)