import numpy as np


def entropy(y):
    """
    Compute entropy of a label vector.

    Entropy measures how mixed the classes are.
    - Entropy = 0 → pure node
    - Entropy = high → mixed node
    """

    classes, counts = np.unique(y, return_counts=True)

    probabilities = counts / counts.sum()

    return -np.sum(probabilities * np.log2(probabilities))

#compute information gain after split
def information_gain(X_column, y, threshold):
   
    parent_entropy = entropy(y)

    # split dataset
    left_mask = X_column <= threshold
    right_mask = X_column > threshold

    y_left = y[left_mask]
    y_right = y[right_mask]

    # invalid split
    if len(y_left) == 0 or len(y_right) == 0:
        return 0

    # weighted child entropy
    w_left = len(y_left) / len(y)
    w_right = len(y_right) / len(y)

    child_entropy = (
        w_left * entropy(y_left) +
        w_right * entropy(y_right)
    )

    return parent_entropy - child_entropy

def best_split_entropy(X, y):
    """
    Find best feature and threshold using information gain.
    """

    best_feature = None
    best_threshold = None
    best_gain = -1

    n_features = X.shape[1]

    for feature in range(n_features):

        values = np.unique(X[:, feature])

        for threshold in values:

            gain = information_gain(X[:, feature], y, threshold)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gain