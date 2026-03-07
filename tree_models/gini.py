import numpy as np

def gini_impurity(y):
    classes, counts = np.unique(y, return_counts=True) #if y={0,0,0,1,1} classes={0,1},counts={3,2}
    probs = counts / counts.sum()
    return 1 - np.sum(probs ** 2) #compute gini index


#compute gini index after split
def split_gini(X_column, y, threshold):

    left_mask = X_column <= threshold
    right_mask = X_column > threshold

    y_left = y[left_mask]
    y_right = y[right_mask]

    g_left = gini_impurity(y_left)
    g_right = gini_impurity(y_right)

    w_left = len(y_left) / len(y)
    w_right = len(y_right) / len(y)

    return w_left * g_left + w_right * g_right

#finding best feature based on gini index 
def best_split_gini(X, y):

    best_feature = None
    best_threshold = None
    best_gini = float("inf")

    n_features = X.shape[1]

    for feature in range(n_features):

        values = np.unique(X[:, feature])

        for threshold in values:

            g = split_gini(X[:, feature], y, threshold)

            if g < best_gini:
                best_gini = g
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold, best_gini