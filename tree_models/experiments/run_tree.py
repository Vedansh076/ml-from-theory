import numpy as np
from sklearn.datasets import load_iris
from tree_models.decision_tree import InterpretableDecisionTree

def main():
    data = load_iris()
    X = data.data
    y = data.target
    feature_names = data.feature_names

    tree = InterpretableDecisionTree(max_depth=3)
    tree.fit(X, y, feature_names=feature_names)

    tree.print_rules()

    print("\nTracing first sample:")
    tree.trace_sample(X[0])

if __name__ == "__main__":
    main()