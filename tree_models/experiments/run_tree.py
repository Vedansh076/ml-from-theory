import numpy as np
from sklearn.datasets import load_iris

# our interpretable tree implementation
from tree_models.decision_tree import InterpretableDecisionTree

# entropy-based split (ID3)
from tree_models.entropy import best_split_entropy


def main():

    # ---------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------
    data = load_iris()

    X = data.data
    y = data.target
    feature_names = data.feature_names


    # ---------------------------------------------------------
    # Train interpretable decision tree
    # ---------------------------------------------------------
    tree = InterpretableDecisionTree(max_depth=3)

    tree.fit(X, y, feature_names=feature_names)


    # ---------------------------------------------------------
    # Print learned rules of the tree
    # ---------------------------------------------------------
    print("\nLearned Decision Rules:")
    tree.print_rules()


    # ---------------------------------------------------------
    # Trace how the first sample moves through the tree
    # ---------------------------------------------------------
    print("\nTracing first sample through the tree:")
    tree.trace_sample(X[0])


    # ---------------------------------------------------------
    # Compute best split using entropy (ID3)
    # ---------------------------------------------------------
    print("\nFinding best split using Entropy (ID3):")

    feature, threshold, gain = best_split_entropy(X, y)

    print("Best feature:", feature_names[feature])
    print("Best threshold:", threshold)
    print("Information gain:", gain)


if __name__ == "__main__":
    main()