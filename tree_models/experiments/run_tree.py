import numpy as np
from sklearn.datasets import load_iris

# our interpretable decision tree implementation
from tree_models.decision_tree import InterpretableDecisionTree

# entropy based split (ID3)
from tree_models.entropy import best_split_entropy

# gini based split (CART)
from tree_models.gini import best_split_gini


def main():

    # Load the Iris dataset from sklearn
    # This gives us:
    #   X → feature matrix (150 samples × 4 features)
    #   y → class labels (0, 1, 2)

    data = load_iris()

    X = data.data
    y = data.target
    feature_names = data.feature_names


    # Train our interpretable decision tree
    # max_depth is kept small so the rules remain readable

    tree = InterpretableDecisionTree(max_depth=3)

    tree.fit(X, y, feature_names=feature_names)


    # Print the rules the tree learned.
    # This helps us see how the model is making decisions.

    print("\nLearned Decision Rules:")
    tree.print_rules()


    # Let's follow how one sample moves through the tree.
    # This shows the exact decisions taken at each node.

    print("\nTracing first sample through the tree:")
    tree.trace_sample(X[0])


    # Now we manually compute the best split using entropy.
    # This corresponds to the ID3 decision tree algorithm.

    print("\nBest split using Entropy (ID3):")

    feature_e, threshold_e, gain = best_split_entropy(X, y)

    print("Feature:", feature_names[feature_e])
    print("Threshold:", threshold_e)
    print("Information Gain:", gain)


    # Next we compute the best split using the Gini index.
    # This is the impurity measure used by CART trees
    # (and by sklearn's DecisionTreeClassifier).

    print("\nBest split using Gini (CART):")

    feature_g, threshold_g, gini_value = best_split_gini(X, y)

    print("Feature:", feature_names[feature_g])
    print("Threshold:", threshold_g)
    print("Gini Reduction:", gini_value)


    # Quick comparison: do both metrics choose the same split?
    # Often they do, but sometimes they differ slightly.

    print("\nComparison Summary")

    if feature_e == feature_g and threshold_e == threshold_g:
        print("Both entropy and gini choose the same split.")
    else:
        print("Entropy and gini choose different splits.")


if __name__ == "__main__":
    main()