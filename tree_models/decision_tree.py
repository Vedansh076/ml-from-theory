import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text


class InterpretableDecisionTree:
    def __init__(self, max_depth=None):
        self.model = DecisionTreeClassifier(max_depth=max_depth)
        self.feature_names = None

    def fit(self, X, y, feature_names=None):
        self.feature_names = feature_names
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def print_rules(self):
        print("=== TREE STRUCTURE ===")
        tree_rules = export_text(self.model, feature_names=self.feature_names)
        print(tree_rules)

    def trace_sample(self, x_sample):
        print("=== DECISION PATH ===")
        node_indicator = self.model.decision_path([x_sample])
        leaf_id = self.model.apply([x_sample])

        feature = self.model.tree_.feature
        threshold = self.model.tree_.threshold

        for node_id in node_indicator.indices:
            if leaf_id[0] == node_id:
                print(f"Reached leaf node {node_id}")
                print("Predicted class:",
                      np.argmax(self.model.tree_.value[node_id]))
                break

            else:
                f = feature[node_id]
                t = threshold[node_id]

                if x_sample[f] <= t:
                    decision = "LEFT"
                else:
                    decision = "RIGHT"

                print(f"Node {node_id}: "
                      f"{self.feature_names[f]} <= {t:.3f} "
                      f"→ Go {decision}")