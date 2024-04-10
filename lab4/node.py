import copy

import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    def gini_best_score(self, y, possible_splits):
        best_gain = -np.inf
        best_idx = 0
        total_count = len(y)

        for idx in possible_splits:
            left_y = y[:idx + 1]
            right_y = y[idx + 1:]

            # Calculate Gini score for the split
            gini_left = 1 - sum((np.sum(left_y == c) / len(left_y)) ** 2 for c in np.unique(left_y))
            gini_right = 1 - sum((np.sum(right_y == c) / len(right_y)) ** 2 for c in np.unique(right_y))

            # Calculate weighted Gini score
            weighted_gini = (len(left_y) / total_count) * gini_left + (len(right_y) / total_count) * gini_right

            # Update best gain and index if current split provides better gain
            if weighted_gini > best_gain:
                best_gain = weighted_gini
                best_idx = idx

        return best_idx, best_gain

    def split_data(self, X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    def find_possible_splits(self, data):
        possible_split_points = []
        for idx in range(data.shape[0] - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None

        for feature_idx in feature_subset:
            possible_splits = self.find_possible_splits(X[:, feature_idx])
            idx, value = self.gini_best_score(y, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (feature_idx, [idx, idx + 1])

        if best_split is None:
            return None, None

        best_value = np.mean(X[best_split[1], best_split[0]])

        return best_split[0], best_value

    def predict(self, x):
        if self.feature_idx is None:
            return self.node_prediction
        if x[self.feature_idx] < self.feature_value:
            return self.left_child.predict(x)
        else:
            return self.right_child.predict(x)

    def train(self, X, y, params):

        if "feature_subset" not in params:
            params["feature_subset"] = range(X.shape[1])  # Use all features by default

        feature_subset = params["feature_subset"]
        if isinstance(feature_subset, int):
            feature_subset = [feature_subset]

        self.node_prediction = np.mean(y)
        if X.shape[0] == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, feature_subset)
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if X_left.shape[0] == 0 or X_right.shape[0] == 0:
            self.feature_idx = None
            return True

        # max tree depth
        if params["depth"] is not None:
            params["depth"] -= 1
        if params["depth"] == 0:
            self.feature_idx = None
            return True

        # create new nodes
        self.left_child, self.right_child = Node(), Node()
        self.left_child.train(X_left, y_left, copy.deepcopy(params))
        self.right_child.train(X_right, y_right, copy.deepcopy(params))
