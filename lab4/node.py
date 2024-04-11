import copy
import numpy as np


class Node:
    def __init__(self):
        self.left_child = None
        self.right_child = None
        self.feature_idx = None
        self.feature_value = None
        self.node_prediction = None

    @staticmethod
    def gini_best_score(y, possible_splits):
        best_gain = -np.inf
        best_idx = None

        for idx in possible_splits:
            left_count = np.sum(y[:idx + 1] == 0)
            right_count = np.sum(y[idx + 1:] == 0)
            total_count = len(y)
            gini_left = 1 - ((left_count / (idx + 1)) ** 2 + ((idx + 1 - left_count) / (idx + 1)) ** 2)
            gini_right = 1 - ((right_count / (total_count - (idx + 1))) ** 2 + (
                        (total_count - (idx + 1) - right_count) / (total_count - (idx + 1))) ** 2)
            gini_score = (idx + 1) / total_count * gini_left + (total_count - (idx + 1)) / total_count * gini_right

            gain = 1 - gini_score
            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        return best_idx, best_gain

    @staticmethod
    def split_data(X, y, idx, val):
        left_mask = X[:, idx] < val
        return (X[left_mask], y[left_mask]), (X[~left_mask], y[~left_mask])

    @staticmethod
    def find_possible_splits(data):
        possible_split_points = []
        for idx in range(len(data) - 1):
            if data[idx] != data[idx + 1]:
                possible_split_points.append(idx)
        return possible_split_points

    def find_best_split(self, X, y, feature_subset):
        best_gain = -np.inf
        best_split = None

        # Select random features if feature_subset is specified
        if feature_subset is not None:
            features = np.random.choice(X.shape[1], feature_subset, replace=False)
        else:
            features = range(X.shape[1])

        for d in features:
            order = np.argsort(X[:, d])
            y_sorted = y[order]
            possible_splits = self.find_possible_splits(X[order, d])
            idx, value = self.gini_best_score(y_sorted, possible_splits)
            if value > best_gain:
                best_gain = value
                best_split = (d, [idx, idx + 1])

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

        self.node_prediction = np.mean(y)
        if len(X) == 1 or self.node_prediction == 0 or self.node_prediction == 1:
            return True

        self.feature_idx, self.feature_value = self.find_best_split(X, y, params["feature_subset"])
        if self.feature_idx is None:
            return True

        (X_left, y_left), (X_right, y_right) = self.split_data(X, y, self.feature_idx, self.feature_value)

        if len(X_left) == 0 or len(X_right) == 0:
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
