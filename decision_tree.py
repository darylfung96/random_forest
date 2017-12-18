import numpy as np

class DecisionTree:
    def __init__(self, max_depth=10, minimum_sample_split=2):
        self.max_depth = max_depth
        self.minimum_sample_split = minimum_sample_split

    def fit(self, X, y):
        pass

    def _build_tree(self):
        pass

    def _find_best_split(self, X, y, feature_indexes):

        for feature_index in feature_indexes:
            current_x = sorted(set(X[:, feature_index]))

            for iteration in range(len(current_x)-1):
                threshold = (current_x[iteration]+current_x[iteration+1])/2  # for continuous variable
                self._split(X, y, feature_index, threshold)

        pass

    def _split(self, X, y, feature_index, threshold):
        X_left = []
        y_left = []
        X_right = []
        y_right = []
        for i in range(len(X)):
            if X[i][feature_index] <= threshold:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        X_left, y_left, X_right, y_right = np.array(X_left), np.array(y_left), np.array(X_right), np.array(y_right)

        return X_left, y_left, X_right, y_right



class TreeNode:
    def __init__(self, leftNode, rightNode):
        self.leftNode = None
        self.rightNode = None





