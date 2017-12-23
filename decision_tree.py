import numpy as np
import random
from collections import Counter

from Information_helper import information_gain

def classification_function(y_value):  # y_value is an array of the target value
    y_counter = Counter(y_value)
    return max(y_counter, key=y_counter.get)

class DecisionTree:
    def __init__(self, max_depth=10, minimum_sample_split=2, type="classification"):
        assert type == "classification" or type == "regression"
        self.leaf_function = lambda x: sum(x) if type == "regression" else lambda x: classification_function(x)

        self.max_depth = max_depth
        self.minimum_sample_split = minimum_sample_split
        self.rootNode = None
        self.type = type

    def fit(self, X, y, is_random_forest=True):
        num_features = X.shape[1]

        if is_random_forest:
            num_features = X.shape[1]
            total_feature_to_extract = np.sqrt(num_features)
            feature_indexes = random.sample(range(num_features), total_feature_to_extract)
        else:
            feature_indexes = num_features

        self.rootNode = self._build_tree(X, y, feature_indexes, depth=0)

    def predict(self, X):
        current_node = self.rootNode
        while type(current_node) is not TreeNode:
            current_threshold = current_node.get_threshold()
            current_feature_index = current_node.get_feature_index()

            if X[current_feature_index] <= current_threshold:
                current_node = current_node.get_left_node()
            else:
                current_node = current_node.get_right_node()

        return current_node


    # do recursion in build tree function
    def _build_tree(self, X, y, feature_indexes, depth):

        if depth == self.max_depth:
            return self.leaf_function(y)

        best_feature_index, best_threshold = self._find_best_split(X, y, feature_indexes=feature_indexes)
        leftNode = self._build_tree(X, y, feature_indexes, depth=depth+1)
        rightNode = self._build_tree(X, y, feature_indexes, depth=depth+1)

        return TreeNode(leftNode, rightNode, best_feature_index, best_threshold)

    def _find_best_split(self, X, y, feature_indexes):

        best_gain = 0
        best_feature_index = 0
        best_threshold = 0

        for feature_index in feature_indexes:
            current_x = sorted(set(X[:, feature_index]))

            for iteration in range(len(current_x)-1):
                threshold = (current_x[iteration]+current_x[iteration+1])/2  # for continuous variable
                x_left, y_left, x_right, y_right = self._split(X, y, feature_index, threshold)
                gain = information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

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
    def __init__(self, leftNode, rightNode, feature_index, threshold):
        self.leftNode = leftNode
        self.rightNode = rightNode
        self.feature_index = feature_index
        self.threshold = threshold

    def get_left_node(self):
        return self.leftNode
    def get_right_node(self):
        return self.rightNode
    def get_feature_index(self):
        return self.feature_index
    def get_threshold(self):
        return self.threshold





decision_tree = DecisionTree()
