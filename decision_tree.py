

class DecisionTree:
    def __init__(self, max_depth=10, minimum_sample_split=2):
        self.max_depth = max_depth
        self.minimum_sample_split = minimum_sample_split

    def _find_best_split(self, X, y, feature_indexes):

        for feature_index in feature_indexes:
            current_x = sorted(set(X[:, feature_index]))

            for iteration in range(len(current_x)-1):
                threshold = (current_x[iteration]+current_x[iteration+1])/2  # for continuous variable
                self._split(X, y, feature_index, threshold)

        pass

    def _split(self, X, y, feature_index, threshold):
        pass


