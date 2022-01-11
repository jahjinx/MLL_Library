"""
=============================================================================
K-Nearest Neighbors Classification
=============================================================================
An implementation of the K-Nearest Neighbors algorithm for classification using numpy and collections.
"""

import numpy as np
from collections import Counter
from ..metrics.MLLL_distance import *
from ..metrics.MLLL_score import *

class KNeighborsClassifier():
    """
    K Nearest Neighbors Classifier.
    Define

    Parameters
    ---------------
    n_neighbors : int, default=3
    metric : callable, default=minkowski, distance metric to be used for measuring distance between points. 
             Available metrics:   manhattan = minkowski @ order-p = 1 
                                  euclidean = minkowski @ order-p = 2 
                                  chebyshev = minkowski @ order-p = ∞
                                  minkowski
    p : int, default=2, power parameter for minkowski metric

    Notes
    ---------------
    Cap 'X' represents the whole, Lower 'x' represents each part of the whole
    """

    def __init__(self, n_neighbors=3, metric=minkowski, p=2):
        """
        Class constructor where we define KNN parameters
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        """
        Fit the model according to the given training data.

        Parameters
        ---------------
        X : training samples
        y : training labels

        Returns
        ---------------
        self : object
        """

        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for given testing data.

        Parameters
        ---------------
        X : test samples

        Returns
        ---------------
        numpy array of predicted labels
        """

        # List Comprehension: for each test sample (x) in test samples (X),
        # run helper function to get predicted label and store to list
        # x represents a single point in X, ex: [5.1, 3.5, 1.4, 0.2]
        predicted_labels = [self._predict(x) for x in X]

        # return above list as array
        return np.array(predicted_labels)

    def _predict(self, x):
        """
        Helper function for predict()

        Parameters
        ---------------
        x : A test point.

        Returns
        ---------------
        Predicted label for given test point.
        """

        # Compute Distances
        # List Comprehension: for each point (x_train) in our training set (X_train), ex: [5.1, 3.5, 1.4, 0.2],
        # get distance to x from test samples (X), ex [4.9, 3. , 1.4, 0.2]
        if self.metric == minkowski:
            distances = [self.metric(x, x_train, self.p) for x_train in self.X_train]


        else:
            distances = [self.metric(x, x_train) for x_train in self.X_train]

        # Sort Distances
        # np.argsort() will sort a 1d array from least to greatest and output the value's original index in the new sorted order
        # This will allow us to retain the proper index to match the closest neighbors to their proper label.
        # get indicies from only k nearest values
        k_indicies = np.argsort(distances)[0:self.n_neighbors]

        # List Comprehension: get k nearest samples & their labels
        k_nearest_labels = [self.y_train[i] for i in k_indicies]

        # get most common class label among k nearest sampples
        # take k_nearest_lables list above and count the [1] most common value. returns list of tuples: [(most common value, number of times it appears)]
        most_common = Counter(k_nearest_labels).most_common(1)

        # access first tuple in list, and first value in tuple
        return most_common[0][0]