import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)

        # NOTE: was originally confused on what n represented, turns out it represents the length of the array itself
        n = len(y_true)     # len(y_true) == len(y_pred) in this case.

        # calculate the y_pred optimized version (using the np.clip() method)
        '''
        the np.clip() method, takes 3 parameters:
        - first is the orignal numpy array itself
        - second is the minimum value
        - third is the maximum value
        '''
        y_pred_clipped = np.clip(y_pred, a_min=1e-7, a_max=1.0-1e-7)
        return np.round((-1/n) * np.sum(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1-(y_pred_clipped))), 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        n = len(y_true)
        y_clip = np.clip(y_pred, a_min=1e-7, a_max=1.0-1e-7)
        return np.round((-1/n) * np.sum(y_true * np.log(y_clip)), 4)
