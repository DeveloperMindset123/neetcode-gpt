import numpy as np
from typing import List


class Solution:
    def forward_and_backward(
        self,
        x: List[float],
        W1: List[List[float]],
        b1: List[float],
        W2: List[List[float]],
        b2: List[float],
        y_true: List[float],
    ) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # first, using architecture provided, implement the forward pass logic that's taking place here
        z1 = np.dot(W1, x) + b1

        relu_res = np.maximum(
            0, z1
        )  # maximum, allows elementwise comparison, modifies the neurons, killing 0s

        # this is also the y_pred
        # This is where the stacking component comes into picture
        # we are recalculating, based on the initial neurons modified by the ReLu activation function
        z2 = (
            np.dot(W2, relu_res) + b2
        )  # utilize the result of the activation function, applying the chain rule here
        loss = np.mean(np.square(z2 - y_true))

        # back propogation logic takes place here
        # here we will be working backwards based on the mathematical formula provided

        dl_dz2 = 2 * (z2 - y_true) / len(z2)  # output gradient, n = len(z2) = len(y_pred) and z2 == y_pred
        dl_dw2 = np.outer(dl_dz2, relu_res)  # layer 2 weight gradient
        dl_db2 = dl_dz2  # layer 2 bias gradient
        dl_da1 = np.dot(dl_dz2, W2)     # gradient through relu
        dl_dz1 = dl_da1 * (z1 > 0)   # gradient through relu, multiplication with the mask applied
        dl_dw1 = np.outer(dl_dz1, x)  # layer 1 weight gradient
        dl_db1 = dl_dz1  # layer 1 bias gradient


        return {
            "loss": np.round(loss, 4),
            "dW1": np.round(dl_dw1, 4),
            "db1": np.round(dl_db1, 4),
            "dW2": np.round(dl_dw2, 4),
            "db2": np.round(dl_db2, 4)
        }
