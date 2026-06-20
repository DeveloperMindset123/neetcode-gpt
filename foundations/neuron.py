import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)

        # this returns an array of vector values
        pre_activation_function = w @ x + b
        ans = 0.0     # can be anything, I just put it as a basic floating point value
        if activation == "sigmoid":
            # represent the sigmoid function
            ans = 1 / (1 + np.exp(-pre_activation_function))

        else:
            # assuming activation is ReLU
            # NOTE: if I don't pass in 0.0, it automatically infers that I want an integer value as the resulting output
            ans = max(0.0, pre_activation_function)
        return np.round(ans,5)
