import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(
        self,
        model_prediction: NDArray[np.float64],
        ground_truth: NDArray[np.float64],
        N: int,
        X: NDArray[np.float64],
        desired_weight: int,
    ) -> float:
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(
        self, X: NDArray[np.float64], weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)

        final_weights = np.array(initial_weights, dtype=np.float64)       # some initial floating point value
        for _ in range(num_iterations):
            model_prediction = self.get_model_prediction(X, final_weights)

            for j in range(len(final_weights)):    # we want to iterate over the index of the weights and modify/optimize the weights themselves
                # This would be the calculation of the derivative itself
                gradient_j = self.get_derivative(model_prediction, Y, len(X), X, j)

                # this would update final_weights in this case based on the learning rate and gradient being recalculated
                final_weights[j] -= self.learning_rate * gradient_j

        return np.round(final_weights, 5)
