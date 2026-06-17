import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        # rather than overcomplicating the implementation
        # I have the numerator, and denominator, and the resulting gets returned using np.round() method
        # This is applicable, since z is an 1D array
        numerator = np.exp(z - np.max(z))
        return np.round(numerator / np.sum(numerator), 4)

    
        
