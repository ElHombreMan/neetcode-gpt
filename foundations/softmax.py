import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        # softmax(x)_i = (e^(z_i - max(z))) * (np.sums())
        
        shifted = z - np.max(z) # z_i - max(z)
        exps = np.exp(shifted) #e ^ (z_i - max(z))
        return np.round((exps / np.sum(exps)), 4) #softmax formula rounded to 4 decimal points
