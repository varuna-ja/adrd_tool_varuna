__all__ = ['ExactExplainer']

from . import BaseExplainer
from typing import Any
from scipy.special import factorial
import numpy as np
import numba

MAX_NUM_FEATURES = 16
DATA_T = np.float_

class ExactExplainer(BaseExplainer):
    """ ... """
    def __init__(self,
        model: Any,
    ):
        """ ... """
        super().__init__(model)

    def shap_values(self, x):
        """ ... """
        # number of inputs and total number of features
        N, S = len(x), len(x[0])

        # due to the exponential complexity, the input size is limited
        assert S <= MAX_NUM_FEATURES

        # evaluate model with all possible coalitions
        v = np.empty((N, 2 ** S), dtype=DATA_T)
        for i in range(N):
            for j in range(2 ** S):
                mask = _int2bin(j, S)
                # v[i, j] = self.model.predict(x[i][np.newaxis, :])[0]
                v[i, j] = self.model(x[i], mask)

        # compute weights
        w = np.empty((S + 1,), dtype=DATA_T)
        for k in range(S + 1):
            w[k] = factorial(k - 1) * factorial(S - k) / factorial(S)

        # calculate exact Shapley values
        return _formula(v, w, N, S)


@numba.njit(cache=True)
def _formula(v, w, M, N):
    """ ... """    
    # calculate shapley values
    phi = np.zeros((M, N), dtype=DATA_T)
    for i in range(M):
        for j in range(2 ** N):
            mask = _int2bin(j, N)
            wp =  w[sum(mask)]
            wn =  w[sum(mask) + 1]
            for k in range(N):
                if mask[k] == 1:
                    phi[i, k] += wp * v[i, j]
                else:
                    phi[i, k] -= wn * v[i, j]
    return phi

@numba.njit(cache=True)
def _int2bin(
    val: int,
    length: int,
):
    """ ... """
    bin = np.zeros(length, dtype=np.int_)
    for i in range(length - 1, -1, -1):
        bin[length - i - 1] = (val >> i) & 1
    return bin
