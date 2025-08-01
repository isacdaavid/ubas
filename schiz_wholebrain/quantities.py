"""
Functions to compute measurements of interest on `Connectivity` data
"""

import numpy as np


def fcs(correlation_matrix: np.array, absolute=False):
    fcs = correlation_matrix.mean(axis=0)
    if absolute:
        return fcs
    fisher = np.arctanh(fcs)
    zscores = (fisher - fisher.mean()) / fisher.std()
    return zscores
