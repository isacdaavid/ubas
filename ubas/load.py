import numpy as np
import scipy.io


def load_matlab(file_path):
    """Load a MATLAB struct."""
    data = scipy.io.loadmat(file_path)
    return data


def load_tsv(file_path):
    """Load matrix in TSV format."""
    data = np.loadtxt(file_path, delimiter='\t', dtype=str)
    columns = tuple(data[0].tolist())
    values = data[1:].astype(float)
    return columns, values
