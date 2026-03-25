"""
Classes to represent connectivity objects
"""

import glob

from bids.layout.models import BIDSFile
import numpy as np
import scipy.io

def load_tsv(file_path):
    """Load matrix in TSV format."""
    data = np.loadtxt(file_path, delimiter='\t', dtype=str)
    columns = tuple(data[0].tolist())
    values = data[1:].astype(float)
    return columns, values

class Connectivity():
    def make_symmetrical(self, connectivity_matrix: str):
        matrix = getattr(self, connectivity_matrix)

        if matrix.shape[0] != matrix.shape[1] or matrix.ndim != 2:
            raise ValueError(f'{connectivity_matrix} is not a square matrix.')

        return matrix + matrix.T

    def __repr__(self):
        return f"{self.__class__.__name__}"


class FunctionalConnectivity(Connectivity):
    def __init__(self, time_series: BIDSFile, outliers: BIDSFile):
        try:
            self._region_labels, self._time_series = load_tsv(time_series.path)
        except FileNotFoundError as error:
            print(error)

        try:
            _, self._motion_outliers = load_tsv(outliers.path)
        except FileNotFoundError as error:
            print(error)

    @property
    def region_labels(self):
        return self._region_labels

    @property
    def time_series(self):
        return self._time_series

    @property
    def correlation_matrix(self):
        return np.corrcoef(self._time_series.T)

    @property
    def motion_outliers(self):
        return self._motion_outliers

    @property
    def motion_outliers_ratio(self):
        return float(self._motion_outliers.mean())


class StructuralConnectivity(Connectivity):
    def __init__(self, connectivity_file: BIDSFile, atlas: str):
        keys = (
            f'atlas_{atlas}Parcels_region_ids',
            f'atlas_{atlas}Parcels_region_labels',
            f'atlas_{atlas}Parcels_radius2_meanlength_connectivity',
            f'atlas_{atlas}Parcels_radius2_count_connectivity',
            f'atlas_{atlas}Parcels_sift_radius2_count_connectivity',
            f'atlas_{atlas}Parcels_sift_invnodevol_radius2_count_connectivity',
        )

        try:
            data = [
                scipy.io.loadmat(connectivity_file.path)[key]
                for key in keys
            ]
        except FileNotFoundError as error:
            print(error)

        self._region_ids = tuple(data[0].flatten().tolist())
        self._region_labels = tuple(str(l[0]) for l in data[1][0, :])
        self._mean_length = data[2]
        self._raw_count = data[3]
        self._sift_count = data[4]
        self._weighted_sift_count = data[5]

    @property
    def region_labels(self):
        return self._region_labels

    @property
    def region_ids(self):
        return self._region_ids

    @property
    def mean_length(self):
        matrix = self.make_symmetrical('_mean_length')
        np.fill_diagonal(matrix, 0)
        # Don't normalize distances, only average with transpose to
        # make symmetrical. Some model parameters are unit-sensitive.
        return matrix / 2

    @property
    def raw_count(self):
        matrix = self.make_symmetrical('_raw_count')
        np.fill_diagonal(matrix, 0)
        return matrix / matrix.max()

    @property
    def sift_count(self):
        matrix = self.make_symmetrical('_sift_count')
        np.fill_diagonal(matrix, 0)
        return matrix / matrix.max()

    @property
    def weighted_sift_count(self):
        matrix = self.make_symmetrical('_weighted_sift_count')
        np.fill_diagonal(matrix, 0)
        return matrix / matrix.max()
