"""
Classes to represent connectivity objects
"""

import glob

import numpy as np

from .load import load_matlab, load_tsv


class Connectivity():
    @property
    def region_labels(self):
        return self._region_labels

    def normalize(self, connectivity_matrix: str):
        matrix = getattr(self, connectivity_matrix)

        if not isinstance(matrix, np.ndarray):
            raise TypeError(
                'Cannot normalize {} of type {}.'.format(
                    connectivity_matrix,
                    type(matrix),
                )
            )

        if matrix.shape[0] != matrix.shape[1] or len(matrix.shape) != 2:
            raise ValueError(
                'Cannot normalize non-square {}.'.format(
                    connectivity_matrix,
                )
            )

        matrix = matrix + matrix.T
        return matrix / np.max(matrix)

    def __repr__(self):
        return f"{self.__class__.__name__}"


class FunctionalConnectivity(Connectivity):
    def __init__(
            self,
            derivatives: str,
            subject: str,
            session: str,
            atlas: str,
    ):
        pattern = f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-rest*_space-fsLR_seg-{atlas}Parcels_stat-mean_timeseries.tsv'

        try:
            runs = glob.glob(f'{derivatives}/{pattern}')
            self._region_labels, self._time_series = load_tsv(runs[0])
        except FileNotFoundError as error:
            print(error)

        pattern = f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-rest_*outliers.tsv'

        try:
            runs = glob.glob(f'{derivatives}/{pattern}')
            _, self._motion_outliers = load_tsv(runs[0])
        except FileNotFoundError as error:
            print(error)

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
    def __init__(
            self,
            derivatives: str,
            subject: str,
            session: str,
            atlas: str,
    ):

        pattern = f'{derivatives}/sub-{subject}/ses-{session}/dwi/sub-{subject}_ses-{session}_space-ACPC_connectivity.mat'
        runs = glob.glob(pattern)

        keys = (
            f'atlas_{atlas}Parcels_region_ids',
            f'atlas_{atlas}Parcels_region_labels',
            f'atlas_{atlas}Parcels_radius2_meanlength_connectivity',
            f'atlas_{atlas}Parcels_radius2_count_connectivity',
            f'atlas_{atlas}Parcels_sift_radius2_count_connectivity',
            f'atlas_{atlas}Parcels_sift_invnodevol_radius2_count_connectivity',
        )

        try:
            data = tuple(load_matlab(runs[0])[key] for key in keys)
        except FileNotFoundError as error:
            print(error)

        self._region_ids = tuple(data[0].flatten().tolist())
        self._region_labels = tuple(str(l[0]) for l in data[1][0, :])
        self._mean_length = data[2]
        self._raw_count = data[3]
        self._sift_count = data[4]
        self._weighted_sift_count = data[5]

    @property
    def region_ids(self):
        return self._region_ids

    @property
    def mean_length(self):
        return self._mean_length

    @property
    def raw_count(self):
        return self._raw_count

    @property
    def sift_count(self):
        return self._sift_count

    @property
    def weighted_sift_count(self):
        return self._weighted_sift_count
