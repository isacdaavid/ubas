"""
File module.
"""

from abc import ABC, abstractmethod
import json
import os
import re
from typing import Dict, Any

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.io

from .common import Member


class FileBIDS(Member, ABC):
    """Abstract class to lazyly represent files in a BIDS Datatype folder."""

    # Regex pattern for BIDS filenames (simplified).
    BIDS_FILENAME_PATTERN = re.compile(
        r'^([a-zA-Z0-9]+-[a-zA-Z0-9]+_)*(?P<suffix>[a-zA-Z0-9]+)\.(?P<extension>[a-zA-Z0-9.]+)$'
    )

    def __init__(
            self,
            filepath: str,
    ):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        self._filepath: str = filepath
        self._filename: str = os.path.basename(filepath)
        self._entities: Dict[str, str] = {}
        self._suffix: str = ""
        self._extension: str = ""

        super().__init__(self._filename)

        self._validate_filename(self._filename)
        self._parse_filename(self._filepath)

    def __reduce__(self):
        return (type(self), (self._filepath,), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _validate_filename(self, filename) -> None:
        """Validate the filename against BIDS conventions."""
        if not self.BIDS_FILENAME_PATTERN.fullmatch(filename):
            raise ValueError(f"Invalid BIDS filename: {filename}")

    def _parse_filename(self, path) -> None:
        """Parse the BIDS filename into entities, suffix, and extension."""
        filename = os.path.basename(path)
        # Split into entities, suffix, and extension
        parts = filename.split("_")
        self._suffix = parts[-1].split('.', 1)[0]
        self._extension = parts[-1].split('.', 1)[1]
        # Parse key-value pairs excluding suffix and extension.
        for part in parts[:-1]:
            if "-" in part:
                key, value = part.split("-", 1)
                self._entities[key] = value
            else:
                raise ValueError(
                    f"Malformed BIDS entity {part} in file name {path}"
                )

    @property
    def filepath(self) -> str:
        """Return the path to the file."""
        return self._filepath

    @property
    def filename(self) -> str:
        """Return the base filename (e.g. 'sub-01_ses-01_T1w.nii.gz')."""
        return self._filename

    @property
    def entities(self) -> Dict[str, str]:
        """Return the parsed entities (e.g. {'sub': '01', 'ses': '01'})."""
        return self._entities

    @property
    def suffix(self) -> str:
        """Return the BIDS suffix (e.g. 'T1w')."""
        return self._suffix

    @property
    def extension(self) -> str:
        """Return the file extension (e.g. 'nii.gz')."""
        return self._extension

    @abstractmethod
    def read(self) -> Any:
        """Read the file contents dynamically (lazy loading)."""

    @abstractmethod
    def write(self, data: Any) -> None:
        """Write data to the file."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._filename})"


class FileBVal(FileBIDS):
    """Lazy concrete class for .bval files (b-values for diffusion MRI)."""
    def read(self) -> np.ndarray:
        """Read b-values as a 1D numpy array."""
        return np.loadtxt(self.filepath, dtype=float)

    def write(self, data: np.ndarray) -> None:
        """Write b-values to file."""
        np.savetxt(self.filepath, data, fmt="%.6f")


class FileBVec(FileBIDS):
    """Lazy concrete class for .bvec files (diffusion gradients)."""
    def read(self) -> np.ndarray:
        """Read gradient directions as a 2D numpy array."""
        return np.loadtxt(self.filepath, dtype=float)

    def write(self, data: np.ndarray) -> None:
        """Write gradient directions to file."""
        np.savetxt(self.filepath, data, fmt="%.6f")


class FileCIFTI(FileBIDS):
    """Lazy concrete class for CIfTI grayordinate files (.dtseries.nii)."""
    def read(self) -> nib.cifti2.Cifti2Image:
        """Read CIfTI file dynamically."""
        return nib.load(self.filepath)

    def write(self, data: nib.cifti2.Cifti2Image) -> None:
        """Write CIfTI data to file."""
        nib.save(data, self.filepath)


class FileGIFTI(FileBIDS):
    """Lazy concrete class for GIfTI files."""
    def read(self) -> nib.gifti.GiftiImage:
        """Read GIfTI file dynamically."""
        return nib.load(self.filepath)

    def write(self, data: nib.gifti.GiftiImage) -> None:
        """Write GIfTI data to file."""
        nib.save(data, self.filepath)


class FileJSON(FileBIDS):
    """Lazy concrete class for JSON files."""

    def read(self) -> dict:
        """Return JSON file dynamically."""
        with open(self.filepath, "r") as f:
            return json.load(f)

    def write(self, data: dict) -> None:
        """Write JSON data to file."""
        with open(self.filepath, "w") as f:
            json.dump(data, f, indent=4)


class FileNIFTI(FileBIDS):
    """Lazy concrete class for NIfTI files."""

    def read(self) -> nib.Nifti1Image:
        """Return NIfTI file dynamically."""
        return nib.load(self.filepath)

    def write(self, data: nib.Nifti1Image) -> None:
        """Write NIfTI data to file."""
        nib.save(data, self.filepath)


class FileMAT(FileBIDS):
    """Lazy concrete class for MATLAB files."""

    def read(self):
        """Read Matlab struct dynamically."""
        return scipy.io.loadmat(self.filepath)

    def write(self, data: dict) -> None:
        """Write dictionary to .mat file."""
        scipy.io.savemat(self.filepath, data)


class FileTSV(FileBIDS):
    """Lazy concrete class for TSV files."""

    def read(self) -> pd.DataFrame:
        """Read TSV file dynamically."""
        return pd.read_csv(self.filepath, sep="\t")

    def write(self, data: pd.DataFrame) -> None:
        """Write TSV data to file."""
        data.to_csv(self.filepath, sep="\t", index=False)


def create(filepath: str) -> FileBIDS:
    """Factory function to create the appropriate FileBIDS object."""
    filename = os.path.basename(filepath)
    extension = filename.split('.', 1)[1]

    if extension in ("nii", "nii.gz"):
        return FileNIFTI(filepath)
    if extension == "json":
        return FileJSON(filepath)
    if extension == "tsv":
        return FileTSV(filepath)
    if extension == "mat":
        return FileMAT(filepath)
    if extension == "bval":
        return FileBVal(filepath)
    if extension == "bvec":
        return FileBVec(filepath)
    if extension in ("gii", "gii.gz"):
        return FileGIFTI(filepath)
    if extension in ("dtseries.nii", "dtseries.nii.gz"):
        return FileCIFTI(filepath)

    raise ValueError(f"Unsupported file extension: {extension}")
