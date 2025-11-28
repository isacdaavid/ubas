"""
Cohort module.
"""

import concurrent.futures
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    TypeVar,
)

from bids import BIDSLayout     # type: ignore
import numpy as np
from tqdm import tqdm

from .common import Collection, Member
from .load import load_matlab
from .subject import Subject

# Type variable for Subject or its subclasses
SubjectT = TypeVar('SubjectT', bound=Subject)
# Generic type for the attribute value
T = TypeVar('T')


class Cohort(Member, Collection):
    """A specialized class to manage a cohort of Subjects."""

    def __init__(self, label: str, subjects: Iterable[Subject]):
        super().__init__(label)
        super(Member, self).__init__(subjects)

    def __reduce__(self):
        return (type(self), (self.label, list(self)), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def _cohort_from_data(
            cls,
            demographics_file: str,
            structural_derivatives: BIDSLayout,
            functional_derivatives: BIDSLayout,
            atlases: Iterable = ('4S156', '4S256', '4S456'),
            sample_size: Optional[int] = None,
    ) -> 'Cohort':
        """Constructs a pre-filled Cohort instance from data.

        This class method skips manual creation of Subjects and creates a Cohort
        by combining demographics data from a MATLAB file with structural and
        functional neuroimaging data provided as BIDSLayout objects. Only
        subjects with complete data are included in the Cohort. Concurrent
        processing is used to load connectivity data.

        Args:
            demographics_file (str):
                Path to the MATLAB (.mat) file containing demographics data.
            structural_derivatives (BIDSLayout):
                Object pointing to the tractography derivatives from QSIrecon.
            functional_derivatives (BIDSLayout):
                Object pointing to the functional derivatives from XCP-D.
            atlases (Iterable[str]):
                Atlas names to load from connectivity data.
            sample_size (Optional[int]):
                Limit Cohort size to at most this many Subjects.

        Returns:
            Cohort:
                A Cohort instance containing Subject objects.
        """
        all_demographics = cls._load_demographics_file(demographics_file)
        all_subject_labels = set(all_demographics[:, 0].tolist())
        structural_subject_labels = set(structural_derivatives.get_subjects())
        functional_subject_labels = set(functional_derivatives.get_subjects())

        # Subset of subjects with complete data.
        subject_labels = tuple(
            set.intersection(
                all_subject_labels,
                structural_subject_labels,
                functional_subject_labels,
            )
        )

        if sample_size is None:
            sample_size = len(subject_labels)

        subject_labels = subject_labels[:min(sample_size, len(subject_labels))]

        # Prepare generators with demographics and session data. We
        # rely on order of `subject_labels` to match them.
        good_demographics = cls._demographics_generator(
            all_demographics, subject_labels
        )
        structural_sessions = (
            # Only use first neuroimaging sessions, if many are present.
            structural_derivatives.get_sessions(subject=label)[0]
            for label in subject_labels
        )
        functional_sessions = (
            functional_derivatives.get_sessions(subject=label)[0]
            for label in subject_labels
        )

        generators = zip(
            subject_labels,
            structural_sessions,
            functional_sessions,
            good_demographics
        )

        # Load IO-heavy connectivity concurrently.
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    Subject,
                    label,
                    structural_session,
                    structural_derivatives.root,
                    functional_session,
                    functional_derivatives.root,
                    atlases,
                    demographics,
                )
                for label, structural_session, functional_session, demographics
                in generators
            }

            subjects = {
                future.result() for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures)
                )
            }

        return cls(subjects)

    # TODO: generality
    @classmethod
    def _load_demographics_file(cls, path: str) -> np.ndarray:
        """Load and clean demographics data from MATLAB file.

        The file must contain an 'alldata' key with a matrix value, where the
        first column represents subject labels.

        Args:
            path (str):
                Path to the MATLAB (.mat) file containing demographics data.

        Returns:
            np.ndarray:
                A 2D NumPy array with the flattened demographics data

        Raises:
            FileNotFoundError:
                Exception is captured and error is printed.
            KeyError:
                If the file dictionary is missing an 'alldata' field.
            TypeError:
                If the array in the file contains unexpected elements.
            ValueError:
                If the array in the file contains unexpected elements.

        """
        try:
            all_data = load_matlab(path)
        except FileNotFoundError as error:
            print(error)

        # Raw data is a matrix of arrays, some of shape (1,) and
        # others of (1,1). We need to further de-encapsulate each datum.
        extract = np.vectorize(
            lambda x: x[0] if x.shape == (1,) else x[0][0]
        )

        clean_data = extract(all_data['alldata'])

        return clean_data

    # TODO: generality
    @classmethod
    def _demographics_generator(
            cls,
            clean_data: np.ndarray,
            subject_labels: Sequence[str],
    ) -> Generator[Dict[str, Any], None, None]:
        """Generates dictionaries of demographics data for select subjects.

        This class method iterates over a sequence of subject labels and
        extracts the corresponding demographics data from a preprocessed 2D
        NumPy array (e.g. from `Cohort._load_demographics_file()`). For each
        subject, it constructs a dictionary containing the subject's age, sex,
        diagnosis ('HC' for healthy controls or 'SSD' for schizophrenia spectrum
        disorder), plus a subdiagnosis category in case of 'SSD'.

        Args:
            clean_data (np.ndarray):
                2D array with clean demographics data.
            subject_labels (Sequence[str]):
                The subjects for which to generate demographics.

        Yields:
            Generator[Dict[str, Any], None, None]:
               A dictionary with the demographics of each Subject.

        Example:
            >>> clean_data = np.array([
            ...     ['Álvaro', 30, 'M', 'No_Known_Disorder'],
            ...     ['Beatriz', 12, 'F', 'Bipolar']
            ... ])
            >>> subject_labels = ['Beatriz']
            >>> demo = Cohort._demographics_generator(clean_data, subject_labels)
            >>> for d in demo:
            ...     print(d)
            {'age': 12, 'sex': 'F', 'subdiagnosis': 'Bipolar', 'diagnosis': 'SSD'}
        """
        for label in subject_labels:
            # Select subject from cohort-wide demographics data.
            subject_demographics = clean_data[clean_data[:, 0] == label, :]

            demographics = {
                'age': int(subject_demographics[0, 1]),
                'sex': str(subject_demographics[0, 2]),
                'subdiagnosis': str(subject_demographics[0, 3]),
            }

            demographics['diagnosis'] = (
                'HC'
                if demographics['subdiagnosis'] == 'No_Known_Disorder'
                else 'SSD'
            )

            yield demographics

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label}, {self.labels})"
