import concurrent.futures
from typing import Any, Callable, Dict, Generator, Iterable, Set

from bids import BIDSLayout
import numpy as np
from tqdm import tqdm

from load import load_matlab
from subject import Subject


class Cohort(set):
    def __init__(self, subjects: Iterable[Subject]):
        super().__init__()

        for subject in subjects:
            self.add(subject)

    @classmethod
    def cohort_from_data(
            cls,
            demographics_file: str,
            structural_derivatives: BIDSLayout,
            functional_derivatives: BIDSLayout,
            atlases: Iterable = ['4S156', '4S256', '4S456'],
    ) -> 'Cohort':
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
                future.result() for future in
                tqdm(concurrent.futures.as_completed(futures), total=len(futures))
            }

        return cls(subjects)

    @classmethod
    def _load_demographics_file(cls, path: str) -> np.ndarray:
        """Load demographics data from MATLAB files."""
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

    @classmethod
    def _demographics_generator(
            cls,
            clean_data,
            subject_labels: Iterable
    ) -> Generator[Dict[str, Any], None, None]:
        for label in subject_labels:
            # Select subject from cohort-wide demographics data.
            subject_demographics = clean_data[clean_data[:, 0] == label, :]

            subject_demographics = {
                'age': int(subject_demographics[0, 1]),
                'sex': str(subject_demographics[0, 2]),
                'subdiagnosis': str(subject_demographics[0, 3]),
            }

            subject_demographics['diagnosis'] = (
                'HC'
                if subject_demographics['subdiagnosis'] == 'No_Known_Disorder'
                else 'SSD'
            )

            yield subject_demographics

    @property
    def demographics(self) -> Dict[str, Dict]:
        return {subject.label: subject.demographics for subject in self}

    @property
    def labels(self) -> Set[str]:
        return {subject.label for subject in self}

    def filter(self, condition: Callable[[Subject], bool]):
        subcohort = filter(condition, self)
        return type(self)(subcohort)

    def __getitem__(self, index):
        for subject in self:
            if subject.label == index:
                return subject
        raise KeyError(f"No subject found with label: {index}")
