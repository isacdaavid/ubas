"""
Class to represent a Cohort of Subjects.
"""

import concurrent.futures
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
)

# Type variable for Subject or its subclasses
SubjectT = TypeVar('SubjectT', bound='Subject')
# Generic type for the attribute value
T = TypeVar('T')

from bids import BIDSLayout
import numpy as np
from tqdm import tqdm

from .load import load_matlab
from .subject import Subject


class Cohort(set):
    """A specialized set to manage a cohort of Subjects in a study.

    Attributes:
        labels: (Set[str]): The labels of all Subjects contained in this Cohort.
    """

    def __init__(self, subjects: Iterable[SubjectT]):
        """Initialize a Cohort with an iterable of Subject objects.

        Args:
            subjects (Iterable[SubjectT]): Subject objects to include.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> print(len(cohort))
            2
        """
        super().__init__(subjects)

        # Cached label->Subject mapping for O(1) access.
        self._label_to_subject = {subject.label: subject for subject in self}

    @property
    def labels(self) -> Set[str]:
        """Set[str]: The labels of all Subjects contained in this Cohort.

        Example:
        >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
        >>> cohort = Cohort(subjects)
        >>> print(cohort.labels)
        {'Álvaro', 'Beatriz'}
        """
        return {subject.label for subject in self}

    def add(self, subject: SubjectT):
        """Overloaded add() which also updates the Subject cache.

        Args:
            subject (SubjectT): The Subject to be added to this Cohort.

        Returns:
            None: Calling Cohort is extended or overwritten as side effect.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects[0])
            >>> print(len(cohort))
            1
            >>> cohort.add(subjects[1])
            >>> print(len(cohort))
            2
        """
        super().add(subject)
        self._label_to_subject[subject.label] = subject

    def remove(self, subject: SubjectT):
        """Overloaded remove() which also updates the Subject cache.

        Args:
            subject (SubjectT): The Subject to be removed from this Cohort.

        Returns:
            None: Calling Cohort might be shrunk as side effect.

        Raises:
            KeyError: If the Subject to remove is not found.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> print(len(cohort))
            2
            >>> cohort.remove(subjects[0])
            >>> print(len(cohort))
            1
        """
        super().remove(subject)
        del self._label_to_subject[subject.label]

    def __getitem__(self, index: str) -> SubjectT:
        """Retrieve a Subject from the Cohort by its label in O(1) time.

        Args:
            index (str): The label of Subject to be retrieved.

        Returns:
            SubjectT: The Subject for the specified label.

        Raises:
            KeyError: If the Subject to retrieve is not found.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> beatriz = cohort['Beatriz']
            >>> print(beatriz.label)
            'Beatriz'
        """
        try:
            return self._label_to_subject[index]
        except KeyError:
            raise KeyError(f"No subject found with label: {index}")

    def __contains__(self, item: Union[str, SubjectT]) -> bool:
        """Check if a Subject or a subject label exists in the Cohort.

        Args:
            item (Union[str, SubjectT]): Subject or label whose membership will
                be checked.

        Returns:
            bool: Whether the Subject or subject labels is part of this Cohort.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> print(subjects[0] in cohort)
            True
            >>> print('Beatriz' in cohort)
            True
            >>> print('Carlos' in cohort)
            False
        """
        if isinstance(item, str):
            return item in self._label_to_subject
        return super().__contains__(item)

    @classmethod
    def cohort_from_data(
            cls,
            demographics_file: str,
            structural_derivatives: BIDSLayout,
            functional_derivatives: BIDSLayout,
            atlases: Iterable = ['4S156', '4S256', '4S456'],
    ) -> 'Cohort':
        """Constructs a pre-filled Cohort instance from data.

        This class method skips manual creation of Subjects and creates a Cohort
        by combining demographics data from a MATLAB file with structural and
        functional neuroimaging data provided as BIDSLayout objects. Only
        subjects with complete data are included in the Cohort. Concurrent
        processing is used to load connectivity data.

        Args:
            demographics_file (str): Path to the MATLAB (.mat) file containing
                demographics data. The file must contain an 'alldata' key with a
                matrix value, where the first column represents subject labels.
            structural_derivatives (BIDSLayout): Object pointing to the
                directory containing tractography derivatives from QSIrecon.
            functional_derivatives (BIDSLayout): Object pointing to the
                directory containing functional derivatives from XCP-D.
            atlases (Iterable[str]): Atlas names to load from connectivity data.

        Returns:
            Cohort: A Cohort instance containing Subject objects.
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
        """Load and clean demographics data from MATLAB file.

        Args:
            path (str): Path to the MATLAB (.mat) file containing demographics
                data. The file must contain an 'alldata' key with a matrix
                value, where the first column represents subject labels.

        Returns:
            np.ndarray: A 2D NumPy array with the flattened demographics data

        Raises:
            FileNotFoundError: Exception is captured and error is printed.
            KeyError: If the file dictionary is missing an 'alldata' field.
            TypeError: If the array in the file contains unexpected elements.
            ValueError: If the array in the file contains unexpected elements.
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
            clean_data (np.ndarray): 2D array with clean demographics data.
            subject_labels (Sequence[str]): The subjects for which to generate
                demographics.

        Yields:
            Generator[Dict[str, Any], None, None]: Dictionaries with demographics.

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

    def collect(
        self,
        attr_name: str,
        default: Optional[T] = None
    ) -> Generator[Tuple[str, Union[T, Any]], None, None]:
        """Extracts a specific attribute from all Subjects in the Cohort.

        This method iterates over all Subjects in the Cohort and yields tuples
        of (subject.label, attribute_value) for each Subject. The attribute can
        be nested, specified using dot notation
        (e.g. "connectivity.functional.time_series").  If a Subject does not
        have the attribute, the default value is yielded instead.

        Args:
            attr_name (str): The path to the attribute to collect.
            default (Optional[T]): The default value to yield if the attribute
                is not found. Defaults to None.

        Yields:
            Tuple[str, Union[T, Any]]: A (subject.label, attribute_value) pair
                for each Subject.

        Example:
            >>> cohort = Cohort([
            ...     Subject('Álvaro', demographics={'age': 30}),
            ...     Subject('Beatriz', demographics={'age': 12})
            ... ])
            >>> dict(cohort.collect("demographics"))
            {'Alice': {'age': 30}, 'Bob': {'age': 12}}
            >>> list(cohort.collect("EEG", default=0))
            [('Alice', 0), ('Bob', 0)]
        """
        for subject in self:
            try:
                # Traverse the nested attribute path
                value = subject
                for attr in attr_name.split('.'):
                    value = gettattr(value, attr)
                yield (subject.label, value)
            except AttributeError:
                yield (subject.label, default)

    def filter(self, condition: Callable[[SubjectT], bool]) -> 'Cohort':
        """Create Cohort subset with Subjects who satisfy the condition.

        This method applies a user-provided function (`condition`) to each
        Subject in the Cohort. Only Subjects for which the function returns
        `True` are included in the new Cohort.

        Args:
            condition (Callable[[SubjectT], bool]): A function that takes a
                Subject and returns a boolean.

        Returns:
            Cohort: A new Cohort containing only satisfactory Subjects.

        Example:
            >>> cohort = Cohort([
            ...     Subject('Álvaro', demographics={'age': 30}),
            ...     Subject('Beatriz', demographics={'age': 12})
            ... ])
            >>> def is_adult(subject):
            ...     return subject.demographics['age'] >= 18
            >>> adult_cohort = cohort.filter(is_adult)
            >>> len(adult_cohort)
            1
        """
        subcohort = filter(condition, self)
        return type(self)(subcohort)
