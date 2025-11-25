"""
Cohort module.
"""

import concurrent.futures
import os
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
    Union,
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


class Cohort(Collection, Member):
    """A specialized class to manage a cohort of Subjects."""

    def __init__(self, subjects: Iterable[Subject]):
        super().__init__(subjects)

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


class Cohort2(set):
    """A specialized set class to manage a cohort of Subjects in a study."""

    def __init__(self, subjects: Iterable[Subject]):
        """Initialize a Cohort with an iterable of Subject objects.

        Args:
            subjects (Iterable[SubjectT]):
                Subject objects to include.

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
        >>> cohort.labels
        {'Álvaro', 'Beatriz'}
        """
        return {subject.label for subject in self}

    def add(self, subject: SubjectT):
        """Overloaded add() which also updates the Subject cache.

        Args:
            subject (SubjectT):
                The Subject to be added to this Cohort.

        Returns:
            None:
                The calling Cohort is extended or overwritten as side effect.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects[0])
            >>> print(len(cohort))
            1
            >>> cohort.add(subjects[1])
            >>> print(len(cohort))
            2
        """
        # TODO: atomic thread-safety
        super().add(subject)
        self._label_to_subject[subject.label] = subject

    def remove(self, subject: SubjectT):
        """Overloaded remove() which also updates the Subject cache.

        Args:
            subject (SubjectT):
                The Subject to be removed from this Cohort.

        Returns:
            None:
                The calling Cohort might be shrunk as side effect.

        Raises:
            KeyError:
                If the Subject to remove is not found.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> print(len(cohort))
            2
            >>> cohort.remove(subjects[0])
            >>> print(len(cohort))
            1
        """
        # TODO: atomic thread-safety
        super().remove(subject)
        del self._label_to_subject[subject.label]

    def __getitem__(self, index: str) -> Subject:
        """Retrieve a Subject from the Cohort by its label in O(1) time.

        Args:
            index (str):
                The label of Subject to be retrieved.

        Returns:
            SubjectT:
                The Subject for the specified label.

        Raises:
            KeyError:
                If the Subject to retrieve is not found.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> beatriz = cohort['Beatriz']
            >>> print(beatriz.label)
            'Beatriz'
        """
        try:
            return self._label_to_subject[index]
        except KeyError as exc:
            raise KeyError(f"No subject found with label: {index}") from exc

    def __contains__(self, item: object) -> bool:
        """Check if a Subject or a subject label exists in the Cohort.

        Args:
            item (Union[str, SubjectT]):
                Subject or label whose membership will be checked.

        Returns:
            bool:
                Whether the Subject or subject label is part of this Cohort.

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

# TODO: parallelize?
    def filter(self, condition: Callable[[SubjectT], bool]) -> 'Cohort':
        """Create Cohort subset with Subjects who satisfy the condition.

        This method applies a user-provided function (`condition`) to each
        Subject in the Cohort. Only Subjects for which the function returns
        `True` are included in the new Cohort.

        Args:
            condition (Callable[[SubjectT], bool]):
                A function that takes a Subject and returns a boolean.

        Returns:
            Cohort:
                A new Cohort containing only satisfactory Subjects.

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

    # TODO: parallelize?
    # TODO: option to shortcircuit on Subject fail.
    def collect(
        self,
        attr_name: str,
        default: Optional[T] = None,
        subject_labels: bool = True,
    ) -> Generator[Union[Any, Tuple[str, Any]], None, None]:
        """Extract a specific attribute from all Subjects in the Cohort.

        This generator iterates over all Subjects in the Cohort and yields the
        requested attribute value for each of them. Nested attributes can be
        collected using dot notation (e.g. "connectivity.functional"), unquoted
        dictionary keys (e.g. "quantities[fcs]") or both. If a Subject does not
        have the attribute, the default value is yielded instead.

        Args:
            attr_name (str):
                The path to the attribute or dictionary value to collect.
            default (Optional[T]):
                The default value to yield if the attribute is not found.
            subject_labels (bool):
                Whether to yield `(subject.label, value)` tuples or just values.

        Yields:
            Union[Any, Tuple[str, Any]]:
                The value of the attribute for each Subject.

        Example:
            >>> cohort = Cohort([
            ...     Subject('Álvaro', demographics={'age': 30}),
            ...     Subject('Beatriz', demographics={'age': 12})
            ... ])
            >>> list(cohort.collect("demographics[age]", subject_labels=False))
            [30, 12]
            >>> dict(cohort.collect("EEG", default=0))
            {'Álvaro': 0, 'Beatriz': 0}
        """
        for subject in self:
            value = subject.collect(attr_name, default)
            yield (subject.label, value) if subject_labels else value

    # TODO: parallelize?
    # TODO: atomic thread-safety
    def store(
            self,
            attr_name: str,
            value: Union[Any, Dict[str, Any]],
            subject_labels: bool = False,
            strict: bool = True,
    ) -> None:
        """Store a value in a specific attribute for each Subject in the Cohort.

        Nested attributes can be stored using dot notation
        (e.g. "connectivity.functional"), unquoted dictionary keys
        (e.g. "quantities[fcs]"), or both. Different values can be stored for
        each `Subject` with the option `subject_labels=True` and passing a
        dictionary with ('label', value) pairs.

        Args:
            attr_name (str):
                The path to the attribute or dictionary value to modify.
            value (Union[Any, Dict[str, Any]]):
                The value(s) to store.
            subject_labels (bool):
                Whether to store a different value per subject using a dict.
            strict (bool):
                Whether to validate labels beforehand if `subject_labels=True`.

        Returns:
            None:
                Subjects' attributes may be modified as side effect.

        Raises:
            TypeError:
                If `subject_labels=True` and `value` isn't instance of Dict.
            KeyError:
                If `strict=True` and value Dict has extra or missing keys.
        Example:
            >>> cohort = Cohort([
            ...     Subject('Álvaro', demographics={'age': 30}),
            ...     Subject('Beatriz', demographics={'age': 12})
            ... ])
            >>> cohort.store('demographics', None)
            >>> cohort['Beatriz'].demographics
            None
            >>> cohort.store(
            ...     'demographics',
            ...     {'Álvaro': {'age': 31}, 'Beatriz': {'age': 13}},
            ...     subject_labels=True
            ... )
            >>> r = cohort.collect("demographics[age]")
            >>> set(r) == {31, 13}
            True
        """
        # Validate input type.
        if subject_labels and not isinstance(value, Dict):
            raise TypeError('Expected a dictionary with values.')

        # Validate subject labels.
        if subject_labels and strict:
            provided_labels = set(value.keys())

            if self.labels != provided_labels:
                extra = provided_labels - self.labels
                missing = self.labels - provided_labels

                messages = []

                if missing:
                    messages.append(f"Missing keys: {missing}")
                if extra:
                    messages.append(f"Extra keys: {extra}")

                messages.append("Bypass mismatch with store(strict=False).")
                raise KeyError("\n".join(messages))

        # Store values.
        for subject in self:
            subject_value = value[subject.label] if subject_labels else value
            subject.store(attr_name, subject_value)

    def compute(
            self,
            quantity: Callable[[SubjectT], Any],
            key: Optional[str] = None,
            store: bool = True,
            max_workers: Optional[int] = None,
            **kwargs: Mapping[str, Any],
    ) -> Union[Set[Any], Dict[str, Any]]:
        """Apply a function to each Subject in parallel.

        This method applies a user-provided function (`quantity`) to each
        Subject in the Cohort and (optionally) stores the result in the
        Subject's `quantities` dictionary. If no `key` is provided, the
        function's name is used as storage key. If `max_workers` is not
        provided, all CPU cores but 2 will be used in parallel.

        Any remaining keyword arguments will be passed as is to `quantity()`.

        Args:
            quantity (Callable[[SubjectT], Any]):
                A function that takes a Subject and returns a computed value.
            key (Optional[str]):
                Override key name under which quantity will be stored.
            store (bool):
                Whether to store result in addition to returning it.
            max_workers (Optional[int]):
                Maximum number of processes to spawn in parallel.
            **kwargs (Mapping[str, Any]):
                Variable named arguments passed as `quantity(subject, **kwargs)`

        Returns:
            Union[Set[Any], Dict[str, Any]]:
                Set[Any]: Returns a set with results.
                Dict[str, Any]: Returns a dict indexed by subject labels.

        Example:
            >>> from math import floor
            >>> cohort = Cohort([Subject('Álvaro', demographics={'age': 31}),
            ...                   Subject('Beatriz', demographics={'age': 12})])
            >>> def decades(subject, round=False):
            ...     if round:
            ...         return floor(subject.demographics['age'] / 10)
            ...     return subject.demographics['age'] / 10
            >>> r = cohort.compute(decades)
            >>> r == {'Álvaro': 3.1, 'Beatriz': 1.2}
            True
            >>> cohort.compute(decades, store=True)
            >>> print(cohort['Beatriz'].quantities['decades'])
            1.2
            >>> r = cohort.compute(decades, key='mykey')
            >>> print(cohort['Beatriz'].quantities['mykey'])
            1.2
            >>> r = cohort.compute(decades, round=True)
            r == {'Álvaro': 3, 'Beatriz': 1}

        """
        if key is None:
            key = quantity.__name__

        if max_workers is None:
            max_workers = (os.cpu_count() or 1) - 2

        max_workers = max(max_workers, 1)

        # Submit task for each Subject in parallel.
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
        ) as executor:
            futures_to_subjects = {
                executor.submit(quantity, subject, **kwargs): subject
                for subject in self
            }

            results = {}
            for future in tqdm(
                    concurrent.futures.as_completed(futures_to_subjects),
                    total=len(futures_to_subjects),
            ):
                subject = futures_to_subjects[future]
                result = future.result()

                if store:
                    subject.quantities[key] = result

                results[subject.label] = result

            return results
