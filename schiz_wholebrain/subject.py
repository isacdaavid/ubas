"""
Class to represent individual participants.
"""

import re
from types import SimpleNamespace
from typing import Any, Callable, Iterable, Mapping, Optional, TypeVar, Union

import numpy as np

from .connectivity import FunctionalConnectivity, StructuralConnectivity
from .load import load_matlab, load_tsv

# Type variable for Subject or its subclasses
SubjectT = TypeVar('SubjectT', bound=Subject)
# Generic type for the attribute value
T = TypeVar('T')


class Subject:
    def __init__(
            self,
            subject_label: str,
            structural_session_label: Optional[str] = None,
            structural_derivatives: Optional[str] = None,
            functional_session_label: Optional[str] = None,
            functional_derivatives: Optional[str] = None,
            atlases: Iterable = ['4S156', '4S256', '4S456'],
            demographics: Optional[Mapping] = None,
    ):
        self._label = subject_label
        self._structural_session = structural_session_label
        self._functional_session = functional_session_label
        self._demographics = demographics
        self._quantities = {}

        self._structural_connectivity = (
            None if structural_derivatives is None
            else self._load_connectivity(
                    StructuralConnectivity,
                    structural_derivatives,
                    subject_label,
                    structural_session_label,
                    atlases,
            )
        )

        self._functional_connectivity = (
            None if functional_derivatives is None
            else self._load_connectivity(
                    FunctionalConnectivity,
                    functional_derivatives,
                    subject_label,
                    functional_session_label,
                    atlases,
            )
        )

        # Populate atlas-named wrapper attributes.
        for atlas in atlases:
            structural = self.structural_connectivity[atlas]
            functional = self.functional_connectivity[atlas]
            bundle = SimpleNamespace(
                structural = structural,
                functional = functional,
            )

            # TODO: Pass properly decorated getters from getter factory instead.
            setattr(self, 'connectivity_' + atlas, bundle)

    @property
    def label(self):
        return self._label

    @property
    def structural_session(self):
        return self._structural_session

    @property
    def functional_session(self):
        return self._functional_session

    @property
    def structural_connectivity(self):
        return self._structural_connectivity

    @property
    def functional_connectivity(self):
        return self._functional_connectivity

    @property
    def demographics(self):
        return self._demographics

    @property
    def quantities(self):
        """Dict[str, Any]: Storage of computed quantities for this Subject."""
        return self._quantities

    def _load_connectivity(
            self,
            constructor: Callable,
            derivatives: str,
            subject: str,
            session: str,
            atlases: str,
    ):
        """Load functional connectivity data for multiple atlases"""
        return {
            atlas: constructor(derivatives, subject, session, atlas)
            for atlas in atlases
        }

    def collect(
        self,
        attr_name: str,
        default: Optional[T] = None,
    ) -> Any:
        """Extracts a specific attribute from Subject.

        Return requested attribute value. Nested attributes can be collected
        using dot notation (e.g. "connectivity.functional"), unquoted dictionary
        keys (e.g. "quantities[fcs]") or both. If a Subject does not have the
        attribute, the default value is returned instead.

        Args:
            attr_name (str):
                The path to the attribute or dictionary value to collect.
            default (Optional[T]):
                The default value to return if the attribute is not found.

        Returns:
            Any:
                The value of the attribute.

        Example:
            >>> sub = Subject('Álvaro', demographics={'age': 30})
            >>> sub.collect("demographics[age]")
            30
            >>> sub.collect("EEG", default=0)
            0
        """
        # Split attr_name, handling both identifiers (left) and brackets (right).
        regex = r"([a-zA-Z_]\w*)|\[([a-zA-Z_]\w*)\]"
        parts = re.findall(regex, attr_name)
        parts = [part[0] or [part[1]] for part in parts]

        try:
            value = self
            for part in parts:
                if isinstance(part, list):
                    # Dictionary key access
                    value = value[part[0]]
                else:
                    # Attribute access
                    value = getattr(value, part)

            return value

        except (AttributeError, KeyError, TypeError):
            return default

    def compute(
            self,
            quantity: Callable[[SubjectT], Any],
            key: Optional[str] = None,
            output: Optional[bool] = False,
            **kwargs: Mapping[str, Any],
    ) -> Union[None, Any]:
        """Compute a quantity for this Subject and store the result.

        This method applies a user-provided function (quantity) to the Subject
        and stores the result in the `quantities` dictionary. If no key is
        provided, the function's name is used as key.

        Args:
            quantity (Callable[[SubjectT], Any]):
                A function that takes a Subject and returns a computed value.
            key (Optional[str]):
                Override key name under which quantity will be stored.
            output (Optional[bool]):
                Whether to return result instead of storing it.
            kwargs (Mapping[str, Any]):
                Variable named arguments passed as `quantity(subject, **kwargs)`

        Example:
        >>> from math import floor
        >>> beatriz = Subject('Beatriz', demographics={'age': 12})
        >>> def decades(subject):
        ...     return floor(subject.demographics['age'] / 10)
        >>> beatriz.compute(decades)
        >>> print(beatriz.quantities['decades'])
        1
        """
        if key is None:
            key = quantity.__name__

        result = quantity(self, **kwargs)

        if output:
            return result

        self.quantities[key] = result

    def __repr__(self):
        return f"Subject({self.label}, {self.demographics['diagnosis']})"

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.label == other.label
        return False
