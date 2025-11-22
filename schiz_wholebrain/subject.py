"""
Class to represent individual participants.
"""

import re
from types import SimpleNamespace
from typing import (
    Any, Callable, Iterable, Mapping, Optional, TypeVar, Union
)

from .connectivity import FunctionalConnectivity, StructuralConnectivity

# Type variable for Subject or its subclasses
SubjectT = TypeVar('SubjectT', bound='Subject')
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
            atlases: Iterable = ('4S156', '4S256', '4S456'),
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
                structural=structural,
                functional=functional,
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

    def _tokenize_attr(
            self,
            attr_name: str
    ) -> list[Union[str, list[str]]]:
        """Helper method to tokenize an attribute path.

        Args:
            attr_name (str):
                The path to the attribute or dictionary value.

        Returns:
            list[Union[str, list[str]]]:
                The list of tokens to traverse. Nested lists encode dict keys.
        """
        # Split attr_name, both identifiers (left) and brackets (right).
        regex = r"([a-zA-Z_]\w*)|\[([a-zA-Z_]\w*)\]"
        tokens = re.findall(regex, attr_name)
        return [token[0] or [token[1]] for token in tokens]

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
        tokens = self._tokenize_attr(attr_name)

        try:
            value = self
            for token in tokens:
                if isinstance(token, list):
                    # Dictionary key access
                    value = value[token[0]]
                else:
                    # Attribute access
                    value = getattr(value, token)

            return value

        except (AttributeError, KeyError, TypeError):
            return default

    def store(self, attr_name: str, value: Any) -> None:
        """Stores a value in a specific attribute of Subject.

        Nested attributes can be stored using dot notation
        (e.g. "connectivity.functional"), unquoted dictionary keys
        (e.g. "quantities[fcs]"), or both.

        Args:
            attr_name (str):
                The path to the attribute or dictionary value to modify.
            value (Any):
                The value to store.

        Example:
            >>> sub = Subject('Álvaro', demographics={'age': 30})
            >>> sub.store("demographics[age]", 31)
            >>> sub.demographics[age]
            31
        """
        tokens = self._tokenize_attr(attr_name)

        obj = self
        for token in tokens[:-1]:
            if isinstance(token, list):
                # Dictionary key access
                obj = obj[token[0]]
            else:
                # Attribute access
                obj = getattr(obj, token)

        attr = tokens[-1]

        if isinstance(attr, list):
            obj[attr] = value
        else:
            setattr(obj, tokens[-1], value)

    def compute(
            self,
            quantity: Callable[[SubjectT], Any],
            key: Optional[str] = None,
            store: bool = True,
            **kwargs: Mapping[str, Any],
    ) -> Any:
        """Compute a quantity for this Subject and store the result.

        This method applies a user-provided function (quantity) to the Subject
        and stores the result in the `quantities` dictionary. If no key is
        provided, the function's name is used as storage key.

        Any remaining keyword arguments will be passed as is to `quantity()`.

        Args:
            quantity (Callable[[SubjectT], Any]):
                A function that takes a Subject and returns a computed value.
            key (Optional[str]):
                Override key name under which quantity will be stored.
            store (bool):
                Whether to store result in addition to returning it.
            kwargs (Mapping[str, Any]):
                Variable named arguments passed as `quantity(subject, **kwargs)`

        Returns:
            Any:
                Result of computing the quantity for this Subject.

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

        if store:
            self.quantities[key] = result

        return result

    def __repr__(self):
        return f"Subject({self.label}, {self.demographics['diagnosis']})"

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.label == other.label
        return False
