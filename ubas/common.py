"""Common abstract classes."""

from __future__ import annotations
import concurrent.futures
import os
import re
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Mapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from tqdm import tqdm

# Generic type for the attribute value
T = TypeVar('T')


class Member:
    """An abstract class for member objects in a Collection."""

    def __init__(self, label: str):
        self._label = label
        self._quantities = {}

    def __reduce__(self):
        return (type(self), (self._label,), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def label(self) -> str:
        """str: The identifier label for this Member.

        Example:
        >>> sub = Subject('Álvaro')
        >>> sub.label
        'Álvaro'
        """
        return self._label

    @property
    def quantities(self):
        """Dict[str, Any]: Storage of computed quantities for this Member."""
        return self._quantities

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

    # TODO: option to error on missing attribute.
    def collect_self(
        self,
        attr_name: str,
        default: Optional[T] = None,
    ) -> Any:
        """Extracts a specific attribute from Member.

        Return requested attribute value. Nested attributes can be collected
        using dot notation (e.g. "ses-01.func"), unquoted dictionary
        keys (e.g. "quantities[fcs]") or both. If a Member does not have the
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

    def store_self(self, attr_name: str, value: Any) -> None:
        """Stores a value in a specific attribute of Member.

        Nested attributes can be stored using dot notation (e.g. "ses-01.func"),
        unquoted dictionary keys (e.g. "quantities[fcs]"), or both.

        Args:
            attr_name (str):
                The path to the attribute or dictionary value to modify.
            value (Any):
                The value to store.

        Example:
            >>> sub = Subject('Álvaro', demographics={'age': 30})
            >>> sub.store("demographics[age]", 31)
            >>> sub.demographics['age']
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

    def compute_self(
            self,
            quantity: Callable[[Member], Any],
            key: Optional[str] = None,
            store: bool = True,
            max_workers: Optional[int] = None,
            **kwargs: Mapping[str, Any],
    ) -> Any:
        """Compute a quantity for this Member and store the result.

        This method applies a user-provided function (quantity) to the Member
        and stores the result in the `quantities` dictionary. If no key is
        provided, the function's name is used as storage key.

        Any remaining keyword arguments will be passed as is to `quantity()`.

        Args:
            quantity (Callable[[Member], Any]):
                A function that takes a Member and returns a computed value.
            key (Optional[str]):
                Override key name under which quantity will be stored.
            store (bool):
                Whether to store result in addition to returning it.
            kwargs (Mapping[str, Any]):
                Variable named arguments passed as `quantity(member, **kwargs)`

        Returns:
            Any:
                Result of computing the quantity for this Member.

        Example:
        >>> from math import floor
        >>> beatriz = Subject('Beatriz', demographics={'age': 12})
        >>> def decades(subject):
        ...     return floor(subject.demographics['age'] / 10)
        >>> beatriz.compute(decades)
        >>> print(beatriz.quantities['decades'])
        1
        """
        # quantity_type = inspect.signature(quantity).parameters.values()
        # quantity_first_arg_type = list(quantity_type)[0].annotation

        # if quantity_first_arg_type is not type(self):
        #     return super(Member, self).compute(
        #         quantity,
        #         key=key,
        #         store=store,
        #         max_workers=max_workers,
        #         **kwargs,
        #     )

        if key is None:
            key = quantity.__name__

        result = quantity(self, **kwargs)

        if store:
            self.quantities[key] = result

        return result

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.label == other.label
        return False


class Collection(set):
    """An abstract class for collection classes in a BIDS data set."""

    def __init__(self, members: Iterable[Member]) -> Collection:
        """Initialize a Collection with an iterable of Member objects.

        Args:
            members (Iterable[Member]):
                Member objects to include in the Collection.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> print(len(cohort))
            2
        """
        super().__init__(members)

        # Cached label->Member mapping for O(1) access.
        self._members = {member.label: member for member in self}

    def __reduce__(self):
        # Reconstruct with the current members
        return (type(self), (list(self),), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._members = {member.label: member for member in self}

    @property
    def labels(self) -> Set[str]:
        """Set[str]: The labels of all Members contained in this Collection.

        Example:
        >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
        >>> cohort = Cohort(subjects)
        >>> cohort.labels
        {'Álvaro', 'Beatriz'}
        """
        return {member.label for member in self}

    @property
    def members(self) -> Dict[str, Member]:
        return set(self._members.values())

    # TODO: atomic thread-safety
    def add(self, member: Member) -> None:
        """Overloaded add() which also updates the cache.

        Args:
            member (Member):
                The Member to be added to this Collection.

        Returns:
            None:
                The calling Collection might be extended as side effect.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects[0])
            >>> print(len(cohort))
            1
            >>> cohort.add(subjects[1])
            >>> print(len(cohort))
            2
        """
        super().add(member)
        self._members[member.label] = member

    # TODO: atomic thread-safety
    def remove(self, member: Member) -> None:
        """Overloaded remove() which also updates the cache.

        Args:
            member (Member):
                The Member to be removed from this Collection.

        Returns:
            None:
                The calling Collection might be shrunk as side effect.

        Raises:
            KeyError:
                If the Member to remove is not found.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> print(len(cohort))
            2
            >>> cohort.remove(subjects[0])
            >>> print(len(cohort))
            1
        """
        super().remove(self._members[member.label])
        del self._members[member.label]

    # TODO: multi-member support.
    def __getitem__(self, index: str) -> Member:
        """Retrieve a Member from the Collection by its label in O(1) time.

        Args:
            index (str):
                The label of the Member to be retrieved.

        Returns:
            Member:
                The Member with the specified label.

        Raises:
            KeyError:
                If the Member to retrieve is not found.

        Example:
            >>> subjects = [Subject('Álvaro'), Subject('Beatriz')]
            >>> cohort = Cohort(subjects)
            >>> beatriz = cohort['Beatriz']
            >>> print(beatriz.label)
            'Beatriz'
        """
        try:
            return self._members[index]
        except KeyError as exc:
            raise KeyError(f"No member found with label: {index}") from exc

    def __contains__(self, item: object) -> bool:
        """Check if a Member or Member label exists in the Collection.

        Args:
            item (Union[str, Member]):
                Member or label whose membership will be checked.

        Returns:
            bool:
                Whether the Member or member label is part of this Collection.

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
            return item in self._members
        return super().__contains__(item)

    def filter(self, condition: Callable[[Member], bool]) -> Collection:
        """Create Collection subset with Members who satisfy the condition.

        This method applies a user-provided function (`condition`) to each
        Member in the Collection. Only Members for which the function returns
        `True` are included in the new Collection.

        Args:
            condition (Callable[[Member], bool]):
                A function that takes a Member and returns a boolean.

        Returns:
            Collection:
                A new Collection containing only satisfactory Members.

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
        subcollection = filter(condition, self)
        return type(self)(self.label, subcollection)

    # TODO: parallelize?
    # TODO: option to shortcircuit on Member fail.
    def collect(
        self,
        attr_name: str,
        default: Optional[T] = None,
        labels: bool = True,
    ) -> Generator[Union[Any, Tuple[str, Any]], None, None]:
        """Extract a specific attribute from all Members in the Collection.

        This generator iterates over all Members in the Collection and yields
        the requested attribute value for each of them. Nested attributes can be
        collected using dot notation (e.g. "connectivity.functional"), unquoted
        dictionary keys (e.g. "quantities[fcs]") or both. If a Member does not
        have the attribute, the default value is yielded instead.

        Args:
            attr_name (str):
                The path to the attribute or dictionary value to collect.
            default (Optional[T]):
                The default value to yield if the attribute is not found.
            labels (bool):
                Whether to yield `(member.label, value)` tuples or just values.

        Yields:
            Union[Any, Tuple[str, Any]]:
                The value of the attribute for each Member.

        Example:
            >>> cohort = Cohort([
            ...     Subject('Álvaro', demographics={'age': 30}),
            ...     Subject('Beatriz', demographics={'age': 12})
            ... ])
            >>> list(cohort.collect("demographics[age]", labels=False))
            [30, 12]
            >>> dict(cohort.collect("EEG", default=0))
            {'Álvaro': 0, 'Beatriz': 0}
        """
        for member in self:
            value = member.collect_self(attr_name, default)
            yield (member.label, value) if labels else value

    # TODO: parallelize?
    # TODO: atomic thread-safety
    def store(
            self,
            attr_name: str,
            value: Union[Any, Mapping[str, Any]],
            labels: bool = False,
            strict: bool = True,
    ) -> None:
        """Store value in attribute for each Member in the Collection.

        Nested attributes can be stored using dot notation (e.g. "ses-01.func"),
        unquoted dictionary keys (e.g. "quantities[fcs]"), or both. Different
        values can be stored for each Member with the option `labels=True` and
        passing a dictionary with ('label', value) pairs.

        Args:
            attr_name (str):
                The path to the attribute or dictionary value to modify.
            value (Union[Any, Dict[str, Any]]):
                The value(s) to store.
            labels (bool):
                Whether to store a different value per Member using a dict.
            strict (bool):
                Whether to validate labels beforehand if `labels=True`.

        Returns:
            None:
                Members' attributes may be modified as side effect.

        Raises:
            TypeError:
                If `labels=True` and `value` isn't instance of Dict.
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
            ...     labels=True
            ... )
            >>> r = cohort.collect("demographics[age]")
            >>> set(r) == {31, 13}
            True
        """
        # Validate input type.
        if labels and not isinstance(value, Mapping):
            raise TypeError('Expected a dictionary with values.')

        provided_labels = set(value.keys()) if labels else self.labels

        # Validate member labels.
        if labels and strict and self.labels != provided_labels:
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
        for label in set.intersection(provided_labels, self.labels):
            member = self[label]
            member_value = value[label] if labels else value
            member.store_self(attr_name, member_value)

    def compute(
            self,
            quantity: Callable[[Member], Any],
            key: Optional[str] = None,
            store: bool = True,
            max_workers: Optional[int] = None,
            **kwargs: Mapping[str, Any],
    ) -> Union[Set[Any], Dict[str, Any]]:
        """Apply a function to each Member in parallel.

        This method applies a user-provided function (`quantity`) to each Member
        in the Collection and (optionally) stores the result in the Members'
        `quantities` dictionary. If no `key` is provided, the function's name is
        used as storage key. If `max_workers` is not provided, all CPU cores but
        2 will be used in parallel.

        Any remaining keyword arguments will be passed as is to `quantity()`.

        Args:
            quantity (Callable[[Member], Any]):
                A function that takes a Member and returns a computed value.
            key (Optional[str]):
                Override key name under which quantity will be stored.
            store (bool):
                Whether to store result in addition to returning it.
            max_workers (Optional[int]):
                Maximum number of processes to spawn in parallel.
            **kwargs (Mapping[str, Any]):
                Variable named arguments passed as `quantity(member, **kwargs)`

        Returns:
            Union[Set[Any], Dict[str, Any]]:
                Set[Any]: Returns a set with results.
                Dict[str, Any]: Returns a dict indexed by labels.

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

        # Submit task for each Member in parallel.
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=max_workers
        ) as executor:
            futures_to_members = {
                executor.submit(quantity, member, **kwargs): member
                for member in self
            }

            results = {}
            for future in tqdm(
                    concurrent.futures.as_completed(futures_to_members),
                    total=len(futures_to_members),
            ):
                member = futures_to_members[future]
                result = future.result()

                if store:
                    member.quantities[key] = result

                results[member.label] = result

            return results
