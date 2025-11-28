"""Subject module."""

from typing import (
    Iterable, Mapping, Optional, TypeVar, Union
)

from .common import Collection, Member
from .session import Session
from .datatype import Datatype

# Generic type for the attribute value
T = TypeVar('T')


class Subject(Member, Collection):
    """A class to hold and manage subject data in a BIDS Cohort."""
    def __init__(
            self,
            label: str,
            sessions: Optional[Iterable[Union[Session, Datatype]]] = None,
            demographics: Optional[Mapping] = None,
    ):
        super().__init__(label)
        super(Member, self).__init__(sessions)
        self._demographics = demographics

    def __reduce__(self):
        return (
            type(self),
            (self.label, list(self), self._demographics),
            self.__dict__
        )

    def __setstate__(self, state):
        self.__dict__.update(state)

    @property
    def demographics(self):
        return self._demographics

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label}, {self.labels})"
