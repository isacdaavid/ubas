"""Subject module."""

from typing import (
    Iterable, Mapping, Optional, TypeVar, Union
)

from bids import BIDSLayout     # type: ignore

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
            contents: Union[Iterable[Session], Iterable[Datatype], BIDSLayout] = set(),
            demographics: Optional[Mapping] = None,
    ):
        super().__init__(label)

        if isinstance(contents, BIDSLayout):
            labels = contents.get(
                subject=label,
                target='session',
                return_type='id'
            )
            if labels:
                contents = [
                    Session(label, contents, subject_label=self.label)
                    for label in labels
                ]
            else:
                labels = contents.get(
                    subject=label,
                    target='datatype',
                    return_type='id'
                )
                contents = [
                    Datatype(
                        label,
                        contents,
                        subject_label=self.label,
                    )
                    for label in labels
                ]

        super(Member, self).__init__(contents)
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
        return f"{self.__class__.__name__}('{self.label}', {self.labels})"
