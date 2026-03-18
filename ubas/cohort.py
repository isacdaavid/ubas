"""
Cohort module.
"""

import concurrent.futures
from typing import (Iterable, Optional, TypeVar, Union)

from bids import BIDSLayout     # type: ignore
from tqdm import tqdm

from .common import Collection, Member
from .subject import Subject

# Type variable for Subject or its subclasses
SubjectT = TypeVar('SubjectT', bound=Subject)
# Generic type for the attribute value
T = TypeVar('T')


class Cohort(Member, Collection):
    """A class to represent a cohort subjects in a BIDS data set."""

    def __init__(
            self,
            label: str,
            contents: Union[Iterable[Subject], BIDSLayout] = (),
    ):
        Member.__init__(self, label)

        if isinstance(contents, BIDSLayout):
            labels = contents.get(target='subject', return_type='id')
            contents = [Subject(label, contents) for label in labels]

        Collection.__init__(self, contents)

    def __reduce__(self):
        return (type(self), (self.label, list(self)), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.label}', subjects={self.labels})"
