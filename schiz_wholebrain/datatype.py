"""
Datatype module.
"""

from typing import Iterable, Optional

from .common import Collection, Member
from .file import FileBIDS


class Datatype(Member, Collection):
    """A class to represent recording modalities in a BIDS Session."""

    def __init__(
            self,
            label: str,
            files: Optional[Iterable[FileBIDS]],
    ):
        super().__init__(label)
        super(Member, self).__init__(files)

    def __reduce__(self):
        return (type(self), (self.label, list(self)), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label}, {self.labels})"
