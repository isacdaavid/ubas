from typing import (
    Any, Callable, Iterable, Mapping, Optional, Sequence, TypeVar, Union
)

from .common import Collection, Member

class Subject(Member, Collection):
    """A class to hold and manage subject data from a BIDS data set."""
    def __init__(
            self,
            label: str,
            sessions: Optional[Sequence[Session]] = None,
            demographics: Optional[Mapping] = None,
    ):
        super().__init__(label)
        super(Member, self).__init__(sessions)
        self._demographics = demographics

    @property
    def demographics(self):
        return self._demographics
