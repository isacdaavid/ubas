"""Session module."""

from typing import Iterable, Optional

from .common import Collection, Member
from .datatype import Datatype


class Session(Member, Collection):
    """A class to represent a neuroimaging session in a BIDS Subject."""
    def __init__(
            self,
            label: str,
            datatypes: Optional[Iterable[Datatype]],
    ):
        super().__init__(label)
        super(Member, self).__init__(datatypes)

    def __reduce__(self):
        return (type(self), (self.label, list(self)), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.label}, {self.labels})"
