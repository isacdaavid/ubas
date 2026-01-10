"""Session module."""

from typing import Iterable, Union

from bids import BIDSLayout     # type: ignore

from .common import Collection, Member
from .datatype import Datatype


class Session(Member, Collection):
    """A class to represent a neuroimaging session in a BIDS Subject."""
    def __init__(
            self,
            label: str,
            contents: Union[Iterable[Datatype], BIDSLayout] = (),
            subject_label: str = '',
    ):
        Member.__init__(self, label)

        if isinstance(contents, BIDSLayout):
            labels = contents.get(
                subject=subject_label,
                session=label,
                target='datatype',
                return_type='id'
            )
            contents = [
                Datatype(
                    label,
                    contents,
                    subject_label=subject_label,
                    session_label=self.label
                )
                for label in labels
            ]

        Collection.__init__(self, contents)

    def __reduce__(self):
        return (type(self), (self.label, list(self)), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.label}', datatypes={self.labels})"
