"""
Datatype module.
"""

from typing import Iterable, Union

from bids import BIDSLayout     # type: ignore

from .common import Collection, Member
from .file import FileBIDS


class Datatype(Member, Collection):
    """A class to represent recording modalities in a BIDS Session."""

    def __init__(
            self,
            label: str,
            contents: Union[Iterable[FileBIDS], BIDSLayout] = (),
            subject_label: str = '',
            session_label: str = '',
    ):
        Member.__init__(self, label)

        if isinstance(contents, BIDSLayout):
            if session_label:
                contents = contents.get(
                    subject=subject_label,
                    session=session_label,
                    datatype=label,
                )
            else:
                contents = contents.get(
                    subject=subject_label,
                    datatype=label,
                )

            for content in contents:
                setattr(content, 'label', content.filename)

        Collection.__init__(self, contents)

    def __reduce__(self):
        return (type(self), (self.label, list(self)), self.__dict__)

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}('{self.label}', files={self.labels})"
