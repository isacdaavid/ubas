"""
Class to represent individual participants.
"""

from types import SimpleNamespace
from typing import Callable, Iterable, Mapping

import numpy as np

from connectivity import FunctionalConnectivity, StructuralConnectivity
from load import load_matlab, load_tsv

class Subject:
    def __init__(
            self,
            subject_label: str,
            structural_session_label: str = None,
            structural_derivatives: str = None,
            functional_session_label: str = None,
            functional_derivatives: str = None,
            atlases: Iterable = ['4S156', '4S256', '4S456'],
            demographics: Mapping = None,
    ):
        self._label = subject_label
        self._structural_session = structural_session_label
        self._functional_session = functional_session_label

        self._demographics = demographics

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
                structural_connectivity = structural,
                functional_connectivity = functional,
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
    def demographics(self):
        return self._demographics

    @property
    def structural_connectivity(self):
        return self._structural_connectivity

    @property
    def functional_connectivity(self):
        return self._functional_connectivity

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

    def __repr__(self):
        return f"Subject({self.label}, {self.demographics['diagnosis']})"

    def __hash__(self):
        return hash(self.label)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.label == other.label
        return False
