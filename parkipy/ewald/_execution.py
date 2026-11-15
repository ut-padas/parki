"""Retained launch arguments for fixed-geometry Ewald applications."""

from dataclasses import dataclass
import numpy as np
import pykokkos as pk


class Launch:
    """Keep array owners and their zero-copy PyKokkos views alive together."""

    def __init__(self, policy, workunit, kwargs, name=None):
        self.policy = policy
        self.workunit = workunit
        self.name = name
        self.owners = kwargs
        self.kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                value = pk.array(value, space=pk.MemorySpace.HostSpace)
            elif hasattr(value, "__cuda_array_interface__"):
                value = pk.array(value)
            self.kwargs[key] = value

    def __call__(self):
        pk.parallel_for(self.name, self.policy, self.workunit, **self.kwargs)


@dataclass
class ParticleStage:
    launch: Launch
    source_list: object = None
    grid: object = None
