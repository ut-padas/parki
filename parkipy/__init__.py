"""
Python package for performance portable Ewald summation.
"""

__version__ = "0.0.1"

try:
    import kokkos
except ModuleNotFoundError as e:
    msg = (
        "parkipy requires pykokkos for parallel execution, "
        "see 'https://kokkos.org/pykokkos/installation.html' for install instructions."
    )
    raise ModuleNotFoundError(msg) from e

from ._celllist import CellList

__parkipy_submodules__ = {"ewald", "distributed"}


def __getattr__(attr):
    match attr:
        case "ewald":
            import parkipy.ewald as ewald

            return ewald
        case "distributed":
            import parkipy.distributed as distributed

            return distributed
        case _:
            raise AttributeError(f"module {__name__!r} has no attribute {attr!r}")
