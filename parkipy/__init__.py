"""
Python package for performance portable Ewald summation.
"""

__version__ = "0.0.1"

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
