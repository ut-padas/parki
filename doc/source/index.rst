.. ParkiPy documentation master file, created by
   sphinx-quickstart on Wed Jul 23 14:46:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ParkiPy documentation
=====================

ParkiPy is a package for performance portable particle kernel interactions in Python.
It is written in Python and uses NumPy like arrays and PyKokkos to execute on 
CPUs and GPUs with excellent performance. ParkiPy provides a `CellList` class
to build data structures for particle interactions within a given 
cutoff radius and the `ewald` module to compute kernel *N-Body* problems 
with Ewald summation.

.. toctree::
   :maxdepth: 2
   :hidden:

   User Guide <user/index>
   API Reference <reference/index>
