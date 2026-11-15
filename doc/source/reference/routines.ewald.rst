.. _routines.ewald:
.. automodule:: parkipy.ewald

Repeated evaluations
--------------------

For a fixed geometry, prepare a reusable :class:`parkipy.ewald.EwaldPlan` with
:func:`parkipy.ewald.prepare_stokes_comb` or
:func:`parkipy.ewald.prepare_stokes_sl`. See :ref:`fixed-geometry` for usage,
output ownership, and GMRES integration. Laplace plans are not currently
available because parameter selection depends on the changing charge norm;
see :ref:`laplace-plan-limitations` for the accuracy policy that remains to
be defined.

.. automethod:: parkipy.ewald.EwaldPlan.apply
