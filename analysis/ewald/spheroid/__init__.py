import numpy as np
from .patches import create_patches_ellipsoid

__all__ = ["spheroid_patches", "spheroid_plain"]


def spheroid_patches(c=1.0, m=8):
    """
    Generate points on a spheroid of aspect ratio ``c``.
    This uses the 6-patch partition of unity discretization.

    Input:
        - ``c`` is the semi-axis in the z direction. The
          semi-axes in the x and y directions are 1.
        - ``m`` is the number of subintervals in each direction
          on a single patch. The total number of grid points
          will be ``Ntot = 6*(m-1)**2``.

    Returns an array of shape ``(3, Ntot)``.
    """
    pts, _ = create_patches_ellipsoid(m, m, 1.0, 1.0, 1.0)

    pts = pts.reshape(-1,3).T
    return pts


def spheroid_plain(c=1.0, N=8):
    """
    Generate points on a spheroid of aspect ratio ``c``.
    This uses uniform points in the angles.

    Input:
        - ``c`` is the semi-axis in the z direction. The
          semi-axes in the x and y directions are 1.
        - ``N`` is the number of subintervals along a meridian
          (from pole to pole). The number of subintervals along
          the equator will be ``2*N``. The total number of grid
          points will be ``Ntot = (N-1)*2*N`` since the points
          at the poles are omitted.

    Returns an array of shape ``(3, Ntot)``.
    """
    th = np.linspace(0, np.pi, num=N + 1)[1:-1]
    ph = np.linspace(0, 2 * np.pi, num=2 * N + 1)[:-1]
    th = th.reshape(-1, 1)
    ph = ph.reshape(1, -1)
    x = np.sin(th) * np.cos(ph)
    y = np.sin(th) * np.sin(ph)
    z = c * np.cos(th) * np.ones(ph.shape)
    pts = np.row_stack((x.flat, y.flat, z.flat))
    return pts
