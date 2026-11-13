"""Repeated applications must preserve the operator and reuse geometry."""

import numpy as np
import pytest
import pykokkos as pk

from parkipy import ewald
from parkipy.utils import get_array_module

GPU_AVAILABLE = pk.kokkos_manager.get_gpu_framework() is not None
DEVICES = [
    "Serial",
    pytest.param(
        "GPU",
        marks=pytest.mark.skipif(
            not GPU_AVAILABLE, reason="no GPU framework available"
        ),
    ),
]


def make_problem(device="Serial", dtype=np.float64, periodicity=1, **kwargs):
    am = get_array_module(device)
    rng = np.random.default_rng(203)
    src = am.asarray(rng.uniform(0.1, 0.9, (3, 15)), dtype=dtype)
    trg = am.asarray(rng.uniform(0.1, 0.9, (3, 11)), dtype=dtype)
    normals = am.asarray(rng.normal(size=src.shape), dtype=dtype)
    density = am.asarray(rng.normal(size=(6, src.shape[1])), dtype=dtype)
    options = ewald.EwaldOptions(
        box=[1.0, 1.0, 1.0],
        periodicity=periodicity,
        tolerance=1e-3,
        execution_space=device,
        cell_size=4,
        **kwargs,
    )
    return am, trg, src, normals, density, options


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("periodicity", [0, 1, 2, 3])
@pytest.mark.parametrize("combined", [False, True])
def test_repeated_densities(device, dtype, periodicity, combined):
    am, trg, src, normals, density, options = make_problem(device, dtype, periodicity)
    if combined:
        plan = ewald.prepare_stokes_comb(trg, src, normals, options)
        reference = lambda q: ewald.stokes_comb(trg, src, q, normals, options)
    else:
        density = density[:3]
        plan = ewald.prepare_stokes_sl(trg, src, options)
        reference = lambda q: ewald.stokes_sl(trg, src, q, options)
    expected = reference(density)
    tolerance = 2e-5 if dtype == np.float32 else 1e-11
    # Zero between nonzero calls catches stale grid, padded density and output.
    for scale in (2, 0, -1):
        am.testing.assert_allclose(
            plan(scale * density), scale * expected, rtol=tolerance, atol=tolerance
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "p2g_method,g2p_method,p2p_method,fft_type",
    [
        ("BASE", "BASE", "GM-1D", "R2C"),
        ("SOURCE", "TARGET", "GM-2D", "C2C"),
        ("GRID", "BASE", "SM-1D", "C2C"),
        ("HYBRID", "TARGET", "SM-2D", "R2C"),
    ],
)
def test_methods(device, p2g_method, g2p_method, p2p_method, fft_type):
    am, trg, src, normals, q, options = make_problem(
        device,
        p2g_method=p2g_method,
        g2p_method=g2p_method,
        p2p_method=p2p_method,
        fft_type=fft_type,
    )
    plan = ewald.prepare_stokes_comb(trg, src, normals, options)
    expected = ewald.stokes_comb(trg, src, q, normals, options)
    # Alternate methods have their own reset and accumulation behavior.
    for scale in (1, 0, -1):
        am.testing.assert_allclose(
            plan(scale * q), scale * expected, rtol=1e-11, atol=1e-11
        )


@pytest.mark.parametrize("device", DEVICES)
def test_output_ownership(device):
    am, trg, src, normals, q, options = make_problem(device)
    plan = ewald.prepare_stokes_comb(trg, src, normals, options)
    original = plan(q)
    saved = original.copy()
    plan(-q)
    out = am.empty_like(original)
    assert plan.apply(q, out=out) is out
    am.testing.assert_array_equal(original, saved)


def test_owned_geometry_and_options():
    am, trg, src, normals, q, options = make_problem()
    expected = ewald.stokes_comb(trg, src, q, normals, options)
    plan = ewald.prepare_stokes_comb(trg, src, normals, options)
    trg.fill(0.2)
    src.fill(0.4)
    normals.fill(0)
    options.box[:] = [2.0, 2.0, 2.0]
    options.p2g_method = "BASE"
    am.testing.assert_allclose(plan(q), expected, rtol=1e-11, atol=1e-11)


def test_no_repeated_preparation_or_view_wrapping(monkeypatch):
    _, trg, src, normals, q, options = make_problem()
    plan = ewald.prepare_stokes_comb(trg, src, normals, options)
    plan(q)  # Warm up runtime dispatch before checking the repeated-call path.
    import parkipy.ewald._ewald as stages
    from parkipy.ewald._prepare import DevicePre

    def fail(*args, **kwargs):
        raise AssertionError("repeated geometry preparation or view wrapping")

    monkeypatch.setattr(stages, "CellList", fail)
    monkeypatch.setattr(DevicePre, "from_particles", fail)
    monkeypatch.setattr(pk.View, "__init__", fail)
    plan(q)


def test_validation_and_aliasing():
    _, _, src, normals, q, options = make_problem()
    plan = ewald.prepare_stokes_comb(src, src, normals, options)
    with pytest.raises(ValueError, match="shape"):
        plan(q[:, :-1])
    with pytest.raises(TypeError, match="dtype"):
        plan(q.astype(np.float32))
    with pytest.raises(ValueError, match="shape"):
        plan(q, out=np.empty((3, 2)))
    with pytest.raises(TypeError, match="dtype"):
        plan(q, out=np.empty(src.shape, dtype=np.float32))
    readonly = np.empty_like(src)
    readonly.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        plan(q, out=readonly)
    expected = plan(q)
    aliased = q.copy()
    plan(aliased, out=aliased[:3])
    np.testing.assert_allclose(aliased[:3], expected)
    # Noncontiguous input and output are copied to/from contiguous workspaces.
    noncontiguous = np.empty((6, 2 * src.shape[1]))[:, ::2]
    noncontiguous[:] = q
    out = np.empty((3, 2 * src.shape[1]))[:, ::2]
    plan(noncontiguous, out=out)
    np.testing.assert_allclose(out, expected)
