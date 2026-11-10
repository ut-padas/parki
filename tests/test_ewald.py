import time
import argparse
import warnings

import pytest
import numpy as np
import parkipy
import pykokkos as pk

from parkipy.ewald._params import se_params_stokeslet

GPU_AVAILABLE = pk.kokkos_manager.get_gpu_framework() is not None

DEVICE_PARAMS = [
    "CPU",
    pytest.param(
        "GPU",
        marks=pytest.mark.skipif(
            not GPU_AVAILABLE, reason="no GPU framework available"
        ),
    ),
]


@pytest.mark.parametrize(
    "kernel",
    [parkipy.ewald.stokes_sl, parkipy.ewald.stokes_comb, parkipy.ewald.laplace],
)
@pytest.mark.parametrize(
    "periodicity",
    [
        pytest.param(0, marks=pytest.mark.xfail(reason="not yet implemented")),
        1,
        2,
        3,
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", DEVICE_PARAMS)
class TestEwald:
    """
    Tests for each Ewald sum configuration in the package.
    """

    def error_rms(self, actual, desired):
        """
        relative rms error between `actual` and `desired`.

        both arrays must have shape ``(3, n)`` (vector-valued) or ``(n,)``
        (scalar-valued).  the pointwise vector norm is computed along axis 0,
        then the rms is taken over the n points and normalised by the rms norm
        of `desired`.

        when ``desired`` is identically zero the denominator is zero, so the
        absolute rms error is returned instead (and a warning is issued).

        returns a dimensionless (or absolute) scalar.
        """
        actual = np.asarray(actual)
        desired = np.asarray(desired)
        pointwise_err = np.linalg.norm(actual - desired, axis=0)
        pointwise_ref = np.linalg.norm(desired, axis=0)
        n = pointwise_err.shape[-1] if pointwise_err.ndim > 0 else 1
        rms_err = np.sqrt(np.sum(pointwise_err**2) / n)
        rms_ref = np.sqrt(np.sum(pointwise_ref**2) / n)
        if rms_ref == 0.0:
            warnings.warn(
                "error_rms: desired is zero; returning absolute rms error.",
                RuntimeWarning,
                stacklevel=2,
            )
            return rms_err
        return rms_err / rms_ref

    def test_self_convergence(self, kernel, periodicity, dtype, device):
        """
        run self convergence test on all supported kernels
        for all periodic directions in single and double
        precision arithmetic.
        """
        if kernel == parkipy.ewald.laplace and periodicity in (1, 2):
            pytest.xfail("not yet implemented")
        run_convergence(fun=kernel, periodicity=periodicity, dtype=dtype, device=device)

    def test_periodic_translation(self, kernel, periodicity, dtype, device):
        """
        Test the translation-invariance of the kernels.
        First, get a reference solution by placing a shape
        in the center of the box, and then translate
        the shape to the periodic edges are confirm the
        answers agree to some finite precision.

        The test data comes from a use case
        where translation invariance initially failed.

        FIX: In version 0.0.1, this test was failing
        due to a finite-precision singularity in the P2P
        kernel. This was fixed by adding a puncture radius
        to EwaldOptions that is passed into the P2P kernel
        """
        if kernel == parkipy.ewald.laplace and periodicity in (1, 2):
            pytest.xfail("not yet implemented")

        am = parkipy.utils.get_array_module(device)

        # TEST PARAMS
        TOL = 1e-7

        # read in test data.
        data = np.load("tests/data/translation_test.npz")
        centered = data["centered"]
        offset = data[
            "offset"
        ]  # this is offset in the 3P case, need to correct for 1P, ...
        normals = data["normals"]
        box = data["box"]
        density = data["density"]

        translate = (
            am.array(
                [box[i] if i < periodicity else box[i] / 2 for i in range(len(box))]
            )
            - am.asarray(box) / 2
        )
        _offset = centered.copy() + translate.reshape(3, 1) + 1e-14
        offset -= am.array(
            [0 if i < periodicity else box[i] / 2 for i in range(len(box))]
        ).reshape(3, 1)
        am.testing.assert_allclose(
            actual=_offset, desired=offset, err_msg="offset array incorrect"
        )
        # offset = _offset.copy()

        # shared Ewald options
        options = parkipy.ewald.EwaldOptions(
            periodicity=periodicity,
            box=box,
            tolerance=1e-6,
            execution_space=device,
            cell_size=64,
        )

        # execute kernel on centered shape
        args = [centered, centered]
        match kernel:
            case parkipy.ewald.stokes_sl:
                args.append(density)
            case parkipy.ewald.stokes_comb:
                args.append(am.vstack([density, density]))
                args.append(normals)
            case parkipy.ewald.laplace:
                args.append(density[0])
            case _:
                raise NotImplementedError(f"Test for kernel {kernel} not implemented")
        centered_potential = kernel(*args, options=options)

        # reset sources/targets to the offset points
        args[0] = offset  # % am.array(box).reshape(3,1)
        args[0][:periodicity] = (
            args[0][:periodicity] % am.array(box).reshape(3, 1)[:periodicity]
        )
        args[1] = args[0].copy()
        offset_potential = kernel(*args, options=options)

        error = self.error_rms(actual=offset_potential, desired=centered_potential)
        if error > TOL:
            raise AssertionError(
                f"error '{error:.2e}' exceeds the tolerance '{TOL:.2e}'."
            )


@pytest.mark.parametrize("p2p_method", ["GM-1D", "GM-2D", "SM-1D", "SM-2D"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_p2p_stokes(p2p_method, device, dtype):
    """
    test different p2p methods for the 1-per stokes solver
    """
    run_convergence(p2p_method=p2p_method, device=device, dtype=dtype)


@pytest.mark.parametrize("p2g_method", ["BASE", "SOURCE", "GRID", "HYBRID"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_p2g_stokes(p2g_method, device, dtype):
    """
    test different p2g method for the 1-per stokes solver
    """
    run_convergence(p2g_method=p2g_method, device=device, dtype=dtype)


@pytest.mark.parametrize("g2p_method", ["BASE", "TARGET"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", DEVICE_PARAMS)
def test_g2p_stokes(g2p_method, device, dtype):
    """
    test different g2p methods for the 1-per stokes solver
    """
    run_convergence(g2p_method=g2p_method, device=device, dtype=dtype)


@pytest.mark.parametrize("periodicity", [1, 2, 3])
@pytest.mark.parametrize("box", [[2.0, 1.0, 1.0], [1.0, 1.5, 1.0]])
def test_rectangular_fourier_resolution(box, periodicity):
    """
    Test grid sizing and Fourier resolution for rectangular periodic boxes.

    Check practical grid sizes and uniform spacing in every periodic direction.
    Decimal aspect ratios must not produce enormous grids through an exact
    rational representation of a floating-point ratio.

    Compare Stokes potentials with different cutoffs and both FFT formats.
    Agreement checks Fourier padding in the free directions for 1P and 2P,
    and Fourier resolution for 3P. The existing convergence tests use cubic
    boxes, so they do not exercise these aspect-ratio-dependent errors.
    Modest aspect ratios keep the padded 1P grids affordable for routine tests.
    """
    params = se_params_stokeslet(
        {"box": box}, tolerance=1e-8, periodicity=periodicity, rc=0.4
    )
    assert np.max(params.grid_shape_ext) < 1000
    np.testing.assert_allclose(
        actual=params.grid_shape_ext[:periodicity] * params.h,
        desired=box[:periodicity],
        rtol=1e-14,
        err_msg="Periodic grid spacing does not match the rectangular box",
    )

    rng = np.random.default_rng(321)
    targets = rng.random((3, 17)) * np.array(box).reshape(3, 1)
    sources = rng.random((3, 29)) * np.array(box).reshape(3, 1)
    density = rng.standard_normal((3, 29))
    potentials = []
    for fft_type, rc in [("R2C", 0.6), ("C2C", 0.6), ("R2C", 0.9)]:
        options = parkipy.ewald.EwaldOptions(
            box=box,
            periodicity=periodicity,
            tolerance=1e-8,
            rc=rc,
            execution_space="CPU",
            fft_type=fft_type,
        )
        potentials.append(parkipy.ewald.stokes_sl(targets, sources, density, options))
    for potential in potentials[1:]:
        error = TestEwald().error_rms(actual=potential, desired=potentials[0])
        assert error < 1e-6, (
            f"Rectangular {periodicity}P Fourier consistency error {error:.2e} "
            "exceeds tolerance 1e-6"
        )


def run_convergence(
    device=None,
    p2p_method="GM-1D",
    p2g_method="HYBRID",
    g2p_method="TARGET",
    nt=312,
    ns=773,
    cell_size=512,
    dtype=np.float64,
    box=[1, 1, 1],
    fun=parkipy.ewald.stokes_comb,
    periodicity=1,
    nutral=False,
    fft_type="R2C",
    torch_fft=False,
):
    """
    Test (1) self-convergence and (2) consistency
    """
    execution_space = parkipy.utils.get_execution_space(device)
    am = parkipy.utils.get_array_module(execution_space)
    # Test 1: Reduce tolerance gradually, fixed cell_list.
    if dtype == np.float64:
        tolvA = [1e-4, 1e-5, 1e-7, 1e-9, 1e-11]
    elif dtype == np.float32:
        tolvA = [1e-1, 1e-4, 1e-5, 1e-6]
    else:
        raise NotImplementedError(f"dtype={dtype}")
    potvA = []
    for tol in tolvA:
        print(f":: Test 1: Running tol={tol}")
        potential = run(
            fun,
            cell_size,
            tol,
            nt,
            ns,
            device,
            p2p_method,
            p2g_method,
            g2p_method,
            dtype,
            box,
            periodicity,
            fft_type,
            torch_fft,
            verbosity=0,
            nutral=nutral,
        )
        if len(potential.shape) == 1:
            potential = potential.reshape(1, -1)
        potential = potential.T
        potvA.append(potential)
    print()

    # Test 2: Vary rc for two different tolerances
    if dtype == np.float64:
        tolvB = [1e-5, 1e-9]
    elif dtype == np.float32:
        tolvB = [1e-3, 1e-5]
    else:
        raise NotImplementedError
    rcvB = am.linspace(0.2, 0.5, 10)
    clszvB = rcvB**3 * max(ns, nt)
    potvB = [[None] * am.size(clszvB), [None] * am.size(clszvB)]
    for j, tol in enumerate(tolvB):
        for k, clsz in enumerate(clszvB):
            print(f":: Test 2: Running tol={tol}, cell_size={clsz}")
            potential = run(
                fun,
                int(clsz),
                tol,
                nt,
                ns,
                device,
                p2p_method,
                p2g_method,
                g2p_method,
                dtype,
                box,
                periodicity,
                fft_type,
                torch_fft,
                verbosity=0,
                nutral=nutral,
            )
            if len(potential.shape) == 1:
                potential = potential.reshape(1, -1)
            potential = potential.T
            potvB[j][k] = potential
    print()

    # Self-convergence
    print("== Self-convergence ==")
    Norm = lambda x, ref: am.max(am.linalg.norm(x - ref, axis=1)) / am.max(
        am.linalg.norm(ref, axis=1)
    )
    errorvA = am.array([Norm(potvA[j], potvA[-1]) for j in range(len(potvA))])
    tolvA = am.asarray(tolvA)
    print("Errors:", errorvA)
    print("Tolerances:", tolvA)
    am.testing.assert_array_less(
        errorvA, 10 * tolvA, err_msg="Self-convergence failed: error exceeds tolerance."
    )

    # Vary rc (consistency check)
    print("== Consistency ==")
    for j, tol in enumerate(tolvB):
        print(f"Tol={tol}:", end=" ")
        ref = potvB[j][0]
        errors = []
        for k in range(1, am.size(clszvB)):
            err = Norm(potvB[j][k], ref)
            errors.append(err)
            print("%.3e" % err, end=" ")
        print()
        errors = am.asarray(errors)
        max_err = am.max(errors)
        try:
            np.testing.assert_(
                max_err < 30 * tol,
                msg=f"Consistency test failed for tol={tol:.3e}: max error = {max_err:.3e}",
            )
        except AssertionError as e:
            print(
                f"Consistency test failed for tol={tol:.3e}: max error = {max_err:.3e}"
            )
            raise AssertionError(
                f"Consistency test failed for tol={tol:.3e}: max error = {max_err:.3e}"
            )

    return


def bytes_to_gb(bytes_value):
    return bytes_value / (1024**3)


def run(
    fun,
    cell_size,
    tolerance,
    nt,
    ns,
    device,
    p2p_method,
    p2g_method,
    g2p_method,
    dtype,
    box,
    periodicity,
    fft_type,
    torch_fft,
    verbosity=0,
    nutral=False,
) -> None:
    am = parkipy.utils.get_array_module(device)
    execution_space = parkipy.utils.get_execution_space(device)

    # GPU arrays
    if verbosity >= 2:
        print("=====allocate GPU arrays=====")

    rng = am.random.default_rng(123)
    trg = rng.random(size=(3, nt), dtype=dtype) * am.array(box, dtype=dtype).reshape(
        -1, 1
    )
    src = rng.random(size=(3, ns), dtype=dtype) * am.array(box, dtype=dtype).reshape(
        -1, 1
    )

    options = parkipy.ewald.EwaldOptions(
        periodicity=periodicity,
        box=box,
        tolerance=tolerance,
        cell_size=cell_size,
        p2p_method=p2p_method,
        g2p_method=g2p_method,
        p2g_method=p2g_method,
        fft_type=fft_type,
        execution_space=execution_space,
        torch_fft=torch_fft,
    )
    if fun == parkipy.ewald.stokes_comb:
        dens_sl = rng.random(size=(3, ns), dtype=dtype)
        dens_dl = rng.random(size=(3, ns), dtype=dtype)
        dens = am.vstack(
            (dens_sl, dens_dl), dtype=dtype
        )  # stack densities for ewald call
        normal = rng.random(size=(3, ns), dtype=dtype)
        if verbosity >= 2:
            print("======Spectral Ewald Sum======")

        pot = fun(
            trg,
            src,
            dens,
            normal,
            options,
        )
    elif fun == parkipy.ewald.laplace:
        charges = rng.standard_normal(size=ns, dtype=dtype)
        charges -= am.mean(charges)
        pot = fun(
            trg,
            src,
            charges,
            options,
        )
    elif fun == parkipy.ewald.stokes_sl:
        dens = rng.random(size=(3, ns), dtype=dtype)

        pot = fun(trg, src, dens, options)
    else:
        raise NotImplementedError

    return pot
