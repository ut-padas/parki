import time
import argparse
import pytest
import numpy as np
import parkipy


@pytest.mark.parametrize(
    "kernel",
    [parkipy.ewald.stokes_sl, parkipy.ewald.stokes_comb, parkipy.ewald.laplace],
)
@pytest.mark.parametrize(
    "periodicity",
    [
        pytest.param(0, marks=pytest.mark.xfail(reason="not yet implemented")),
        1,
        pytest.param(2, marks=pytest.mark.xfail(reason="not yet implemented")),
        3,
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_self_convergence(kernel, periodicity, dtype):
    """
    run self convergence test on all supported kernels
    for all periodic directions in single and double
    precision arithmetic.
    """
    if kernel == parkipy.ewald.laplace and periodicity == 1:
        pytest.xfail("not yet implemented")
    run_convergence(fun=kernel, periodicity=periodicity, dtype=dtype)


def test_FGC_R2R():
    """
    test the R2C FFT paired with C2R IFFT
    gives the same results as C2C FFT, IFFT
    for real densities
    """
    raise NotImplementedError


def test_gpu():
    """
    test that code compiles, runs, and is correct
    on all available devices for any solver
    """
    run_convergence(device="GPU")


@pytest.mark.parametrize("p2p_method", ["GM-1D", "GM-2D", "SM-1D", "SM-2D"])
def test_p2p_stokes(p2p_method):
    """
    test different p2p methods for the 1-per stokes solver
    """
    run_convergence(p2p_method=p2p_method)


@pytest.mark.parametrize("p2g_method", ["BASE", "SOURCE", "GRID", "HYBRID"])
def test_p2g_stokes(p2g_method):
    """
    test different p2g method for the 1-per stokes solver
    """
    run_convergence(p2g_method=p2g_method)


@pytest.mark.parametrize("g2p_method", ["BASE", "TARGET"])
def test_g2p_stokes(g2p_method):
    """
    test different g2p methods for the 1-per stokes solver
    """
    run_convergence(g2p_method=g2p_method)


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
        tolvA = [1e-1, 1e-2, 1e-3, 1e-5, 1e-6]
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
        errorvA, 5 * tolvA, err_msg="Self-convergence failed: error exceeds tolerance."
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
                max_err < 20 * tol,
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
    verbosity=0,
    nutral=False,
) -> None:
    am = parkipy.utils.get_array_module(device)
    execution_space = parkipy.utils.get_execution_space(device)

    # GPU arrays
    if verbosity >= 2:
        print("=====allocate GPU arrays=====")

    am.random.seed(123)  # seed random numbers
    trg = am.random.rand(3, nt).astype(dtype) * am.array(box, dtype=dtype).reshape(
        -1, 1
    )
    src = am.random.rand(3, ns).astype(dtype) * am.array(box, dtype=dtype).reshape(
        -1, 1
    )
    if dtype == np.float32:
        trg = am.where(trg < 1e-5, 0.5, trg)
        src = am.where(src < 1e-5, 0.5, src)

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
    )
    if fun == parkipy.ewald.stokes_comb:
        dens_sl = am.random.randn(3, ns).astype(dtype)
        dens_dl = am.random.randn(3, ns).astype(dtype)
        dens = am.vstack(
            (dens_sl, dens_dl), dtype=dtype
        )  # stack densities for ewald call
        normal = am.random.randn(3, ns).astype(dtype)
        if dtype == "fp32":
            dens = am.where(dens < 1e-5, 0.5, dens)
            normal = am.where(normal < 1e-5, 0.5, normal)
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
        charges = am.random.randn(ns).astype(dtype)
        charges -= am.mean(charges).astype(dtype)
        if dtype == "fp_32":
            charges = am.where(charges < 1e-5, 0.5, charges)
        pot = fun(
            trg,
            src,
            charges,
            options,
        )
    elif fun == parkipy.ewald.stokes_sl:
        dens = am.random.randn(3, ns).astype(dtype)
        if dtype == "fp32":
            dense = am.where(dens < 1e-5, 0.5, dens)

        pot = fun(trg, src, dens, options)
    else:
        raise NotImplementedError

    return pot
