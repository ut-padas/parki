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
@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_self_convergence(kernel, periodicity, dtype, device):
    """
    run self convergence test on all supported kernels
    for all periodic directions in single and double
    precision arithmetic.
    """
    if kernel == parkipy.ewald.laplace and periodicity == 1:
        pytest.xfail("not yet implemented")
    run_convergence(fun=kernel, periodicity=periodicity, dtype=dtype, device=device)


@pytest.mark.parametrize("p2p_method", ["GM-1D", "GM-2D", "SM-1D", "SM-2D"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_p2p_stokes(p2p_method, device, dtype):
    """
    test different p2p methods for the 1-per stokes solver
    """
    run_convergence(p2p_method=p2p_method, device=device, dtype=dtype)


@pytest.mark.parametrize("p2g_method", ["BASE", "SOURCE", "GRID", "HYBRID"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_p2g_stokes(p2g_method, device, dtype):
    """
    test different p2g method for the 1-per stokes solver
    """
    run_convergence(p2g_method=p2g_method, device=device, dtype=dtype)


@pytest.mark.parametrize("g2p_method", ["BASE", "TARGET"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_g2p_stokes(g2p_method, device, dtype):
    """
    test different g2p methods for the 1-per stokes solver
    """
    run_convergence(g2p_method=g2p_method, device=device, dtype=dtype)


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
