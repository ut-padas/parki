import time
import argparse
import pytest
import numpy as np

from parkipy.utils import get_array_module, get_execution_space, get_dtype
from parkipy.ewald import stokes_comb, laplace, EwaldOptions


@pytest.mark.parametrize("fft_type", ["R2C", "C2C"])
def test_fft_type(fft_type):
    run_convergence(fft_type=fft_type)


@pytest.mark.parametrize("device", ["OpenMP", "CUDA"])
@pytest.mark.parametrize("dtype", ["fp32", "fp64"])
def test_laplace_3p(device, dtype):
    run_convergence(fun=laplace, periodicity=3, device=device, dtype=dtype, nutral=True, p2g_method="HYBRID")


@pytest.mark.parametrize("box", [[1.0, 1.0, 1.0], [1.15, 1, 1]])
def test_box(box):
    run_convergence(box=box)


@pytest.mark.parametrize("dtype", ["fp32", "fp64"])
def test_dtype(dtype):
    run_convergence(dtype=dtype)


@pytest.mark.parametrize("device", ["OpenMP", "CUDA"])
def test_stokes_1p(device):
    run_convergence(device=device)


# @pytest.mark.parametrize("periodicity", [0,1,2,3])
@pytest.mark.parametrize("periodicity", [1, 3])
@pytest.mark.parametrize("p2p_method", ["GM-1D", "GM-2D", "SM-1D", "SM-2D"])
def test_p2p_stokes(p2p_method, periodicity):
    run_convergence(p2p_method=p2p_method, periodicity=periodicity)


# @pytest.mark.parametrize("periodicity", [0,1,2,3])
@pytest.mark.parametrize("periodicity", [1, 3])
@pytest.mark.parametrize("p2g_method", ["BASE", "SOURCE", "GRID", "HYBRID"])
def test_p2g_stokes(p2g_method, periodicity):
    run_convergence(p2g_method=p2g_method, periodicity=periodicity)


# @pytest.mark.parametrize("periodicity", [0,1,2,3])
@pytest.mark.parametrize("periodicity", [1, 3])
@pytest.mark.parametrize("g2p_method", ["BASE", "TARGET"])
def test_g2p_stokes(g2p_method, periodicity):
    run_convergence(g2p_method=g2p_method, periodicity=periodicity)


def run_convergence(
    device="Cuda",
    p2p_method="GM-1D",
    p2g_method="BASE",
    g2p_method="BASE",
    nt=5000,
    cell_size=512,
    dtype="fp64",
    box=[1.0, 1.0, 1.0],
    fun=stokes_comb,
    periodicity=1,
    nutral=False,
    fft_type="R2C",
):
    """
    Test (1) self-convergence and (2) consistency
    """
    execution_space = get_execution_space(device)
    am = get_array_module(execution_space)
    # Test 1: Reduce tolerance gradually, fixed cell_list.
    if dtype == "fp64":
        tolvA = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]
    if dtype == "fp32":
        tolvA = [1e-1, 1e-2, 1e-3, 1e-5, 1e-6]
    potvA = []
    for tol in tolvA:
        print(f":: Test 1: Running tol={tol}")
        potential = run(
            fun,
            cell_size,
            tol,
            nt,
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
    if dtype == "fp64":
        tolvB = [1e-5, 1e-9]
    if dtype == "fp32":
        tolvB = [1e-3, 1e-5]
    rcvB = am.linspace(0.2, 0.5, 10)
    clszvB = rcvB**3 * 33600
    potvB = [[None] * am.size(clszvB), [None] * am.size(clszvB)]
    for j, tol in enumerate(tolvB):
        for k, clsz in enumerate(clszvB):
            print(f":: Test 2: Running tol={tol}, cell_size={clsz}")
            potential = run(
                fun,
                int(clsz),
                tol,
                nt,
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
                max_err < 90 * tol,
                msg=f"Consistency test failed for tol={tol:.3e}: max error = {max_err:.3e}",
            )
        except AssertionError as e:
            raise e

    return


def bytes_to_gb(bytes_value):
    return bytes_value / (1024**3)


def run(
    fun,
    cell_size,
    tolerance,
    nt,
    device,
    p2p_method,
    p2g_method,
    g2p_method,
    dtype,
    box,
    periodicity,
    fft_type,
    up=1,  # TODO test me
    verbosity=0,
    nutral=False,
) -> None:
    am = get_array_module(device)
    execution_space = get_execution_space(device)

    # deterministic arguments
    ns = nt * up
    # GPU arrays
    if verbosity >= 2:
        print("=====allocate GPU arrays=====")

    am.random.seed(123)  # seed random numbers
    trg = am.random.rand(3, nt).astype(get_dtype(dtype)) * am.array(
        box, dtype=get_dtype(dtype)
    ).reshape(-1, 1)
    src = am.random.rand(3, nt).astype(get_dtype(dtype)) * am.array(
        box, dtype=get_dtype(dtype)
    ).reshape(-1, 1)
    if dtype == 'fp32':
        trg = am.where(trg < 1e-5, 0.5, trg)
        src = am.where(src < 1e-5, 0.5, src)

    options = EwaldOptions(
            periodicity=periodicity, box=box, tolerance=tolerance, cell_size=cell_size,
            p2p_method=p2p_method, g2p_method=g2p_method, p2g_method=p2g_method,
            fft_type=fft_type, execution_space=execution_space) 
    if fun == stokes_comb:
        dens_sl = am.random.randn(3, ns).astype(get_dtype(dtype))
        dens_dl = am.random.randn(3, ns).astype(get_dtype(dtype))
        dens = am.vstack(
            (dens_sl, dens_dl), dtype=get_dtype(dtype)
        )  # stack densities for ewald call
        normal = am.random.randn(3, ns).astype(get_dtype(dtype))
        if dtype == 'fp32':
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
    elif fun == laplace:
        charges = am.random.randn(ns).astype(get_dtype(dtype))
        charges -= am.mean(charges).astype(get_dtype(dtype))
        if dtype == 'fp_32':
            charges = am.where(charges < 1e-5, 0.5, charges)
        pot = fun(
            trg,
            src,
            charges,
            options,
        )
    else:
        pass

    return pot
