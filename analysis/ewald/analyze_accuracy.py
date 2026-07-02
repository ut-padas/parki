"""
Generate a table showing the achieved accuracy
for different tolerances.
"""

from pathlib import Path
import argparse
import parkipy
import pandas as pd


def float_format(num):
    try:
        num = float(num)
    except:
        return num

    sci_str = f"{num:.1e}"

    coeff_str, exp_str = sci_str.split("e")

    coeff = float(coeff_str)
    exp = int(exp_str)  # int() removes leading zeros like in '-09'

    if coeff == 1.0:
        return rf"$10^{{{exp}}}$"
    else:
        if coeff.is_integer():
            coeff = int(coeff)
        return rf"${coeff} \times 10^{{{exp}}}$"


def main(args):
    base_dir = Path(args.output_dir).resolve()
    tables_dir = base_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    xp = parkipy.utils.get_array_module()

    tols = [10 ** (-i) for i in range(1, 8)]
    ref_tol = 1e-12

    # script constants
    N = 4_000_000
    s = 512
    box = [1, 1, 1]

    rng = xp.random.default_rng(2005)

    # set up particles and densities
    x = rng.random((3, N), dtype=float)
    y = rng.random((3, N), dtype=float)
    q = rng.random(N, dtype=float)
    q -= q.mean()  # charge neutrality
    f = rng.random((3, N), dtype=float)
    h = rng.random((6, N), dtype=float)
    n = rng.random((3, N), dtype=float)
    n /= xp.linalg.norm(n, axis=0)

    # get reference solutions
    o1p = parkipy.ewald.EwaldOptions(
        periodicity=1,
        box=box,
        tolerance=ref_tol,
        cell_size=s,
    )
    o3p = parkipy.ewald.EwaldOptions(
        periodicity=3,
        box=box,
        tolerance=ref_tol,
        cell_size=s,
    )

    # get potential at tols
    l3p_ref = parkipy.ewald.laplace(x, y, q, o3p)
    sl1p_ref = parkipy.ewald.stokes_sl(x, y, f, o1p)
    sl3p_ref = parkipy.ewald.stokes_sl(x, y, f, o3p)
    sldl1p_ref = parkipy.ewald.stokes_comb(x, y, h, n, o1p)
    sldl3p_ref = parkipy.ewald.stokes_comb(x, y, h, n, o3p)

    # define error norm
    Norm = lambda x, ref: xp.max(xp.linalg.norm(x - ref, axis=-1)) / xp.max(
        xp.linalg.norm(ref, axis=-1)
    )

    # error dicts
    el3p = dict()
    esl1p = dict()
    esl3p = dict()
    esldl1p = dict()
    esldl3p = dict()

    for tol in tols:
        print(f"***tolerance {tol:.1e}***")
        o1p = parkipy.ewald.EwaldOptions(
            periodicity=1,
            box=box,
            tolerance=tol,
            cell_size=s,
        )
        o3p = parkipy.ewald.EwaldOptions(
            periodicity=3,
            box=box,
            tolerance=tol,
            cell_size=s,
        )

        # get potential at tols
        l3p = parkipy.ewald.laplace(x, y, q, o3p)
        sl1p = parkipy.ewald.stokes_sl(x, y, f, o1p)
        sl3p = parkipy.ewald.stokes_sl(x, y, f, o3p)
        sldl1p = parkipy.ewald.stokes_comb(x, y, h, n, o1p)
        sldl3p = parkipy.ewald.stokes_comb(x, y, h, n, o3p)

        # get error against reference
        el3p[tol] = Norm(l3p, l3p_ref)
        esl1p[tol] = Norm(sl1p, sl1p_ref)
        esl3p[tol] = Norm(sl3p, sl3p_ref)
        esldl1p[tol] = Norm(sldl1p, sldl1p_ref)
        esldl3p[tol] = Norm(sldl3p, sldl3p_ref)

    # build accuracy table: rows = kernels, columns = tolerances
    table = pd.DataFrame(
        {
            r"P--$3\mathcal{P}$": el3p,
            r"SL--$1\mathcal{P}$": esl1p,
            r"SL--$3\mathcal{P}$": esl3p,
            r"SL\&DL--$1\mathcal{P}$": esldl1p,
            r"SL\&DL--$3\mathcal{P}$": esldl3p,
        }
    )
    table.index.name = "Tol"
    print(table)

    # export to latex
    table.style.format(float_format).format_index(float_format).to_latex(
        buf=tables_dir / "accuracy.tex",
        caption=rf"Maximum component-wise $L_2$ self-convergence error for $N={float_format(N)[1:-1]}$ particles with $s={s}$ for different kernels and tolerances; "
        rf"$\#\mathcal{{P}}$ is the number of dimensions with periodic tessellations; "
        "P is the Poisson kernel, SL is the Stokeslet, DL is the Stresslet. "
        "Particles, densities, and normal vectors are sampled from a uniform distribution."
        rf"Reference solution has tolerance {float_format(ref_tol)}.",
        label="t:accuracy",
        position_float="centering",
        hrules=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ewald self-convergence test for different kernels."
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Path to the directory where data, tables, and figs folders will be created.",
    )
    args = parser.parse_args()

    exit(main(args))
