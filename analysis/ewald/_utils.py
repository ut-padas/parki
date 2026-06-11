import numpy as np

DEVICE_CONSTANTS = {
    "a100": {
        "bandwidth": 1555,
        "gflops": 9.7e3,
        "bandwidth shmem": 20e3,
        "intensity": 9.7e3 / 1555,
        "peak flops": 9.7e3 * 1e9,
        "peak band": 1555 * 1e9,
    },
    "h200": {
        "bandwidth": 4000,
        "gflops": 33.5e3,
        "bandwidth shmem": np.inf,
        "intensity": 33.5e3 / 4000,
        "peak flops": 33.5e3 * 1e9,
        "peak band": 4000 * 1e9,
    },
    "mi300a": {
        "bandwidth": 5300,
        "gflops": 61.3e3,
        "bandwidth shmem": np.inf,
        "intensity": 61.3e3 / 5300,
        "peak flops": 61.3e3 * 1e9,
        "peak band": 5300 * 1e9,
    },
    "grace": {
        "bandwidth": 1000 / 2,
        "gflops": 7.1e3 / 2,
        "bandwidth shmem": np.inf,
        "intensity": 7.1e3 / 1000,
        "peak flops": 7.1e3 * 1e9 / 2,
        "peak band": 1000 * 1e9 / 2,
    },
    "epyc": {
        "bandwidth": 204.8 * 2,
        "gflops": 5e3,
        "bandwidth shmem": np.inf,
        "intensity": 5e3 / (204.8 * 2),
        "peak flops": 5e3 * 1e9 / 2,
        "peak band": 204.8 * 2 * 1e9 / 2,
    },
}


OPERATION_CONSTANTS = {
    "a100": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 29,
        "frsqrt": 19,
        "fdiv": 24,
        "fexpn": 45,
        "fsinh": 150,
        "ferf": 86,
    },
    "h200": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 41,
        "frsqrt": 30,
        "fdiv": 35,
        "fexpn": 50,
        "fsinh": 361,
        "ferf": 155,
    },
    "mi300a": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 41,
        "frsqrt": 30,
        "fdiv": 35,
        "fexpn": 50,
        "fsinh": 361,
        "ferf": 155,
    },
    "grace": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 41,
        "frsqrt": 30,
        "fdiv": 35,
        "fexpn": 50,
        "fsinh": 361,
        "ferf": 155,
    },
    "epyc": {
        "fadd": 2,
        "fmul": 2,
        "fsqrt": 41,
        "frsqrt": 30,
        "fdiv": 35,
        "fexpn": 50,
        "fsinh": 361,
        "ferf": 155,
    },
}


def determine_degree(P):
    if P < 2:
        raise ValueError("P cannot be smaller than 2")
    elif P % 2 != 0:
        raise ValueError("P must be even")
    elif P <= 10:
        return P // 2 + 1
    else:
        return min(P // 2 + 2, 9)


def p2g_count_flops(method, Ns, P):
    if method.upper() == "HYBRID":
        return Ns * (P) ** 3 * 35
    elif method.upper() in ["BASE", "SOURCE"]:
        return Ns * (P) ** 3 * 37
    elif method.upper() == "GRID":
        nu = determine_degree(P)
        return Ns * (27 * P / 2**3 * 3 + P**3 * (3 * (2 * nu)) * 35)
    else:
        raise ValueError(f"method {method} not analyzed.")

    return 0


def p2g_count_mops_hbm(Ns, P, b_fs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    return 27 * Ns * (P / 2) ** 3 * 12 * dsize / b_fs


def p2g_count_mops_shmem(method, Ns, P, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    isize = 4
    if method.upper() == "HYBRID":
        return Ns * (P) ** 3 * (12 * dsize + 3 * isize)
    return 0


def p2g_intensity(
    method,
    p2g_count_flops,
    p2g_count_hbm,
    p2g_count_shmem,
    P,
    b_fs,
    bandwidth_hbm,
    bandwidth_shmem,
    dp,
):

    return p2g_count_flops(method, 1, P) / (
        p2g_count_hbm(1, P, b_fs, dp)
        + p2g_count_shmem(method, 1, P, dp) * bandwidth_hbm / bandwidth_shmem
    )


def p2g_get_counts(
    method, ns, P, fs_cell_size, bandwidth_hbm, bandwidth_shmem, dp_flag=True
):
    flops = p2g_count_flops(method, ns, P)
    mops = (
        p2g_count_mops_hbm(ns, P, fs_cell_size, dp_flag)
        + p2g_count_mops_shmem(method, ns, P, dp_flag) * bandwidth_hbm / bandwidth_shmem
    )
    intensity = p2g_intensity(
        method,
        p2g_count_flops,
        p2g_count_mops_hbm,
        p2g_count_mops_shmem,
        P,
        fs_cell_size,
        bandwidth_hbm,
        bandwidth_shmem,
        dp_flag,
    )
    return intensity, flops, mops


def p2g_efficiency(dev, arch, method, time, ns, P, fs_cell_size, dp_flag):
    dev_name = ""
    if dev.upper() == "CUDA":
        if arch == 80:
            dev_name = "a100"
        elif arch == 90:
            dev_name = "h200"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HIP":
        if arch == 94:
            dev_name = "mi300a"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HOST":
        if arch == 0 or arch == "aarch64":
            dev_name = "grace"
        elif arch == 1 or arch == "x86_64":
            dev_name = "epyc"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    else:
        raise ValueError(f"Unknown device {dev}")

    dev_cons = DEVICE_CONSTANTS[dev_name]
    string = ""
    p2g_intensity, p2g_flops, p2g_mops = p2g_get_counts(
        method,
        ns,
        P,
        fs_cell_size,
        dev_cons["bandwidth"],
        dev_cons["bandwidth shmem"],
        dp_flag,
    )
    if p2g_intensity > dev_cons["intensity"]:
        string = f"${p2g_flops/dev_cons['peak flops']/time:.0%}$\nflops"
    else:
        string = f"${p2g_mops/dev_cons['peak band']/time:.0%}$\nmops"
    string = string.replace("%", "\\%")
    return string


def p2p_cnt_flop(op_cons, Nt, s):

    flops = 27 * Nt * s * (
        37 * op_cons["fmul"]
        + 17 * op_cons["fadd"]
        + 36
        + op_cons["frsqrt"]
        + op_cons["fdiv"]
        + op_cons["fexpn"]
        + op_cons["ferf"]
    ) * np.pi / 6 + 27 * Nt * s * (4 * op_cons["fadd"] + 4 + op_cons["fmul"])

    return flops


def p2p_cnt_mop(dev_cons, method, Nt, s, bt, bs, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    if method == "GM-1D":
        mop = Nt * (3 + 27 * s * 12 + 3) * dsize
    elif method == "SM-1D":
        mop = (
            Nt / bt * (bt * 3 + 27 * s * 12 + bt * 3) * dsize
            + Nt
            * (3 + 27 * s * 12)
            * dsize
            * dev_cons["bandwidth"]
            / dev_cons["bandwidth shmem"]
        )
    elif method == "GM-2D":
        mop = Nt * bs * (3 + 27 * s * 12 / bs + 3) * dsize
    elif method == "SM-2D":
        mop = (
            Nt / bt * (bt * 3 + 27 * s * 12 + bt * 3) * dsize
            + Nt
            * bs
            * (3 + 27 * s * 12 / bs)
            * dsize
            * dev_cons["bandwidth"]
            / dev_cons["bandwidth shmem"]
        )
    else:
        raise ValueError(f"performance model for {method} does not exist.")

    return mop


def p2p_cnt_intns(op_cons, dev_cons, method, Nt, s, bt, bs, dp=True):

    return p2p_cnt_flop(op_cons, Nt, s) / p2p_cnt_mop(
        dev_cons, method, Nt, s, bt, bs, dp
    )


def p2p_efficiency(dev, arch, method, time, nt, s, bt, bs, dp=True, both=False):
    dev_name = ""
    if dev.upper() == "CUDA":
        if arch == 80:
            dev_name = "a100"
        elif arch == 90:
            dev_name = "h200"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HIP":
        if arch == 94:
            dev_name = "mi300a"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HOST":
        if arch == 0 or arch == "aarch64":
            dev_name = "grace"
        elif arch == 1 or arch == "x86_64":
            dev_name = "epyc"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    else:
        raise ValueError(f"Unknown device {dev}")

    dev_cons = DEVICE_CONSTANTS[dev_name]
    op_cons = OPERATION_CONSTANTS[dev_name]
    string = ""

    p2p = {}
    p2p["flop"] = p2p_cnt_flop(op_cons, nt, s)
    p2p["mop"] = p2p_cnt_mop(dev_cons, method, nt, s, bt, bs, dp)
    p2p["intense"] = p2p_cnt_intns(op_cons, dev_cons, method, nt, s, bt, bs, dp)

    if p2p["intense"] > dev_cons["intensity"]:
        string = f"${p2p['flop']/dev_cons['peak flops']/time:.0%}$ (flops)"
    else:
        string = f"${p2p['mop']/dev_cons['peak band']/time:.0%}$ (mops)"
    string = string.replace("%", "\\%")
    if both:
        string_flops = f"${p2p['flop']/dev_cons['peak flops']/time:.0%}$"
        string_flops = string_flops.replace("%", "\\%")
        string_mops = f"${p2p['mop']/dev_cons['peak band']/time:.0%}$"
        string_mops = string_mops.replace("%", "\\%")
        string = [string_flops, string_mops]
    return string


def g2p_count_flops(method, Nt, P):
    return Nt * P**3 * (2 * 3 + 1)


def g2p_count_mops_hbm(Nt, P, dp=True):
    if dp:
        dsize = 8
    else:
        dsize = 4

    return Nt * P**3 * 3 * dsize


def g2p_intensity(
    method,
    g2p_count_flops,
    g2p_count_hbm,
    P,
    dp,
):

    return g2p_count_flops(method, 1, P) / (g2p_count_hbm(1, P, dp))


def g2p_get_counts(method, nt, P, dp_flag=True):
    flops = g2p_count_flops(method, nt, P)
    mops = g2p_count_mops_hbm(nt, P, dp_flag)
    intensity = g2p_intensity(
        method,
        g2p_count_flops,
        g2p_count_mops_hbm,
        P,
        dp_flag,
    )
    return intensity, flops, mops


def g2p_efficiency(dev, arch, method, time, ns, P, dp_flag):
    dev_name = ""
    if dev.upper() == "CUDA":
        if arch == 80:
            dev_name = "a100"
        elif arch == 90:
            dev_name = "h200"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HIP":
        if arch == 94:
            dev_name = "mi300a"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    elif dev.upper() == "HOST":
        if arch == 0 or arch == "aarch64":
            dev_name = "grace"
        elif arch == 1 or arch == "x86_64":
            dev_name = "epyc"
        else:
            raise ValueError(f"Unknown architecture {arch}")
    else:
        raise ValueError(f"Unknown device {dev}")

    dev_cons = DEVICE_CONSTANTS[dev_name]

    string = ""
    g2p_intensity, g2p_flops, g2p_mops = g2p_get_counts(method, ns, P, dp_flag)
    if g2p_intensity > dev_cons["intensity"]:
        string = f"${g2p_flops/dev_cons['peak flops']/time:.0%}$ (flops)"
    else:
        string = f"${g2p_mops/dev_cons['peak band']/time:.0%}$ (mops)"
    string = string.replace("%", "\\%")
    return string
