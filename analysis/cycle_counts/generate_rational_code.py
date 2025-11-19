"""
Generate code for the Horner form of rational function evaluation
for erf(x)/x for ``main.cu``.

This code reads ``rational_coefficients_xmax{}.txt`` and writes
code to ``rational_code_xmax{}.cu``, where ``{}`` is a positive
real number.
"""


def main():
    xmax = 6

    # Read input
    with open(f"rational_coefficients_xmax{xmax}.txt") as f:
        coefficients_str = f.read()

    # Create output
    output = ""
    for line in coefficients_str.splitlines():
        if not line:
            continue
        elif line.startswith("M="):
            M = int(line[len("M=") :])
            a = []
            b = []
        elif line in {"a=", "b="} or "Vector" in line:
            continue
        else:
            if len(a) < M:
                a.append(line.strip())
            elif len(b) < M:
                b.append(line.strip())
            if len(b) == M:
                output += write_output(xmax, M, a, b)

    # Write output
    fname_out = f"rational_code_xmax{xmax}.cu"
    with open(fname_out, "w") as f:
        f.write(output)

    print(f"Wrote output to {fname_out!r}")
    return


def write_output(xmax, M, a, b):
    s = f"""\
__device__ double rational_erfoverx_M{M}(double x)
{{
    /* Horner's rule uses M-1 FMAs each for the numerator and denominator,
     * followed by a single division.
     */
    // This function uses M={M} and values for x in [0, {xmax}].
    double numerator = {a[-1]};
    double denominator = {b[-1]};
"""
    for i in range(M - 2, -1, -1):
        s += f"""\
    numerator = {a[i]} + x*numerator;
    denominator = {b[i]} + x*denominator;
"""
    s += f"""\
    return numerator / denominator;
}}

"""
    return s


if __name__ == "__main__":
    main()
