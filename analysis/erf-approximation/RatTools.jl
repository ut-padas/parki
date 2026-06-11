module RatTools
    export MonomialRational, monomial_from_barycentric, monomial_from_pole_zero, combo_sum;

    using RatApproxAAA;

    """
    `MonomialRational`

    Representation of a rational function as a ratio of two polynomials in monomial form.
    This is the return type of `monomial_from_barycentric(_)` and `monomial_from_pole_zero(_)`.

    An object `r::MonomialRational` has the following members:
      - `r.ratfun` is the rational function itself. It can be called using
        `r.ratfun(x)` or directly by `r(x)`.
      - `r.num_coeffs` is a vectors containing the coefficients of the polynomial
        in the numerator, in order of ascending exponents (lowest to highest).
      - `r.den_coeffs` is a vectors containing the coefficients of the polynomial
        in the denominator, in order of ascending exponents (lowest to highest).
    """
    struct MonomialRational
        ratfun::Function
        num_coeffs
        den_coeffs
    end

    """
    `(r::MonomialRational)(x) -> y`

    Apply the rational function represented by the `MonomialRational` object `r`
    and return the result `y`. Same as `r.ratfun(x)`.
    """
    function (r::MonomialRational)(x)
        return r.ratfun(x);
    end

    """
    `monomial_from_barycentric(ra::AAA) -> r::MonomialRational`

    Compute the monomial form of a rational function based on the barycentric form.
    """
    function monomial_from_barycentric(ra::AAA)
        M = length(ra.supp_w);
        a = zeros(eltype(ra.supp_w), M);
        b = zeros(eltype(ra.supp_w), M);

        for j in 1:M
            tmp = ra.supp_w[j] * ra.supp_f[j];
            for i in 0:M-1
                com = combo_sum(-ra.supp_z[vcat(1:j-1,j+1:end)], M-i-1);
                a[i+1] += tmp * com;
                b[i+1] += ra.supp_w[j] * com;
            end
        end

        r = z -> horner_reval(z, a, b);
        return MonomialRational(r, a, b);
    end

    """
    `monomial_from_pole_zero(ra::AAA; t0=zero) -> r::MonomialRational`

    Compute the monomial form of a rational function based on the poles and zeros.

    The input `t0` is used to determine the prefactor in the pole-zero form.
    It should not be equal to any of the poles of `ra`. It may be a function
    (taking a data type) or a value.
    """
    function monomial_from_pole_zero(ra::AAA; t0=zero, imag_tol=1e-14)
        M = length(ra.poles) + 1;
        a = zeros(eltype(ra.supp_w), M);
        b = zeros(eltype(ra.supp_w), M);

        # Compute the prefactor c
        if isa(t0, Function)
            t0 = t0(eltype(ra.poles));
        end
        tmp = prod(t0 .- ra.zeros) / prod(t0 .- ra.poles);
        c = ra(t0) / tmp;
        c = real_if_real(c, ra.supp_w, "c", imag_tol=imag_tol);

        for i in 0:M-1
            com = combo_sum(-ra.zeros, M-i-1);
            com = real_if_real(com, ra.supp_w, "sum_zeros", imag_tol=imag_tol);
            a[i+1] = c * com;
            com = combo_sum(-ra.poles, M-i-1);
            com = real_if_real(com, ra.supp_w, "sum_poles", imag_tol=imag_tol);
            b[i+1] = com;
        end

        r = z -> horner_reval(z, a, b);
        return MonomialRational(r, a, b);
    end

    function real_if_real(val, check, name; imag_tol=1e-14)
        if isreal(check)
            if abs(imag(val)) / abs(real(val)) > imag_tol
                @warn "Truncating imaginary part of $name = $val";
            end
            val = real(val);
        end
        return val;
    end

    """
    `horner_reval(z, aj, bj) -> r`

    Evaluate rational function in monomial form using Horner's rule. Computes
    and returns `r`, the value of the rational function at the point `z`.
    The vector `aj` are the monomial coefficients of the polynomial in the
    numerator, and `bj` those of the polynomial in the denominator.

    Note: `horner_reval` assumes that `z` is a single point. Use dot notation
    to evaluate the rational function in multiple points at once.

    See also `monomial_from_barycentric` and `monomial_from_pole_zero`.
    """
    function horner_reval(z, aj, bj)
        num = aj[end];
        den = bj[end];
        for i in length(aj)-1:-1:1
            num = aj[i] + z*num;
            den = bj[i] + z*den;
        end
        return num / den;
    end

    """
    `combo_sum(v, i) -> s`

    Given a sequence `v`, compute the combinatoric sum where the terms are
    all possible way of selecting `i` elements of `v` without repetitions,
    and without taking order into account.

    For example, if `v = [1, 2, 3]` and `i=2`, the sum would be
    `s = 1*2 + 1*3 + 2*3 = 11`. Note that the sum has `binomial(n,i)` terms,
    where `n` is `length(v)`. Every term in the sum has `i` factors.

    Note that `combo_sum(v, 0)` is equal to `1`.
    """
    function combo_sum(v, i)
        if i == 0
            return one(eltype(v));
        end
        s = zero(eltype(v));
        for j in 1:length(v)
            s += v[j] * combo_sum(v[j+1:end], i-1);
        end
        return s;
    end
end

# vim:set softtabstop=4 shiftwidth=4 expandtab:
