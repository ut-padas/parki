module RatApproxAAA
    export AAA, aaa, aaamod;

    using LinearAlgebra;
    IntOrNothing = Union{Integer,Nothing};

    """
    `AAA`

    The return type of `aaa(_)`, representing a rational approximant
    and accompanying metadata.

    If `a::AAA` is the object, the following data can be obtained:
      - `a.ratfun` is the rational function itself. It can be called using
        `a.ratfun(x)` or directly by `a(x)`.
      - `a.poles` is a vector containing the poles of `a.ratfun`.
      - `a.residues` is a vector containing the residues of `a.ratfun`.
      - `a.zeros` is a vector containing the zeros of `a.ratfun`.
      - `a.supp_z` is a vector containing the support points `ZJ` of `a.ratfun`.
      - `a.supp_f` is a vector containing the approximation values `FJ = a(ZJ)`
        of `a.ratfun`.
      - `a.supp_w` is a vector containing the weights `WJ` of the barycentric
        representation of `a.ratfun`.
      - `a.errvec` is a vector of errors `||f-r||_Inf` in successive iterative
        steps of AAA. Note that the rational degrees are not `1,2,...,length(a.errvec)`
        but `0,1,...,length(a.errvec)-1`.
      - `a.lawson_weights` is the final weight vector of the Lawson iteration,
        if Lawson iteration was used (otherwise the vector is filled with `NaN`).
    """
    struct AAA
        ratfun::Function # r
        poles # pol
        residues # res
        zeros # zer
        supp_z # zj
        supp_f # fj
        supp_w # wj
        errvec
        lawson_weights # wt
    end

    """
    `(a::AAA)(x) -> y`

    Apply the rational function represented by the `AAA` object `a`
    and return the result `y`. Same as `a.ratfun(x)`.
    """
    function (a::AAA)(x)
        return a.ratfun(x);
    end

    """
    `aaa(F, [Z]; tol=1e-13, degree::IntOrNothing=nothing, mmax::IntOrNothing=nothing,
         dom=[-1,1], cleanup::Bool=true, cleanuptol=nothing, lawson::IntOrNothing=nothing) -> AAA`

    AAA and AAA-Lawson (near-minimax) real or complex rational approximation.
    Computes the AAA rational approximant to data `F` on the set of sample
    points `Z` and returns an `AAA` object. `F` may be given by its values at
    `Z` or as a function handle.

    `aaa` takes the following keyword arguments:
      - `tol` is the relative tolerance (default: `1e-13`)
      - `degree` is the maximal degree `N` (default: `99`). Output rational
        approximant will be at most of rational type (`N`,`N`). Identical
        to setting `mmax=N+1`. By default, this will turn on Lawson iteration
        (see next paragraph).
      - `mmax` is the maximal number of terms in the barycentric representation
        (default: `100`). Identical to setting `degree=mmax-1`. Also turns on
        Lawson iteration.
      - `dom` is the domain (default: `[-1, 1]`). No effect if `Z` is provided.
      - `cleanup=false` turns off automatic removal of numerical Froissart
        doublets (spurious pole-zero pairs); default is `true`.
      - `cleanuptol` sets the cleanup tolerance (default: `tol`). Poles with
        residues less than this number times the geometric mean size of `F`
        times the minimum distance to `Z` are deemed spurious by the cleanup
        procedure. If `tol==0`, then `cleanuptol` defaults to `1e-13`.
      - `lawson` is the number of Lawson iterations (iteratively reweighted
        least-squares steps) to bring the approximation closer to minimax;
        specifying `lawson=0` ensures there is no Lawson iteration. The
        default, `nothing`, selects the number of iterations adaptively
        (but only if `degree` or `mmax` is specified). See next paragraph.

    If `degree` or equivalently `mmax` is specified and `lawson` is not, then
    `aaa` attempts to find a minimax approximant of degree `N` by Lawson
    iteration. This will generally be successful only if the minimax error is
    well above machine precision, and is more reliable for complex problems
    than real ones.  If `degree` and `lawson` are both specified, then exactly
    `lawson` Lawson steps are taken (so `lawson=0` corresponds to AAA
    approximation with no Lawson iteration). If neither `degree` nor `mmax` is
    specified, Lawson iteration is by default turned off.

    Note that the rational approximant may have fewer than `N` poles and zeros.
    This may happen, for example, if `N` is too large, or if `F` is even and
    `N` is odd, or if `F` is odd and `N` is even.

    The input `Z` is optional. If `F` is a vector, omitting `Z` is equivalent
    to `Z = range(dom..., length=length(F))`. If `F` is a function handle and
    `Z` is omitted, `aaa` attempts to resolve `F` on its domain `dom`.

    # Examples

    ```
    julia> using PyPlot;
    julia> r = aaa(exp);
    julia> xx = range(-1, 1, length=50); plot(xx, r.(xx)-exp.(xx));

    julia> r = aaa(exp, degree=4);
    julia> xx = range(-1, 1, length=50); plot(xx, r.(xx)-exp.(xx));

    julia> Z = exp.(2im*pi*range(0, 1, length=500));
    julia> r = aaa(tan, Z); display(hcat(r.poles, r.residues));

    julia> X = range(-1, 1, length=1000); F = tanh.(20*X);
    julia> subplot(1, 2, 1);
    julia> r = aaa(F, X, degree=15, lawson=0); plot(X, F-r.(X));
    julia> r = aaa(F, X, degree=15); plot(X, F-r.(X));

    julia> Z = exp.(1im*pi*range(-1, 1, length=1000)); G = exp.(Z);
    julia> subplot(1, 2, 2);
    julia> r = aaa(G, Z, degree=3, lawson=0); z=G-r.(Z);
    julia> plot(real(z), imag(z)); axis("equal");
    julia> r = aaa(G, Z, degree=3); z=G-r.(Z);
    julia> plot(real(z), imag(z)); axis("equal");
    ```

    # References

    [1] Yuji Nakatsukasa, Olivier Sete, Lloyd N. Trefethen, "The AAA algorithm
    for rational approximation", SIAM J. Sci. Comp. 40 (2018), A1494-A1522.

    [2] Yuji Nakatsukasa and Lloyd N. Trefethen, An algorithm for real and
    complex rational minimax approximation, SIAM J. Sci. Comp. 42 (2020),
    A3157-A3179.

    This Julia function is based on the MATLAB function `aaa` from Chebfun
    (Git commit `8f4b8de14202a1dcf813d961acfc372212aff30f`, 2022-09-04).

    Differences from the Chebfun version of `aaa`:
      - `F` cannot be a chebfun.

    Copyright 2017 by The University of Oxford and The Chebfun Developers.
    See http://www.chebfun.org/ for Chebfun information.
    """
    function aaa(F, Z=nothing; tol=1e-13, degree::IntOrNothing=nothing, mmax::IntOrNothing=nothing,
                 dom=[-1,1], cleanup::Bool=true, cleanuptol=nothing,
                 lawson::IntOrNothing=nothing, alt_cleanup::Bool=false)
        # Parse inputs
        (F, Z, mmax, cleanuptol, needZ, mmax_flag, lawson) =
            parseInputs(F, Z, tol, degree, mmax, dom, cleanuptol, lawson);

        if needZ
            # Z was not provided. Try to resolve F on its domain.
            out = aaa_autoZ(F, dom, tol, mmax, cleanup, alt_cleanup, cleanuptol, mmax_flag, lawson);
            return out;
        end

        # Remove any Inf or NaN function values (avoid SVD failures):
        toKeep = isfinite.(F);
        F = F[toKeep]; Z = Z[toKeep];

        # Remove repeated elements of Z and corresponding elements of F:
        uni = unique(n -> Z[n], 1:length(Z));
        Z = Z[uni]; F = F[uni];

        M = length(Z);

        # Relative tolerance:
        reltol = tol * norm(F, Inf);

        # Left scaling matrix:
        SF = Diagonal(F);

        # Initialization for AAA iteration:
        J = collect(1:M);
        zj = zeros(eltype(Z), 0);
        fj = zeros(eltype(F), 0);
        C = zeros(eltype(F), M, 0);
        errvec = zeros(real(eltype(F)), 0);
        R = mean(F);

        # AAA iteration:
        for m = 1:mmax
            # Select next support point where error is largest:
            jj = argmax(abs.(F .- R));          # Select next support point.
            zj = push!(zj, Z[jj]);              # Update support points.
            fj = push!(fj, F[jj]);              # Update data values.
            deleteat!(J, J .== jj);             # Update index vector.
            C = hcat(C, 1 ./ (Z .- Z[jj]));     # Next column of Cauchy matrix.

            # Compute weights:
            Sf = Diagonal(copy(fj));            # Right scaling matrix.
            A = SF*C - C*Sf;                    # Loewner matrix.
            full = length(J) < size(A,2);       # Mimick MATLAB behaviour.
            V = svd(A[J,:], full=full).V;       # Reduced SVD.
            wj = V[:,m];                        # weight vector = min sing vector

            # Rational approximant on Z:
            N = C*(wj.*fj);                     # Numerator
            D = C*wj;                           # Denominator
            R = copy(F);
            R[J] = N[J]./D[J];

            # Error in the sample points:
            maxerr = norm(F - R, Inf);
            errvec = push!(errvec, maxerr);

            # Check if converged:
            if maxerr <= reltol
                break;
            end
        end
        maxerrAAA = errvec[end];                # error at end of AAA

        # When M == 2, one weight is zero and r is constant.
        # To obtain a good approximation, interpolate in both sample points.
        if M == 2
            zj = copy(Z);
            fj = copy(F);
            wj = [1, -1];       # Only pole at infinity.
            wj = wj/norm(wj);   # Impose norm(w) = 1 for consistency.
            errvec = [errvec[1], 0.0];
            maxerrAAA = 0.0;
        end

        # We now enter Lawson iteration: barycentric IRLS = iteratively reweighted
        # least-squares if `lawson` is specified with `lawson > 0` or `mmax` is
        # specified and `lawson` is not.  In the latter case the number of steps
        # is chosen adaptively.  Note that the Lawson iteration is unlikely to be
        # successful when the errors are close to machine precision.

        wj0 = wj; fj0 = fj;                         # Save parameters in case Lawson fails
        wt = fill(NaN, M); wt_new = ones(eltype(wj), M);
        if isnothing(lawson) || lawson > 0          # Lawson iteration
            maxerrold = maxerrAAA;
            maxerr = maxerrold;
            nj = length(zj);
            A = zeros(eltype(F), length(Z), 0);
            for j = 1:nj                            # Cauchy/Loewner matrix
                A = [A 1 ./ (Z .- zj[j]) F ./ (Z .- zj[j])];
            end
            for j = 1:nj
                i = findall(Z .== zj[j]);           # support pt rows are special
                A[i,:] .= 0;
                A[i,2*j-1] .= 1;
                A[i,2*j] = F[i];
            end
            stepno = 0;
            c = nothing;
            while ( !isnothing(lawson) && stepno < lawson ) ||
                  ( isnothing(lawson) && stepno < 20 ) ||
                  ( isnothing(lawson) && maxerr/maxerrold < .999 && stepno < 1000 )
                stepno += 1;
                wt = wt_new;
                W = Diagonal(sqrt.(wt));
                full = size(A,1) < size(A,2);
                V = svd(W*A, full=full).V;
                c = V[:,end];
                denom = zeros(M); num = zeros(M);
                for j = 1:nj
                    denom = denom + c[2*j] ./ (Z .- zj[j]);
                    num = num - c[2*j-1] ./ (Z .- zj[j]);
                end
                R = num ./ denom;
                for j = 1:nj
                    i = findall(Z .== zj[j]);       # support pt rows are special
                    R[i] .= -c[2*j-1]/c[2*j];
                end
                err = F - R; abserr = abs.(err);
                wt_new = wt.*abserr; wt_new = wt_new/norm(wt_new,Inf);
                maxerrold = maxerr;
                maxerr = maximum(abserr);
            end
            wj = c[2:2:end];
            fj = -c[1:2:end]./wj;
            # If Lawson has not reduced the error, return to pre-Lawson values.
            if maxerr > maxerrAAA && isnothing(lawson)
                wj = wj0; fj = fj0;
            end
        end

        # Remove support points with zero weight:
        I = wj .== 0;
        deleteat!(zj, I);
        deleteat!(wj, I);
        deleteat!(fj, I);

        # Construct function handle:
        r = z -> reval(z, zj, fj, wj);

        # Compute poles, residues and zeros:
        (pol, res, zer) = prz(zj, fj, wj);

        if cleanup && lawson == 0
            if !alt_cleanup                             # Remove Froissart doublets
                (r, pol, res, zer, zj, fj, wj) =
                    cleanup1(r, pol, res, zer, zj, fj, wj, Z, F, cleanuptol);
            else                                        # Alternative cleanup. For the
                                                        # moment this is an undocumented
                                                        # feature, pending further
                                                        # investigation.
                (zj, fj, wj) = cleanup2(zj, fj, wj, Z, F, max(cleanuptol, eps()));
                r = z -> reval(z, zj, fj, wj);
                (pol, res, zer) = prz(zj, fj, wj);
            end
        end

        out = AAA(r, pol, res, zer, zj, fj, wj, errvec, wt);
        return out;
    end # of aaa()

    """Input parsing for AAA."""
    function parseInputs(F, Z, tol, degree, mmax, dom, cleanuptol, lawson)
        # Check if F is empty:
        if !isa(F, Function) && isempty(F)
            error("No function given.");
        end

        # Check if Z is given:
        if !isnothing(Z) && isempty(Z)
            error("If sample set is provided, it must be nonempty.");
        end

        # Check degree/mmax:
        mmax_flag = !isnothing(degree) || !isnothing(mmax); # Degree or mmax manually specified
        mmax_out = 100; # default value
        if !isnothing(degree)
            if !isnothing(mmax) && degree != mmax-1
                error("mmax must equal degree+1.");
            end
            mmax_out = degree+1;
        elseif !isnothing(mmax)
            mmax_out = mmax;
        end

        # Check domain:
        if length(dom) != 2
            error("Given domain must be an interval [a,b], i.e. a 2-element vector.");
        end

        # Cleanup tolerance:
        if isnothing(cleanuptol)
            if tol > 0
                cleanuptol = tol;
            else
                cleanuptol = 1e-13;
            end
        end

        # Deal with Z and F:
        if isnothing(Z) && !isa(F, Function)
            # F is given as data values, pick same number of sample points:
            Z = range(dom[1], dom[2], length=length(F));
        end

        if !isnothing(Z)
            # Z is given:
            needZ = false;

            # Function values:
            if isa(F, Function)
                # Sample F on Z:
                F = F.(Z);
            else
                # Check that the vector has correct length.
                if length(F) != length(Z)
                    error("Inputs F and Z must have the same length.");
                end
            end
        else
            # Z was not given. Set flag that Z needs to be determined.
            needZ = true;
        end

        if !mmax_flag && isnothing(lawson)
            lawson = 0; # turn off Lawson iteration
        end

        return (F, Z, mmax_out, cleanuptol, needZ, mmax_flag, lawson);
    end # of parseInputs()

    """
    Remove spurious pole-zero pairs.

    In June 2022 the residue size test was changed to be relative to
    the distance to the approximation set Z.
    """
    function cleanup1(r, pol, res, zer, z, f, w, Z, F, cleanuptol)
        # Find negligible residues:
        if all(F .== 0)
           geometric_mean_of_absF = 0;
        else
           geometric_mean_of_absF = exp(mean(log.(abs.(F[F .!= 0]))));
        end
        Zdistances = fill(NaN, length(pol));
        for j = 1:length(Zdistances);
           Zdistances[j] = minimum(abs.(pol[j].-Z));
        end
        ii = findall(abs.(res)./Zdistances .< cleanuptol * geometric_mean_of_absF);
        ni = length(ii);
        if ni == 0
            # Nothing to do.
            return (r, pol, res, zer, z, f, w);
        elseif ni == 1
            @info "1 Froissart doublet (spurious pole-zero pair) removed";
        else
            @info "$(ni) Froissart doublets (spurious pole-zero pairs) removed";
        end

        # For each spurious pole find and remove closest support point:
        for j = 1:ni
            azp = abs.(z .- pol[ii[j]]);
            jj = findfirst(azp .== minimum(azp));

            # Remove support point:
            deleteat!(z, jj);
            deleteat!(f, jj);
        end

        # Remove support points z from sample set:
        for jj = 1:length(z)
            I = findall(Z .== z[jj]);
            deleteat!(Z, I);
            deleteat!(F, I);
        end
        m = length(z);
        M = length(Z);

        # Build Loewner matrix:
        SF = Diagonal(F);
        Sf = Diagonal(f);
        C = 1 ./ (Z .- transpose(z));       # Cauchy matrix.
        A = SF*C - C*Sf;                    # Loewner matrix.

        # Solve least-squares problem to obtain weights:
        full = size(A,1) < size(A,2);
        V = svd(A, full=full).V;
        w = V[:,m];

        # Build function handle and compute poles, residues and zeros:
        r = ze -> reval(ze, z, f, w);
        (pol, res, zer) = prz(z, f, w);

        return (r, pol, res, zer, z, f, w);
    end # of cleanup1()

    """
    Alternative cleanup procedure to remove spurious pole-zero pairs.
    This considers pole-zero distances.  Stefano Costa, August 2022.
    """
    function cleanup2(z, f, w, Z, F, cleanuptol)
        (pol, res, zer) = prz(z, f, w);

        niter = 0;
        while true
            niter += 1;
            ii = zeros(Int64, 0);
            for jj = 1:length(pol)
                if !isempty(zer)
                    dz = minimum(abs.(zer .- pol[jj]));
                else
                    dz = 1e100;
                end
                dS = abs.(Z .- pol[jj]);
                ds = minimum(dS);
                if all(F .== 0)
                    q = 4*pi*abs.(F).*dS;
                    Q = mean(q);                # Arithmetic mean
                else
                    Q = 0;
                end
                R = 8*cleanuptol*Q/(4*pi);      # Equivalent residue value

                # Conditions to expunge poles
                # Expunge if either minimum distance is zero
                if ds==0 || dz==0
                    push!(ii, jj);
                # Expunge if Z is a real interval
                elseif isreal(Z) && abs(imag(pol[jj]))<eps() &&
                    real(pol[jj])>=minimum(Z) && real(pol[jj])<=maximum(Z)
                    push!(ii, jj);
                # Expunge if Z is the unit disk
                elseif all(abs.(Z) .== 1) && abs(abs(pol[jj])-1)<eps()
                    push!(ii, jj);
                # Expunge if distance to closest zero is undetectable
                elseif dz/ds<1 && dz<max(cleanuptol^2,eps())
                    push!(ii, jj);
                # Expunge if a nearby zero exists and residue is below the
                # equivalent value R. Two choices for real and complex F
                elseif dz/ds<sqrt(cleanuptol)
                    if all(imag(F) .== 0) && abs(real(res[jj])) < R
                        push!(ii, jj);
                    elseif abs(res[jj]) < R
                        push!(ii, jj);
                    end
                end
            end
            ii = unique(ii);

            ni = length(ii);
            if ni == 0
                # Nothing to do.
                break;
            elseif ni == 1
                @info "1 Froissart doublet (spurious pole-zero pair) removed, niter = $(niter)";
            else
                @info "$(ni) Froissart doublets (spurious pole-zero pairs) removed, niter = $(niter)";
            end

            # For each spurious pole find and remove closest support point:
            for j = 1:ni
                azp = abs.(z .- pol[ii[j]]);
                jj = findfirst(azp .== minimum(azp));

                # Remove support point(s):
                deleteat!(z, jj);
                deleteat!(f, jj);
            end

            # Remove support points z from sample set:
            for jj = 1:length(z)
                deleteat!(F, Z .== z[jj]);
                deleteat!(Z, Z .== z[jj]);
            end
            m = length(z);
            M = length(Z);

            # Build Loewner matrix:
            SF = Diagonal(F);
            Sf = Diagonal(f);
            C = 1 ./ (Z .- transpose(z));       # Cauchy matrix.
            A = SF*C - C*Sf;                    # Loewner matrix.

            # Solve least-squares problem to obtain weights:
            full = size(A,1) < size(A,2);
            V = svd(A, full=full).V;
            w = V[:,m];

            # Compute poles, residues and zeros:
            (pol, res, zer) = prz(z, f, w);
        end # of while loop
        return (z, f, w);
    end # of cleanup2()

    """Automated choice of sample set."""
    function aaa_autoZ(F, dom, tol, mmax, cleanup, alt_cleanup, cleanuptol, mmax_flag, lawson)
        # Flag if function has been resolved:
        isResolved = false;
        out = nothing;

        # Main loop:
        for n = 5:14
            # Sample points:
            # Next lines enables us to do pretty well near poles
            d = dom[2]-dom[1];
            Z = range(dom[1]+1.37e-8*d, dom[2]-3.08e-9*d, length=1 + 2^n);

            out = aaa(F, Z, tol=tol, mmax=mmax, cleanup=cleanup, cleanuptol=cleanuptol,
                      lawson=lawson, alt_cleanup=alt_cleanup);

            # Test if rational approximant is accurate:
            reltol = tol * norm(F.(Z), Inf);

            # On Z(n):
            err1 = norm(F.(Z) - out.(Z), Inf);

            Zrefined = range(dom[1]+1.37e-8*d, dom[2]-3.08e-9*d,
                             length=Integer(round(1.5 * (1 + 2^(n+1)))));
            err2 = norm(F.(Zrefined) - out.(Zrefined), Inf);

            if all([err1,err2] .< reltol)
                # Final check that the function is resolved, inspired by sampleTest().
                # Pseudo random sample points in [-1, 1]:
                xeval = [-0.357998918959666, 0.036785641195074];
                # Scale to dom:
                xeval = (dom[2] - dom[1])/2 * xeval .+ (dom[2] + dom[1])/2;

                if norm(F.(xeval) - out.(xeval), Inf) < reltol
                    isResolved = true;
                    break;
                end
            end
        end

        if !isResolved && !mmax_flag
            @warn("Function not resolved using $(length(Z)) pts.");
        end
        return out;
    end # of aaa_autoZ()

    """
    `reval(z, zj, fj, wj) -> r`

    Evaluate rational function in barycentric form. Computes and returns
    `r`, the value of the barycentric rational function with support points
    `zj`, function values `fj`, and barycentric weights `wj`, evaluated at
    the point `z`.

    Note: `reval` assumes that `z` is a single point. Use dot notation to
    evaluate the rational function in multiple points at once.

    See also `reval_vec`, `aaa` and `prz`.

    Copyright 2018 by The University of Oxford and The Chebfun Developers.
    See http://www.chebfun.org/ for Chebfun information.
    """
    function reval(z, zj, fj, wj)
        if isinf(z)
            # Deal with input Inf: r(Inf) = lim r(zz) = sum(w.*f) / sum(w):
            r = sum(wj.*fj)./sum(wj);
        else
            CC = wj ./ (z .- zj); # Cauchy "matrix" times wj
            r = cdot(CC, fj) / sum(CC); # scalar value
        end

        # Deal with NaN:
        if isnan(r)
            if isnan(z) || !any(z .== zj)
                # r(NaN) = NaN is fine.
                # The second case may happen if r(z) = 0/0
            else
                # Clean up values NaN = Inf/Inf at support points.
                # Find the corresponding node and set entry to correct value:
                r = fj[findfirst(z .== zj)];
            end
        end
        return r;
    end # of reval()

    """
    `reval_vec(zz, zj, fj, wj) -> r`

    Evaluate rational function in barycentric form. Computes and returns
    `r` (vector of floats), the values of the barycentric rational function
    with support points `zj`, function values `fj`, and barycentric
    weights `wj`, evaluated at the points `zz`.

    Note: Unlike `reval`, `reval_vec` assumes that `zz` is a vector
    (or matrix) of points, and behaves as the original Chebfun function
    ``reval``.

    See also `reval`, `aaa` and `prz`.

    Copyright 2018 by The University of Oxford and The Chebfun Developers.
    See http://www.chebfun.org/ for Chebfun information.
    """
    function reval_vec(zz, zj, fj, wj)
        zv = vec(zz);                           # vectorize zz if necessary
        CC = 1 ./ (zv .- transpose(zj));        # Cauchy matrix
        r = (CC*(wj.*fj))./(CC*wj);             # vector of values

        # Deal with input Inf: r(Inf) = lim r(zz) = sum(w.*f) / sum(w):
        r[isinf.(zv)] = sum(wj.*fj)./sum(wj);

        # Deal with NaN:
        ii = findall(isnan.(r));
        for i = ii
            if isnan(zv[i]) || !any(zv[i] .== zj)
                # r(NaN) = NaN is fine.
                # The second case may happen if r(zv[i]) = 0/0 at some point.
            else
                # Clean up values NaN = Inf/Inf at support points.
                # Find the corresponding node and set entry to correct value:
                r[i] = fj[findfirst(zv[i] .== zj)];
            end
        end

        # Reshape to input format:
        r = reshape(r, size(zz));
        return r;
    end # of reval_vec()

    """
    `prz(zj, fj, wj) -> (pol, res, zer)`

    Computes poles, residues, and zeros of a rational function in
    barycentric form. Returns vectors of poles `pol`, residues `res`,
    and zeros `zer` of the rational function defined by support
    points `zj`, function values `fj`, and barycentric weights `wj`.

    See also `aaa` and `reval`.

    Copyright 2018 by The University of Oxford and The Chebfun Developers.
    See http://www.chebfun.org/ for Chebfun information.
    """
    function prz(zj, fj, wj)
        m = length(wj);

        # Compute poles via generalized eigenvalue problem:
        B = Matrix(I, m+1, m+1);
        B[1,1] = 0;
        E = [0 transpose(wj); ones(m,1) diagm(zj)];
        # (Sort eigenvalues by magnitude, this is what MATLAB does.)
        pol = eigen(E, B, sortby=(λ -> -abs(λ))).values;
        # Remove zeros of denominator at infinity:
        pol = pol[isfinite.(pol)];

        # Compute residues via formula for res of quotient of analytic functions:
        N(t) = cdot(1 ./ (t .- zj), fj.*wj);
        Ddiff(t) = -cdot(1 ./ (t .- zj).^2, wj);
        res = N.(pol) ./ Ddiff.(pol);

        # Compute zeros via generalized eigenvalue problem:
        E = [0 transpose(wj.*fj); ones(m,1) diagm(zj)];
        zer = eigen(E, B, sortby=(λ -> -abs(λ))).values;
        # Remove zeros of numerator at infinity:
        zer = zer[isfinite.(zer)];

        return (pol, res, zer);
    end # of prz()

    mean(x) = sum(x)/length(x);
    cdot(x,y) = dot(conj(x),y); # seems faster than sum(x .* y)

## Modified AAA algorithm (started 2022-11-25) ##

    """
    `aaamod(F, [Z]; degree::IntOrNothing=nothing, dom=[-1,1],
            cleanup::Bool=false, cleanuptol=1e-13, lawson::IntOrNothing=0) -> AAA`

    AAA and AAA-Lawson (near-minimax) real or complex rational approximation.
    Computes the AAA rational approximant to data `F` on the set of sample
    points `Z` and returns an `AAA` object. `F` may be given by its values at
    `Z` or as a function handle.

    `aaamod` takes the following keyword arguments:
      - `degree` is the maximal degree `N`. Output rational approximant will be
        at most of rational type (`N`,`N`). The default value is
        `Integer(floor(Npts/2-1))`, where `Npts` is the number of sampling points.
        The degree must be given if `Z` is not given. The maximal number of terms
        in the barycentric representation is always `degree+1`.
      - `dom` is the domain (default: `[-1, 1]`). No effect if `Z` is provided.
      - `cleanup=true` turns on automatic removal of numerical Froissart
        doublets (spurious pole-zero pairs); default is `false`.
      - `cleanuptol` sets the cleanup tolerance (default: `1e-13`). Poles with
        residues less than this number times the geometric mean size of `F`
        times the minimum distance to `Z` are deemed spurious by the cleanup
        procedure.
      - `lawson` is the number of Lawson iterations (iteratively reweighted
        least-squares steps) to bring the approximation closer to minimax;
        the default `lawson=0` ensures there is no Lawson iteration.
        Setting `lawson=nothing` selects the number of iterations adaptively;
        see the next paragraph.

    If `lawson=nothing`, then `aaamod` attempts to find a minimax approximant
    of degree `N` by Lawson iteration. This will generally be successful only
    if the minimax error is well above machine precision, and is more reliable
    for complex problems than real ones.

    Note that the rational approximant may have fewer than `N` poles and zeros.
    This may happen, for example, if `N` is too large, or if `F` is even and
    `N` is odd, or if `F` is odd and `N` is even.

    The input `Z` is optional. If `F` is a vector, omitting `Z` is equivalent
    to `Z = range(dom..., length=length(F))`. If `F` is a function handle and
    `Z` is omitted, then `degree` must be given and `aaamod` will select
    `2*(degree+1)` uniform points.

    # References

    This function is based on the `aaa` function; see that function.

    Copyright 2017 by The University of Oxford and The Chebfun Developers.
    See http://www.chebfun.org/ for Chebfun information.
    """
    function aaamod(F, Z=nothing; degree::IntOrNothing=nothing, dom=[-1,1],
                    cleanup::Bool=false, cleanuptol=1e-13, lawson::IntOrNothing=0,
                    alt_cleanup::Bool=false, support_points=M->collect(1:2:M))
        # Parse inputs
        (F, Z, m) = parseInputsMod(F, Z, degree, dom);

        # Remove any Inf or NaN function values (avoid SVD failures):
        toKeep = isfinite.(F);
        F = F[toKeep]; Z = Z[toKeep];

        # Remove repeated elements of Z and corresponding elements of F:
        uni = unique(n -> Z[n], 1:length(Z));
        Z = Z[uni]; F = F[uni];

        M = length(Z);

        # Left scaling matrix:
        SF = Diagonal(F);

        # Select support points
        J = collect(1:M);
        jj = support_points(M); m = length(jj);
        zj = Z[jj];
        fj = F[jj];
        # Cauchy matrix
        C = zeros(eltype(F), M, 0);
        for j=jj
            deleteat!(J, J .== j);
            C = hcat(C, 1 ./ (Z .- Z[j]));
        end
        errvec = ones(real(eltype(F)), m);

        # AAA algorithm

        # Compute weights:
        Sf = Diagonal(copy(fj));            # Right scaling matrix.
        A = SF*C - C*Sf;                    # Loewner matrix.
        full = length(J) < size(A,2);       # Mimick MATLAB behaviour.
        V = svd(A[J,:], full=full).V;       # Reduced SVD.
        wj = V[:,m];                        # weight vector = min sing vector

        # Rational approximant on Z:
        N = C*(wj.*fj);                     # Numerator
        D = C*wj;                           # Denominator
        R = copy(F);
        R[J] = N[J]./D[J];

        # Error in the sample points:
        maxerr = norm(F - R, Inf);
        errvec = maxerr*errvec;
        maxerrAAA = errvec[end];                # error at end of AAA

        # When M == 2, one weight is zero and r is constant.
        # To obtain a good approximation, interpolate in both sample points.
        if M == 2
            zj = copy(Z);
            fj = copy(F);
            wj = [1, -1];       # Only pole at infinity.
            wj = wj/norm(wj);   # Impose norm(w) = 1 for consistency.
            errvec = [errvec[1], 0.0];
            maxerrAAA = 0.0;
        end

        # We now enter Lawson iteration: barycentric IRLS = iteratively reweighted
        # least-squares if `lawson` is specified with `lawson > 0` or `mmax` is
        # specified and `lawson` is not.  In the latter case the number of steps
        # is chosen adaptively.  Note that the Lawson iteration is unlikely to be
        # successful when the errors are close to machine precision.

        wj0 = wj; fj0 = fj;                         # Save parameters in case Lawson fails
        wt = fill(NaN, M); wt_new = ones(eltype(wj), M);
        if isnothing(lawson) || lawson > 0          # Lawson iteration
            maxerrold = maxerrAAA;
            maxerr = maxerrold;
            nj = length(zj);
            A = zeros(eltype(F), length(Z), 0);
            for j = 1:nj                            # Cauchy/Loewner matrix
                A = [A 1 ./ (Z .- zj[j]) F ./ (Z .- zj[j])];
            end
            for j = 1:nj
                i = findall(Z .== zj[j]);           # support pt rows are special
                A[i,:] .= 0;
                A[i,2*j-1] .= 1;
                A[i,2*j] = F[i];
            end
            stepno = 0;
            c = nothing;
            while ( !isnothing(lawson) && stepno < lawson ) ||
                  ( isnothing(lawson) && stepno < 20 ) ||
                  ( isnothing(lawson) && maxerr/maxerrold < .999 && stepno < 1000 )
                stepno += 1;
                wt = wt_new;
                W = Diagonal(sqrt.(wt));
                full = size(A,1) < size(A,2);
                V = svd(W*A, full=full).V;
                c = V[:,end];
                denom = zeros(M); num = zeros(M);
                for j = 1:nj
                    denom = denom + c[2*j] ./ (Z .- zj[j]);
                    num = num - c[2*j-1] ./ (Z .- zj[j]);
                end
                R = num ./ denom;
                for j = 1:nj
                    i = findall(Z .== zj[j]);       # support pt rows are special
                    R[i] .= -c[2*j-1]/c[2*j];
                end
                err = F - R; abserr = abs.(err);
                wt_new = wt.*abserr; wt_new = wt_new/norm(wt_new,Inf);
                maxerrold = maxerr;
                maxerr = maximum(abserr);
            end
            wj = c[2:2:end];
            fj = -c[1:2:end]./wj;
            # If Lawson has not reduced the error, return to pre-Lawson values.
            if maxerr > maxerrAAA && isnothing(lawson)
                wj = wj0; fj = fj0;
            end
        end

        # Remove support points with zero weight:
        I = wj .== 0;
        deleteat!(zj, I);
        deleteat!(wj, I);
        deleteat!(fj, I);

        # Construct function handle:
        r = z -> reval(z, zj, fj, wj);

        # Compute poles, residues and zeros:
        (pol, res, zer) = prz(zj, fj, wj);

        if cleanup && lawson == 0
            if !alt_cleanup                             # Remove Froissart doublets
                (r, pol, res, zer, zj, fj, wj) =
                    cleanup1(r, pol, res, zer, zj, fj, wj, Z, F, cleanuptol);
            else                                        # Alternative cleanup. For the
                                                        # moment this is an undocumented
                                                        # feature, pending further
                                                        # investigation.
                (zj, fj, wj) = cleanup2(zj, fj, wj, Z, F, max(cleanuptol, eps()));
                r = z -> reval(z, zj, fj, wj);
                (pol, res, zer) = prz(zj, fj, wj);
            end
        end

        out = AAA(r, pol, res, zer, zj, fj, wj, errvec, wt);
        return out;
    end # of aaa()

    """Input parsing for `aaamod`."""
    function parseInputsMod(F, Z, degree, dom)
        # Check if F is empty:
        if !isa(F, Function) && isempty(F)
            error("No function given.");
        end

        # Check if Z is given:
        if !isnothing(Z) && isempty(Z)
            error("If sample set is provided, it must be nonempty.");
        end

        # Check degree:
        if isnothing(degree)
            if isnothing(Z)
                error("If sample set is omitted, then degree must be given.");
            end
            Npts = length(Z);
            degree = Integer(floor(Npts/2-1));
        end
        m = degree+1;

        # Check domain:
        if length(dom) != 2
            error("Given domain must be an interval [a,b], i.e. a 2-element vector.");
        end

        # Deal with Z and F:
        if isnothing(Z) && !isa(F, Function)
            # F is given as data values, pick same number of sample points:
            Z = range(dom[1], dom[2], length=length(F));
        end

        if !isnothing(Z)
            # Z is given.

            # Function values:
            if isa(F, Function)
                # Sample F on Z:
                F = F.(Z);
            else
                # Check that the vector has correct length.
                if length(F) != length(Z)
                    error("Inputs F and Z must have the same length.");
                end
            end
        else
            # Z was not given. Select based on the degree.
            Npts = 2*(degree+1);
            Z = range(dom[1], dom[2], length=Npts);
        end

        return (F, Z, m);
    end # of parseInputsMod()

end

# vim:set softtabstop=4 shiftwidth=4 expandtab:
