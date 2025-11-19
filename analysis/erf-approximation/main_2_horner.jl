# Run this by executing ``include("main_2_horner.jl");`` in the Julia interpreter.

include("fix_julia_load_path.jl");
using SpecialFunctions;
using PyPlot;
using RatApproxAAA;
using RatTools;

# Function to approximate
function f(t)
  if t == 0
    return 2/sqrt(pi);
  else
    return erf(t)/t;
  end
end

# Interval to approximate on
t_max = 6;
tv = [0, t_max];

# Plot function on interval
tf = range(tv[1], tv[end], length=1000);
figure(1);
clf();
plot(tf, f.(tf), label="Function");
grid();
xlabel("\$t\$");
title("\$\\text{erf}(t)/t\$");
tight_layout();
legend();

# Perform rational approximation of f using AAA
M = 6;
print("Starting rational approximation ...");
r = aaa(f, mmax=M, dom=tv);
print(" done.\n");
print("M=$(length(r.supp_z)) support points\n");

# Compute monomial form using two different methods
r1 = monomial_from_barycentric(r);
r2 = monomial_from_pole_zero(r);

# Evaluate error of barycentric rational approximant (relative to f(0))
tf = range(tv[1], tv[end], length=1000);
err_r = abs.(f.(tf) - r.(tf)) / abs(f(0));
maxerr_r = maximum(err_r);
print("Maximum pointwise error (barycentric): $(maxerr_r)\n")

# Evaluate error of monomial rational approximants
err_r1 = abs.(f.(tf) - r1.(tf)) / abs(f(0));
err_r2 = abs.(f.(tf) - r2.(tf)) / abs(f(0));
maxerr_r1 = maximum(err_r1);
maxerr_r2 = maximum(err_r2);
print("Maximum pointwise error (monomial 1): $(maxerr_r1)\n")
print("Maximum pointwise error (monomial 2): $(maxerr_r2)\n")

# Plot error
figure(2);
clf();
semilogy(tf, err_r, label="Error (barycentric)");
semilogy(tf, err_r1, label="Error (monomial 1)");
semilogy(tf, err_r2, label="Error (monomial 2)");
grid();
xlabel("\$t\$");
title("\$\\text{erf}(t)/t\$");
tight_layout();
legend();

# Conclusion: `monomial_from_pole_zero` (monomial 2) is better.
# It more closely follows the error of the barycentric form for
# large M (checked up to M=14), while `monomial_from_barycentric`
# (monomial 1) has a larger error. Using monomial 2 should be
# fine for erf(t)/t; it is as good as the barycentric form.
