# Run this by executing ``include("main.jl");`` in the Julia interpreter.

include("fix_julia_load_path.jl");
using SpecialFunctions;
using PyPlot;
using RatApproxAAA;
using Polynomials;

# Function to approximate
function f(t)
  if t == 0
    return 2/sqrt(pi);
  else
    return erf(t)/t;
  end
end

# Interval to approximate on
tv = [0, 6];

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

# Evaluate error of rational approximant (relative to f(0))
tf = range(tv[1], tv[end], length=1000);
err_r = abs.(f.(tf) - r.(tf)) / abs(f(0));
maxerr_r = maximum(err_r);
print("Maximum pointwise error: $(maxerr_r)\n")

# Perform polynomial (least-squares) fit
N = 16;
print("\nStarting polynomial fit ...");
tpv = range(tv[1], tv[end], length=500);
p = fit(tpv, f.(tpv), N-1);
print(" done.\n");
print("N=$(N) coefficients\n");

# Evaluate error of polynomial approximant (relative to f(0))
err_p = abs.(f.(tf) - p.(tf)) / abs(f(0));
maxerr_p = maximum(err_p);
print("Maximum pointwise error: $(maxerr_p)\n")

"""
# Perform Taylor expansion
# This is really bad.
print("\nTaylor polynomial around t=3\n");
N = 3;
coeff = [erf(3)/3, 2/(3*exp(9)*sqrt(pi))-erf(3)/9, (1/27)*(erf(3)-60/(exp(9)*sqrt(pi))),
         (1/81)*(366/(exp(9)*sqrt(pi))), (1/243)*(erf(3)-1581/(exp(9)*sqrt(pi)))];
function polyeval(c, t, shift)
  N = length(c);
  out = 0;
  if N >= 1
    out += c[1];
  end
  ts = t - shift;
  tsk = ts;
  for i in 2:N
    out += c[i]*tsk;
    tsk *= ts;
  end
  return out;
end
p2 = t -> polyeval(coeff[1:N], t, 3);

# Error
err_p2 = abs.(f.(tf) - p2.(tf)) / abs(f(0));
maxerr_p2 = maximum(err_p2);
"""
#print("Maximum pointwise error: $(maxerr_p2)\n")

# Plot error
figure(2);
clf();
semilogy(tf, err_r, label="Error (rational)");
semilogy(tf, err_p, label="Error (polyfit)");
#semilogy(tf, err_p2, label="Error (Taylor t=3)");
grid();
xlabel("\$t\$");
title("\$\\text{erf}(t)/t\$");
tight_layout();
legend();
