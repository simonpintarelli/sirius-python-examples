import matplotlib.pyplot as plt
import numpy as np
from sirius import DFT_ground_state_find
from sirius.coefficient_array import diag, inner, l2norm
from sirius.ot import ApplyHamiltonian, Energy
from sirius import Logger
from sirius.edft.neugebaur import _solve
from sirius.edft.neugebaur import loewdin, grad_eta
from sirius.edft import (kb,
                         make_fermi_dirac_smearing,
                         make_gaussian_spline_smearing)
# Teter preconditioner
from sirius.edft.preconditioner import make_kinetic_precond2

from sirius.edft.free_energy import FreeEnergy
from copy import deepcopy

np.set_printoptions(precision=4, linewidth=120)
plt.interactive(True)

T = 300
kT = kb*T
kappa = 0.3

logger = Logger()

res = DFT_ground_state_find(1, config='sirius.json')
ctx = res['ctx']
kset = res['kpointset']
potential = res['potential']
density = res['density']
H = ApplyHamiltonian(potential, kset)
E = Energy(kset, potential, density, H)
X = kset.C
fn = kset.fn
fo = deepcopy(fn)

smearing = make_fermi_dirac_smearing(T, kset.ctx(), kset)
# smearing = make_gaussian_spline_smearing(T, kset.ctx(), kset)

M = FreeEnergy(E=E, T=T, smearing=smearing)
kw = kset.w
F0, _ = M(X, fn)
print('initial energy:', F0)
ne = ctx.unit_cell().num_valence_electrons
eta0 = kset.e
# eta0 = diag(kset.e)
w, U = diag(eta0).eigh()
ek = w
# rotate (e.g. in this case permute X)
X = X@U
eta = diag(w)
fn, mu = smearing.fn(ek)

# evaluate total energy, gradient, overlap
F0i, Hx = M(X, fn)

logger('Free energy after setting F_n:', F0i)

# compute gradients
HX = Hx * kw
Hij = X.H @ HX

g_eta = grad_eta(Hij, ek, fn, T, kw, mo=kset.ctx().max_occupancy())

K = make_kinetic_precond2(kset)
XhKHX = X.H @ (K @ HX)
XhKSX = X.H @ (K @ X)
# Lagrange multipliers
LL = _solve(XhKSX, XhKHX)

g_X = HX*fn - X@LL
delta_X = -K*(HX - X@LL) / kw
delta_eta = kappa * (Hij - kw*diag(ek)) / kw

G_X = delta_X
G_eta = delta_eta
dts = np.linspace(0, 3, 15)
fx = 0

slope = np.real(2*inner(g_X, fx*G_X) + inner(g_eta, G_eta))
Fs = []
Hs = []  # gradients along lines
dt_slope = 1e-3
dts = np.concatenate([np.array([0, dt_slope]), dts[1:]])
for dt in dts:
    X_new = X + dt * fx * G_X
    eta_new = eta + dt * G_eta
    w, Ul = eta_new.eigh()
    print('w', w)
    Q_new = loewdin(X_new) @ Ul
    # update occupation numbers
    fn_new, mu = smearing.fn(w)
    logger('orth err: %.3g, mu: %.5g' % (l2norm(X_new@Ul-Q_new), mu))
    Floc, Hloc = M(Q_new, fn_new)
    Fs.append(Floc)
    Hs.append(Hloc)


plt.plot(dts, Fs, '-x')
plt.grid(True)
fit = np.polyfit(dts, Fs, deg=2)
ts = np.linspace(min(dts), max(dts), 300)
# plt.plot(ts, np.polyval(fit, ts), label='quadratic (l2)')

logger('slope (fit): %.6g' % fit[1])
logger('slope      : %.6g' % slope)
slope_fd = (Fs[1]-Fs[0])/dt_slope
logger('slope (fd) : %.6g' % slope_fd)

fid = 10  # trial point index (for ts)
F1 = Fs[fid]
xi_trial = dts[fid]
b = slope
c = Fs[0]
a = (F1 - b*xi_trial - c) / xi_trial**2
fitl = np.array([a, b, c])
ts = np.linspace(min(dts), max(dts), 1000)

plt.plot(ts, np.polyval(fitl, ts), label='quad. approx')
plt.plot(dts, Fs, '--kx', label='evaluated')

plt.legend(loc='best')
plt.show()
