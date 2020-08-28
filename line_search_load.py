import matplotlib.pyplot as plt
import numpy as np
from sirius.coefficient_array import PwCoeffs, diag, inner
from sirius.ot import ApplyHamiltonian, Energy
from sirius.edft.neugebaur import grad_eta, kb, _solve
from sirius.edft.preconditioner import make_kinetic_precond2
from sirius.edft.ortho import loewdin
from sirius import load_state
from sirius.edft import FreeEnergy
from sirius import DFT_ground_state_find

np.set_printoptions(precision=3, linewidth=120)
input_file_name = 'sirius.json'
lname = 'nlcg_dump0108*.h5'
# input variables
# TODO: read from nlcg.yaml
T = 1500
kappa = 0.0
load_gradients = False
smearing_type = 'gaussian-spline'


def make_smearing(label, T, ctx, kset):
    """
    smearing factory
    """
    from sirius.edft import make_fermi_dirac_smearing, make_gaussian_spline_smearing
    if label == 'fermi-dirac':
        return make_fermi_dirac_smearing(T, ctx, kset)
    elif label == 'gaussian-spline':
        return make_gaussian_spline_smearing(T, ctx, kset)
    else:
        raise NotImplementedError('invalid smearing: ', label)


def make_pwcoeffs(coefficient_array):
    out = PwCoeffs(dtype=np.complex, ctype=np.matrix)
    out._data = coefficient_array._data
    return out


np.set_printoptions(precision=4, linewidth=120)
plt.interactive(True)

res = DFT_ground_state_find(1, config=input_file_name)
ctx = res['ctx']
m = ctx.max_occupancy()
# not yet implemented for single spin channel system
assert m == 1
kset = res['kpointset']
potential = res['potential']
density = res['density']
H = ApplyHamiltonian(potential, kset)
E = Energy(kset, potential, density, H)

kT = kb*T
kw = kset.w
ne = kset.ctx().unit_cell().num_valence_electrons

smearing = make_smearing(smearing_type, T, kset.ctx(), kset)
M = FreeEnergy(E, T=T, smearing=smearing)
X = make_pwcoeffs(load_state(lname, kset, 'X', np.complex))
f = load_state(lname, kset, 'fn', np.double)
f = f.asarray().flatten()
eta = make_pwcoeffs(load_state(lname, kset, 'eta', np.complex))
if load_gradients:
    G_eta = make_pwcoeffs(load_state(lname, kset, 'G_eta', np.complex))
    G_X = make_pwcoeffs(load_state(lname, kset, 'G_X', np.complex))
    g_X = make_pwcoeffs(load_state(lname, kset, 'g_X', np.complex))
    g_eta = make_pwcoeffs(load_state(lname, kset, 'g_eta', np.complex))

F0, Hx = M(X, f)

if not load_gradients:
    K = make_kinetic_precond2(kset)
    HX = Hx * kset.w
    Hij = X.H @ HX
    XhKHXF = X.H @ (K @ HX)
    XhKX = X.H @ (K @ X)
    LL = _solve(XhKX, XhKHXF)
    delta_X = -K * (HX - X @ LL) / kw
    g_X = (HX*f - X@LL)
    G_X = delta_X
    delta_eta = kappa * (Hij - kw*eta) / kw
    G_eta = delta_eta
    g_eta = grad_eta(Hij, diag(eta), f, T, kw, mo=m)

    # Teter kinetic preconditioner

print('loaded energy: %.9f' % F0)

slope = np.real(2*inner(g_X, G_X) + inner(g_eta, kappa*G_eta))

print('slope:', slope)

dts = np.linspace(0, 0.1, 10)
Fs = []
dt_slope = 1e-7
dts = np.concatenate([np.array([0, dt_slope]), dts[1:]])
for dt in dts:
    X_new = X + dt*G_X
    eta_new = eta + dt*kappa*G_eta
    w, Ul = eta_new.eigh()
    Q_new = loewdin(X_new)
    fn_new, _ = smearing.fn(w)
    print('entropy: {0}'.format(smearing.entropy(fn_new)))

    Floc, _ = M(Q_new @ Ul, fn_new)
    Fs.append(Floc)
plt.plot(dts, Fs, '-x')
plt.grid(True)
fit = np.polyfit(dts, Fs, deg=2)

fid = 5
xi_trial = dts[fid]
F1 = Fs[fid]
print('compute slope by fd')
fd_slope = (Fs[1]-Fs[0])/dt_slope
b = slope
c = Fs[0]
print('fd slope: %.10f' % fd_slope)
a = (F1 - b*xi_trial - c) / xi_trial**2

fitl = np.array([a, b, c])
ts = np.linspace(min(dts), max(dts), 1000)

plt.plot(ts, np.polyval(fit, ts), '--', label='quadratic fit')
plt.plot(ts, np.polyval(fitl, ts), label='quad. approx')
plt.plot(xi_trial, F1, 's', color='red')
plt.plot(dts, Fs, '-x', color='black', label='line eval')
plt.plot(0, Fs[0], 's', color='green')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('plot.pdf')
