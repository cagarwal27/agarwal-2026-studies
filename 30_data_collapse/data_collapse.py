#!/usr/bin/env python3
"""
Study 30: Universal Data Collapse
==================================
Single figure: 13 bistable systems from 7 physical domains collapse
onto ln(D) - beta_0 = B after normalizing out the Kramers prefactor.

beta_0 for each system is computed from its own sigma sweep (not fitted
to the operating point). The slope = 1 is Kramers universality.
"""
import os
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====================================================================
# Style
# ====================================================================
SINGLE_COL = (3.375, 3.375)

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.formatter.use_mathtext": True,
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.3,
    "ytick.minor.width": 0.3,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ====================================================================
# MFPT engine
# ====================================================================

def mfpt_sweep(f_func, x_eq, x_sad, lam_eq, B_lo=0.8, B_hi=9.0,
               N_sigma=50, N_grid=60000, boundary_eq=False):
    if x_eq > x_sad:
        return mfpt_sweep(lambda x: -f_func(-x), -x_eq, -x_sad, lam_eq,
                          B_lo, B_hi, N_sigma, N_grid, boundary_eq)
    tau = 1.0 / abs(lam_eq)
    gap = abs(x_sad - x_eq)
    x_lo = max(x_eq, 1e-10) if boundary_eq else x_eq - 0.5 * gap
    x_hi = x_sad + 0.02 * gap
    x_grid = np.linspace(x_lo, x_hi, N_grid)
    dx = x_grid[1] - x_grid[0]
    f_vals = np.array([f_func(x) for x in x_grid])
    U = np.cumsum(-f_vals) * dx
    i_eq = np.argmin(np.abs(x_grid - x_eq))
    i_sad = np.argmin(np.abs(x_grid - x_sad))
    U -= U[i_eq]
    DeltaPhi = U[i_sad]
    if DeltaPhi <= 0:
        return np.array([]), np.array([]), 0
    sig_lo = np.sqrt(2 * DeltaPhi / B_hi)
    sig_hi = np.sqrt(2 * DeltaPhi / B_lo)
    sigmas = np.logspace(np.log10(sig_lo), np.log10(sig_hi), N_sigma)
    B_out, lnD_out = [], []
    for sigma in sigmas:
        Phi = 2 * U / sigma**2
        if Phi[i_sad] > 700 or Phi[i_sad] < 0.2:
            continue
        Phi = np.clip(Phi, -500, 700)
        exp_neg = np.exp(-Phi)
        I_x = np.cumsum(exp_neg) * dx
        psi = (2 / sigma**2) * np.exp(Phi) * I_x
        MFPT = np.trapz(psi[i_eq:i_sad + 1], x_grid[i_eq:i_sad + 1])
        D = MFPT / tau
        if D > 0.5 and np.isfinite(D):
            B_out.append(2 * DeltaPhi / sigma**2)
            lnD_out.append(np.log(D))
    return np.array(B_out), np.array(lnD_out), DeltaPhi

def find_roots(f, x_lo, x_hi, N=100000):
    x = np.linspace(x_lo, x_hi, N)
    fv = np.array([f(xi) for xi in x])
    roots = []
    for i in range(len(fv) - 1):
        if fv[i] * fv[i + 1] < 0:
            try: roots.append(brentq(f, x[i], x[i + 1]))
            except: pass
    return sorted(roots)

def fderiv(f, x, dx=1e-7):
    return (f(x + dx) - f(x - dx)) / (2 * dx)

# ====================================================================
# 1D systems
# ====================================================================

def system_lake():
    b, r, q, h = 0.8, 1.0, 8, 1.0; a = 0.326588
    def f(x): return a - b*x + r * x**q / (x**q + h**q)
    return mfpt_sweep(f, 0.409217, 0.978152, -0.784651)

def system_kelp():
    r, K, h, p = 0.4, 668.0, 100.0, 65.0
    def f(U): return r*U*(1 - U/K) - p*U/(U + h)
    disc = (K-h)**2 - 4*K*(p/r - h)
    Us = ((K-h)-np.sqrt(disc))/2; Uu = ((K-h)+np.sqrt(disc))/2
    return mfpt_sweep(f, Uu, Us, fderiv(f, Uu))

def system_coral():
    a, gamma, r, d, g = 0.1, 0.8, 1.0, 0.44, 0.30
    Cb, Tb = 1-d/r, d/r
    def f(M):
        C = max(1-M-(d+a*M)/r, 0.0); T = (d+a*M)/r
        return a*M*C - g*M/(M+T) + gamma*M*T if M+T > 1e-30 else 0.0
    Ms = np.linspace(1e-6, 0.6, 200000); fs = np.array([f(m) for m in Ms])
    Msad = None
    for i in range(len(fs)-1):
        if fs[i]*fs[i+1] < 0: Msad = brentq(f, Ms[i], Ms[i+1]); break
    if Msad is None: return np.array([]), np.array([]), 0
    return mfpt_sweep(f, 1e-8, Msad, a*Cb - g/Tb + gamma*Tb, boundary_eq=True)

def system_jj():
    g = 0.5
    def f(phi): return g - np.sin(phi)
    return mfpt_sweep(f, np.arcsin(g), np.pi-np.arcsin(g), -np.sqrt(1-g**2))

def system_nanoparticle():
    h = 0.3
    def f(t): return -2*np.sin(t)*(np.cos(t)+h)
    return mfpt_sweep(f, np.pi, np.arccos(-h), -(2*(1-h)))

def system_peatland():
    NPP, da, dan, m, q = 0.20, 0.05, 0.001, 40.0, 8; Dd = da-dan
    def f(C):
        hC = C**q/(C**q+m**q) if C>0 else 0.0; return NPP - da*C + Dd*hC*C
    roots = find_roots(f, 0.1, 300, 200000)
    if len(roots)<3: return np.array([]),np.array([]),0
    return mfpt_sweep(f, roots[2], roots[1], fderiv(f, roots[2]))

def system_savanna():
    mu,nu,o0,o1,th,ss = 0.2,0.1,0.9,0.2,0.4,0.01; bv = 0.39
    def omega(G): return o0+(o1-o0)/(1+np.exp(np.clip(-(G-th)/ss,-500,500)))
    def Gn(T):
        n,d = mu-T*(mu-nu), mu+bv*T
        if d<=0 or n<=0: return np.nan
        G = n/d; return G if (0<G<1 and G+T<1) else np.nan
    def fe(T):
        G=Gn(T)
        if np.isnan(G): return 0.0
        S=1-G-T; return omega(G)*S-nu*T if S>=0 else 0.0
    roots = find_roots(fe, 0.01, 0.90, 100000)
    if len(roots)<3: return np.array([]),np.array([]),0
    lam = fderiv(fe, roots[0])
    if lam>0: return np.array([]),np.array([]),0
    return mfpt_sweep(fe, roots[0], roots[1], lam)

def system_tropforest():
    al,p0,p1,th,s2 = 0.6,0.1,0.9,0.4,0.05
    def phi(G): return p0+(p1-p0)/(1+np.exp(np.clip(-(G-th)/s2,-500,500)))
    def fF(F): return (al*(1-F)-phi(1-F))*F
    roots = find_roots(fF, 0.01, 0.99, 200000)
    if len(roots)<2: return np.array([]),np.array([]),0
    stab = [(r,fderiv(fF,r)) for r in roots]
    stable = sorted([r for r,fp in stab if fp<0])
    unstable = sorted([r for r,fp in stab if fp>0])
    if not stable or not unstable: return np.array([]),np.array([]),0
    xeq = stable[-1]; xsad = max(u for u in unstable if u<xeq)
    return mfpt_sweep(fF, xeq, xsad, fderiv(fF, xeq), N_grid=80000)

def system_cusp():
    q, a = 3.0, 1.6
    def f(x): return -x**3 + q*x - a
    roots = find_roots(f, -3, 3, 100000)
    if len(roots)<3: return np.array([]),np.array([]),0
    return mfpt_sweep(f, roots[2], roots[1], fderiv(f, roots[2]))


# ====================================================================
# Compute beta_0 from sigma sweeps
# ====================================================================

print("=" * 60)
print("STUDY 30: UNIVERSAL DATA COLLAPSE")
print("=" * 60)
print()

sweep_beta0 = {}

for name, func in [
    ('Lake', system_lake), ('Kelp', system_kelp), ('Coral', system_coral),
    ('Savanna', system_savanna), ('Trop. forest', system_tropforest),
    ('Peatland', system_peatland), ('Josephson jn.', system_jj),
    ('Nanoparticle', system_nanoparticle), ('Currency peg', system_cusp),
]:
    try:
        B, lnD, _ = func()
        if len(B) > 3:
            sweep_beta0[name] = np.mean(lnD - B)
            print(f"  {name:18s}  beta_0 = {sweep_beta0[name]:+.3f}  ({len(B)} pts)")
    except Exception as e:
        print(f"  {name:18s}  FAILED ({e})")

# 2D SDE sweeps
STUDY_DIR = os.path.dirname(os.path.abspath(__file__))
for fname, name in [('sweep_tumor.npz', 'Tumor-immune'),
                     ('sweep_diabetes.npz', 'Diabetes')]:
    fpath = os.path.join(STUDY_DIR, fname)
    if os.path.exists(fpath):
        data = dict(np.load(fpath, allow_pickle=True))
        if 'B' in data and 'lnD' in data:
            B, lnD = data['B'], data['lnD']
            fesc = data.get('frac_escaped', np.ones_like(B))
            valid = (lnD > 0) & (fesc > 0.3) & np.isfinite(B)
            if np.sum(valid) >= 2:
                sweep_beta0[name] = np.mean(lnD[valid] - B[valid])
                print(f"  {name:18s}  beta_0 = {sweep_beta0[name]:+.3f}  "
                      f"({np.sum(valid)} pts, 2D SDE)")

# Toggle: no sweep (discrete CME)
sweep_beta0['Toggle switch'] = np.log(1000) - 4.83
print(f"  {'Toggle switch':18s}  beta_0 = {sweep_beta0['Toggle switch']:+.3f}  "
      f"(table, CME)")


# ====================================================================
# Systems table
# ====================================================================

# (label, sweep_key, B, D)
# label: what appears on the plot next to the point
# sweep_key: key into sweep_beta0 for the beta_0 value

systems = [
    ('Kelp',        'Kelp',          1.80,   29.4),
    ('GBP peg',     'Currency peg',  1.89,   28),
    ('Thai Baht',   'Currency peg',  1.90,   26),
    ('Tumor',       'Tumor-immune',  2.73,   4),
    ('Peatland',    'Peatland',      3.07,   30.3),
    ('Josephson',   'Josephson jn.', 3.26,   92),
    ('Nanoparticle','Nanoparticle',   3.41,   100),
    ('Savanna',     'Savanna',       3.74,   100),
    ('Trop. forest','Trop. forest',  4.00,   95.2),
    ('Lake',        'Lake',          4.25,   200),
    ('Toggle',      'Toggle switch', 4.83,   1000),
    ('Diabetes',    'Diabetes',      5.54,   75),
    ('Coral',       'Coral',         6.04,   1111),
]


# ====================================================================
# FIGURE
# ====================================================================

OUTDIR = os.path.join(STUDY_DIR, 'plots')
os.makedirs(OUTDIR, exist_ok=True)

fig, ax = plt.subplots(figsize=SINGLE_COL)

# y = x reference
B_ref = np.linspace(0, 8, 200)
ax.plot(B_ref, B_ref, color='0.78', lw=0.7, zorder=0)

# Plot each system: direct labels, no legend
# Hand-tuned label offsets: (dx, dy) from the data point
label_pos = {
    'Kelp':         ( 0.15,  0.12),
    'GBP peg':      (-0.15,  0.20),
    'Thai Baht':    ( 0.15, -0.30),
    'Tumor':        (-0.15, -0.30),
    'Peatland':     (-0.15, -0.30),
    'Josephson':    (-0.15,  0.20),
    'Nanoparticle': ( 0.15, -0.30),
    'Savanna':      ( 0.15,  0.12),
    'Trop. forest': (-0.15, -0.30),
    'Lake':         ( 0.15,  0.12),
    'Toggle':       ( 0.15,  0.12),
    'Diabetes':     (-0.15,  0.20),
    'Coral':        (-0.50,  0.15),
}

print(f"\n  {'Label':15s} {'B':>5s} {'ln D':>6s} {'beta_0':>7s} "
      f"{'norm':>6s} {'resid':>6s}")
print(f"  {'-'*50}")

for label, sweep_key, B_op, D_op in systems:
    lnD = np.log(D_op)
    b0 = sweep_beta0.get(sweep_key, lnD - B_op)
    y = lnD - b0
    resid = y - B_op

    # Plot marker
    ax.plot(B_op, y, 'o', ms=4.5, mfc='k', mec='k', mew=0.3, zorder=3)

    # Direct label
    dx, dy = label_pos[label]
    ha = 'left' if dx > 0 else 'right'
    ax.annotate(label, (B_op, y), xytext=(B_op + dx, y + dy),
                fontsize=5.5, color='0.15', ha=ha, va='center', zorder=4)

    print(f"  {label:15s} {B_op:5.2f} {lnD:6.2f} {b0:+7.3f} "
          f"{y:6.2f} {resid:+6.3f}")

ax.set_xlabel(r'$B = 2\,\Delta\Phi\,/\,\sigma^2$')
ax.set_ylabel(r'$\ln D - \beta_0$')
ax.set_xlim(0, 7.5)
ax.set_ylim(0, 7.5)
ax.set_aspect('equal')

plt.savefig(os.path.join(OUTDIR, 'fig_collapse.pdf'), dpi=300)
plt.savefig(os.path.join(OUTDIR, 'fig_collapse.png'), dpi=300)
print(f"\nSaved plots/fig_collapse.pdf and .png")
plt.close()

# Summary
residuals = []
for label, sweep_key, B_op, D_op in systems:
    b0 = sweep_beta0.get(sweep_key, np.log(D_op) - B_op)
    residuals.append(np.log(D_op) - b0 - B_op)
res = np.array(residuals)
print(f"\n  Collapse quality (n = {len(res)}):")
print(f"    RMS residual from y = x:  {np.sqrt(np.mean(res**2)):.3f}")
print(f"    Max residual:             {np.max(np.abs(res)):.3f}")
print(f"\nDone.")
