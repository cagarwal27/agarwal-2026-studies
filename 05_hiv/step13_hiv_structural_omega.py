"""
Step 13: HIV Structural Omega — MAP via Freidlin-Wentzell Theory
================================================================
Computes the Minimum Action Path (MAP) from PTC to saddle in the full
5D Conway-Perelson state space using the geometric Freidlin-Wentzell action,
then derives Omega*_structural from the Kramers bridge identity.

Kramers formula:  D = exp(2*Omega*s*) / (pf_rate * tau_relax)
where s* = quasipotential barrier per unit Omega (from 5D MAP),
pf_rate = |lambda_u|/(2pi) * sqrt(prod|eig_ptc|/prod'|eig_sad|),
tau_relax = 1/|slowest eigenvalue at PTC|.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.linalg import solve_continuous_lyapunov
import warnings, time
warnings.filterwarnings('ignore')

# ============================================================
# Model parameters
# ============================================================
P = dict(lam=10000.0, dT=0.01, beta=1.5e-8, delta=1.0, p=2000.0,
         c=23.0, a=0.001, dL=0.004, rho=0.0045, alpha_L=1e-6,
         lam_E=1.0, bE=1.0, KB=0.1, dE=2.0, KD=5.0, mu=2.0,
         m=0.42, eps=0.0)
D_PRODUCT = 13.0
OM_SDE = 2296.0
SDE = {500:0.35, 1000:1.24, 2000:8.22, 2500:17.24,
       3000:21.67, 3500:24.85, 4000:23.82, 4500:28.26}
VN = ['T','L','I','V','E']

def rhs(y):
    T,L,I,V,E = y
    inf = (1-P['eps'])*P['beta']*V*T
    return np.array([
        P['lam']-P['dT']*T-inf,
        P['alpha_L']*inf+(P['rho']-P['a']-P['dL'])*L,
        (1-P['alpha_L'])*inf-P['delta']*I+P['a']*L-P['m']*E*I,
        P['p']*I-P['c']*V,
        P['lam_E']+P['bE']*I/(P['KB']+I)*E-P['dE']*I/(P['KD']+I)*E-P['mu']*E])

def jac(y):
    T,L,I,V,E = y; b = (1-P['eps'])*P['beta']
    J = np.zeros((5,5))
    J[0,0]=-P['dT']-b*V;           J[0,3]=-b*T
    J[1,0]=P['alpha_L']*b*V;       J[1,1]=P['rho']-P['a']-P['dL']
    J[1,3]=P['alpha_L']*b*T
    J[2,0]=(1-P['alpha_L'])*b*V;   J[2,1]=P['a']
    J[2,2]=-P['delta']-P['m']*E;   J[2,3]=(1-P['alpha_L'])*b*T
    J[2,4]=-P['m']*I
    J[3,2]=P['p'];                  J[3,3]=-P['c']
    J[4,2]=(P['bE']*P['KB']/(P['KB']+I)**2-P['dE']*P['KD']/(P['KD']+I)**2)*E
    J[4,4]=P['bE']*I/(P['KB']+I)-P['dE']*I/(P['KD']+I)-P['mu']
    return J

def D_raw(y):
    """CLE diffusion matrix (Omega-independent). D_CLE = D_raw/Omega."""
    T,L,I,V,E = np.maximum(y, 1e-30)  # clamp to non-negative
    aL = P['alpha_L']
    rs = [(P['lam'],[1,0,0,0,0]), (P['dT']*T,[-1,0,0,0,0]),
          ((1-P['eps'])*P['beta']*V*T,[-1,aL,1-aL,0,0]),
          (P['rho']*L,[0,1,0,0,0]), (P['dL']*L,[0,-1,0,0,0]),
          (P['a']*L,[0,-1,1,0,0]), (P['delta']*I,[0,0,-1,0,0]),
          (P['m']*E*I,[0,0,-1,0,0]), (P['p']*I,[0,0,0,1,0]),
          (P['c']*V,[0,0,0,-1,0]), (P['lam_E'],[0,0,0,0,1]),
          (P['bE']*I/(P['KB']+I)*E,[0,0,0,0,1]),
          (P['dE']*I/(P['KD']+I)*E,[0,0,0,0,-1]),
          (P['mu']*E,[0,0,0,0,-1])]
    D = np.zeros((5,5))
    for r,s in rs:
        sv = np.array(s); D += max(r,0)*np.outer(sv,sv)
    return D

# ============================================================
# Geometric action
# ============================================================

def seg_action(ya, yb):
    """Geometric FW action for one segment: sqrt(a*b) - c."""
    mid = 0.5*(ya+yb)
    dy = yb - ya
    f = rhs(mid)
    D = D_raw(mid) + np.eye(5)*1e-20
    Q = np.linalg.inv(D)
    a = dy@Q@dy; b = f@Q@f; c = dy@Q@f
    if a > 0 and b > 0:
        return max(np.sqrt(a*b) - c, 0.0)
    return 0.0

def full_action(path):
    return sum(seg_action(path[i], path[i+1]) for i in range(len(path)-1))

# ============================================================
# MAP via coordinate descent with non-negativity constraints
# ============================================================

def find_MAP(yp, ys, N=12, n_iter=500):
    """Coordinate descent with local action evaluation and physical bounds."""
    sc = ys - yp
    sc[np.abs(sc) < 1e-30] = 1.0
    Ni = N - 2

    al = np.linspace(0, 1, N)
    z = np.array([al[i+1]*np.ones(5) for i in range(Ni)])

    # Bounds: all physical coords must be non-negative
    # y_phys = yp + z * sc >= 0  =>  z >= -yp/sc (for sc > 0)
    z_min = np.zeros(5)
    for k in range(5):
        if sc[k] > 0:
            z_min[k] = -yp[k] / sc[k]
        else:
            z_min[k] = -100  # very loose for negative sc

    def pt(i):
        if i == 0: return yp.copy()
        if i == N-1: return ys.copy()
        return yp + z[i-1]*sc

    S = sum(seg_action(pt(i), pt(i+1)) for i in range(N-1))
    S_best = S; z_best = z.copy()
    print(f"  N={N}, init S = {S:.8e}", flush=True)

    deltas = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
              -0.005, -0.01, -0.02, -0.05, -0.1, -0.2, -0.5, -1.0, -2.0]

    for it in range(n_iter):
        improved = False
        for i in range(Ni):
            pi = i + 1
            for k in range(5):
                z_old = z[i,k]
                S_loc_old = seg_action(pt(pi-1), pt(pi)) + seg_action(pt(pi), pt(pi+1))
                best_d = 0; best_imp = 0

                for d in deltas:
                    z_new = z_old + d
                    # Enforce non-negativity
                    if z_new < z_min[k]:
                        continue
                    z[i,k] = z_new
                    S_loc_new = seg_action(pt(pi-1), pt(pi)) + seg_action(pt(pi), pt(pi+1))
                    imp = S_loc_old - S_loc_new
                    if imp > best_imp:
                        best_imp = imp; best_d = d

                if best_d != 0:
                    z[i,k] = z_old + best_d
                    S -= best_imp; improved = True
                else:
                    z[i,k] = z_old

        if S < S_best:
            S_best = S; z_best = z.copy()

        if it % 50 == 0 or it == n_iter-1:
            S = sum(seg_action(pt(i), pt(i+1)) for i in range(N-1))
            if S < S_best: S_best = S; z_best = z.copy()
            print(f"  iter {it}: S = {S_best:.8e}", flush=True)

        if not improved:
            # Fine deltas
            any_fine = False
            for i in range(Ni):
                pi = i+1
                for k in range(5):
                    z_old = z[i,k]
                    S_loc = seg_action(pt(pi-1),pt(pi)) + seg_action(pt(pi),pt(pi+1))
                    for d in [0.001,-0.001,0.002,-0.002,0.003,-0.003]:
                        z_new = z_old + d
                        if z_new < z_min[k]: continue
                        z[i,k] = z_new
                        S_new = seg_action(pt(pi-1),pt(pi)) + seg_action(pt(pi),pt(pi+1))
                        if S_new < S_loc - 1e-15:
                            S_loc = S_new; z_old = z[i,k]; any_fine = True
                        else:
                            z[i,k] = z_old
            S = sum(seg_action(pt(j), pt(j+1)) for j in range(N-1))
            if S < S_best: S_best = S; z_best = z.copy()
            if not any_fine:
                print(f"  Converged at iter {it}, S = {S_best:.8e}", flush=True)
                break

    z[:] = z_best
    path = np.array([pt(i) for i in range(N)])
    return path, S_best

# ============================================================
# Prefactors
# ============================================================

def compute_prefactors(yp, ys):
    Jp = jac(yp); Js = jac(ys)
    ep = np.linalg.eigvals(Jp); es = np.linalg.eigvals(Js)
    lu = np.max(np.real(es))
    pp = np.prod(np.abs(ep))
    iu = np.argmax(np.real(es))
    mk = np.ones(5, dtype=bool); mk[iu] = False
    ps = np.prod(np.abs(es[mk]))
    pf_det = abs(lu)/(2*np.pi)*np.sqrt(pp/ps)

    # Quasi-potential prefactor
    Dp = D_raw(yp)
    Sig = solve_continuous_lyapunov(Jp, -Dp)
    eSig = np.linalg.eigvalsh(Sig)
    pf_qp = pf_det
    if np.all(eSig > 0):
        detQp = np.linalg.det(np.linalg.inv(Sig))
        Ds = D_raw(ys)
        evs, evecs = np.linalg.eig(Js)
        vu = np.real(evecs[:,iu]); vu /= np.linalg.norm(vu)
        evl, evecl = np.linalg.eig(Js.T)
        ilu = np.argmax(np.real(evl))
        wu = np.real(evecl[:,ilu]); wu /= np.dot(wu,vu)
        Jreg = Js - 2*lu*np.outer(vu,wu)
        SigS = solve_continuous_lyapunov(Jreg, -Ds)
        eSS = np.linalg.eigvalsh(SigS)
        if np.all(eSS > 0):
            eQs = np.sort(np.linalg.eigvalsh(np.linalg.inv(SigS)))
            dQsp = np.prod(eQs[1:])
            pf_qp = abs(lu)/(2*np.pi)*np.sqrt(abs(detQp/dQsp))

    tau = 1.0/np.min(np.abs(np.real(ep)[np.real(ep) < -1e-12]))
    return pf_det, pf_qp, lu, tau, ep, es

# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("="*72)
    print("STEP 13: HIV STRUCTURAL OMEGA — 5D MAP")
    print("="*72, flush=True)

    # 1. Fixed points
    print("\n1. FIXED POINTS", flush=True)
    yp = fsolve(rhs, np.array([999967., 6.57e-4, 0.252, 21.9, 0.725]))
    ys = fsolve(rhs, np.array([999901., 1.99e-3, 0.762, 66.3, 0.724]))
    for lbl, y in [('PTC',yp),('SAD',ys)]:
        print(f"  {lbl}: {[f'{v:.6e}' for v in y]}")
        print(f"       |f| = {np.linalg.norm(rhs(y)):.2e}")

    ep = np.sort(np.real(np.linalg.eigvals(jac(yp))))
    es = np.sort(np.real(np.linalg.eigvals(jac(ys))))
    print(f"  PTC eigs: {ep}")
    print(f"  SAD eigs: {es}")
    tau = 1.0/np.min(np.abs(ep[ep < -1e-12]))
    lu = np.max(es)
    print(f"  tau_relax = {tau:.2f} days, lambda_u = {lu:.6f} day^-1")

    Dr = D_raw(yp)
    print(f"\n  D_raw diagonal at PTC:")
    for k in range(5):
        print(f"    {VN[k]}: {Dr[k,k]:.4e}")
    print(f"  Anisotropy D_TT/D_LL = {Dr[0,0]/Dr[1,1]:.4e}")

    # 2. PHASE 1: MAP
    print("\n" + "="*72)
    print("PHASE 1: MINIMUM ACTION PATH")
    print("="*72, flush=True)

    # Reference
    Nref = 50
    al = np.linspace(0,1,Nref)
    pline = np.outer(1-al, yp) + np.outer(al, ys)
    Sline = full_action(pline)
    s_1D = 0.00528/5000.0
    print(f"\n  Straight-line S_line = {Sline:.8e}")
    print(f"  1D projection s_1D  = {s_1D:.8e}")

    # MAP optimization
    results = []
    for N in [8, 10, 12, 15]:
        print(f"\n  === N={N} ===", flush=True)
        path, S = find_MAP(yp, ys, N=N, n_iter=500)
        results.append((f'N{N}', path, S))
        print(f"  [{time.time()-t0:.0f}s elapsed]", flush=True)

    results.sort(key=lambda x: x[2])
    bn, bp, s_star = results[0]
    print(f"\n  Best: {bn}, s* = {s_star:.8e}")
    for nm, _, S in results:
        print(f"    {nm}: {S:.8e}")

    # 3. PHASE 2-3: Prefactor & Omega*
    print("\n" + "="*72)
    print("PHASE 2-3: PREFACTOR & OMEGA*")
    print("="*72, flush=True)

    pf_det, pf_qp, lu, tau, eigs_p, eigs_s = compute_prefactors(yp, ys)
    print(f"  Rate prefactor (det)  = {pf_det:.6e} day^-1")
    print(f"  Rate prefactor (qp)   = {pf_qp:.6e} day^-1")
    print(f"  tau_relax             = {tau:.2f} days")
    print(f"  lambda_u              = {lu:.6f} day^-1")
    print(f"  pf_det * tau          = {pf_det*tau:.6f}")

    # Correct Kramers formula: D = exp(2*Omega*s*) / (pf_rate * tau_relax)
    # => Omega* = ln(D_product * pf_rate * tau) / (2*s*)
    print(f"\n  Bridge identity: D = exp(2*Omega*s*) / (pf * tau)")
    Om_star_det = np.log(D_PRODUCT * pf_det * tau) / (2*s_star)
    Om_star_qp = np.log(D_PRODUCT * pf_qp * tau) / (2*s_star)
    print(f"\n  *** Omega*_structural (det pf) = {Om_star_det:.0f} mL ***")
    print(f"      ln(D*pf*tau) = ln({D_PRODUCT*pf_det*tau:.4f}) = {np.log(D_PRODUCT*pf_det*tau):.6f}")
    print(f"      2*s* = {2*s_star:.8e}")

    if pf_qp * tau * D_PRODUCT > 0:
        print(f"  *** Omega*_structural (qp pf)  = {Om_star_qp:.0f} mL ***")
    print(f"      Omega*_SDE                 = {OM_SDE:.0f} mL")
    ratio = Om_star_det / OM_SDE
    pct_err = abs(ratio - 1) * 100
    print(f"      Ratio Om*_det / Om*_SDE    = {ratio:.4f} ({pct_err:.1f}% error)")

    # K-factor sensitivity
    print(f"\n  K-factor sensitivity:")
    for Kn, Kv in [('K=1.0',1.0),('K=0.55',0.55),('K=0.34',0.34)]:
        Om = np.log(D_PRODUCT * pf_det * tau / Kv) / (2*s_star)
        print(f"    {Kn}: Om* = {Om:.0f} mL (ratio = {Om/OM_SDE:.3f})")

    # 4. PHASE 4: Cross-checks
    print("\n" + "="*72)
    print("PHASE 4: CROSS-CHECKS")
    print("="*72, flush=True)

    # 4a. D_Kramers vs D_exact
    print(f"\n  4a. D_Kramers_5D vs D_exact")
    print(f"  Formula: D = exp(2*Om*s*) / (pf*tau)")
    print(f"  {'Om':>6} {'D_exact':>8} {'D_Kramers':>10} {'ratio':>7}")
    ratios = []
    for Om, Dex in sorted(SDE.items()):
        exp_val = 2 * Om * s_star
        Dkr = np.exp(exp_val) / (pf_det * tau)
        r = Dkr / Dex
        ratios.append(r)
        print(f"  {Om:>6} {Dex:>8.2f} {Dkr:>10.2f} {r:>7.3f}")
    print(f"  Mean ratio (Om 1000-3000): {np.mean([r for (O,_),r in zip(sorted(SDE.items()),ratios) if 1000<=O<=3000]):.3f}")

    # 4b. MAP geometry
    print(f"\n  4b. MAP path geometry")
    Np = len(bp)
    arc = np.zeros(5)
    for i in range(Np-1): arc += np.abs(bp[i+1]-bp[i])
    tot = np.sum(arc)
    for k in range(5):
        print(f"    {VN[k]}: {arc[k]:.4e} ({arc[k]/tot*100:.1f}%)")

    # Check all values non-negative
    min_vals = bp.min(axis=0)
    print(f"\n  Min values along path: {[f'{v:.4e}' for v in min_vals]}")
    has_negative = any(v < -1e-10 for v in min_vals)
    if has_negative:
        print(f"  WARNING: Path has negative values!")
    else:
        print(f"  All values non-negative. ✓")

    # Path at key points
    print(f"\n  Path at key points:")
    print(f"  {'i':>3} {'T':>12} {'L':>12} {'I':>10} {'V':>10} {'E':>10}")
    for idx in [0, Np//4, Np//2, 3*Np//4, Np-1]:
        y = bp[idx]
        print(f"  {idx:>3} {y[0]:>12.1f} {y[1]:>12.6f} {y[2]:>10.6f} {y[3]:>10.4f} {y[4]:>10.6f}")

    # 4c. Extensive scaling
    print(f"\n  4c. Extensive scaling")
    print(f"  S(Omega) = Omega * s* by construction (D_raw^-1 is Omega-independent)")
    print(f"  s* = {s_star:.8e}")
    for Om in [500, 1000, 2000, 3000, 5000]:
        S_Om = Om * s_star
        print(f"    S(Om={Om}) = {S_Om:.6e}")

    # 4d. Action decomposition
    print(f"\n  4d. Action per coordinate (approximate)")
    S_total = 0
    coord_action = np.zeros(5)
    for i in range(Np-1):
        mid = 0.5*(bp[i]+bp[i+1])
        dy = bp[i+1]-bp[i]
        f = rhs(mid)
        D = D_raw(mid)+np.eye(5)*1e-20
        Q = np.linalg.inv(D)
        # Fraction of action from each coordinate
        Qdy = Q@dy; Qf = Q@f
        a = dy@Qdy; b = f@Qf; c = dy@Qf
        if a > 0 and b > 0:
            S_seg = np.sqrt(a*b) - c
            S_total += S_seg
            for k in range(5):
                # Approximate: contribution = |dy[k]*Qdy[k]| / a * S_seg
                coord_action[k] += abs(dy[k]*Qdy[k]) / max(a,1e-30) * S_seg
    if S_total > 0:
        for k in range(5):
            print(f"    {VN[k]}: {coord_action[k]/S_total*100:.1f}%")

    # 5. VERDICT
    print("\n" + "="*72)
    print("VERDICT")
    print("="*72, flush=True)

    Om_primary = Om_star_det
    ratio = Om_primary / OM_SDE
    pct = abs(ratio - 1) * 100

    if 1500 <= Om_primary <= 3500:
        verdict = "STRUCTURAL PASS"
        print(f"  {verdict}")
        print(f"  Omega*_structural = {Om_primary:.0f} mL")
        print(f"  Omega*_SDE        = {OM_SDE:.0f} mL")
        print(f"  Match: {pct:.1f}%")
        print()
        print(f"  The Conway-Perelson model's 5D noise structure and barrier geometry")
        print(f"  predict Omega*_structural = {Om_primary:.0f} mL, matching the SDE-determined")
        print(f"  Omega*_SDE = {OM_SDE:.0f} mL within {pct:.1f}%. The framework identifies the")
        print(f"  lymphoid tissue volume as the effective reaction compartment without")
        print(f"  anatomical input.")
    elif 500 <= Om_primary <= 7000:
        verdict = "QUALIFIED PASS"
        factor = max(ratio, 1/ratio)
        print(f"  {verdict}")
        print(f"  Omega*_structural = {Om_primary:.0f} mL (within {factor:.1f}x of SDE)")
    elif Om_primary > 10000 or Om_primary < 200:
        verdict = "STRUCTURAL MISS"
        print(f"  {verdict}")
        print(f"  Omega*_structural = {Om_primary:.0f} mL (far from {OM_SDE:.0f})")
    else:
        verdict = "INDETERMINATE"
        print(f"  {verdict}: Omega*_structural = {Om_primary:.0f} mL")

    # Summary
    print(f"\n  FULL SUMMARY:")
    print(f"    s* (MAP action per Omega)     = {s_star:.8e}")
    print(f"    S_straight                     = {Sline:.8e}")
    print(f"    s_1D (old linear projection)   = {s_1D:.8e}")
    print(f"    s*/S_line ratio                = {s_star/Sline:.4f}")
    print(f"    s*/s_1D ratio                  = {s_star/s_1D:.1f}")
    print(f"    Rate prefactor (det)           = {pf_det:.6e} day^-1")
    print(f"    Rate prefactor (qp)            = {pf_qp:.6e} day^-1")
    print(f"    tau_relax                      = {tau:.2f} days")
    print(f"    lambda_u                       = {lu:.6f} day^-1")
    print(f"    Omega*_structural (det)        = {Om_star_det:.0f} mL")
    print(f"    Omega*_structural (qp)         = {Om_star_qp:.0f} mL")
    print(f"    Omega*_SDE                     = {OM_SDE:.0f} mL")
    print(f"    D_product                      = {D_PRODUCT:.1f}")
    print(f"    Verdict                        = {verdict}")
    print(f"\n  Total computation time: {time.time()-t0:.1f}s")

    return dict(s_star=s_star, S_line=Sline, s_1D=s_1D,
                pf_det=pf_det, pf_qp=pf_qp, tau=tau, lu=lu,
                Om_det=Om_star_det, Om_qp=Om_star_qp,
                verdict=verdict), bp

if __name__ == "__main__":
    summary, path = main()
