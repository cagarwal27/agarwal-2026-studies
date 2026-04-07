"""
Step 13b: Reduced-Dimension MAP for HIV Conway-Perelson Model
==============================================================
Computes the Minimum Action Path (MAP) from PTC to saddle using L-BFGS-B,
testing dimension reductions: 3D (T,I,V), 4D (T,L,I,V), 4D (T,I,V,E),
and full 5D. The geometric FW action is always the full 5D formula.

Performance note: Each action evaluation requires N-1 5x5 matrix inversions.
We limit N and number of starts to keep runtime under 10 min.
"""

import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.linalg import solve_continuous_lyapunov
import warnings, time
warnings.filterwarnings('ignore')

P = dict(lam=10000.0, dT=0.01, beta=1.5e-8, delta=1.0, p=2000.0,
         c=23.0, a=0.001, dL=0.004, rho=0.0045, alpha_L=1e-6,
         lam_E=1.0, bE=1.0, KB=0.1, dE=2.0, KD=5.0, mu=2.0,
         m=0.42, eps=0.0)

D_PRODUCT = 13.0; OM_SDE = 2296.0
SDE = {500:0.35,1000:1.24,2000:8.22,2500:17.24,
       3000:21.67,3500:24.85,4000:23.82,4500:28.26}
VN = ['T','L','I','V','E']

def rhs5(y):
    T,L,I,V,E = y
    inf = (1-P['eps'])*P['beta']*V*T
    return np.array([
        P['lam']-P['dT']*T-inf,
        P['alpha_L']*inf+(P['rho']-P['a']-P['dL'])*L,
        (1-P['alpha_L'])*inf-P['delta']*I+P['a']*L-P['m']*E*I,
        P['p']*I-P['c']*V,
        P['lam_E']+P['bE']*I/(P['KB']+I)*E-P['dE']*I/(P['KD']+I)*E-P['mu']*E])

def jac5(y):
    T,L,I,V,E = y; b = (1-P['eps'])*P['beta']
    J = np.zeros((5,5))
    J[0,0]=-P['dT']-b*V;         J[0,3]=-b*T
    J[1,0]=P['alpha_L']*b*V;     J[1,1]=P['rho']-P['a']-P['dL']
    J[1,3]=P['alpha_L']*b*T
    J[2,0]=(1-P['alpha_L'])*b*V; J[2,1]=P['a']
    J[2,2]=-P['delta']-P['m']*E; J[2,3]=(1-P['alpha_L'])*b*T
    J[2,4]=-P['m']*I
    J[3,2]=P['p'];               J[3,3]=-P['c']
    J[4,2]=(P['bE']*P['KB']/(P['KB']+I)**2-P['dE']*P['KD']/(P['KD']+I)**2)*E
    J[4,4]=P['bE']*I/(P['KB']+I)-P['dE']*I/(P['KD']+I)-P['mu']
    return J

def D_raw5(y):
    T,L,I,V,E = np.maximum(y,1e-30); aL=P['alpha_L']
    rs=[(P['lam'],[1,0,0,0,0]),(P['dT']*T,[-1,0,0,0,0]),
        ((1-P['eps'])*P['beta']*V*T,[-1,aL,1-aL,0,0]),
        (P['rho']*L,[0,1,0,0,0]),(P['dL']*L,[0,-1,0,0,0]),
        (P['a']*L,[0,-1,1,0,0]),(P['delta']*I,[0,0,-1,0,0]),
        (P['m']*E*I,[0,0,-1,0,0]),(P['p']*I,[0,0,0,1,0]),
        (P['c']*V,[0,0,0,-1,0]),(P['lam_E'],[0,0,0,0,1]),
        (P['bE']*I/(P['KB']+I)*E,[0,0,0,0,1]),
        (P['dE']*I/(P['KD']+I)*E,[0,0,0,0,-1]),
        (P['mu']*E,[0,0,0,0,-1])]
    D=np.zeros((5,5))
    for r,s in rs:
        sv=np.array(s); D+=max(r,0)*np.outer(sv,sv)
    return D

# Pre-compute Q matrices for all midpoints (cache)
_Q_cache = {}

def total_action(path):
    N = len(path)
    S = 0.0
    for i in range(N-1):
        mid = 0.5*(path[i]+path[i+1])
        dy = path[i+1]-path[i]
        f = rhs5(mid)
        D = D_raw5(mid); D[np.diag_indices(5)] += 1e-20
        Q = np.linalg.inv(D)
        a = dy@Q@dy; b = f@Q@f; c = dy@Q@f
        if a > 0 and b > 0:
            v = np.sqrt(a*b)-c
            if v > 0: S += v
    return S

def optimize_map(yp, ys, N, free_dims, n_starts=3, label=""):
    Ni = N-2; ndim = len(free_dims)
    fixed_dims = [d for d in range(5) if d not in free_dims]
    al = np.linspace(0,1,N)
    fixed_arr = {d: (1-al)*yp[d]+al*ys[d] for d in fixed_dims}

    def unpack(x):
        interior = x.reshape(Ni, ndim)
        path = np.zeros((N,5))
        path[0]=yp; path[-1]=ys
        for i in range(Ni):
            for d in fixed_dims: path[i+1,d]=fixed_arr[d][i+1]
            for j,d in enumerate(free_dims): path[i+1,d]=interior[i,j]
        return path

    def obj(x):
        return total_action(unpack(x))

    bounds = [(1e-30,None)]*(Ni*ndim)
    best_S=np.inf; best_x=None

    for r in range(n_starts):
        np.random.seed(42+r*13)
        x0=np.zeros(Ni*ndim)
        for j,d in enumerate(free_dims):
            x0[j::ndim]=(1-al[1:-1])*yp[d]+al[1:-1]*ys[d]
        if r>0:
            sc=np.array([max(abs(ys[d]-yp[d]),1e-10) for d in free_dims])
            for j in range(ndim):
                x0[j::ndim] += np.random.randn(Ni)*sc[j]*(0.1+0.15*r)
            x0=np.maximum(x0,1e-30)

        res=minimize(obj,x0,method='L-BFGS-B',bounds=bounds,
                     options={'maxiter':3000,'maxfun':30000,'ftol':1e-14,'gtol':1e-11})
        tag = " *NEW*" if res.fun<best_S else ""
        print(f"    r{r}: S={res.fun:.8e} nit={res.nit}{tag}", flush=True)
        if res.fun<best_S: best_S=res.fun; best_x=res.x.copy()

    return unpack(best_x), best_S

def compute_prefactors(yp, ys):
    Jp=jac5(yp); Js=jac5(ys)
    ep=np.linalg.eigvals(Jp); es=np.linalg.eigvals(Js)
    lu=np.max(np.real(es)); pp=np.prod(np.abs(ep))
    iu=np.argmax(np.real(es)); mk=np.ones(5,dtype=bool); mk[iu]=False
    ps=np.prod(np.abs(es[mk]))
    pf=abs(lu)/(2*np.pi)*np.sqrt(pp/ps)
    neg=np.real(ep)[np.real(ep)<-1e-12]
    tau=1.0/np.min(np.abs(neg))
    return pf,lu,tau,ep,es

def main():
    t0=time.time()
    print("="*72)
    print("STEP 13b: REDUCED-DIMENSION MAP FOR HIV CONWAY-PERELSON")
    print("="*72, flush=True)

    # 1. Fixed points
    print("\n1. FIXED POINTS", flush=True)
    yp=fsolve(rhs5,np.array([999967.,6.57e-4,0.252,21.9,0.725]))
    ys=fsolve(rhs5,np.array([999901.,1.99e-3,0.762,66.3,0.724]))
    for lbl,y in [('PTC',yp),('SAD',ys)]:
        print(f"  {lbl}: T={y[0]:.4f} L={y[1]:.6e} I={y[2]:.6f} V={y[3]:.4f} E={y[4]:.6f}")
        print(f"       |f|={np.linalg.norm(rhs5(y)):.2e}")
    D5=D_raw5(yp)
    print(f"  Anisotropy: D_TT/D_LL={D5[0,0]/D5[1,1]:.2e}")

    # 2. Reference
    print("\n2. REFERENCE", flush=True)
    al50=np.linspace(0,1,50)
    pline=np.outer(1-al50,yp)+np.outer(al50,ys)
    S_line=total_action(pline)
    print(f"  Straight line (N=50): S={S_line:.8e}")
    print(f"  5D Step13 (partial): s*=2.47e-3")
    print(f"  SDE target: s*~9.69e-4")

    # 3. Optimization: N=10 for all configs (fast), then N=15 for best
    all_res = []
    configs = [
        ([0,2,3],      "3D(T,I,V)"),
        ([0,1,2,3],    "4D(T,L,I,V)"),
        ([0,2,3,4],    "4D(T,I,V,E)"),
        ([0,1,2,3,4],  "5D(full)"),
    ]

    # Phase 1: N=10, 3 starts each
    print("\n" + "="*72)
    print("3. PHASE 1: N=10, 3 random starts per config")
    print("="*72, flush=True)
    for free,lbl in configs:
        print(f"\n  [{lbl}] N=10:", flush=True)
        t1=time.time()
        path,S=optimize_map(yp,ys,10,free,n_starts=3,label=lbl)
        all_res.append((S,f"{lbl}/N10",path,10,lbl))
        print(f"  -> {S:.8e} ({time.time()-t1:.0f}s)", flush=True)

    # Phase 2: N=15, 3 starts for 4D and 5D
    print("\n" + "="*72)
    print("4. PHASE 2: N=15, 3 random starts (4D and 5D only)")
    print("="*72, flush=True)
    for free,lbl in configs[1:]:  # skip 3D
        print(f"\n  [{lbl}] N=15:", flush=True)
        t1=time.time()
        path,S=optimize_map(yp,ys,15,free,n_starts=3,label=lbl)
        all_res.append((S,f"{lbl}/N15",path,15,lbl))
        print(f"  -> {S:.8e} ({time.time()-t1:.0f}s)", flush=True)

    # Phase 3: N=20 for 5D only (the key test)
    print("\n" + "="*72)
    print("5. PHASE 3: N=20, 5D full (3 starts)")
    print("="*72, flush=True)
    print(f"\n  [5D(full)] N=20:", flush=True)
    t1=time.time()
    path,S=optimize_map(yp,ys,20,[0,1,2,3,4],n_starts=3,label="5D/N20")
    all_res.append((S,"5D(full)/N20",path,20,"5D(full)"))
    print(f"  -> {S:.8e} ({time.time()-t1:.0f}s)", flush=True)

    # Summary
    print("\n" + "="*72)
    print("6. ALL RESULTS")
    print("="*72, flush=True)
    all_res.sort(key=lambda x:x[0])
    print(f"  {'Rank':>4} {'Config':<22} {'s*':>14}")
    print(f"  {'-'*42}")
    for i,(S,lab,_,_,_) in enumerate(all_res):
        m=" <-- BEST" if i==0 else ""
        print(f"  {i+1:>4} {lab:<22} {S:>14.8e}{m}")

    s_star=all_res[0][0]; best_path=all_res[0][2]; best_label=all_res[0][1]

    # Best per config
    by_cfg={}
    for S,_,_,_,cfg in all_res:
        if cfg not in by_cfg or S<by_cfg[cfg]: by_cfg[cfg]=S
    print(f"\n  Best by config:")
    for cfg,S in sorted(by_cfg.items(), key=lambda x:x[1]):
        print(f"    {cfg:<22} {S:.8e}")

    # 7. Omega*
    print("\n" + "="*72)
    print("7. OMEGA*", flush=True)
    pf,lu,tau,_,_=compute_prefactors(yp,ys)
    print(f"  pf={pf:.6e}, tau={tau:.2f}, lu={lu:.6f}")
    arg=D_PRODUCT*pf*tau
    Om=np.log(arg)/(2*s_star)
    print(f"  ln(D*pf*tau) = {np.log(arg):.6f}")
    print(f"  Omega* = {Om:.0f} mL  (SDE: {OM_SDE})")

    # Kramers cross-check
    print(f"\n  D_Kramers vs D_exact:")
    print(f"  {'Om':>6} {'D_ex':>7} {'D_Kr':>9} {'ratio':>7}")
    for O,Dex in sorted(SDE.items()):
        Dkr=np.exp(2*O*s_star)/(pf*tau)
        print(f"  {O:>6} {Dex:>7.2f} {Dkr:>9.2f} {Dkr/Dex:>7.3f}")

    # 8. Path geometry
    print("\n" + "="*72)
    print("8. PATH GEOMETRY", flush=True)
    Nb=len(best_path)
    coord_act=np.zeros(5); Stot=0
    for i in range(Nb-1):
        ya,yb=best_path[i],best_path[i+1]
        mid=0.5*(ya+yb); dy=yb-ya; f=rhs5(mid)
        D=D_raw5(mid)+np.eye(5)*1e-20; Q=np.linalg.inv(D)
        Qdy=Q@dy; a=dy@Qdy; b=f@Q@f; c=dy@Q@f
        if a>0 and b>0:
            Ss=max(np.sqrt(a*b)-c,0.0); Stot+=Ss
            for k in range(5):
                coord_act[k]+=abs(dy[k]*Qdy[k])/max(a,1e-30)*Ss
    if Stot>0:
        print(f"  Action fraction:")
        for k in range(5): print(f"    {VN[k]}: {coord_act[k]/Stot*100:.2f}%")

    print(f"\n  Path points:")
    print(f"  {'i':>3} {'T':>12} {'L':>12} {'I':>10} {'V':>10} {'E':>10}")
    for idx in [0,Nb//4,Nb//2,3*Nb//4,Nb-1]:
        y=best_path[idx]
        print(f"  {idx:>3} {y[0]:>12.1f} {y[1]:>12.6e} {y[2]:>10.6f} {y[3]:>10.4f} {y[4]:>10.6f}")

    # 9. Verdict
    print("\n" + "="*72)
    print("9. VERDICT", flush=True)
    s5o=2.47e-3; s_sde=9.69e-4
    print(f"  s*_best    = {s_star:.8e}")
    print(f"  s*_5D(old) = {s5o:.8e}")
    print(f"  s*_SDE     = {s_sde:.8e}")
    imp=s_star<s5o
    if imp:
        print(f"  Improved over 5D: YES ({(1-s_star/s5o)*100:.1f}% reduction)")
    else:
        print(f"  Improved over 5D: NO ({s_star/s5o:.2f}x higher)")
    print(f"  Below 1.5e-3: {'YES' if s_star<1.5e-3 else 'NO'}")

    if 1500<=Om<=3500: v="STRUCTURAL PASS"
    elif 500<=Om<=7000: v="QUALIFIED PASS"
    elif Om>10000 or Om<200: v="STRUCTURAL MISS"
    else: v="INDETERMINATE"
    print(f"  Omega*={Om:.0f} mL, Verdict={v}")
    print(f"\n  Time: {time.time()-t0:.1f}s")
    return s_star,Om,v

if __name__=="__main__":
    main()
