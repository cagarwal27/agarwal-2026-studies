"""
Step 13c: Global optimization of MAP via differential evolution.
"""
import numpy as np
from scipy.optimize import fsolve, differential_evolution
import time
t0 = time.time()

P = dict(lam=10000.0,dT=0.01,beta=1.5e-8,delta=1.0,p=2000.0,c=23.0,a=0.001,
         dL=0.004,rho=0.0045,alpha_L=1e-6,lam_E=1.0,bE=1.0,KB=0.1,dE=2.0,
         KD=5.0,mu=2.0,m=0.42,eps=0.0)

def rhs(y):
    T,L,I,V,E=y; inf=(1-P['eps'])*P['beta']*V*T
    return np.array([P['lam']-P['dT']*T-inf,
        P['alpha_L']*inf+(P['rho']-P['a']-P['dL'])*L,
        (1-P['alpha_L'])*inf-P['delta']*I+P['a']*L-P['m']*E*I,
        P['p']*I-P['c']*V,
        P['lam_E']+P['bE']*I/(P['KB']+I)*E-P['dE']*I/(P['KD']+I)*E-P['mu']*E])

def D_raw(y):
    T,L,I,V,E=np.maximum(y,1e-30); aL=P['alpha_L']
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

yp=fsolve(rhs,np.array([999967.,6.57e-4,0.252,21.9,0.725]))
ys=fsolve(rhs,np.array([999901.,1.99e-3,0.762,66.3,0.724]))
sc = ys - yp; sc[np.abs(sc)<1e-30] = 1.0

def full_action(path):
    S=0.0
    for i in range(len(path)-1):
        mid=0.5*(path[i]+path[i+1]); dy=path[i+1]-path[i]; f=rhs(mid)
        D=D_raw(mid)+np.eye(5)*1e-20; Q=np.linalg.inv(D)
        a=dy@Q@dy; b=f@Q@f; c=dy@Q@f
        if a>0 and b>0: S+=max(np.sqrt(a*b)-c,0)
    return S

# ============================================================
# Config: fix L at linear interp, optimize T,I,V,E
# ============================================================
pf=6.284274e-4; tau=2000.21

for N in [6, 8]:
    Ni = N-2
    al = np.linspace(0,1,N)

    def action(zf, N_=N, Ni_=Ni, al_=al):
        z = zf.reshape(Ni_, 4)
        path = np.zeros((N_, 5))
        path[0]=yp; path[-1]=ys
        for i in range(Ni_):
            t = al_[i+1]
            L_val = yp[1]+t*(ys[1]-yp[1])
            path[i+1] = np.array([yp[0]+z[i,0]*sc[0], L_val,
                                   yp[2]+z[i,1]*sc[2], yp[3]+z[i,2]*sc[3],
                                   yp[4]+z[i,3]*sc[4]])
        return full_action(path)

    bounds = []
    for i in range(Ni):
        bounds.append((-0.5, 2.0))   # T (normalized)
        bounds.append((-0.5, 5.0))   # I
        bounds.append((-0.5, 5.0))   # V
        bounds.append((-50, 50))     # E (very small scale, need wide bounds)

    z0 = np.array([[al[i+1]]*4 for i in range(Ni)]).flatten()
    S0 = action(z0)
    print(f"=== N={N} ({Ni*4}D) ===", flush=True)
    print(f"Straight line: {S0:.6e}", flush=True)

    best = [S0]
    def cb(xk, convergence):
        S = action(xk)
        if S < best[0] * 0.95:  # only print significant improvements
            best[0] = S
            print(f"  S={S:.6e} (t={time.time()-t0:.0f}s)", flush=True)

    result = differential_evolution(action, bounds,
                                    maxiter=1000, popsize=25, tol=1e-12,
                                    mutation=(0.5, 1.5), recombination=0.9,
                                    seed=42, callback=cb, polish=True)

    s_star = result.fun
    Om = np.log(13.0*pf*tau)/(2*s_star)
    print(f"Result: s*={s_star:.6e}, Om*={Om:.0f} mL (target 2296)")
    print(f"Time: {time.time()-t0:.0f}s\n", flush=True)

# ============================================================
# Also try: full 5D with N=6 (only 20D)
# ============================================================
print("=== Full 5D, N=6 (20D) ===", flush=True)
N=6; Ni=4; al=np.linspace(0,1,N)

def action_5d(zf):
    z=zf.reshape(Ni,5)
    path=np.zeros((N,5))
    path[0]=yp; path[-1]=ys
    for i in range(Ni):
        path[i+1]=yp+z[i]*sc
    return full_action(path)

bounds_5d=[]
for i in range(Ni):
    for k in range(5):
        lo=-yp[k]/sc[k] if sc[k]>0 else -10
        bounds_5d.append((max(lo,-5), 5.0))

z0=np.array([[al[i+1]]*5 for i in range(Ni)]).flatten()
print(f"Straight line: {action_5d(z0):.6e}", flush=True)

best5=[action_5d(z0)]
def cb5(xk, conv):
    S=action_5d(xk)
    if S<best5[0]*0.95:
        best5[0]=S
        print(f"  S={S:.6e} (t={time.time()-t0:.0f}s)", flush=True)

result5=differential_evolution(action_5d, bounds_5d,
                               maxiter=1000, popsize=30, tol=1e-12,
                               mutation=(0.5,1.5), recombination=0.9,
                               seed=42, callback=cb5, polish=True)

s5=result5.fun
Om5=np.log(13.0*pf*tau)/(2*s5)
print(f"Result: s*={s5:.6e}, Om*={Om5:.0f} mL")

# Reconstruct best path and show
z_opt=result5.x.reshape(Ni,5)
path=np.zeros((N,5))
path[0]=yp; path[-1]=ys
for i in range(Ni): path[i+1]=yp+z_opt[i]*sc
print(f"\nBest 5D path:")
print(f"  pt    T          L          I        V        E")
for i in range(N):
    y=path[i]
    print(f"  {i:2d} {y[0]:11.1f} {y[1]:10.6f} {y[2]:8.4f} {y[3]:8.2f} {y[4]:8.4f}")

print(f"\nMin values: {[f'{v:.4e}' for v in path.min(axis=0)]}")
print(f"Total time: {time.time()-t0:.0f}s")
