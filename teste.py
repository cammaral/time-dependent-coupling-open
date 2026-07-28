# -*- coding: utf-8 -*-
# Rodar na RAIZ do repositorio, no mesmo ambiente das simulacoes (com qutip).
import numpy as np
import qutip as qt

# ---- USA O CODIGO QUE JA TINHAMOS NO REPO ----
from quantum.operators import get_operators, get_collapse
from quantum.hamiltonian import h_open
from quantum.run import solve
from quantum.non_classicality import coerence      # <<< a funcao auditada, sem tocar nela
from utils.utils import diagonaliza

np.set_printoptions(precision=4, suppress=True, linewidth=120)
L = "="*80

# ===== mesmos parametros e estado inicial de coerence.py =====
N, Nb = 2, 45
alpha = np.sqrt(5)                                  # <<< igual ao codigo (nao e 1!)
args = {'g0': 1, 'w': np.pi/20, 'phi': 0,
        'kappa': 1e-1, 'gamma': 0, 'gamma_phi': 1e-2, 'coupling': 'cos'}
t = np.linspace(0, 50, 160)

sz, sp, sm, b, nb, I = get_operators(N, Nb)
c_ops = get_collapse(args, sm, sz, b)
phi0 = qt.tensor(qt.basis(N, 0) + qt.basis(N, 1), qt.coherent(Nb, alpha)).unit()

# ===== dinamica real (mesma funcao solve do repo) =====
sol = solve(h_open(b, sp, sm), phi0, t, c_ops, [], args)
states = sol.states
print(L); print(" TESTE DE COERENCIA — coerence() do repo  vs  Eq.(7)"); print(L)
print(f" alpha={alpha:.4f} (|alpha|^2={alpha**2:.1f})  Nb={Nb}  pontos={len(t)}")

# ===== 1) diagnostico: o que 'estado' e, e o que a multiplicacao faz =====
rhoQ = states[80].ptrace(0)                         # instante generico (misto)
quad = rhoQ * rhoQ.dag()                            # exatamente a linha do repo
print("\n Diagnostico em t=%.2f:" % t[80])
print("  type(states[i].ptrace(0)) =", rhoQ.type, " (e 'oper', NAO 'ket')")
print("  Tr[estado*estado.dag()]   =", round(quad.tr().real, 4),
      " -> !=1  => virou rho_Q^2 (nao e |psi><psi|)")
print("  pureza de rho_Q           =", round((rhoQ*rhoQ).tr().real, 4))
print("  rho_Q =\n", rhoQ.full().real)
print("  estado*estado.dag() (=rho_Q^2) =\n", quad.full().real)

# ===== 2) coerence() REAL do repo  vs  Eq.(7) correta =====
C_repo = np.array(coerence(states))                 # tua funcao, com rho^2 e base e
def C_eq7(states, base=2):                           # definicao correta
    return np.array([qt.entropy_vn(diagonaliza(s.ptrace(0)), base=base)
                     - qt.entropy_vn(s.ptrace(0), base=base) for s in states])
C_ok = C_eq7(states, base=2)

nr, nc = C_repo/C_repo.max(), C_ok/C_ok.max()        # normalizadas (como nas Figs. 7-9)
print("\n   t   | coerence() repo |  Eq.(7) correta | dif % || norm repo | norm Eq7 | dif norm")
print("  " + "-"*88)
for i in np.linspace(0, len(t)-1, 14).astype(int):
    dp = 100*(C_repo[i]-C_ok[i])/C_ok[i] if abs(C_ok[i])>1e-9 else 0
    print("  %5.2f | %14.6f | %14.6f | %+6.1f || %8.5f | %8.5f | %+7.4f"
          % (t[i], C_repo[i], C_ok[i], dp, nr[i], nc[i], nr[i]-nc[i]))

m = np.abs(C_ok) > 1e-6
print("\n Max dif relativa (valores)      : %.1f %%" % (100*np.max(np.abs((C_repo[m]-C_ok[m])/C_ok[m]))))
print(" Max dif nas curvas NORMALIZADAS : %.4f  (>0 => a FORMA difere)" % np.max(np.abs(nr-nc)))
print(" Correlacao repo vs Eq.(7)       : %.4f" % np.corrcoef(C_repo, C_ok)[0,1])
print("\n Se as colunas diferem e a dif normalizada > 0, esta confirmado que")
print(" coerence() calcula sobre rho^2 (base e), nao a Eq.(7).")
print(" Correcao: apagar 'estado = estado*estado.dag()' e usar base=2.")
print(L)
