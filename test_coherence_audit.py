# -*- coding: utf-8 -*-
"""
==================================================================================
 TESTE DE AUDITORIA — métrica de coerência (Eq. 7 vs. implementação em coerence())
==================================================================================
Objetivo: verificar, de forma reproduzível, se a função `coerence()` do repositório
calcula a entropia relativa de coerência da Eq. (7) — ou se, por causa da linha
`estado = estado * estado.dag()`, ela calcula sobre rho^2.

Como rodar (na raiz do repo, com o mesmo ambiente das simulações):
    python test_coherence_audit.py

O script roda em duas partes independentes:
  PARTE A  usa o CÓDIGO REAL do repo (precisa de qutip + pacote quantum/).
  PARTE B  é uma reprodução independente em NumPy/SciPy puro (não precisa de qutip).
Cada parte imprime uma tabela e um VEREDITO. As duas devem concordar.
==================================================================================
"""
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=120)

LINE = "=" * 82
def title(s): print("\n" + LINE + "\n" + s + "\n" + LINE)


# ==================================================================================
# PARTE A — roda a função REAL do repositório sobre uma dinâmica real
# ==================================================================================
def parte_A():
    title("PARTE A — função REAL coerence() do repo, sobre dinâmica de Lindblad real")
    try:
        import qutip as qt
        from quantum.operators import get_operators, get_collapse
        from quantum.hamiltonian import h_open
        from quantum.run import solve
        from quantum.non_classicality import coerence          # <<< a função auditada
        from utils.utils import diagonaliza
    except Exception as e:
        print("  [PULANDO PARTE A] não consegui importar o repo/qutip:", repr(e))
        print("  -> rode este arquivo na RAIZ do repositório, no ambiente com qutip.")
        return

    # ---- mesmos parâmetros/estado inicial de coerence.py ----
    N, Nb = 2, 45
    alpha = np.sqrt(5)                       # <<< igual ao código (NÃO é 1)
    args = {'g0': 1, 'w': np.pi/20, 'phi': 0,
            'kappa': 1e-1, 'gamma': 0, 'gamma_phi': 1e-2, 'coupling': 'cos'}
    t = np.linspace(0, 50, 160)

    sz, sp, sm, b, nb, I = get_operators(N, Nb)
    c_ops = get_collapse(args, sm, sz, b)
    phi0 = qt.tensor(qt.basis(N, 0) + qt.basis(N, 1), qt.coherent(Nb, alpha)).unit()

    H = h_open(b, sp, sm)
    sol = solve(H, phi0, t, c_ops, [], args)         # equação mestra (mesolve)
    states = sol.states
    print(f"  alpha = {alpha:.4f}  (|alpha|^2 = {alpha**2:.1f})   Nb = {Nb}   pontos = {len(t)}")

    # ---- (1) a função REAL do repo ----
    C_repo = np.array(coerence(states))

    # ---- (2) a Eq. (7) CORRETA (sem o quadrado, base 2) ----
    def C_eq7(states, base=2):
        out = np.zeros(len(states))
        for i, st in enumerate(states):
            rhoQ = st.ptrace(0)
            out[i] = qt.entropy_vn(diagonaliza(rhoQ), base=base) - qt.entropy_vn(rhoQ, base=base)
        return out
    C_correto = C_eq7(states, base=2)

    # ---- (3) diagnósticos: o que 'estado' é e o que a multiplicação faz ----
    rhoQ = states[80].ptrace(0)                       # um instante genérico (misto)
    quad = rhoQ * rhoQ.dag()
    print("\n  Diagnóstico em t = %.2f :" % t[80])
    print("   type(states[i].ptrace(0)) =", rhoQ.type, " (deveria ser 'oper', não 'ket')")
    print("   pureza de rho_Q          =", round((rhoQ*rhoQ).tr().real, 4))
    print("   Tr[ estado*estado.dag() ] =", round(quad.tr().real, 4),
          " -> se != 1, a multiplicação fez rho_Q^2 (não é |psi><psi|)")
    print("   rho_Q =\n", np.round(rhoQ.full().real, 4))
    print("   estado*estado.dag() (=rho_Q^2) =\n", np.round(quad.full().real, 4))

    _tabela_e_veredito(t, C_repo, C_correto, nome_repo="coerence() do repo")


# ==================================================================================
# PARTE B — reprodução INDEPENDENTE em NumPy/SciPy (sem qutip, sem o pacote quantum/)
# ==================================================================================
def parte_B():
    title("PARTE B — reprodução independente (NumPy/SciPy puro, sem qutip)")
    from scipy.integrate import solve_ivp
    from math import lgamma

    N, Nb = 2, 30
    alpha = np.sqrt(5)
    kappa, gamma_phi = 1e-1, 1e-2
    w = np.pi/20

    # operadores
    def destroy(n):
        return np.diag(np.sqrt(np.arange(1, n)), 1)
    a = np.kron(np.eye(2), destroy(Nb))
    adag = a.conj().T
    sp = np.kron(np.array([[0, 1], [0, 0]]), np.eye(Nb))   # |e><g|
    sm = sp.conj().T
    sz = np.kron(np.array([[1, 0], [0, -1]]), np.eye(Nb))
    Hbase = sp @ a + sm @ adag

    # estado inicial (|e>+|g>)/sqrt2 (x) |alpha>
    n = np.arange(Nb)
    coh = np.exp(-abs(alpha)**2/2 + n*np.log(alpha+0j) - 0.5*np.array([lgamma(k+1) for k in n]))
    coh = coh/np.linalg.norm(coh)
    q = np.array([1, 1])/np.sqrt(2)
    psi0 = np.kron(q, coh)
    rho0 = np.outer(psi0, psi0.conj())

    cops = [np.sqrt(kappa)*a, np.sqrt(gamma_phi)*sz]
    pre = [c.conj().T @ c for c in cops]

    def rhs(tt, y):
        rho = y.reshape(2*Nb, 2*Nb)
        H = np.cos(w*tt) * Hbase
        d = -1j*(H@rho - rho@H)
        for c, cdc in zip(cops, pre):
            d += c@rho@c.conj().T - 0.5*(cdc@rho + rho@cdc)
        return d.ravel()

    t = np.linspace(0, 50, 140)
    print(f"  integrando Lindblad... (dim={2*Nb}, {len(t)} pontos)")
    sol = solve_ivp(rhs, (t[0], t[-1]), rho0.ravel(), t_eval=t,
                    method="RK45", rtol=1e-7, atol=1e-9)
    rhos = [sol.y[:, k].reshape(2*Nb, 2*Nb) for k in range(len(t))]

    def ptrace_qubit(rho):
        R = rho.reshape(2, Nb, 2, Nb)
        return np.einsum('akbk->ab', R)

    def S(M, base):
        w_ = np.linalg.eigvalsh((M+M.conj().T)/2)
        w_ = w_[w_ > 1e-12]
        return float(-np.sum(w_*np.log(w_))/np.log(base))

    def diagonaliza(M):
        return np.diag(np.diag(M))

    # replica EXATAMENTE a lógica de coerence(): rho_Q -> rho_Q^2 -> base e
    def C_repo(rhos):
        out = np.zeros(len(rhos))
        for i, rho in enumerate(rhos):
            estado = ptrace_qubit(rho)
            estado = estado @ estado.conj().T          # <<< o passo do repo (= rho_Q^2)
            out[i] = S(diagonaliza(estado), np.e) - S(estado, np.e)
        return out

    # Eq. (7) correta: rho_Q, base 2
    def C_correto(rhos):
        out = np.zeros(len(rhos))
        for i, rho in enumerate(rhos):
            rhoQ = ptrace_qubit(rho)
            out[i] = S(diagonaliza(rhoQ), 2) - S(rhoQ, 2)
        return out

    _tabela_e_veredito(t, C_repo(rhos), C_correto(rhos), nome_repo="lógica do repo (rho^2)")


# ==================================================================================
# Impressão de tabela + veredito (compartilhado pelas duas partes)
# ==================================================================================
def _tabela_e_veredito(t, C_repo, C_correto, nome_repo):
    # normalizadas pelo próprio máximo (como nas Figs. 7-9)
    nr = C_repo/np.max(C_repo)
    nc = C_correto/np.max(C_correto)

    print("\n  %6s | %14s | %14s | %8s || %10s | %10s | %8s" %
          ("t", nome_repo, "Eq.(7) correta", "dif %", "norm repo", "norm Eq7", "dif norm"))
    print("  " + "-"*94)
    idx = np.linspace(0, len(t)-1, 14).astype(int)
    for i in idx:
        difp = 100*(C_repo[i]-C_correto[i])/C_correto[i] if abs(C_correto[i]) > 1e-9 else 0.0
        difn = nr[i]-nc[i]
        print("  %6.2f | %14.6f | %14.6f | %+7.1f || %10.5f | %10.5f | %+8.4f" %
              (t[i], C_repo[i], C_correto[i], difp, nr[i], nc[i], difn))

    # métricas globais
    mask = np.abs(C_correto) > 1e-6
    max_rel = np.max(np.abs((C_repo[mask]-C_correto[mask])/C_correto[mask]))*100
    max_norm = np.max(np.abs(nr-nc))
    corr = np.corrcoef(C_repo, C_correto)[0, 1]
    print("\n  Máx. diferença relativa (valores)      : %.1f %%" % max_rel)
    print("  Máx. diferença nas curvas NORMALIZADAS : %.4f   (se >0, a FORMA difere)" % max_norm)
    print("  Correlação repo vs Eq.(7)              : %.4f" % corr)

    igual_valores = max_rel < 1.0
    igual_forma = max_norm < 1e-3
    print("\n  VEREDITO:")
    if igual_valores and igual_forma:
        print("   [OK] A implementação COINCIDE com a Eq. (7). Nenhum problema de coerência.")
    else:
        print("   [PROBLEMA CONFIRMADO] A implementação NÃO reproduz a Eq. (7):")
        if not igual_valores:
            print(f"      - valores diferem em até {max_rel:.1f}% (efeito do rho^2 + base e).")
        if not igual_forma:
            print(f"      - a FORMA das curvas normalizadas difere (máx {max_norm:.4f}),")
            print("        ou seja, normalizar pelo máximo NÃO corrige o erro.")
        print("   Correção: apagar 'estado = estado * estado.dag()' e usar base=2.")


if __name__ == "__main__":
    print(LINE)
    print(" AUDITORIA DA COERÊNCIA — Eq. (7) vs. coerence()   [rode e leia os VEREDITOS]")
    print(LINE)
    parte_A()
    parte_B()
    print("\n" + LINE + "\n Fim. As duas partes devem apontar o mesmo veredito.\n" + LINE)
