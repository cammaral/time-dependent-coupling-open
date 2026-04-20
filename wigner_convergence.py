import os
import time
import numpy as np
import qutip as qt
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ============================================
# ESTADO INICIAL
# ============================================
N = 2
Nb = 45
alpha = np.sqrt(5)

psi0 = qt.tensor(
    qt.basis(N, 0),
    (qt.coherent(Nb, alpha) + qt.coherent(Nb, -alpha)).unit()
)

# pega só o modo do campo
rho_field = qt.ptrace(psi0, 1)


# ============================================
# NEGATIVIDADE DA WIGNER
# ============================================
def wigner_negativity_from_W(W, xvec, pvec):
    dx = xvec[1] - xvec[0]
    dp = pvec[1] - pvec[0]
    return 0.5 * np.sum(np.abs(W) - W) * dx * dp


# ============================================
# TESTE DE CONVERGÊNCIA
# ============================================
def convergence_wigner_plot(
    rho,
    L=20,
    n_list=(40, 60, 80, 100, 120, 160, 200, 250, 300, 400),
    method="clenshaw",
    save_dir="results/convergence_wigner"
):
    os.makedirs(save_dir, exist_ok=True)

    results = []

    for n in tqdm(n_list, desc="Convergência"):
        xvec = np.linspace(-L, L, n)
        pvec = np.linspace(-L, L, n)

        t0 = time.perf_counter()
        W = qt.wigner(rho, xvec, pvec, method=method)
        elapsed = time.perf_counter() - t0

        neg = wigner_negativity_from_W(W, xvec, pvec)

        results.append({
            "n_points": n,
            "negativity": neg,
            "time_sec": elapsed,
            "W_min": W.min(),
            "W_max": W.max(),
        })

    df = pd.DataFrame(results)
    df["delta_neg"] = df["negativity"].diff().abs()

    # salva tabela
    df.to_csv(os.path.join(save_dir, "convergence.csv"), index=False)

    # =========================
    # PLOT 1: negatividade
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(df["n_points"], df["negativity"], marker="o")
    plt.xlabel("Número de pontos em x e p")
    plt.ylabel("Negatividade da Wigner")
    plt.title(f"Convergência da negatividade (L = {L})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "negativity_convergence.png"), dpi=200)
    plt.show()

    # =========================
    # PLOT 2: diferença entre malhas consecutivas
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(df["n_points"][1:], df["delta_neg"][1:], marker="o")
    plt.xlabel("Número de pontos em x e p")
    plt.ylabel(r"$|\Delta$ negatividade$|$")
    plt.title(f"Erro entre resoluções consecutivas (L = {L})")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "delta_convergence.png"), dpi=200)
    plt.show()

    # =========================
    # PLOT 3: custo computacional
    # =========================
    plt.figure(figsize=(8, 5))
    plt.plot(df["n_points"], df["time_sec"], marker="o")
    plt.xlabel("Número de pontos em x e p")
    plt.ylabel("Tempo de cálculo (s)")
    plt.title(f"Custo computacional da Wigner (L = {L})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "time_convergence.png"), dpi=200)
    plt.show()

    return df


# ============================================
# EXECUÇÃO
# ============================================
df_conv = convergence_wigner_plot(
    rho=rho_field,
    L=10,
    n_list=[40, 60, 80, 100, 120, 160, 200, 250, 300, 400],
    method="clenshaw",
    save_dir="results/convergence_wigner_initial"
)

print(df_conv)