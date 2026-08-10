# -*- coding: utf-8 -*-
"""
run_coherence_article_corrected.py

Roda a COERENCIA do CASO DO ARTIGO (os DOIS decaimentos, kappa e gamma_phi),
na VARREDURA COMPLETA (a que gera o grafico DENSO), calculando a coerencia
nas DUAS formas para manter o mesmo padrao de run_coherence_old_and_correct.py:

  * "old"     -> funcao coerence() do repo            (usa rho_Q^2, base e)   [BUGADA]
  * "correct" -> entropia relativa de coerencia Eq.7  (rho_Q,   base 2)       [CORRETA]

FIDELIDADE DE PARAMETROS:
  - Dissipacao: os DOIS canais, via loss_args("specific_parameters")
        kappa = 1e-1,  gamma = 0,  gamma_phi = 1e-2       (= texto do artigo)
  - Varredura COMPLETA (mesma dos blocos only_*), com os parametros do artigo:
        gauss/zeta : T=25, sigma=-1, 6 < zeta < 10   (100 valores; av -> indice 50)
        gauss/T    : zeta=8, sigma=-1, 15 < T < 35    (100 valores; av -> indice 50)
        cos/omega  : phi=0, 0 < omega < pi/5          (200 valores; min ->50, av ->100)
  - Estado inicial, N, Nb, alpha, grade de tempo e acoplamentos: IMPORTADOS de
    run_lista_simulacoes_open_system.py (identicos aos demais runs).

Como rodar (na raiz do repo, no .venv com qutip):
    python run_coherence_article_corrected.py
Para teste rapido: SCAN_LIMIT = 3.

Saida: results/coherence_article_corrected/<case_id>/ com
    const_aberto.npy (old)   const_aberto_correct.npy (correct)
    var_aberto.npy   (old)   var_aberto_correct.npy   (correct)
    t.npy, scan_values.npy, args_per_scan.csv, case_metadata.json, summary.csv
"""
import os
import json
import numpy as np
import pandas as pd
import qutip as qt
from tqdm import tqdm

# --- primitivas do repo (as MESMAS dos scripts antigos) ---
from quantum.operators import get_operators, get_collapse
from quantum.hamiltonian import h_open, h_closed
from quantum.run import solve
from quantum.non_classicality import coerence as coherence_old   # funcao ORIGINAL, intacta
from utils.utils import diagonaliza

# --- reaproveita EXATAMENTE os primitivos/parametros do run recente ---
import run_lista_simulacoes_open_system as R

# ============================================================
# CONFIG
# ============================================================
DST_ROOT = "results/coherence_article_corrected"
LOSS_BLOCK = "specific_parameters"   # loss_args -> kappa=1e-1, gamma=0, gamma_phi=1e-2 (dois decaimentos)
SCAN_LIMIT = None                    # None = varredura completa (pesado). Ex.: 3 p/ teste rapido.
SKIP_EXISTING = True                 # pula caso ja finalizado (arquivo DONE.txt)
# ============================================================

N, Nb, ALPHA = R.N, R.Nb, R.ALPHA


# Casos do artigo: DOIS decaimentos + varredura COMPLETA (identica aos blocos only_*).
def build_article_cases():
    return [
        {
            "case_id": "article_coherence_gauss_zeta",
            "block": LOSS_BLOCK, "resource": "coherence", "coupling": "gauss",
            "scan_name": "epsilon", "scan_display_name": "zeta",
            "scan_values": np.linspace(6.0, 10.0, 100, endpoint=True),
            "fixed_args": {"T": 25.0, "sigma": -1.0},
            "tmax": 50.0,
            "note": "Article coherence (kappa=1e-1, gamma_phi=1e-2), gaussian varying width; av index=50.",
        },
        {
            "case_id": "article_coherence_gauss_T",
            "block": LOSS_BLOCK, "resource": "coherence", "coupling": "gauss",
            "scan_name": "T", "scan_display_name": "T",
            "scan_values": np.linspace(15.0, 35.0, 100, endpoint=True),
            "fixed_args": {"epsilon": 8.0, "sigma": -1.0},
            "tmax": 50.0,
            "note": "Article coherence (kappa=1e-1, gamma_phi=1e-2), gaussian varying peak time; av index=50.",
        },
        {
            "case_id": "article_coherence_cos_omega",
            "block": LOSS_BLOCK, "resource": "coherence", "coupling": "cos",
            "scan_name": "w", "scan_display_name": "omega",
            "scan_values": np.linspace(0.0, np.pi / 5.0, 200, endpoint=True),
            "fixed_args": {"phi": 0.0},
            "tmax": 50.0,
            "note": "Article coherence (kappa=1e-1, gamma_phi=1e-2), cosine varying frequency; min index=50, av index=100.",
        },
    ]


def coherence_correct(states, base=2):
    """Entropia relativa de coerencia da Eq.(7): S(rho_diag) - S(rho), base 2 (SEM rho^2)."""
    out = np.zeros(len(states))
    for i, st in enumerate(states):
        rhoQ = st.ptrace(0)
        out[i] = qt.entropy_vn(diagonaliza(rhoQ), base=base) - qt.entropy_vn(rhoQ, base=base)
    return out


def summarize(t, y, tag):
    y = np.asarray(y, float)
    return {
        f"{tag}_initial": float(y[0]),
        f"{tag}_max": float(np.nanmax(y)),
        f"{tag}_mean": float(np.nanmean(y)),
        f"{tag}_final": float(y[-1]),
        f"{tag}_time_at_max": float(t[int(np.nanargmax(y))]),
    }


def run_case(case, dst_root):
    case_dir = os.path.join(dst_root, case["case_id"])
    os.makedirs(case_dir, exist_ok=True)
    done = os.path.join(case_dir, "DONE.txt")
    if SKIP_EXISTING and os.path.exists(done):
        print(f"[skip] {case['case_id']}")
        return

    t = R.time_grid(case["tmax"])
    args0 = R.base_args(case["block"], case["coupling"])   # inclui loss_args(specific_parameters)
    args0.update(case["fixed_args"])

    sz, sp, sm, b, nb, I = get_operators(N, Nb)
    c_ops = get_collapse(args0, sm, sz, b)
    state0 = R.initial_state("coherence", N, Nb, ALPHA)

    scan_values = np.asarray(case["scan_values"], float)
    if SCAN_LIMIT is not None:
        scan_values = scan_values[:int(SCAN_LIMIT)]

    meta = {k: case[k] for k in ("case_id", "block", "resource", "coupling",
                                 "scan_name", "scan_display_name", "tmax", "note")}
    meta.update({"N": int(N), "Nb": int(Nb), "alpha": float(ALPHA),
                 "base_args": dict(args0), "n_scan": int(len(scan_values))})
    with open(os.path.join(case_dir, "case_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(case_dir, "t.npy"), t)
    np.save(os.path.join(case_dir, "scan_values.npy"), scan_values)

    print(f"\n[{case['case_id']}]  n_scan={len(scan_values)}  tmax={case['tmax']}  "
          f"kappa={args0.get('kappa')} gamma_phi={args0.get('gamma_phi')}")

    rows = []

    # ---- acoplamento CONSTANTE (baseline), sistema aberto: usa h_closed ----
    solc = solve(h_closed(args0, b, sp, sm), state0, t, c_ops, [], args0)
    C_old = coherence_old(solc.states)
    C_new = coherence_correct(solc.states)
    np.save(os.path.join(case_dir, "const_aberto.npy"), np.array(C_old))
    np.save(os.path.join(case_dir, "const_aberto_correct.npy"), np.array(C_new))
    rows.append({"label": "const_aberto", "scan_value": np.nan,
                 **summarize(t, C_old, "old"), **summarize(t, C_new, "correct")})

    # ---- varredura, sistema aberto: usa h_open (time-dependent) ----
    var_old, var_new, args_rows = [], [], []
    for i, val in enumerate(tqdm(scan_values, desc=case["case_id"])):
        args = dict(args0)
        args[case["scan_name"]] = float(val)
        sol = solve(h_open(b, sp, sm), state0, t, c_ops, [], args)
        co = coherence_old(sol.states)
        cn = coherence_correct(sol.states)
        var_old.append(co); var_new.append(cn)
        args_rows.append({**args, "label": f"var_aberto_{i:04d}",
                          "scan_name": case["scan_name"], "scan_value": float(val)})
        rows.append({"label": f"var_aberto_{i:04d}", "scan_value": float(val),
                     **summarize(t, co, "old"), **summarize(t, cn, "correct")})

    np.save(os.path.join(case_dir, "var_aberto.npy"), np.array(var_old, dtype=float))
    np.save(os.path.join(case_dir, "var_aberto_correct.npy"), np.array(var_new, dtype=float))
    pd.DataFrame(args_rows).to_csv(os.path.join(case_dir, "args_per_scan.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(case_dir, "summary.csv"), index=False)
    with open(done, "w") as f:
        f.write("done\n")


def main():
    cases = build_article_cases()
    os.makedirs(DST_ROOT, exist_ok=True)
    print(f"Casos de coerencia (ARTIGO, dois decaimentos, varredura completa): {len(cases)}")
    print(f"loss_args({LOSS_BLOCK}) = {R.loss_args(LOSS_BLOCK)}")
    print(f"alpha={ALPHA:.4f}  Nb={Nb}   -> salvando em {DST_ROOT}/")

    for case in cases:
        run_case(case, DST_ROOT)

    print(f"\nTudo salvo em {DST_ROOT}/  (var_aberto_correct.npy = Eq.7 corrigida = usar no denso).")


if __name__ == "__main__":
    main()
