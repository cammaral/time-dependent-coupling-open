# -*- coding: utf-8 -*-
"""
run_coherence_old_and_correct.py

Roda a DINAMICA e calcula a COERENCIA na hora (gera estados + coerencia juntos,
salva SO a coerencia), nas DUAS formas, para comparar:

  * "old"     -> exatamente a funcao coerence() do repo  (usa rho_Q^2, base e)
  * "correct" -> entropia relativa de coerencia da Eq.(7) (rho_Q, base 2)

FIDELIDADE DE PARAMETROS:
  Os casos (blocos de dissipacao, varreduras, estado inicial, N, Nb, alpha, grade
  de tempo, acoplamentos) sao IMPORTADOS de run_lista_simulacoes_open_system.py
  via build_cases(). Ou seja, sao EXATAMENTE os mesmos que voce ja usava:
      only_dephasing        (kappa=0,    gamma_phi=1e-2)
      only_cavity_damping   (kappa=1e-1, gamma_phi=0)
      specific_parameters   (kappa=1e-1, gamma_phi=1e-2)
  estado inicial (|e>+|g>)/sqrt2 (x) |alpha=sqrt(5)>, Nb=45, t ate 50.

Como rodar (na raiz do repo, no .venv com qutip):
    python run_coherence_old_and_correct.py

Para um teste rapido, defina SCAN_LIMIT = 3 (usa so 3 pontos por varredura).

Saida: results/coherence_old_vs_correct/<case_id>/ com
    const_aberto.npy            (old)      const_aberto_correct.npy     (correct)
    var_aberto.npy              (old)      var_aberto_correct.npy       (correct)
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
from quantum.non_classicality import coerence as coherence_old   # <<< funcao ORIGINAL, intacta
from utils.utils import diagonaliza

# --- reaproveita EXATAMENTE os casos/parametros do run recente ---
import run_lista_simulacoes_open_system as R

# ============================================================
# CONFIG
# ============================================================
DST_ROOT = "results/coherence_old_vs_correct"
BLOCKS = ["only_dephasing", "only_cavity_damping", "specific_parameters"]
SCAN_LIMIT = None      # None = varredura completa (pesado). Ex.: 3 p/ testar rapido.
SKIP_EXISTING = True   # pula caso ja finalizado (arquivo DONE.txt)
# ============================================================

N, Nb, ALPHA = R.N, R.Nb, R.ALPHA


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
    args0 = R.base_args(case["block"], case["coupling"])
    args0.update(case["fixed_args"])

    sz, sp, sm, b, nb, I = get_operators(N, Nb)
    c_ops = get_collapse(args0, sm, sz, b)
    state0 = R.initial_state("coherence", N, Nb, ALPHA)

    scan_values = np.asarray(case["scan_values"], float)
    if SCAN_LIMIT is not None:
        scan_values = scan_values[:int(SCAN_LIMIT)]

    # metadados (mesmos campos-chave do run original)
    meta = {k: case[k] for k in ("case_id", "block", "resource", "coupling",
                                 "scan_name", "scan_display_name", "tmax", "note")}
    meta.update({"N": int(N), "Nb": int(Nb), "alpha": float(ALPHA),
                 "base_args": dict(args0), "n_scan": int(len(scan_values))})
    with open(os.path.join(case_dir, "case_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(case_dir, "t.npy"), t)
    np.save(os.path.join(case_dir, "scan_values.npy"), scan_values)

    print(f"\n[{case['case_id']}]  n_scan={len(scan_values)}  tmax={case['tmax']}")

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
    # restringe build_cases() a coerencia e aos blocos escolhidos
    R.RESOURCES = ["coherence"]
    R.RUN_BLOCKS = BLOCKS
    R.SCAN_LIMIT = None            # o corte fino fica por conta do SCAN_LIMIT deste script
    cases = [c for c in R.build_cases() if c["resource"] == "coherence"]

    os.makedirs(DST_ROOT, exist_ok=True)
    print(f"Casos de coerencia: {len(cases)}  (blocos: {BLOCKS})")
    print(f"alpha={ALPHA:.4f}  Nb={Nb}   -> salvando em {DST_ROOT}/")

    for case in cases:
        run_case(case, DST_ROOT)

    print(f"\nTudo salvo em {DST_ROOT}/  (arquivos *_correct.npy = Eq.7; os outros = coerence() original)")


if __name__ == "__main__":
    main()
