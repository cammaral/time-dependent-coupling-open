# -*- coding: utf-8 -*-
"""
recompute_correct_coherence.py

Recalcula a COERENCIA CORRETA (entropia relativa de coerencia, Eq. 7, base 2)
para TODOS os casos de coerencia ja rodados por run_lista_simulacoes_open_system.py
— incluindo os 3 blocos de dissipacao:
    only_dephasing        (kappa=0,    gamma_phi=1e-2)
    only_cavity_damping   (kappa=1e-1, gamma_phi=0)
    specific_parameters   (kappa=1e-1, gamma_phi=1e-2)

POR QUE NAO PRECISA RE-SIMULAR:
  O run salvou, em *_observables.csv, os valores esperados do qubit
  <X>, <Y>, <Z> em cada instante. Para um qubit, rho_Q = 1/2 (I + <X>X + <Y>Y + <Z>Z),
  entao a coerencia correta e uma funcao exata desses tres numeros:
        r  = sqrt(<X>^2 + <Y>^2 + <Z>^2)
        C  = H2((1+<Z>)/2) - H2((1+r)/2)     [base 2, na base sigma_z {|e>,|g>}]
  Isso usa EXATAMENTE os mesmos estados do run original -> maxima fidelidade.

O que a funcao coerence() do repo fazia de errado (para comparacao):
  usava rho_Q^2 (linha 'estado = estado*estado.dag()') e base e.

Como rodar (na raiz do repo):
    python recompute_correct_coherence.py

Saida: results/correct_coerence/<case_id>/<label>.npy  (+ var_aberto.npy, summary.csv,
        comparacao correta-vs-original) e um indice global comparison_index.csv.
"""
import os
import re
import glob
import json
import shutil
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================
SRC_ROOT = "results/lista_simulacoes_open_system"   # onde estao os runs originais
DST_ROOT = "results/correct_coerence"               # <<< pasta pedida
# ============================================================


def find_latest_run(src_root):
    cands = sorted(glob.glob(os.path.join(src_root, "open_system_sims*")))
    if not cands:
        raise SystemExit(f"Nao achei nenhum run em {src_root}. "
                         f"Rode antes: python run_lista_simulacoes_open_system.py")
    # ordena pelo numero no fim do nome
    def k(p):
        m = re.search(r"(\d+)$", os.path.basename(p))
        return int(m.group(1)) if m else -1
    return sorted(cands, key=k)[-1]


def H2(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def coherence_correct_from_bloch(x, y, z):
    """Entropia relativa de coerencia (Eq. 7, base 2) a partir do vetor de Bloch."""
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    r = np.clip(np.sqrt(x**2 + y**2 + z**2), 0.0, 1.0)
    return H2((1 + z) / 2.0) - H2((1 + r) / 2.0)


def summarize(t, y):
    y = np.asarray(y, float)
    return {
        "metric_initial": float(y[0]),
        "metric_max": float(np.nanmax(y)),
        "metric_mean": float(np.nanmean(y)),
        "metric_final": float(y[-1]),
        "time_at_max": float(t[int(np.nanargmax(y))]),
    }


def var_index(label):
    m = re.search(r"var_aberto_(\d+)_", label)
    return int(m.group(1)) if m else 10**9


def process_case(case_dir, dst_root, index_rows):
    case_id = os.path.basename(case_dir)
    dst = os.path.join(dst_root, case_id)
    os.makedirs(dst, exist_ok=True)

    # copia metadados/eixos para a pasta corrigida
    for fn in ["t.npy", "scan_values.npy", "case_metadata.json", "args_per_scan.csv"]:
        src_fn = os.path.join(case_dir, fn)
        if os.path.exists(src_fn):
            shutil.copy2(src_fn, os.path.join(dst, fn))

    t = np.load(os.path.join(case_dir, "t.npy"))

    obs_files = sorted(glob.glob(os.path.join(case_dir, "*_observables.csv")))
    summary_rows = []
    var_labels = []

    for obs in obs_files:
        label = os.path.basename(obs)[:-len("_observables.csv")]
        df = pd.read_csv(obs)
        C_ok = coherence_correct_from_bloch(
            df["expect_X_qubit"].values,
            df["expect_Y_qubit"].values,
            df["expect_Z_qubit"].values,
        )
        np.save(os.path.join(dst, f"{label}.npy"), C_ok)

        # comparacao com o valor original (bugado) se existir
        buggy_path = os.path.join(case_dir, f"{label}.npy")
        cmp = {}
        if os.path.exists(buggy_path):
            C_bug = np.load(buggy_path)
            n = min(len(C_bug), len(C_ok))
            a, bb = C_ok[:n], C_bug[:n]
            na = a / (np.nanmax(a) if np.nanmax(a) > 0 else 1)
            nb = bb / (np.nanmax(bb) if np.nanmax(bb) > 0 else 1)
            cmp = {
                "C0_correct": float(a[0]), "C0_original": float(bb[0]),
                "max_abs_diff": float(np.nanmax(np.abs(a - bb))),
                "max_norm_diff": float(np.nanmax(np.abs(na - nb))),
            }

        row = {"case_id": case_id, "label": label, **summarize(t, C_ok), **cmp}
        summary_rows.append(row)
        index_rows.append(row)
        if label.startswith("var_aberto_"):
            var_labels.append(label)

    # reconstroi var_aberto.npy (empilhado na ordem do scan)
    var_labels.sort(key=var_index)
    if var_labels:
        stack = np.array([np.load(os.path.join(dst, f"{lb}.npy")) for lb in var_labels], dtype=float)
        np.save(os.path.join(dst, "var_aberto.npy"), stack)

    pd.DataFrame(summary_rows).to_csv(os.path.join(dst, "summary.csv"), index=False)
    return len(summary_rows)


def main():
    run_dir = find_latest_run(SRC_ROOT)
    print(f"Run original: {run_dir}")
    os.makedirs(DST_ROOT, exist_ok=True)

    case_dirs = sorted(
        d for d in glob.glob(os.path.join(run_dir, "*"))
        if os.path.isdir(d) and "coherence" in os.path.basename(d)
        and os.path.exists(os.path.join(d, "t.npy"))
    )
    if not case_dirs:
        raise SystemExit("Nenhum caso de coerencia encontrado no run.")

    print(f"Casos de coerencia encontrados: {len(case_dirs)}")
    index_rows = []
    total = 0
    for cd in case_dirs:
        n = process_case(cd, DST_ROOT, index_rows)
        total += n
        print(f"  [ok] {os.path.basename(cd):55s} ({n} curvas)")

    idx = pd.DataFrame(index_rows)
    idx.to_csv(os.path.join(DST_ROOT, "comparison_index.csv"), index=False)

    print("\n" + "=" * 78)
    print(f"Curvas recalculadas (corretas): {total}")
    if "max_norm_diff" in idx:
        print(f"Maior diferenca de FORMA (normalizada) entre correta e original: "
              f"{idx['max_norm_diff'].max():.4f}")
        print(f"Maior diferenca ABSOLUTA de valor:                              "
              f"{idx['max_abs_diff'].max():.4f}")
    print(f"Tudo salvo em: {DST_ROOT}/")
    print(f"Indice comparativo: {os.path.join(DST_ROOT, 'comparison_index.csv')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
