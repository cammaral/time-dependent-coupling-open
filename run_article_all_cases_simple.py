"""
run_article_all_cases_simple.py

Roda os casos do artigo/projeto para:
  - coerência atômica
  - quantum magic atômica via SRE-2
  - negatividade da Wigner do campo
  - emaranhamento átomo-campo

A ideia é manter a estrutura simples dos scripts atuais:
  - usa os módulos quantum.* já existentes no projeto
  - usa apenas 3 pontos por varredura: min, av, max
  - não gera vídeo
  - salva métrica principal + valores esperados + g(t)

Coloque este arquivo na raiz do projeto, no mesmo nível da pasta quantum/.
Rode com:
    python run_article_all_cases_simple.py
"""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import qutip as qt
from tqdm import tqdm

from quantum.hamiltonian import g_t, h_closed, h_open
from quantum.operators import get_operators, get_collapse
from quantum.run import solve
from quantum.non_classicality import coerence, entanglement, wigner_negativity


# ==========================================================
# CONFIGURAÇÃO GERAL
# ==========================================================
ROOT = "results/article_all_cases"
N = 2
Nb = 45

# ATENÇÃO:
# O texto do artigo menciona alpha = 1 e perdas 1e-2 / 1e-3,
# mas os códigos do projeto que geram as figuras usam alpha=sqrt(5),
# kappa=1e-1 e gamma_phi=1e-2. Mantive o padrão do projeto.
ALPHA = np.sqrt(5)
KAPPA = 1e-1
GAMMA = 0.0
GAMMA_PHI = 1e-2

X_MIN, X_MAX, NX = -7.5, 7.5, 200
P_MIN, P_MAX, NP = -7.5, 7.5, 200

RUN_CLOSED = False  # deixe True se também quiser salvar const.npy e var.npy fechados


# ==========================================================
# FUNÇÕES PEQUENAS
# ==========================================================
def make_exp_folder(base_name="article_all_cases", root="results"):
    os.makedirs(root, exist_ok=True)
    k = 1
    while True:
        folder = os.path.join(root, f"{base_name}{k}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        k += 1


def time_grid(tmax):
    """Mesma lógica dos scripts: grade densa no começo e depois até tmax."""
    return np.concatenate([
        np.linspace(0, 1, 100, endpoint=False),
        np.linspace(1, tmax, 200 if tmax == 50 else 100, endpoint=True),
    ])


def selected_rows(values, labels=("min", "av", "max"), display_values=None):
    values = np.asarray(values, dtype=float)
    if display_values is None:
        display_values = values
    rows = []
    for idx, label in enumerate(labels):
        rows.append({
            "label": label,
            "index": int(idx),
            "value": float(values[idx]),
            "display_value": float(display_values[idx]),
        })
    return rows


def eval_g_single(tt, args):
    try:
        return float(g_t(float(tt), args))
    except TypeError:
        return float(g_t(args, float(tt)))


def coupling_curve(t, args, constant=False):
    if constant:
        return np.full_like(t, float(args.get("g0", 1.0)), dtype=float)
    return np.array([eval_g_single(tt, args) for tt in t], dtype=float)


def expect_real(op, state):
    val = qt.expect(op, state)
    val = np.real_if_close(val)
    return float(np.real(val))


def save_observables(folder, label, states, t, args, sz, nb, constant=False):
    exp_N = np.array([expect_real(nb, s) for s in states], dtype=float)
    exp_N2 = np.array([expect_real(nb ** 2, s) for s in states], dtype=float)
    exp_Z = np.array([expect_real(sz, s) for s in states], dtype=float)
    var_N = exp_N2 - exp_N ** 2
    g_vals = coupling_curve(t, args, constant=constant)

    np.save(os.path.join(folder, f"{label}_expect_N.npy"), exp_N)
    np.save(os.path.join(folder, f"{label}_expect_N2.npy"), exp_N2)
    np.save(os.path.join(folder, f"{label}_var_N.npy"), var_N)
    np.save(os.path.join(folder, f"{label}_expect_Z.npy"), exp_Z)
    np.save(os.path.join(folder, f"{label}_g_t.npy"), g_vals)

    df = pd.DataFrame({
        "time": t,
        "g_t": g_vals,
        "expect_N": exp_N,
        "expect_N2": exp_N2,
        "var_N": var_N,
        "expect_Z": exp_Z,
    })
    df.to_csv(os.path.join(folder, f"{label}_observables.csv"), index=False)
    return df


def initial_state(kind, N, Nb, alpha):
    e = qt.basis(N, 0)
    g = qt.basis(N, 1)
    coh_p = qt.coherent(Nb, alpha)
    coh_m = qt.coherent(Nb, -alpha)

    if kind == "coherence":
        return qt.tensor(e + g, coh_p).unit()

    if kind == "magic":
        theta = np.arccos(1 / np.sqrt(3))
        qubit = np.cos(theta / 2) * g + np.exp(1j * np.pi / 4) * np.sin(theta / 2) * e
        return qt.tensor(qubit, coh_p).unit()

    if kind == "wigner":
        return qt.tensor(e, coh_p + coh_m).unit()

    if kind == "entanglement":
        # Mantido como no código do projeto.
        return (qt.tensor(e, coh_m) + qt.tensor(g, coh_p)).unit()

    raise ValueError(f"Estado inicial desconhecido: {kind}")


def atomic_sre2_single_state(state):
    rho = state.ptrace(0)
    if rho.isket:
        rho = rho.proj()

    paulis = [qt.qeye(2), qt.sigmay(), qt.sigmax(), qt.sigmaz()]
    vals = np.array([qt.expect(P, rho) for P in paulis], dtype=complex)

    s2 = float(np.sum(np.abs(vals) ** 2))
    s4 = float(np.sum(np.abs(vals) ** 4))

    if s2 <= 0 or s4 <= 0:
        return 0.0

    return float(-np.log(s4 / s2))


def atomic_magic_sre2(states):
    return np.array([atomic_sre2_single_state(s) for s in states], dtype=float)


def compute_metric(resource, states, xvec=None, pvec=None):
    if resource == "coherence":
        return np.array(coerence(states), dtype=float)
    if resource == "magic":
        return atomic_magic_sre2(states)
    if resource == "wigner":
        return np.array(wigner_negativity(states, xvec, pvec), dtype=float)
    if resource == "entanglement":
        return np.array(entanglement(states), dtype=float)
    raise ValueError(f"Recurso desconhecido: {resource}")


def base_args(coupling):
    args = {
        "g0": 1.0,
        "eta": 1.0,
        "kappa": KAPPA,
        "gamma": GAMMA,
        "gamma_phi": GAMMA_PHI,
        "coupling": coupling,
    }
    if coupling == "cos":
        args.update({"w": 0.0, "phi": 0.0})
    if coupling == "gauss":
        args.update({"sigma": -1.0, "epsilon": None, "T": None})
    return args


def summarize_metric(t, y):
    y = np.asarray(y, dtype=float)
    return {
        "metric_initial": float(y[0]),
        "metric_max": float(np.nanmax(y)),
        "metric_mean": float(np.nanmean(y)),
        "metric_final": float(y[-1]),
        "time_at_max": float(t[int(np.nanargmax(y))]),
    }


# ==========================================================
# MAPA DOS CASOS DO ARTIGO + PROJETO
# ==========================================================
def build_cases():
    omega_units = np.array([6.0, 8.0, 10.0])
    omega_50 = omega_units * np.pi / 50.0

    wigner_wmax = ((50.0 / 15.0) * np.pi) / 5.0
    wigner_omega = np.array([wigner_wmax / 4.0, wigner_wmax / 2.0, wigner_wmax])

    cases = [
        # ------------------------
        # Coherence: Figs. 1-3
        # ------------------------
        {
            "case_id": "fig1_coherence_gauss_epsilon",
            "article_fig": "Fig. 1",
            "resource": "coherence",
            "state_kind": "coherence",
            "coupling": "gauss",
            "scan_name": "epsilon",
            "scan_display_name": "zeta",
            "scan_values": np.array([6.0, 8.0, 10.0]),
            "scan_display_values": np.array([6.0, 8.0, 10.0]),
            "fixed_args": {"T": 25.0, "sigma": -1.0},
            "tmax": 50,
            "metric_label": "C_q(t)",
        },
        {
            "case_id": "fig2_coherence_gauss_T",
            "article_fig": "Fig. 2",
            "resource": "coherence",
            "state_kind": "coherence",
            "coupling": "gauss",
            "scan_name": "T",
            "scan_display_name": "T",
            "scan_values": np.array([15.0, 25.0, 35.0]),
            "scan_display_values": np.array([15.0, 25.0, 35.0]),
            "fixed_args": {"epsilon": 8.0, "sigma": -1.0},
            "tmax": 50,
            "metric_label": "C_q(t)",
        },
        {
            "case_id": "fig3_coherence_cos_omega",
            "article_fig": "Fig. 3",
            "resource": "coherence",
            "state_kind": "coherence",
            "coupling": "cos",
            "scan_name": "w",
            "scan_display_name": "omega_units",
            "scan_values": omega_50,
            "scan_display_values": omega_units,
            "fixed_args": {"phi": 0.0},
            "tmax": 50,
            "metric_label": "C_q(t)",
        },

        # ------------------------
        # Quantum magic: usa os mesmos blocos de acoplamento da coerência
        # ------------------------
        {
            "case_id": "magic_gauss_epsilon",
            "article_fig": "Magic / Eq. 9 + Fig. 1 parameters",
            "resource": "magic",
            "state_kind": "magic",
            "coupling": "gauss",
            "scan_name": "epsilon",
            "scan_display_name": "zeta",
            "scan_values": np.array([6.0, 8.0, 10.0]),
            "scan_display_values": np.array([6.0, 8.0, 10.0]),
            "fixed_args": {"T": 25.0, "sigma": -1.0},
            "tmax": 50,
            "metric_label": "M_2(t)",
        },
        {
            "case_id": "magic_gauss_T",
            "article_fig": "Magic / Eq. 9 + Fig. 2 parameters",
            "resource": "magic",
            "state_kind": "magic",
            "coupling": "gauss",
            "scan_name": "T",
            "scan_display_name": "T",
            "scan_values": np.array([15.0, 25.0, 35.0]),
            "scan_display_values": np.array([15.0, 25.0, 35.0]),
            "fixed_args": {"epsilon": 8.0, "sigma": -1.0},
            "tmax": 50,
            "metric_label": "M_2(t)",
        },
        {
            "case_id": "magic_cos_omega",
            "article_fig": "Magic / Eq. 9 + Fig. 3 parameters",
            "resource": "magic",
            "state_kind": "magic",
            "coupling": "cos",
            "scan_name": "w",
            "scan_display_name": "omega_units",
            "scan_values": omega_50,
            "scan_display_values": omega_units,
            "fixed_args": {"phi": 0.0},
            "tmax": 50,
            "metric_label": "M_2(t)",
        },

        # ------------------------
        # Wigner: Figs. 4-6
        # ------------------------
        {
            "case_id": "fig4_wigner_gauss_epsilon",
            "article_fig": "Fig. 4",
            "resource": "wigner",
            "state_kind": "wigner",
            "coupling": "gauss",
            "scan_name": "epsilon",
            "scan_display_name": "zeta",
            "scan_values": np.array([1.8, 2.4, 3.0]),
            "scan_display_values": np.array([1.8, 2.4, 3.0]),
            "fixed_args": {"T": 7.5, "sigma": -1.0},
            "tmax": 15,
            "metric_label": "N_W(t)",
        },
        {
            "case_id": "fig5_wigner_gauss_T",
            "article_fig": "Fig. 5",
            "resource": "wigner",
            "state_kind": "wigner",
            "coupling": "gauss",
            "scan_name": "T",
            "scan_display_name": "T",
            "scan_values": np.array([4.5, 7.5, 10.5]),
            "scan_display_values": np.array([4.5, 7.5, 10.5]),
            "fixed_args": {"epsilon": 2.4, "sigma": -1.0},
            "tmax": 15,
            "metric_label": "N_W(t)",
        },
        {
            "case_id": "fig6_wigner_cos_omega",
            "article_fig": "Fig. 6",
            "resource": "wigner",
            "state_kind": "wigner",
            "coupling": "cos",
            "scan_name": "w",
            "scan_display_name": "omega",
            "scan_values": wigner_omega,
            "scan_display_values": wigner_omega,
            "fixed_args": {"phi": 0.0},
            "tmax": 15,
            "metric_label": "N_W(t)",
        },

        # ------------------------
        # Entanglement: Figs. 7-9
        # ------------------------
        {
            "case_id": "fig7_entanglement_gauss_epsilon",
            "article_fig": "Fig. 7",
            "resource": "entanglement",
            "state_kind": "entanglement",
            "coupling": "gauss",
            "scan_name": "epsilon",
            "scan_display_name": "zeta",
            "scan_values": np.array([6.0, 8.0, 10.0]),
            "scan_display_values": np.array([6.0, 8.0, 10.0]),
            "fixed_args": {"T": 25.0, "sigma": -1.0},
            "tmax": 50,
            "metric_label": "E_N(t)",
        },
        {
            "case_id": "fig8_entanglement_gauss_T",
            "article_fig": "Fig. 8",
            "resource": "entanglement",
            "state_kind": "entanglement",
            "coupling": "gauss",
            "scan_name": "T",
            "scan_display_name": "T",
            "scan_values": np.array([15.0, 25.0, 35.0]),
            "scan_display_values": np.array([15.0, 25.0, 35.0]),
            "fixed_args": {"epsilon": 8.0, "sigma": -1.0},
            "tmax": 50,
            "metric_label": "E_N(t)",
        },
        {
            "case_id": "fig9_entanglement_cos_omega",
            "article_fig": "Fig. 9",
            "resource": "entanglement",
            "state_kind": "entanglement",
            "coupling": "cos",
            "scan_name": "w",
            "scan_display_name": "omega_units",
            "scan_values": omega_50,
            "scan_display_values": omega_units,
            "fixed_args": {"phi": 0.0},
            "tmax": 50,
            "metric_label": "E_N(t)",
        },
    ]
    return cases


def run_case(case, root_dir):
    case_dir = os.path.join(root_dir, case["case_id"])
    os.makedirs(case_dir, exist_ok=True)

    t = time_grid(case["tmax"])
    xvec = np.linspace(X_MIN, X_MAX, NX)
    pvec = np.linspace(P_MIN, P_MAX, NP)

    args0 = base_args(case["coupling"])
    args0.update(case["fixed_args"])

    rows = selected_rows(
        case["scan_values"],
        labels=("min", "av", "max"),
        display_values=case["scan_display_values"],
    )

    sz, sp, sm, b, nb, I = get_operators(N, Nb)
    obs_list = [sz, nb, nb ** 2]
    c_ops = get_collapse(args0, sm, sz, b)
    state0 = initial_state(case["state_kind"], N, Nb, ALPHA)

    # --------------------------
    # constante aberto
    # --------------------------
    print(f"\n[{case['case_id']}] constante aberto")
    H_const = h_closed(args0, b, sp, sm)
    sol_const_aberto = solve(H_const, state0, t, c_ops, obs_list, args0)
    metric_const_aberto = compute_metric(case["resource"], sol_const_aberto.states, xvec=xvec, pvec=pvec)
    np.save(os.path.join(case_dir, "const_aberto.npy"), metric_const_aberto)
    save_observables(case_dir, "const_aberto", sol_const_aberto.states, t, args0, sz, nb, constant=True)

    if RUN_CLOSED:
        sol_const = solve(H_const, state0, t, None, obs_list, args0, open=False)
        metric_const = compute_metric(case["resource"], sol_const.states, xvec=xvec, pvec=pvec)
        np.save(os.path.join(case_dir, "const.npy"), metric_const)
        save_observables(case_dir, "const", sol_const.states, t, args0, sz, nb, constant=True)

    # --------------------------
    # variável aberto: 3 pontos
    # --------------------------
    var_metrics = []
    var_g = []
    args_used = []
    summary_rows = [{
        "label": "const_aberto",
        "case_id": case["case_id"],
        "scan_name": "const",
        "scan_value": np.nan,
        "display_value": np.nan,
        **summarize_metric(t, metric_const_aberto),
    }]

    for row in tqdm(rows, desc=case["case_id"]):
        args = dict(args0)
        args[case["scan_name"]] = float(row["value"])

        H_var = h_open(b, sp, sm)
        sol_var_aberto = solve(H_var, state0, t, c_ops, obs_list, args)
        metric = compute_metric(case["resource"], sol_var_aberto.states, xvec=xvec, pvec=pvec)

        label = f"var_aberto_{row['label']}"
        np.save(os.path.join(case_dir, f"{label}.npy"), metric)
        save_observables(case_dir, label, sol_var_aberto.states, t, args, sz, nb, constant=False)

        var_metrics.append(metric)
        var_g.append(coupling_curve(t, args, constant=False))
        args_used.append({
            **args,
            "label": row["label"],
            "index": row["index"],
            "scan_name": case["scan_name"],
            "scan_value": row["value"],
            "scan_display_name": case["scan_display_name"],
            "scan_display_value": row["display_value"],
        })
        summary_rows.append({
            "label": label,
            "case_id": case["case_id"],
            "scan_name": case["scan_name"],
            "scan_value": float(row["value"]),
            "display_value": float(row["display_value"]),
            **summarize_metric(t, metric),
        })

    var_metrics = np.array(var_metrics, dtype=float)
    var_g = np.array(var_g, dtype=float)

    np.save(os.path.join(case_dir, "var_aberto.npy"), var_metrics)
    np.save(os.path.join(case_dir, "g_var_aberto.npy"), var_g)
    np.save(os.path.join(case_dir, "t.npy"), t)
    np.save(os.path.join(case_dir, "scan_values.npy"), np.asarray(case["scan_values"], dtype=float))
    np.save(os.path.join(case_dir, "scan_display_values.npy"), np.asarray(case["scan_display_values"], dtype=float))

    pd.DataFrame(rows).to_csv(os.path.join(case_dir, "scan_points_used.csv"), index=False)
    pd.DataFrame(args_used).to_csv(os.path.join(case_dir, "args_per_scan_used.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(case_dir, "summary.csv"), index=False)

    metadata = dict(case)
    metadata["scan_values"] = list(np.asarray(case["scan_values"], dtype=float))
    metadata["scan_display_values"] = list(np.asarray(case["scan_display_values"], dtype=float))
    metadata["fixed_args"] = dict(case["fixed_args"])
    metadata["alpha_used_in_project_code"] = float(ALPHA)
    metadata["N"] = int(N)
    metadata["Nb"] = int(Nb)
    metadata["kappa"] = float(KAPPA)
    metadata["gamma"] = float(GAMMA)
    metadata["gamma_phi"] = float(GAMMA_PHI)
    metadata["tmax"] = float(case["tmax"])
    metadata["n_times"] = int(len(t))
    metadata["run_closed"] = bool(RUN_CLOSED)

    with open(os.path.join(case_dir, "case_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def main():
    root_dir = make_exp_folder(base_name="article_all_cases", root=ROOT)
    cases = build_cases()

    run_info = {
        "created_at": datetime.now().isoformat(),
        "root_dir": root_dir,
        "notes": [
            "Casos escolhidos a partir das figuras do artigo e dos blocos equivalentes do projeto.",
            "Cada varredura usa apenas min, av, max.",
            "Nao gera videos; salva metricas, observaveis e g(t).",
            "ALPHA, KAPPA e GAMMA_PHI seguem os scripts do projeto, nao necessariamente o texto atual do artigo.",
        ],
        "alpha": float(ALPHA),
        "kappa": float(KAPPA),
        "gamma": float(GAMMA),
        "gamma_phi": float(GAMMA_PHI),
        "N": int(N),
        "Nb": int(Nb),
    }

    with open(os.path.join(root_dir, "run_info.json"), "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2, ensure_ascii=False)

    map_rows = []
    for case in cases:
        metadata = run_case(case, root_dir)
        map_rows.append({
            "case_id": metadata["case_id"],
            "article_fig": metadata["article_fig"],
            "resource": metadata["resource"],
            "state_kind": metadata["state_kind"],
            "coupling": metadata["coupling"],
            "scan_name": metadata["scan_name"],
            "scan_display_name": metadata["scan_display_name"],
            "fixed_args": json.dumps(metadata["fixed_args"], ensure_ascii=False),
            "tmax": metadata["tmax"],
            "metric_label": metadata["metric_label"],
            "case_dir": metadata["case_id"],
        })

    pd.DataFrame(map_rows).to_csv(os.path.join(root_dir, "case_map.csv"), index=False)
    print(f"\n✅ Tudo salvo em: {root_dir}")


if __name__ == "__main__":
    main()
