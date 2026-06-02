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


# ============================================================
# UTILIDADES DE I/O
# ============================================================
def make_exp_folder(base_name="expected_values_exp", root="results/expected_values"):
    os.makedirs(root, exist_ok=True)
    k = 1
    while True:
        folder = os.path.join(root, f"{base_name}{k}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        k += 1


def selected_w_points(w_list):
    """
    Mantém o mesmo padrão do arquivo wigner_all_time_cos_var_w.py:
    - omega_min = omega_max / 4
    - omega_av  = omega_max / 2
    - omega_max

    Aqui w_list já contém apenas esses três valores.
    """
    labels = ["omega_min", "omega_av", "omega_max"]
    rows = []
    for idx, name in enumerate(labels):
        rows.append({
            "w_label": name,
            "w_index": int(idx),
            "w_value": float(w_list[idx]),
        })
    return rows


def eval_g_single(tt, args):
    """
    Wrapper pequeno para evitar erro caso g_t no projeto tenha assinatura
    g_t(t, args) ou g_t(args, t).
    """
    try:
        return float(g_t(tt, args))
    except TypeError:
        return float(g_t(args, tt))


def coupling_curve(t, args, constant=False):
    """
    Curva g(t).

    Para o caso constante, usa g0.
    Para o caso variável, tenta usar a função g_t do projeto.
    """
    if constant:
        return np.full_like(t, float(args.get("g0", 1.0)), dtype=float)

    vals = []
    for tt in t:
        vals.append(eval_g_single(float(tt), args))
    return np.array(vals, dtype=float)


# ============================================================
# SALVAMENTO DOS VALORES ESPERADOS
# ============================================================
def save_expected_values(
    sol,
    t,
    save_root,
    label,
    args_for_coupling,
    constant_coupling=False,
):
    """
    Salva SOMENTE os valores esperados calculados pelo QuTiP.

    Este script NÃO calcula Wigner, NÃO salva frames e NÃO gera vídeo.

    A lista de observáveis usada no solve é:
        obs_list = [sz, nb, nb**2]

    Saídas principais:
    - <label>_expect_Z.npy
    - <label>_expect_N.npy
    - <label>_expect_N2.npy
    - <label>_var_N.npy
    - <label>_g_t.npy
    - <label>_observables.csv
    - <label>_timeseries.csv
    - <label>_info.json
    """
    os.makedirs(save_root, exist_ok=True)

    if sol.expect is None or len(sol.expect) < 3:
        raise ValueError(
            "sol.expect não contém os 3 observáveis esperados. "
            "Rode solve(..., obs_list=[sz, nb, nb**2], store_state=False)."
        )

    expect_Z = np.real_if_close(np.asarray(sol.expect[0], dtype=complex)).real.astype(float)
    expect_N = np.real_if_close(np.asarray(sol.expect[1], dtype=complex)).real.astype(float)
    expect_N2 = np.real_if_close(np.asarray(sol.expect[2], dtype=complex)).real.astype(float)
    var_N = expect_N2 - expect_N**2

    # Corrige resíduos numéricos negativos muito pequenos na variância.
    var_N[np.isclose(var_N, 0.0, atol=1e-12)] = 0.0

    g_values = coupling_curve(t, args_for_coupling, constant=constant_coupling)

    np.save(os.path.join(save_root, f"{label}_expect_Z.npy"), expect_Z)
    np.save(os.path.join(save_root, f"{label}_expect_N.npy"), expect_N)
    np.save(os.path.join(save_root, f"{label}_expect_N2.npy"), expect_N2)
    np.save(os.path.join(save_root, f"{label}_var_N.npy"), var_N)
    np.save(os.path.join(save_root, f"{label}_g_t.npy"), g_values)

    df = pd.DataFrame({
        "time_index": np.arange(len(t), dtype=int),
        "time": np.asarray(t, dtype=float),
        "g_t": g_values,
        "expect_N": expect_N,
        "expect_N2": expect_N2,
        "var_N": var_N,
        "expect_Z": expect_Z,
    })

    df.to_csv(os.path.join(save_root, f"{label}_observables.csv"), index=False)
    df.to_csv(os.path.join(save_root, f"{label}_timeseries.csv"), index=False)

    with open(os.path.join(save_root, f"{label}_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "n_times": int(len(t)),
            "saved_quantities": [
                "g_t",
                "expect_N",
                "expect_N2",
                "var_N",
                "expect_Z",
            ],
            "wigner_saved": False,
            "video_saved": False,
            "args": args_for_coupling,
        }, f, indent=2, ensure_ascii=False)

    print(f"Salvo somente valores esperados: {save_root}")
    return df


# ============================================================
# PARAMETERS
# ============================================================
eps = 1e-10
limite = 1e-1

t = np.concatenate([
    np.linspace(0, 1, 100, endpoint=False),
    np.linspace(1, 50, 200, endpoint=True),
])

N = 2      # Qubit Base Size
Nb = 45    # Field Base Size


# ============================================================
# COS PARAMETERS
# ============================================================
# Mantido como no bloco original do cosseno.
args = {
    "g0": 1,
    "eta": 1,
    "w": 2 * np.pi / 50,
    "kappa": 1e-1,
    "gamma": 0,
    "gamma_phi": 1e-2,
    "coupling": "cos",
    "phi": 0,
    # "zeta": 0.7,
    # "lambda": 0.1,
}

extra = "cos_paper_omegas_expected_values_open_only"


# ============================================================
# OMEGAS DA FIGURA
# ============================================================
# omega_max ≈ 2.09, omega_min = omega_max/4, omega_av = omega_max/2.
wmax = ((50 / 15) * np.pi) / 5
wmin = wmax / 4
wav = wmax / 2

w_list = np.array([wmin, wav, wmax], dtype=float)
w_points = selected_w_points(w_list)


# ============================================================
# INITIAL STATE
# ============================================================
alpha = np.sqrt(5)

phi0 = qt.tensor(
    qt.basis(N, 0),
    qt.coherent(Nb, alpha) + qt.coherent(Nb, -alpha),
).unit()

state0 = phi0.copy()


# ============================================================
# CREATE OUTPUT FOLDER + SAVE RUN INFO
# ============================================================
save_dir = make_exp_folder(
    base_name="expected_values_cos_paper_omegas_all_times",
    root="results/expected_values",
)

args_init = dict(args)

with open(os.path.join(save_dir, "run_info.txt"), "w", encoding="utf-8") as f:
    f.write("=== EXPECTED VALUES ALL TIMES RUN INFO ===\n")
    f.write(f"created_at: {datetime.now().isoformat()}\n\n")

    f.write(f"eps: {eps}\n")
    f.write(f"limite: {limite}\n")
    f.write(f"N: {N}\n")
    f.write(f"Nb: {Nb}\n")
    f.write(f"alpha: {float(alpha)}\n")
    f.write(f"extra: {extra}\n")
    f.write(f"len(t): {len(t)}\n")
    f.write(f"t_min: {float(np.min(t))}\n")
    f.write(f"t_max: {float(np.max(t))}\n\n")

    f.write(f"wmin = wmax/4: {float(wmin)}\n")
    f.write(f"wav = wmax/2: {float(wav)}\n")
    f.write(f"wmax: {float(wmax)}\n")
    f.write(f"len(w_list usada): {len(w_list)}\n")
    f.write(f"phi fixo: {float(args_init['phi'])}\n")
    f.write("w_points usados:\n")
    for row in w_points:
        f.write(f"  {row['w_label']}: index={row['w_index']}, w={row['w_value']}\n")

    f.write("\nargs initial:\n")
    f.write(json.dumps(args_init, indent=2, ensure_ascii=False))
    f.write("\n")

np.save(os.path.join(save_dir, "t.npy"), t)
np.save(os.path.join(save_dir, "w_list_paper_omegas.npy"), w_list)
pd.DataFrame(w_points).to_csv(os.path.join(save_dir, "w_points_used.csv"), index=False)

with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
    json.dump(args_init, f, indent=2, ensure_ascii=False)


# ============================================================
# OPERATORS
# ============================================================
sz, sp, sm, b, nb, I = get_operators(N, Nb)
obs_list = [sz, nb, nb**2]


# ============================================================
# DECAY AND DEPHASING
# ============================================================
c_ops = get_collapse(args, sm, sz, b)


# ============================================================
# CONSTANT HAMILTONIAN - OPEN ONLY
# ============================================================
H1 = h_closed(args, b, sp, sm)

print("Rodando caso constante aberto, somente valores esperados...")
args_const = dict(args_init)
sol_const_aberto = solve(
    H1,
    state0,
    t,
    c_ops,
    obs_list,
    args_const,
    store_state=False,
    open=True,
)

save_expected_values(
    sol_const_aberto,
    t,
    save_root=os.path.join(save_dir, "const_aberto_expected_values"),
    label="const_aberto",
    args_for_coupling=args_const,
    constant_coupling=True,
)


# ============================================================
# OPEN HAMILTONIAN - 3 OMEGAS DA FIGURA, phi FIXED
# ============================================================
args_per_w = []

for row in tqdm(w_points, desc="w points"):
    w_label = row["w_label"]
    w_index = row["w_index"]
    w_value = row["w_value"]

    args["w"] = float(w_value)
    args["phi"] = 0

    print(
        f"Rodando variável aberto, somente valores esperados | {w_label} | "
        f"w_index={w_index} | omega={w_value} | phi={args['phi']}"
    )

    H = h_open(b, sp, sm)

    sol_var_aberto = solve(
        H,
        state0,
        t,
        c_ops,
        obs_list,
        args,
        store_state=False,
        open=True,
    )

    label = f"var_aberto_{w_label}_idx{w_index:03d}_omega{w_value:.4f}_phi{args['phi']:.2f}"

    save_expected_values(
        sol_var_aberto,
        t,
        save_root=os.path.join(save_dir, f"{label}_expected_values"),
        label=label,
        args_for_coupling=dict(args),
        constant_coupling=False,
    )

    args_per_w.append({
        **args,
        "w_label": w_label,
        "w_index": w_index,
        "w_value": w_value,
    })


pd.DataFrame(args_per_w).to_csv(os.path.join(save_dir, "args_per_w_used.csv"), index=False)

print(f"✅ Tudo salvo em: {save_dir}")
