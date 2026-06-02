import numpy as np
import qutip as qt
from tqdm import tqdm
from quantum.hamiltonian import h_closed, h_open
import pandas as pd
import matplotlib.pyplot as plt
from quantum.operators import get_operators, get_collapse
from quantum.run import solve
from quantum.non_classicality import stabilizer_renyi_entropy_qubit_from_expectations
import os
import json
from datetime import datetime


def make_exp_folder(base_name="magic_exp", root="results/magic"):
    os.makedirs(root, exist_ok=True)
    k = 1
    while True:
        folder = os.path.join(root, f"{base_name}{k}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        k += 1


def get_snapshot_indices(t, t_min=0.0, t_max=10.0, n_snapshots=10):
    target_times = np.linspace(t_min, t_max, n_snapshots)
    indices = [int(np.argmin(np.abs(t - tt))) for tt in target_times]
    actual_times = [float(t[i]) for i in indices]
    return target_times, indices, actual_times


def magic_from_solution(sol):
    """Compute M2 from solver expectations [<X>, <Y>, <Z>, ...]."""
    px = np.asarray(sol.expect[0])
    py = np.asarray(sol.expect[1])
    pz = np.asarray(sol.expect[2])
    M2, S2, S4, purity, magic_proxy_l1 = stabilizer_renyi_entropy_qubit_from_expectations(px, py, pz)
    return {
        "M2": np.asarray(M2, dtype=float),
        "px": np.real_if_close(px),
        "py": np.real_if_close(py),
        "pz": np.real_if_close(pz),
        "S2": np.asarray(S2, dtype=float),
        "S4": np.asarray(S4, dtype=float),
        "purity": np.asarray(purity, dtype=float),
        "magic_proxy_l1": np.asarray(magic_proxy_l1, dtype=float),
    }


def save_magic_dict(save_dir, prefix, data):
    """Save M2 as .npy and all diagnostics as .npz/.csv."""
    np.save(os.path.join(save_dir, f"{prefix}.npy"), data["M2"])
    np.savez(
        os.path.join(save_dir, f"{prefix}_details.npz"),
        M2=data["M2"],
        px=data["px"],
        py=data["py"],
        pz=data["pz"],
        S2=data["S2"],
        S4=data["S4"],
        purity=data["purity"],
        magic_proxy_l1=data["magic_proxy_l1"],
    )
    pd.DataFrame({
        "M2": data["M2"],
        "px": np.real_if_close(data["px"]),
        "py": np.real_if_close(data["py"]),
        "pz": np.real_if_close(data["pz"]),
        "S2": data["S2"],
        "S4": data["S4"],
        "purity": data["purity"],
        "magic_proxy_l1": data["magic_proxy_l1"],
    }).to_csv(os.path.join(save_dir, f"{prefix}_details.csv"), index=False)


def save_qubit_magic_snapshots(sol, t, save_root, label, subsystem=0):
    """Save reduced qubit states and magic values at selected times."""
    os.makedirs(save_root, exist_ok=True)
    target_times, indices, actual_times = get_snapshot_indices(t, t_min=0.0, t_max=min(10.0, float(np.max(t))), n_snapshots=10)

    X = qt.sigmax()
    Y = qt.sigmay()
    Z = qt.sigmaz()
    rows = []

    for k, (ttarget, idx, tactual) in enumerate(zip(target_times, indices, actual_times)):
        state = sol.states[idx]
        rho_q = state.ptrace(subsystem)
        qt.qsave(rho_q, os.path.join(save_root, f"{label}_rho_qubit_{k:02d}"))

        px = qt.expect(X, rho_q)
        py = qt.expect(Y, rho_q)
        pz = qt.expect(Z, rho_q)
        M2, S2, S4, purity, magic_proxy_l1 = stabilizer_renyi_entropy_qubit_from_expectations(px, py, pz)

        rows.append({
            "snapshot_id": k,
            "target_time": float(ttarget),
            "actual_time": float(tactual),
            "time_index": int(idx),
            "rho_file": f"{label}_rho_qubit_{k:02d}.qu",
            "M2": float(np.real_if_close(M2)),
            "px": float(np.real_if_close(px)),
            "py": float(np.real_if_close(py)),
            "pz": float(np.real_if_close(pz)),
            "S2": float(np.real_if_close(S2)),
            "S4": float(np.real_if_close(S4)),
            "purity": float(np.real_if_close(purity)),
            "magic_proxy_l1": float(np.real_if_close(magic_proxy_l1)),
        })

    pd.DataFrame(rows).to_csv(os.path.join(save_root, f"{label}_magic_snapshots.csv"), index=False)


# ==========================
# PARAMETERS
# ==========================
eps = 1e-10
limite = 1e-1
t = np.concatenate([
    np.linspace(0, 1, 100, endpoint=False),
    np.linspace(1, 50, 200, endpoint=True)
])

N = 2      # Qubit Base Size
Nb = 45    # Field Base Size

# Gaussian coupling. This file scans T, matching the gauss files in the project.
args = {
    "g0": 1,
    "eta": 1,
    "sigma": -1,
    "kappa": 1e-1,
    "gamma": 0,
    "gamma_phi": 1e-2,
    "coupling": "gauss",
    "epsilon": 8,
    "T": 25,
}

extra = "gauss_magic_sre2_varep"

# ======= GAUSS =======
epmax = 10
#Tmax = 35
#T_list = np.linspace(15, Tmax, 100, endpoint=True)
ep_list = np.linspace(6, epmax, 100, endpoint=True)

# ==========================
# INITIAL STATE
# ==========================
alpha = np.sqrt(5)
theta = np.arccos(1/np.sqrt(3))
# Same style as the project files. You can swap to cat-like states below.
phi0 = qt.tensor(np.cos(theta/2)*qt.basis(N, 0) + np.exp(1j*np.pi/4)*np.sin(theta/2)*qt.basis(N, 1), qt.coherent(Nb, alpha)).unit()
# phi0 = (qt.tensor(qt.basis(N, 0), qt.coherent(Nb, -alpha)) + qt.tensor(qt.basis(N, 1), qt.coherent(Nb, alpha))).unit()
# phi0 = qt.tensor((0.888074 * qt.basis(N, 1) + np.exp(1j*np.pi/4) * 0.459701 * qt.basis(N, 0)), qt.coherent(Nb, np.sqrt(2))).unit()

# ==========================
# CREATE OUTPUT FOLDER + SAVE RUN INFO
# ==========================
save_dir = make_exp_folder(base_name=f"magic_{args['coupling']}", root="results/magic")
args_init = dict(args)

with open(os.path.join(save_dir, "run_info.txt"), "w", encoding="utf-8") as f:
    f.write("=== QUANTUM MAGIC RUN INFO ===\n")
    f.write(f"created_at: {datetime.now().isoformat()}\n\n")

    f.write(f"measure: Stabilizer Renyi entropy order 2 / SRE-2\n")
    f.write(f"formula_single_qubit: M2 = log((1 + |<X>|^2 + |<Y>|^2 + |<Z>|^2)/(1 + |<X>|^4 + |<Y>|^4 + |<Z>|^4))\n\n")

    f.write(f"eps: {eps}\n")
    f.write(f"limite: {limite}\n")
    f.write(f"N: {N}\n")
    f.write(f"Nb: {Nb}\n")
    f.write(f"alpha: {float(alpha)}\n")
    f.write(f"extra: {extra}\n")
    f.write(f"epmax: {float(epmax)}\n")
    f.write(f"len(ep_list): {len(ep_list)}\n")
    f.write(f"len(t): {len(t)}\n\n")

    f.write("args (initial):\n")
    f.write(json.dumps(args_init, indent=2, ensure_ascii=False))
    f.write("\n\n")

    f.write("t summary:\n")
    f.write(f"  t_min: {float(np.min(t))}\n")
    f.write(f"  t_max: {float(np.max(t))}\n")
    f.write(f"  first_5: {t[:5].tolist()}\n")
    f.write(f"  last_5: {t[-5:].tolist()}\n\n")

    f.write("ep_list summary:\n")
    f.write(f"  ep_min: {float(np.min(ep_list))}\n")
    f.write(f"  ep_max: {float(np.max(ep_list))}\n")
    f.write(f"  first_5: {ep_list[:5].tolist()}\n")
    f.write(f"  last_5: {ep_list[-5:].tolist()}\n")

np.save(os.path.join(save_dir, "t.npy"), t)
np.save(os.path.join(save_dir, "ep_list.npy"), ep_list)

with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
    json.dump(args_init, f, indent=2, ensure_ascii=False)

# ==========================
# OPERATORS
# ==========================
sz, sp, sm, b, nb, I = get_operators(N, Nb)
sx = sp + sm
sy = -1j * (sp - sm)
obs_list = [sx, sy, sz, nb, nb**2]

# ==========================
# DECAY AND DEPHASING
# ==========================
c_ops = get_collapse(args, sm, sz, b)

# ==========================
# CLOSED HAMILTONIAN (CONST)
# ==========================
H1 = h_closed(args, b, sp, sm)
state0 = phi0.copy()

sol_const_aberto = solve(H1, state0, t, c_ops, obs_list, args)
sol_const = solve(H1, state0, t, None, obs_list, args, open=False)

magic_const_aberto = magic_from_solution(sol_const_aberto)
magic_const = magic_from_solution(sol_const)

save_magic_dict(save_dir, "const_aberto", magic_const_aberto)
save_magic_dict(save_dir, "const", magic_const)

save_qubit_magic_snapshots(
    sol_const_aberto,
    t,
    save_root=os.path.join(save_dir, "snapshots_const_aberto"),
    label="const_aberto",
)
save_qubit_magic_snapshots(
    sol_const,
    t,
    save_root=os.path.join(save_dir, "snapshots_const"),
    label="const",
)

# ==========================
# OPEN HAMILTONIAN (VAR over T)
# ==========================
magic_list_aberto = []
magic_list = []

px_list_aberto = []
py_list_aberto = []
pz_list_aberto = []
purity_list_aberto = []
proxy_list_aberto = []

px_list = []
py_list = []
pz_list = []
purity_list = []
proxy_list = []

args_per_ep = []

for ep in tqdm(ep_list):
    args["epsilon"] = float(ep)
    H = h_open(b, sp, sm)

    sol_var_aberto = solve(H, state0, t, c_ops, obs_list, args)
    sol_var = solve(H, state0, t, None, obs_list, args, open=False)

    magic_var_aberto = magic_from_solution(sol_var_aberto)
    magic_var = magic_from_solution(sol_var)

    magic_list_aberto.append(magic_var_aberto["M2"])
    magic_list.append(magic_var["M2"])

    px_list_aberto.append(magic_var_aberto["px"])
    py_list_aberto.append(magic_var_aberto["py"])
    pz_list_aberto.append(magic_var_aberto["pz"])
    purity_list_aberto.append(magic_var_aberto["purity"])
    proxy_list_aberto.append(magic_var_aberto["magic_proxy_l1"])

    px_list.append(magic_var["px"])
    py_list.append(magic_var["py"])
    pz_list.append(magic_var["pz"])
    purity_list.append(magic_var["purity"])
    proxy_list.append(magic_var["magic_proxy_l1"])

    args_per_ep.append({**args})

# salvar var / var_aberto como matrizes: (len(T_list), len(t))
magic_list_aberto = np.array(magic_list_aberto, dtype=float)
magic_list = np.array(magic_list, dtype=float)

np.save(os.path.join(save_dir, "var_aberto.npy"), magic_list_aberto)
np.save(os.path.join(save_dir, "var.npy"), magic_list)

np.savez(
    os.path.join(save_dir, "var_aberto_details.npz"),
    M2=magic_list_aberto,
    px=np.array(px_list_aberto),
    py=np.array(py_list_aberto),
    pz=np.array(pz_list_aberto),
    purity=np.array(purity_list_aberto),
    magic_proxy_l1=np.array(proxy_list_aberto),
)

np.savez(
    os.path.join(save_dir, "var_details.npz"),
    M2=magic_list,
    px=np.array(px_list),
    py=np.array(py_list),
    pz=np.array(pz_list),
    purity=np.array(purity_list),
    magic_proxy_l1=np.array(proxy_list),
)

pd.DataFrame(args_per_ep).to_csv(os.path.join(save_dir, "args_per_T.csv"), index=False)

print(f"✅ Quantum magic salvo em: {save_dir}")
