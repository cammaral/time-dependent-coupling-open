import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Change this to the folder created by quantum_magic_cos_var_w.py or quantum_magic_gauss_varT.py.
# Example: save_dir = "results/magic/magic_cos1"
save_dir = "results/magic/magic_cos1"


# ==========================
# LOAD DATA
# ==========================
t = np.load(os.path.join(save_dir, "t.npy"))

if os.path.exists(os.path.join(save_dir, "w_list.npy")):
    scan = np.load(os.path.join(save_dir, "w_list.npy"))
    scan_name = "w"
    scan_label = r"$\omega$"
elif os.path.exists(os.path.join(save_dir, "T_list.npy")):
    scan = np.load(os.path.join(save_dir, "T_list.npy"))
    scan_name = "T"
    scan_label = r"$T$"
else:
    raise FileNotFoundError("Nao encontrei w_list.npy nem T_list.npy no save_dir.")

const = np.load(os.path.join(save_dir, "const.npy"))
const_aberto = np.load(os.path.join(save_dir, "const_aberto.npy"))
var = np.load(os.path.join(save_dir, "var.npy"))
var_aberto = np.load(os.path.join(save_dir, "var_aberto.npy"))

plot_dir = os.path.join(save_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)


# ==========================
# CONST COMPARISON
# ==========================
plt.figure(figsize=(8, 5))
plt.plot(t, const, lw=2.0, label="const")
plt.plot(t, const_aberto, lw=2.0, linestyle="--", label="const_aberto")
plt.xlabel("t")
plt.ylabel(r"$M_2$")
plt.title("Quantum magic/SRE-2: constant coupling")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "magic_const_comparison.png"), dpi=200)
plt.close()


# ==========================
# HEATMAPS
# ==========================
for name, arr in [("var", var), ("var_aberto", var_aberto)]:
    plt.figure(figsize=(9, 5))
    plt.imshow(
        arr,
        aspect="auto",
        origin="lower",
        extent=[float(t[0]), float(t[-1]), float(scan[0]), float(scan[-1])],
    )
    plt.colorbar(label=r"$M_2$")
    plt.xlabel("t")
    plt.ylabel(scan_label)
    plt.title(f"Quantum magic/SRE-2: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"magic_heatmap_{name}_{scan_name}.png"), dpi=200)
    plt.close()


# ==========================
# FINAL-TIME SCAN
# ==========================
plt.figure(figsize=(8, 5))
plt.plot(scan, var[:, -1], lw=2.0, label="var")
plt.plot(scan, var_aberto[:, -1], lw=2.0, linestyle="--", label="var_aberto")
plt.xlabel(scan_label)
plt.ylabel(r"$M_2(t_{final})$")
plt.title("Quantum magic/SRE-2 at final time")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"magic_final_time_scan_{scan_name}.png"), dpi=200)
plt.close()


# ==========================
# SUMMARY CSV
# ==========================
summary = pd.DataFrame({
    scan_name: scan,
    "var_M2_max": np.max(var, axis=1),
    "var_M2_mean": np.mean(var, axis=1),
    "var_M2_final": var[:, -1],
    "var_aberto_M2_max": np.max(var_aberto, axis=1),
    "var_aberto_M2_mean": np.mean(var_aberto, axis=1),
    "var_aberto_M2_final": var_aberto[:, -1],
})
summary.to_csv(os.path.join(save_dir, f"magic_summary_per_{scan_name}.csv"), index=False)

print(f"✅ Figuras e resumo salvos em: {plot_dir}")
