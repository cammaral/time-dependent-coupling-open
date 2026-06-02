import numpy as np
import qutip as qt
from tqdm import tqdm
from quantum.hamiltonian import g_t, h_closed, h_open
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from quantum.operators import get_operators, get_collapse
from quantum.run import solve
import os
import json
from datetime import datetime


def make_exp_folder(base_name="wigner_exp", root="results"):
    os.makedirs(root, exist_ok=True)
    k = 1
    while True:
        folder = os.path.join(root, f"{base_name}{k}")
        if not os.path.exists(folder):
            os.makedirs(folder)
            return folder
        k += 1


def selected_T_points(T_list):
    """
    Usa apenas 3 pontos do sweep:
    - inicial
    - meio
    - máximo/final
    """
    idxs = [0, len(T_list) // 2, len(T_list) - 1]
    rows = []
    for name, idx in zip(["inicial", "meio", "maximo"], idxs):
        rows.append({
            "T_label": name,
            "T_index": int(idx),
            "T_value": float(T_list[idx]),
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
    Curva g(t) para colocar no vídeo.

    Para o caso constante, usa g0.
    Para o caso variável, tenta usar a função g_t do projeto.
    """
    if constant:
        return np.full_like(t, float(args.get("g0", 1.0)), dtype=float)

    vals = []
    for tt in t:
        vals.append(eval_g_single(float(tt), args))
    return np.array(vals, dtype=float)


def trapz2d(A, xvec, pvec):
    """
    Integral 2D usando a malha da Wigner.
    A tem shape [p, x].
    """
    try:
        int_x = np.trapezoid(A, xvec, axis=1)
        return float(np.trapezoid(int_x, pvec, axis=0))
    except AttributeError:
        int_x = np.trapz(A, xvec, axis=1)
        return float(np.trapz(int_x, pvec, axis=0))


def wigner_negativity_from_W(W, xvec, pvec):
    """
    Negatividade da Wigner calculada diretamente da Wigner completa.

    Usa:
        delta = 1/2 * integral (|W| - W) dx dp

    Essa forma é mais robusta do que assumir integral(W)=1 exatamente,
    porque usa a normalização numérica real da grade.
    """
    int_abs = trapz2d(np.abs(W), xvec, pvec)
    int_w = trapz2d(W, xvec, pvec)
    return float(0.5 * (int_abs - int_w))


def expect_real(op, state):
    """
    Valor esperado real de um operador.
    Funciona para ket ou matriz densidade.
    """
    val = qt.expect(op, state)
    val = np.real_if_close(val)
    return float(np.real(val))


def set_curve_ylim(ax, values):
    """
    Coloca uma margem segura no eixo y, inclusive para curvas constantes.
    """
    values = np.asarray(values, dtype=float)
    ymin = float(np.nanmin(values))
    ymax = float(np.nanmax(values))

    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = -1.0, 1.0

    if np.isclose(ymin, ymax):
        pad = 0.1 * max(1.0, abs(ymin))
    else:
        pad = 0.08 * (ymax - ymin)

    ax.set_ylim(ymin - pad, ymax + pad)


def save_wigner_all_times_and_video(
    sol,
    t,
    xvec,
    pvec,
    save_root,
    label,
    args_for_coupling,
    sz_op,
    nb_op,
    constant_coupling=False,
    fps=12,
):
    """
    Salva a Wigner em TODOS os instantes de tempo da solução.

    Saídas principais:
    - wigner_frames/<label>_wigner_tXXXX.npy : uma Wigner por tempo
    - <label>_wigner_all_times.npy           : stack completo [tempo, p, x]
    - <label>_timeseries.csv                 : tempo, g(t), <N>, <Z>, negatividade
    - <label>_expect_N.npy                   : array com <N>(t)
    - <label>_expect_Z.npy                   : array com <Z>(t)
    - <label>_wigner_negativity.npy          : array com negatividade da Wigner
    - <label>_wigner_video.mp4               : vídeo com Wigner + g(t), <N>, <Z>, negatividade

    Observação:
    - Não salva todos os estados em disco; salva o resultado Wigner.
    - A escala de cor é fixa e simétrica em torno de zero.
    - Não salva GIF. Se o ffmpeg não estiver disponível, o script avisa e para.
    """
    os.makedirs(save_root, exist_ok=True)
    frames_dir = os.path.join(save_root, "wigner_frames")
    os.makedirs(frames_dir, exist_ok=True)

    W_list = []
    meta_rows = []
    expect_N_list = []
    expect_Z_list = []
    negativity_list = []

    print(f"Calculando e salvando Wigner, <N>, <Z> e negatividade: {label}")

    for idx, (state, tt) in enumerate(tqdm(list(zip(sol.states, t)), desc=f"Wigner {label}")):
        # pega só o modo bosônico para calcular a Wigner
        rho_field = qt.ptrace(state, 1)

        # calcula Wigner completa
        W = qt.wigner(rho_field, xvec, pvec)
        W = np.asarray(W)

        # observáveis no estado completo
        expect_N = expect_real(nb_op, state)
        expect_Z = expect_real(sz_op, state)

        # negatividade calculada diretamente da Wigner completa
        negativity = wigner_negativity_from_W(W, xvec, pvec)

        # salva uma Wigner por tempo
        frame_file = f"{label}_wigner_t{idx:04d}.npy"
        np.save(os.path.join(frames_dir, frame_file), W)

        W_list.append(W)
        expect_N_list.append(expect_N)
        expect_Z_list.append(expect_Z)
        negativity_list.append(negativity)

        meta_rows.append({
            "time_index": int(idx),
            "time": float(tt),
            "wigner_file": os.path.join("wigner_frames", frame_file),
            "expect_N": float(expect_N),
            "expect_Z": float(expect_Z),
            "wigner_negativity": float(negativity),
        })

    W_stack = np.stack(W_list, axis=0)
    expect_N_values = np.array(expect_N_list, dtype=float)
    expect_Z_values = np.array(expect_Z_list, dtype=float)
    negativity_values = np.array(negativity_list, dtype=float)

    np.save(os.path.join(save_root, f"{label}_wigner_all_times.npy"), W_stack)
    np.save(os.path.join(save_root, f"{label}_expect_N.npy"), expect_N_values)
    np.save(os.path.join(save_root, f"{label}_expect_Z.npy"), expect_Z_values)
    np.save(os.path.join(save_root, f"{label}_wigner_negativity.npy"), negativity_values)

    # curva de acoplamento para salvar e colocar no vídeo
    g_values = coupling_curve(t, args_for_coupling, constant=constant_coupling)

    df_ts = pd.DataFrame(meta_rows)
    df_ts["g_t"] = g_values
    df_ts.to_csv(os.path.join(save_root, f"{label}_timeseries.csv"), index=False)

    # compatibilidade com a versão anterior
    df_ts[["time_index", "time", "wigner_file"]].to_csv(
        os.path.join(save_root, f"{label}_wigner_times.csv"),
        index=False
    )

    pd.DataFrame({
        "time": t,
        "g_t": g_values,
        "expect_N": expect_N_values,
        "expect_Z": expect_Z_values,
        "wigner_negativity": negativity_values,
    }).to_csv(os.path.join(save_root, f"{label}_observables.csv"), index=False)

    # escala fixa de cor para o vídeo
    max_abs = float(np.nanmax(np.abs(W_stack)))
    if max_abs == 0 or not np.isfinite(max_abs):
        max_abs = 1.0

    vmin = -0.2#-max_abs
    vmax = 0.2#max_abs

    # ==========================
    # VIDEO MP4 ONLY
    # ==========================
    print(f"Gerando vídeo MP4: {label}")

    if not FFMpegWriter.isAvailable():
        raise RuntimeError(
            "FFmpeg não está disponível para o Matplotlib. "
            "Instale ffmpeg para gerar .mp4. Nenhum GIF será salvo."
        )

    fig = plt.figure(figsize=(12.8, 7.2))  # 12.8*150 = 1920 px, dimensao par para H.264
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=[1.25, 1.0],
        height_ratios=[1, 1, 1, 1],
        wspace=0.35,
        hspace=0.55,
    )

    ax_w = fig.add_subplot(gs[:, 0])
    ax_g = fig.add_subplot(gs[0, 1])
    ax_N = fig.add_subplot(gs[1, 1])
    ax_Z = fig.add_subplot(gs[2, 1])
    ax_neg = fig.add_subplot(gs[3, 1])

    im = ax_w.imshow(
        W_stack[0],
        origin="lower",
        extent=[xvec.min(), xvec.max(), pvec.min(), pvec.max()],
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
    )
    cbar = fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)
    cbar.set_label("W")

    ax_w.set_xlabel("x")
    ax_w.set_ylabel("p")
    ax_w.set_title(f"{label} | t={t[0]:.3f}")

    # painel g(t)
    ax_g.plot(t, g_values, lw=1.8)
    marker_g, = ax_g.plot([t[0]], [g_values[0]], marker="o", markersize=6)
    ax_g.set_ylabel("g(t)")
    ax_g.set_title("Acoplamento")
    ax_g.grid(alpha=0.3)
    set_curve_ylim(ax_g, g_values)

    # painel <N>
    ax_N.plot(t, expect_N_values, lw=1.8)
    marker_N, = ax_N.plot([t[0]], [expect_N_values[0]], marker="o", markersize=6)
    ax_N.set_ylabel(r"$\langle N \rangle$")
    ax_N.set_title(r"Valor esperado de $N$")
    ax_N.grid(alpha=0.3)
    set_curve_ylim(ax_N, expect_N_values)

    # painel <Z>
    ax_Z.plot(t, expect_Z_values, lw=1.8)
    marker_Z, = ax_Z.plot([t[0]], [expect_Z_values[0]], marker="o", markersize=6)
    ax_Z.set_ylabel(r"$\langle Z \rangle$")
    ax_Z.set_title(r"Valor esperado de $Z$")
    ax_Z.grid(alpha=0.3)
    set_curve_ylim(ax_Z, expect_Z_values)

    # painel negatividade
    ax_neg.plot(t, negativity_values, lw=1.8)
    marker_neg, = ax_neg.plot([t[0]], [negativity_values[0]], marker="o", markersize=6)
    ax_neg.set_xlabel("t")
    ax_neg.set_ylabel("Neg.")
    ax_neg.set_title("Negatividade da Wigner")
    ax_neg.grid(alpha=0.3)
    set_curve_ylim(ax_neg, negativity_values)

    # remove x labels dos painéis superiores para ficar mais limpo
    for ax in [ax_g, ax_N, ax_Z]:
        ax.tick_params(labelbottom=False)

    info_text = ax_w.text(
        0.02,
        0.98,
        "",
        transform=ax_w.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    def update(frame):
        im.set_data(W_stack[frame])
        ax_w.set_title(f"{label} | t={t[frame]:.3f}")

        marker_g.set_data([t[frame]], [g_values[frame]])
        marker_N.set_data([t[frame]], [expect_N_values[frame]])
        marker_Z.set_data([t[frame]], [expect_Z_values[frame]])
        marker_neg.set_data([t[frame]], [negativity_values[frame]])

        info_text.set_text(
            f"t = {t[frame]:.3f}\n"
            f"g(t) = {g_values[frame]:.6g}\n"
            f"<N> = {expect_N_values[frame]:.6g}\n"
            f"<Z> = {expect_Z_values[frame]:.6g}\n"
            f"Neg. = {negativity_values[frame]:.6g}"
        )
        return im, marker_g, marker_N, marker_Z, marker_neg, info_text

    mp4_path = os.path.join(save_root, f"{label}_wigner_video.mp4")

    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        metadata={"artist": "matplotlib"},
        extra_args=[
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
        ],
    )
    with writer.saving(fig, mp4_path, dpi=150):
        for frame in tqdm(range(len(t)), desc=f"Video {label}"):
            update(frame)
            writer.grab_frame()

    plt.close(fig)

    # metadados finais
    with open(os.path.join(save_root, f"{label}_wigner_video_info.json"), "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "n_times": int(len(t)),
            "wigner_stack_shape": list(W_stack.shape),
            "saved_observables": [
                "expect_N",
                "expect_Z",
                "wigner_negativity",
                "g_t",
            ],
            "color_scale": {
                "vmin": vmin,
                "vmax": vmax,
                "symmetric": True,
                "fixed_in_video": True,
            },
            "video_file": os.path.basename(mp4_path),
            "video_format": "mp4",
            "gif_saved": False,
        }, f, indent=2, ensure_ascii=False)

    print(f"Salvo: {save_root}")


# ==========================
# PARAMETERS
# ==========================
eps = 1e-10
limite = 1e-1

t = np.concatenate([
    np.linspace(0, 1, 100, endpoint=False),
    np.linspace(1, 15, 100, endpoint=True)
])

N = 2      # Qubit Base Size
Nb = 45    # Field Base Size


args = {
    "g0": 1,
    "eta": 1,
    "sigma": -1,
    "kappa": 1e-1,
    "gamma": 0,
    "gamma_phi": 1e-2,
    "coupling": "gauss",
    "epsilon": 8 / (50 / 15),
    "T": None,
}


extra = "gauss6_new_params_wigner_all_times_open_only_obs_video"

xvec = np.linspace(-7.5, 7.5, 200)
pvec = np.linspace(-7.5, 7.5, 200)


#======= GAUSS ========
Tmax = 35 / (50 / 15)
T_list = np.linspace(15 / (50 / 15), Tmax, 100, endpoint=True)
T_points = selected_T_points(T_list)


# ==========================
# INITIAL STATE
# ==========================
alpha = np.sqrt(5)

phi0 = qt.tensor(
    qt.basis(N, 0),
    qt.coherent(Nb, alpha) + qt.coherent(Nb, -alpha)
).unit()

print("oi")


# ==========================
# CREATE OUTPUT FOLDER + SAVE RUN INFO
# ==========================
save_dir = make_exp_folder(base_name=f"wigner_{args['coupling']}_all_times", root="results")

args_init = dict(args)

with open(os.path.join(save_dir, "run_info.txt"), "w", encoding="utf-8") as f:
    f.write("=== WIGNER ALL TIMES RUN INFO ===\n")
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

    f.write(f"Tmax: {float(Tmax)}\n")
    f.write(f"len(T_list original): {len(T_list)}\n")
    f.write("T_points usados:\n")
    for row in T_points:
        f.write(f"  {row['T_label']}: index={row['T_index']}, T={row['T_value']}\n")

    f.write("\nargs initial:\n")
    f.write(json.dumps(args_init, indent=2, ensure_ascii=False))
    f.write("\n")

np.save(os.path.join(save_dir, "t.npy"), t)
np.save(os.path.join(save_dir, "T_list_original.npy"), T_list)
pd.DataFrame(T_points).to_csv(os.path.join(save_dir, "T_points_used.csv"), index=False)

with open(os.path.join(save_dir, "args.json"), "w", encoding="utf-8") as f:
    json.dump(args_init, f, indent=2, ensure_ascii=False)


# ==========================
# OPERATORS
# ==========================
sz, sp, sm, b, nb, I = get_operators(N, Nb)
obs_list = [sz, nb, nb**2]


# ==========================
# DECAY AND DEPHASING
# ==========================
c_ops = get_collapse(args, sm, sz, b)


# ==========================
# CONSTANT HAMILTONIAN - OPEN ONLY
# ==========================
H1 = h_closed(args, b, sp, sm)

state0 = phi0.copy()

print("Rodando caso constante aberto...")
args_const = dict(args_init)
sol_const_aberto = solve(H1, state0, t, c_ops, obs_list, args_const)

save_wigner_all_times_and_video(
    sol_const_aberto,
    t,
    xvec,
    pvec,
    save_root=os.path.join(save_dir, "const_aberto_all_times"),
    label="const_aberto",
    args_for_coupling=args_const,
    sz_op=sz,
    nb_op=nb,
    constant_coupling=True,
    fps=12,
)


# ==========================
# OPEN HAMILTONIAN - ONLY 3 T POINTS
# ==========================
args_per_T = []

for row in tqdm(T_points, desc="T points"):
    T_label = row["T_label"]
    T_index = row["T_index"]
    T_value = row["T_value"]

    args["T"] = float(T_value)

    print(f"Rodando variável aberto | {T_label} | T_index={T_index} | T={T_value}")

    H = h_open(b, sp, sm)

    # aberto apenas
    sol_var_aberto = solve(H, state0, t, c_ops, obs_list, args)

    label = f"var_aberto_T_{T_label}_idx{T_index:03d}_T{T_value:.4f}"

    save_wigner_all_times_and_video(
        sol_var_aberto,
        t,
        xvec,
        pvec,
        save_root=os.path.join(save_dir, label),
        label=label,
        args_for_coupling=dict(args),
        sz_op=sz,
        nb_op=nb,
        constant_coupling=False,
        fps=12,
    )

    args_per_T.append({
        **args,
        "T_label": T_label,
        "T_index": T_index,
        "T_value": T_value,
    })


pd.DataFrame(args_per_T).to_csv(os.path.join(save_dir, "args_per_T_used.csv"), index=False)

print(f"✅ Tudo salvo em: {save_dir}")
