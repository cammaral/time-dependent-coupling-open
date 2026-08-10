"""Generate a self-contained PDF report about mixed-state atomic magic.

The report is built exclusively with Matplotlib (including text, equations,
tables and plots).  It reads the simulations already present in ``results``;
it does not rerun the Jaynes--Cummings dynamics.
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
MPL_CACHE = PROJECT_ROOT / "tmp" / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


OUTPUT_DIR = PROJECT_ROOT / "output" / "pdf" / "mixed_state_magic_report"
OUTPUT_PDF = OUTPUT_DIR / "mixed_state_magic_report.pdf"
PAGE_SIZE = (8.27, 11.69)  # A4 portrait, inches

BLUE = "#1f5a85"
LIGHT_BLUE = "#dfeef7"
GREEN = "#27856f"
RED = "#b84d4d"
ORANGE = "#d17a22"
GRAY = "#666666"
LIGHT_GRAY = "#f2f4f5"
INK = "#20262b"

CASE_CONFIGS = (
    {
        "case_id": "magic_gauss_epsilon",
        "full_scan_folder": "magic_gauss1",
        "title": r"Gaussiano: largura $\zeta$ variável ($T=25$, $\sigma=-1$)",
        "scan_label": r"largura $\zeta$",
        "scan_short": r"$\zeta$",
        "limits": (6.0, 10.0),
        "curve_labels": {
            "const": "constante",
            "min": r"$\zeta=6$",
            "av": r"$\zeta=8$",
            "max": r"$\zeta=10$",
        },
        "interpretation": (
            "A largura menor preserva magia exata por mais tempo. Aumentar a largura "
            "antecipa a perda: a duração com W1/2>0 cai de 18,62 para 13,90."
        ),
    },
    {
        "case_id": "magic_gauss_T",
        "full_scan_folder": "magic_gauss2",
        "title": r"Gaussiano: instante de pico $T$ variável ($\zeta=8$, $\sigma=-1$)",
        "scan_label": r"instante de pico $T$",
        "scan_short": r"$T$",
        "limits": (15.0, 35.0),
        "curve_labels": {
            "const": "constante",
            "min": r"$T=15$",
            "av": r"$T=25$",
            "max": r"$T=35$",
        },
        "interpretation": (
            "Deslocar o pulso para tempos tardios desloca também a janela de interação "
            "e prolonga a magia inicial: a duração sobe de 7,87 para 26,87."
        ),
    },
    {
        "case_id": "magic_cos_omega",
        "full_scan_folder": "magic_cos1",
        "title": r"Cosseno: frequência $\omega$ variável ($\phi=0$)",
        "scan_label": r"frequência $\omega$",
        "scan_short": r"$\omega$",
        "limits": (np.pi / 20.0, np.pi / 5.0),
        "curve_labels": {
            "const": "constante",
            "min": r"$\omega=\pi/20$",
            "av": r"$\omega=\pi/10$",
            "max": r"$\omega=\pi/5$",
        },
        "interpretation": (
            "A frequência alta produz muitos revivals. Para o ponto selecionado pi/5, "
            "W1/2>0 ocupa 35,75 unidades de tempo, mas W2 e W3 quase não certificam "
            "essa magia devido à sua menor sensibilidade."
        ),
    },
)

CURVE_ORDER = ("const", "min", "av", "max")
CURVE_STYLES = {
    "const": dict(color="#555555", linestyle="--", linewidth=1.65),
    "min": dict(color="#c44e52", linestyle=":", linewidth=1.75),
    "av": dict(color="#4c72b0", linestyle="-.", linewidth=1.75),
    "max": dict(color="#2a9d78", linestyle="-", linewidth=1.85),
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "axes.edgecolor": "#45525b",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def latest_numbered_folder(parent: Path, prefix: str) -> Path:
    def suffix(path: Path) -> int:
        match = re.search(r"(\d+)$", path.name)
        return int(match.group(1)) if match else -1

    folders = [path for path in parent.glob(f"{prefix}*") if path.is_dir()]
    if not folders:
        raise FileNotFoundError(f"Nenhuma pasta {prefix}* em {parent}")
    return max(folders, key=suffix)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_all_data() -> tuple[Path, Path, list[dict[str, object]]]:
    mixed_run = latest_numbered_folder(
        PROJECT_ROOT / "results" / "mixed_state_magic_witnesses", "mixed_magic"
    )
    expected_run = latest_numbered_folder(
        PROJECT_ROOT / "results" / "expected_values_all_cases_xy",
        "expected_values_all_cases_xy",
    )
    expected_relative = expected_run.relative_to(PROJECT_ROOT / "results")
    cases: list[dict[str, object]] = []

    for config in CASE_CONFIGS:
        full_path = (
            mixed_run
            / "derived"
            / "magic"
            / str(config["full_scan_folder"])
            / "var_aberto_details_mixed_magic.npz"
        )
        selected_metric_dir = (
            mixed_run
            / "derived"
            / expected_relative
            / str(config["case_id"])
        )
        selected_source_dir = expected_run / str(config["case_id"])
        curves: dict[str, dict[str, np.ndarray]] = {}
        for key in CURVE_ORDER:
            stem = "const_aberto" if key == "const" else f"var_aberto_{key}"
            metrics = load_npz(selected_metric_dir / f"{stem}_observables_mixed_magic.npz")
            source = load_npz(selected_source_dir / f"{stem}_observables.npz")
            metrics["g_t"] = np.asarray(source["g_t"], dtype=float)
            curves[key] = metrics
        cases.append(
            {
                "config": config,
                "full": load_npz(full_path),
                "curves": curves,
                "full_path": full_path,
                "selected_metric_dir": selected_metric_dir,
                "selected_source_dir": selected_source_dir,
            }
        )
    return mixed_run, expected_run, cases


def positive_duration(t: np.ndarray, values: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(values, dtype=float)
    total = 0.0
    for t0, t1, y0, y1 in zip(t[:-1], t[1:], y[:-1], y[1:]):
        dt = float(t1 - t0)
        p0, p1 = y0 > 0.0, y1 > 0.0
        if p0 and p1:
            total += dt
        elif p0 != p1 and y1 != y0:
            frac = float(np.clip(-y0 / (y1 - y0), 0.0, 1.0))
            total += dt * (frac if p0 else 1.0 - frac)
    return total


def first_loss(t: np.ndarray, values: np.ndarray) -> float:
    t = np.asarray(t, dtype=float)
    y = np.asarray(values, dtype=float)
    for index in range(len(y) - 1):
        if y[index] > 0.0 and y[index + 1] <= 0.0:
            fraction = -y[index] / (y[index + 1] - y[index])
            return float(t[index] + fraction * (t[index + 1] - t[index]))
    return float("nan")


def count_revivals(values: np.ndarray) -> int:
    positive = np.asarray(values, dtype=float) > 0.0
    return int(np.count_nonzero((~positive[:-1]) & positive[1:]))


def selected_summary(case: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    config = case["config"]
    for key in CURVE_ORDER:
        curve = case["curves"][key]
        t = np.asarray(curve["time"], dtype=float)
        w = np.asarray(curve["W_half"], dtype=float)
        rows.append(
            {
                "key": key,
                "label": config["curve_labels"][key],
                "duration": positive_duration(t, w),
                "first_loss": first_loss(t, w),
                "revivals": count_revivals(w),
                "final_w": float(w[-1]),
                "final_magic": bool(np.asarray(curve["is_magic_exact"])[-1]),
            }
        )
    return rows


def scan_statistics(case: dict[str, object]) -> dict[str, object]:
    config = case["config"]
    full = case["full"]
    scan = np.asarray(full["scan_values"], dtype=float)
    t = np.asarray(full["time"], dtype=float)
    low, high = config["limits"]
    mask = (scan >= low - 1e-12) & (scan <= high + 1e-12)
    scan = scan[mask]
    w = np.asarray(full["W_half"], dtype=float)[mask]
    durations = np.array([positive_duration(t, row) for row in w])
    positive_area = np.trapezoid(np.maximum(w, 0.0), t, axis=1)
    final_magic = w[:, -1] > 0.0
    return {
        "scan": scan,
        "time": t,
        "W_half": w,
        "M_2": np.asarray(full["M_2"], dtype=float)[mask],
        "W_2": np.asarray(full["W_2"], dtype=float)[mask],
        "W_3": np.asarray(full["W_3"], dtype=float)[mask],
        "durations": durations,
        "positive_area": positive_area,
        "best_duration_index": int(np.argmax(durations)),
        "best_area_index": int(np.argmax(positive_area)),
        "final_magic_fraction": float(np.mean(final_magic)),
    }


def new_page(title: str, subtitle: str | None = None) -> plt.Figure:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.text(0.07, 0.955, title, fontsize=18, fontweight="bold", color=INK, va="top")
    if subtitle:
        fig.text(0.07, 0.924, subtitle, fontsize=9.5, color=GRAY, va="top")
    fig.add_artist(Line2D([0.07, 0.93], [0.905, 0.905], color=BLUE, linewidth=1.4))
    return fig


def add_footer(fig: plt.Figure, page_number: int, mixed_run: Path) -> None:
    fig.add_artist(Line2D([0.07, 0.93], [0.037, 0.037], color="#c8cdd1", linewidth=0.7))
    fig.text(
        0.07,
        0.019,
        f"Relatório reprodutível · dados: {mixed_run.name} · Matplotlib",
        fontsize=7.2,
        color="#737b80",
        va="bottom",
    )
    fig.text(0.93, 0.019, str(page_number), fontsize=7.2, color="#737b80", ha="right", va="bottom")


def save_page(pdf: PdfPages, fig: plt.Figure, page_number: int, mixed_run: Path) -> None:
    add_footer(fig, page_number, mixed_run)
    pdf.savefig(fig)
    plt.close(fig)


def wrapped(text: str, width: int) -> str:
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)


def add_box(
    fig: plt.Figure,
    xy: tuple[float, float],
    size: tuple[float, float],
    title: str,
    body: str,
    *,
    facecolor: str = LIGHT_GRAY,
    edgecolor: str = "#c8d0d5",
    title_color: str = BLUE,
    body_size: float = 9.0,
    width: int = 82,
) -> None:
    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        transform=fig.transFigure,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.9,
    )
    fig.add_artist(patch)
    fig.text(x + 0.018, y + h - 0.025, title, fontsize=10.3, fontweight="bold", color=title_color, va="top")
    fig.text(
        x + 0.018,
        y + h - 0.058,
        wrapped(body, width),
        fontsize=body_size,
        color=INK,
        va="top",
        linespacing=1.35,
    )


def page_cover(pdf: PdfPages, page: int, mixed_run: Path, expected_run: Path) -> None:
    fig = plt.figure(figsize=PAGE_SIZE)
    fig.add_artist(FancyBboxPatch((0, 0.73), 1, 0.27, transform=fig.transFigure, facecolor=BLUE, edgecolor="none"))
    fig.text(0.075, 0.92, "MAGIA QUÂNTICA DO ÁTOMO", color="white", fontsize=11.5, fontweight="bold", va="top")
    fig.text(0.075, 0.86, "Reanálise de estados mistos", color="white", fontsize=27, fontweight="bold", va="top")
    fig.text(0.075, 0.805, "Teoria, witnesses, monotones e resultados do artigo", color="#dcecf5", fontsize=13, va="top")

    fig.text(
        0.075,
        0.675,
        "Objetivo",
        color="white",
        fontsize=12,
        fontweight="bold",
        va="top",
    )
    fig.text(
        0.075,
        0.64,
        wrapped(
            "Separar o que foi efetivamente calculado no código daquilo que pode ser "
            "afirmado como magia de um estado misto. O relatório usa os valores de X, Y e Z "
            "já simulados, recalcula os witnesses de Haug e Tarabunga e compara os três "
            "protocolos de modulação presentes na seção de magia do artigo.",
            100,
        ),
        fontsize=10.3,
        color="#edf5fa",
        va="top",
        linespacing=1.45,
    )

    add_box(
        fig,
        (0.075, 0.42),
        (0.85, 0.15),
        "Conclusão central",
        "As curvas antigas de M2 foram reproduzidas com erro máximo de 1,166×10⁻¹⁵, "
        "portanto não há falha algébrica no cálculo antigo. O problema é interpretativo: "
        "M2 positivo não certifica magia quando o átomo está misto. Para este subsistema de "
        "um qubit, W1/2>0 — equivalente a |x|+|y|+|z|>1 — é um teste exato.",
        facecolor="#edf6f2",
        edgecolor="#a8cfc1",
        title_color=GREEN,
        body_size=10,
        width=92,
    )

    add_box(
        fig,
        (0.075, 0.215),
        (0.405, 0.145),
        "O que foi calculado",
        "M2 antigo; pureza e S2; W1/2, W2 e W3; critério exato do octaedro; duração, primeira perda, revivals e valor final.",
        width=42,
    )
    add_box(
        fig,
        (0.52, 0.215),
        (0.405, 0.145),
        "O que não foi calculado",
        "Robustez logarítmica (LR), fidelidade ao conjunto estabilizador (DF) e outros monotones que exigem otimização. Eles aparecem apenas na comparação teórica.",
        width=42,
    )

    fig.text(0.075, 0.115, f"Resultados: {mixed_run}", fontsize=7.7, color=GRAY)
    fig.text(0.075, 0.093, f"Curvas selecionadas: {expected_run}", fontsize=7.7, color=GRAY)
    fig.text(0.075, 0.071, "Gerado diretamente dos NPZ existentes; nenhuma dinâmica foi re-simulada.", fontsize=7.7, color=GRAY)
    add_footer(fig, page, mixed_run)
    pdf.savefig(fig)
    plt.close(fig)


def page_model(pdf: PdfPages, page: int, mixed_run: Path) -> None:
    fig = new_page("1. Modelo físico e convenções numéricas", "O que está escrito nas equações e o que o código realmente executa")

    fig.text(0.075, 0.865, "Hamiltoniano de interação ressonante", fontsize=11, fontweight="bold", color=BLUE)
    fig.text(
        0.16,
        0.815,
        r"$H(t)=g(t)\left(\sigma_+ a+\sigma_- a^\dagger\right),\qquad g_0=1,$",
        fontsize=15,
        color=INK,
    )
    fig.text(0.075, 0.77, wrapped("O código quantum/hamiltonian.py usa exatamente as duas modulações abaixo para os casos de magia.", 105), fontsize=9.5, color=INK)
    fig.text(0.13, 0.715, r"$g_{\cos}(t)=g_0\cos(\omega t+\phi),\quad \phi=0,$", fontsize=13)
    fig.text(0.13, 0.67, r"$g_{\rm G}(t)=g_0\exp\!\left[\sigma\left(\frac{t-T}{\zeta}\right)^2\right],\quad \sigma=-1.$", fontsize=13)

    fig.text(0.075, 0.605, "Equação mestra e taxas", fontsize=11, fontweight="bold", color=BLUE)
    fig.text(
        0.09,
        0.55,
        r"$\dot\rho=-i[H(t),\rho]+\sum_j\left(C_j\rho C_j^\dagger-\frac{1}{2}\{C_j^\dagger C_j,\rho\}\right),$",
        fontsize=12.5,
    )
    fig.text(
        0.115,
        0.495,
        r"$C_\kappa=\sqrt{\kappa}\,a,\qquad C_\gamma=\sqrt{\gamma}\,\sigma_-,\qquad C_\phi=\sqrt{\gamma_\phi}\,\sigma_z,$",
        fontsize=12.2,
    )
    add_box(
        fig,
        (0.075, 0.36),
        (0.85, 0.105),
        "Taxa não é operador de colapso",
        "Os parâmetros fornecidos ao solver são kappa=10⁻¹, gamma=0 e gamma_phi=10⁻². "
        "O código faz corretamente sqrt(kappa), sqrt(gamma) e sqrt(gamma_phi) ao construir "
        "os operadores Cj. Ao expandir C rho C†, as taxas aparecem sem raiz na dissipação.",
        facecolor="#fff6e8",
        edgecolor="#e3c593",
        title_color=ORANGE,
        body_size=9.4,
        width=96,
    )

    fig.text(0.075, 0.305, "Estado inicial usado na seção de magia", fontsize=11, fontweight="bold", color=BLUE)
    fig.text(
        0.087,
        0.25,
        r"$|\Psi(0)\rangle=\left[\cos\!\left(\frac{\theta}{2}\right)|g\rangle+e^{i\pi/4}\sin\!\left(\frac{\theta}{2}\right)|e\rangle\right]\otimes|\alpha\rangle,$",
        fontsize=12.5,
    )
    fig.text(0.235, 0.205, r"$\cos\theta=1/\sqrt{3},\qquad \alpha=\sqrt{5},\qquad \langle n\rangle=|\alpha|^2=5.$", fontsize=12.0)
    fig.text(
        0.075,
        0.145,
        wrapped(
            "A truncagem do modo é Nb=45 nas novas curvas selecionadas. O objeto analisado "
            "é o estado reduzido do átomo, obtido indiretamente pela tomografia de Pauli "
            "x(t)=<X>, y(t)=<Y>, z(t)=<Z>. Referências do modelo: Jaynes–Cummings [1] e "
            "formalismo de Lindblad [2].",
            104,
        ),
        fontsize=9.3,
        color=INK,
        va="top",
        linespacing=1.35,
    )
    save_page(pdf, fig, page, mixed_run)


def page_geometry(pdf: PdfPages, page: int, mixed_run: Path) -> None:
    fig = new_page("2. Geometria da magia para um único qubit", "Aqui existe um critério necessário e suficiente — mais forte que um witness genérico")

    ax = fig.add_axes([0.07, 0.43, 0.48, 0.43], projection="3d")
    vertices = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 1],
            [0, 0, -1],
        ],
        dtype=float,
    )
    faces = [
        [vertices[i], vertices[j], vertices[k]]
        for i in (0, 1)
        for j in (2, 3)
        for k in (4, 5)
        if np.linalg.det(np.vstack([vertices[i], vertices[j], vertices[k]])) != 0
    ]
    poly = Poly3DCollection(faces, alpha=0.18, facecolor="#4c9ed9", edgecolor=BLUE, linewidth=0.8)
    ax.add_collection3d(poly)
    r = 1 / np.sqrt(3)
    ax.quiver(0, 0, 0, r, r, r, color=RED, linewidth=2.5, arrow_length_ratio=0.12)
    ax.scatter([r], [r], [r], color=RED, s=35, label="estado inicial")
    ax.set(xlim=(-1, 1), ylim=(-1, 1), zlim=(-1, 1), xlabel="x", ylabel="y", zlabel="z")
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=36)
    ax.set_title("Octaedro dos estados estabilizadores", color=INK, pad=8)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    fig.text(0.58, 0.84, r"$\rho_A=\frac{1}{2}(I+xX+yY+zZ)$", fontsize=14, color=INK)
    fig.text(0.58, 0.77, r"$\rho_A\in\mathrm{STAB}\;\Longleftrightarrow\;$", fontsize=12, color=INK)
    fig.text(0.61, 0.71, r"$|x|+|y|+|z|\leq1$", fontsize=17, color=GREEN, fontweight="bold")
    fig.text(0.58, 0.63, r"$\rho_A\;\mathrm{m\acute{a}gica}\;\Longleftrightarrow\;$", fontsize=12, color=INK)
    fig.text(0.61, 0.57, r"$|x|+|y|+|z|>1$", fontsize=17, color=RED, fontweight="bold")
    fig.text(
        0.58,
        0.475,
        wrapped(
            "O conjunto livre é o invólucro convexo dos seis autoestados de X, Y e Z. "
            "Logo, a desigualdade é exata inclusive para estados mistos de um qubit [3].",
            43,
        ),
        fontsize=9.2,
        color=INK,
        va="top",
        linespacing=1.35,
    )

    add_box(
        fig,
        (0.075, 0.245),
        (0.85, 0.13),
        "Estado inicial",
        "A fase pi/4 e cos(theta)=1/sqrt(3) produzem x(0)=y(0)=z(0)=1/sqrt(3). "
        "Assim, ||r||1=sqrt(3)>1 e o átomo começa no ponto puro de magia máxima para M2: "
        "W1/2(0)=0,623811; M2(0)=W2(0)=ln(3/2)=0,405465; W3(0)=0,293893.",
        facecolor="#edf6f2",
        edgecolor="#afd0c5",
        title_color=GREEN,
        width=98,
    )
    fig.text(
        0.075,
        0.185,
        wrapped(
            "Importante: essa equivalência geométrica vale porque o subsistema atômico é um único qubit. "
            "Ela não se transfere diretamente para dois qubits nem para o sistema átomo–campo completo.",
            105,
        ),
        fontsize=9.4,
        color=RED,
        va="top",
        linespacing=1.35,
    )
    save_page(pdf, fig, page, mixed_run)


def page_metrics(pdf: PdfPages, page: int, mixed_run: Path) -> None:
    fig = new_page("3. O que cada quantidade significa", "SRE, witnesses e monotones não são sinônimos")
    fig.text(0.075, 0.86, "Definições usadas na reanálise [5]", fontsize=11, fontweight="bold", color=BLUE)
    fig.text(0.095, 0.81, r"$A_\alpha(\rho)=2^{-n}\sum_{P\in\mathcal{P}_n}|\mathrm{Tr}(\rho P)|^{2\alpha},\qquad S_2(\rho)=-\ln\mathrm{Tr}(\rho^2),$", fontsize=12.0)
    fig.text(0.095, 0.755, r"$M_\alpha(\rho)=\frac{\ln A_\alpha(\rho)+S_2(\rho)}{1-\alpha},\qquad \mathcal{W}_\alpha(\rho)=M_\alpha(\rho)-2S_2(\rho).$", fontsize=12.0)
    fig.text(0.095, 0.70, r"$\mathcal{W}_{1/2}=2\ln\!\left(\frac{1+|x|+|y|+|z|}{2}\right)\quad(n=1).$", fontsize=13.2, color=GREEN)

    ax = fig.add_axes([0.065, 0.32, 0.87, 0.32])
    ax.axis("off")
    columns = ["Quantidade", "Tipo", "Se > 0", "Limitação principal"]
    rows = [
        [r"$M_2$ antigo", "SRE estendida", "não certifica no misto", "pode ser >0 em estado livre"],
        [r"$\mathcal{W}_{1/2}$", "witness / teste exato (1 qubit)", "magia; e, aqui, sse", "exatidão especial de n=1"],
        [r"$\mathcal{W}_2$", "witness", "certifica magia", "W<=0 é inconclusivo"],
        [r"$\mathcal{W}_3$", "witness eficiente", "certifica magia", "W<=0 é inconclusivo"],
        [r"$\mathrm{LR}$", "monotone genuíno", "quantifica recurso", "requer otimização"],
        [r"$D_F$", "monotone por fidelidade", "quantifica recurso", "requer otimização sobre STAB"],
    ]
    table = ax.table(cellText=rows, colLabels=columns, loc="center", cellLoc="left", colLoc="left", colWidths=[0.17, 0.28, 0.22, 0.33])
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    table.scale(1.0, 1.75)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd2d6")
        if row == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f7f8")

    add_box(
        fig,
        (0.075, 0.15),
        (0.85, 0.115),
        "Leitura correta",
        "Para alpha>=1/2, W_alpha>0 é condição suficiente de magia [5]. Em geral ela não é "
        "necessária e W_alpha não é monotone: pode aumentar sob operações Clifford e pode ser "
        "não positivo em estados mágicos. O caso W1/2 torna-se necessário e suficiente aqui "
        "somente porque toda a geometria de STAB para um qubit é conhecida.",
        facecolor="#fff6e8",
        edgecolor="#e3c593",
        title_color=ORANGE,
        width=98,
    )
    fig.text(0.075, 0.095, wrapped("A desigualdade 2 LR >= 2 ln D >= W_alpha permite interpretar um witness positivo como um limite inferior, mas este relatório não resolve as otimizações de LR ou DF.", 110), fontsize=8.8, color=INK)
    save_page(pdf, fig, page, mixed_run)


def page_counterexample(pdf: PdfPages, page: int, mixed_run: Path) -> None:
    fig = new_page("4. O M2 antigo pode enganar em estados mistos", "Um contraexemplo analítico dentro do próprio octaedro estabilizador")
    x = np.linspace(0.0, 1.0, 500)
    purity = (1.0 + x**2) / 2.0
    a2 = (1.0 + x**4) / 2.0
    s2 = -np.log(purity)
    m2 = -(np.log(a2) + s2)
    w2 = m2 - 2.0 * s2
    wh = 2.0 * np.log((1.0 + x) / 2.0)

    ax = fig.add_axes([0.105, 0.49, 0.79, 0.36])
    ax.plot(x, m2, color=RED, linewidth=2.1, label=r"$M_2$ antigo")
    ax.plot(x, w2, color=BLUE, linewidth=1.8, label=r"$\mathcal{W}_2$")
    ax.plot(x, wh, color=GREEN, linewidth=1.8, label=r"$\mathcal{W}_{1/2}$")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.fill_between(x, ax.get_ylim()[0], 0, color="#e8f2ed", alpha=0.55)
    ax.set_xlabel(r"$x$ em $\rho_x=(I+xX)/2$")
    ax.set_ylabel("valor")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.set_title("Toda a família é mistura dos estabilizadores |+> e |->")

    fig.text(0.105, 0.425, r"$\rho_x=\frac{1+x}{2}|+\rangle\langle+|+\frac{1-x}{2}|-\rangle\langle-|,\qquad 0\leq x\leq1.$", fontsize=12.5)
    fig.text(0.105, 0.37, r"$M_2(\rho_x)=\ln\!\left(\frac{1+x^2}{1+x^4}\right)>0\quad\mathrm{para}\quad0<x<1,$", fontsize=12.5, color=RED)
    fig.text(0.105, 0.322, r"$\mathcal{W}_{1/2}(\rho_x)\leq0,\quad\mathcal{W}_2(\rho_x)\leq0,$", fontsize=12.5, color=GREEN)

    add_box(
        fig,
        (0.075, 0.145),
        (0.85, 0.12),
        "Consequência para o artigo",
        "A reprodução numérica perfeita de M2 não valida a frase 'a magia permanece em regime "
        "estacionário'. Depois que o átomo se mistura com o campo e com o ambiente, M2 positivo "
        "pode refletir apenas o espectro de Pauli e a pureza. A afirmação de magia deve ser baseada "
        "em W1/2>0 (exato aqui) ou, de modo conservador, em W2/W3>0.",
        facecolor="#faecec",
        edgecolor="#d8aaaa",
        title_color=RED,
        width=100,
    )
    save_page(pdf, fig, page, mixed_run)


def page_pipeline(pdf: PdfPages, page: int, mixed_run: Path, expected_run: Path) -> None:
    fig = new_page("5. Dados, processamento e validação", "Os gráficos seguintes vêm das simulações já salvas")
    steps = [
        ("Dinâmica", "QuTiP / Lindblad\nX(t), Y(t), Z(t)"),
        ("Estado reduzido", "rho_A=(I+xX+yY+zZ)/2"),
        ("Reanálise", "pureza, S2, A_alpha\nM_alpha e W_alpha"),
        ("Diagnóstico", "octaedro exato\ndurações e revivals"),
    ]
    x_positions = [0.075, 0.295, 0.515, 0.735]
    for idx, ((title, body), x0) in enumerate(zip(steps, x_positions)):
        patch = FancyBboxPatch((x0, 0.72), 0.19, 0.12, boxstyle="round,pad=0.008", transform=fig.transFigure, facecolor=LIGHT_BLUE if idx < 3 else "#e6f2ed", edgecolor="#9ab8c9", linewidth=1.0)
        fig.add_artist(patch)
        fig.text(x0 + 0.015, 0.81, title, fontsize=9.8, fontweight="bold", color=BLUE, va="top")
        fig.text(x0 + 0.015, 0.772, body, fontsize=8.2, color=INK, va="top", linespacing=1.35)
        if idx < 3:
            fig.text(x0 + 0.202, 0.775, "→", fontsize=19, color="#7c8c95", ha="center", va="center")

    fig.text(0.075, 0.655, "Cobertura da reanálise", fontsize=11, fontweight="bold", color=BLUE)
    ax = fig.add_axes([0.075, 0.43, 0.85, 0.19])
    ax.axis("off")
    rows = [
        ["fontes encontradas", "909"],
        ["fontes processadas", "849"],
        ["pontos recalculados", "492 900"],
        ["fontes sem X/Y/Z ou inválidas", "36"],
        ["CSVs duplicados, NPZ preferido", "24"],
        ["maior erro ao reproduzir M2 antigo", "1,166 × 10⁻¹⁵"],
    ]
    table = ax.table(cellText=rows, colLabels=["verificação", "resultado"], colLoc="left", cellLoc="left", colWidths=[0.72, 0.28], loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.7)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd2d6")
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f7f8")

    add_box(
        fig,
        (0.075, 0.255),
        (0.85, 0.115),
        "Casos selecionados do artigo",
        "Gaussiano por largura: zeta={6,8,10}, T=25. Gaussiano por pico: T={15,25,35}, "
        "zeta=8. Cosseno: omega={pi/20, pi/10, pi/5}, phi=0. Todos usam t em [0,50], "
        "kappa=10⁻¹, gamma=0, gamma_phi=10⁻² e o mesmo estado inicial mágico.",
        width=100,
    )
    fig.text(0.075, 0.195, "Proveniência", fontsize=10.3, fontweight="bold", color=BLUE)
    fig.text(0.075, 0.165, wrapped(f"Witnesses: {mixed_run}", 115), fontsize=7.7, color=GRAY)
    fig.text(0.075, 0.137, wrapped(f"Curvas selecionadas e g(t): {expected_run}", 115), fontsize=7.7, color=GRAY)
    fig.text(0.075, 0.095, wrapped("Nos mapas completos do cosseno, o arquivo contém omega=0; para ficar fiel à figura/seção de magia, o relatório restringe a análise a [pi/20, pi/5].", 108), fontsize=8.7, color=RED)
    save_page(pdf, fig, page, mixed_run)


def case_result_page(pdf: PdfPages, page: int, mixed_run: Path, case: dict[str, object]) -> None:
    config = case["config"]
    stats = scan_statistics(case)
    fig = new_page(str(config["title"]), r"Mapa completo e curvas selecionadas de $\mathcal{W}_{1/2}$; positivo significa magia exata do átomo")

    ax_map = fig.add_axes([0.10, 0.635, 0.72, 0.225])
    scan = stats["scan"]
    t = stats["time"]
    values = stats["W_half"]
    finite = values[np.isfinite(values)]
    norm = TwoSlopeNorm(vmin=float(np.min(finite)), vcenter=0.0, vmax=float(np.max(finite)))
    image = ax_map.pcolormesh(t, scan, values, cmap="RdBu_r", norm=norm, shading="auto", rasterized=True)
    ax_map.contour(t, scan, values, levels=[0.0], colors="black", linewidths=0.55)
    ax_map.set(xlabel="tempo t", ylabel=str(config["scan_label"]), xlim=(0, 50), ylim=config["limits"])
    ax_map.set_title(r"Varredura completa: contorno preto em $\mathcal{W}_{1/2}=0$")
    cax = fig.add_axes([0.84, 0.635, 0.025, 0.225])
    cb = fig.colorbar(image, cax=cax)
    cb.set_label(r"$\mathcal{W}_{1/2}$")

    ax_curve = fig.add_axes([0.10, 0.355, 0.765, 0.22])
    for key in CURVE_ORDER:
        curve = case["curves"][key]
        ax_curve.plot(curve["time"], curve["W_half"], label=config["curve_labels"][key], **CURVE_STYLES[key])
    ax_curve.axhline(0, color="black", linewidth=0.8)
    ax_curve.set(xlabel="tempo t", ylabel=r"$\mathcal{W}_{1/2}$", xlim=(0, 50))
    ax_curve.grid(alpha=0.18)
    ax_curve.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    inset = ax_curve.inset_axes([0.66, 0.10, 0.30, 0.32])
    for key in CURVE_ORDER:
        curve = case["curves"][key]
        inset.plot(curve["time"], curve["g_t"], **CURVE_STYLES[key])
    inset.set(xlim=(0, 50), xlabel="t", ylabel="g(t)")
    inset.tick_params(labelsize=6.5)
    inset.grid(alpha=0.14)

    ax_table = fig.add_axes([0.075, 0.135, 0.85, 0.15])
    ax_table.axis("off")
    rows = []
    for row in selected_summary(case):
        rows.append(
            [
                str(row["label"]),
                f"{row['duration']:.2f}",
                f"{row['first_loss']:.2f}",
                str(row["revivals"]),
                f"{row['final_w']:+.3f}",
                "sim" if row["final_magic"] else "não",
            ]
        )
    columns = ["curva", "tempo W1/2>0", "1ª perda", "revivals", "W1/2 final", "magia final"]
    table = ax_table.table(cellText=rows, colLabels=columns, cellLoc="center", colLoc="center", loc="center", colWidths=[0.22, 0.19, 0.13, 0.12, 0.15, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(7.7)
    table.scale(1, 1.45)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd2d6")
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f7f8")

    fig.text(0.075, 0.09, wrapped(str(config["interpretation"]), 112), fontsize=8.9, color=INK, va="top", linespacing=1.3)
    save_page(pdf, fig, page, mixed_run)


def page_witness_comparison(pdf: PdfPages, page: int, mixed_run: Path, cases: list[dict[str, object]]) -> None:
    fig = new_page("9. Comparação entre os witnesses", "Sensibilidade diferente não significa contradição")
    labels = ["const."]
    curves = [cases[0]["curves"]["const"]]
    plain_labels = (
        ("ζ=6", "ζ=8", "ζ=10"),
        ("T=15", "T=25", "T=35"),
        ("ω=π/20", "ω=π/10", "ω=π/5"),
    )
    for case, case_labels in zip(cases, plain_labels):
        for key, label in zip(("min", "av", "max"), case_labels):
            labels.append(label)
            curves.append(case["curves"][key])
    metrics = ("W_half", "W_2", "W_3")
    colors = (GREEN, BLUE, ORANGE)
    durations = {metric: [] for metric in metrics}
    for curve in curves:
        for metric in metrics:
            durations[metric].append(positive_duration(curve["time"], curve[metric]))

    ax = fig.add_axes([0.085, 0.57, 0.84, 0.29])
    x = np.arange(len(labels))
    width = 0.24
    for offset, metric, color, label in zip((-width, 0.0, width), metrics, colors, (r"$\mathcal{W}_{1/2}$", r"$\mathcal{W}_2$", r"$\mathcal{W}_3$")):
        ax.bar(x + offset, durations[metric], width, color=color, alpha=0.88, label=label)
    ax.set_ylabel("duração com witness > 0")
    ax.set_xticks(x, labels, rotation=33, ha="right")
    ax.set_ylim(0, 40)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_title("Curvas selecionadas")

    ax2 = fig.add_axes([0.105, 0.245, 0.79, 0.235])
    markers = ("o", "s", "^")
    for case, marker in zip(cases, markers):
        stats = scan_statistics(case)
        ax2.scatter(stats["M_2"][:, -1], stats["W_half"][:, -1], s=20, alpha=0.72, marker=marker, label=case["config"]["scan_label"])
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set(xlabel=r"$M_2$ antigo no tempo final", ylabel=r"$\mathcal{W}_{1/2}$ no tempo final")
    ax2.grid(alpha=0.18)
    ax2.legend(frameon=False, fontsize=7.5)
    ax2.set_title("M2 permanece positivo mesmo quando o teste exato classifica o estado como livre")

    add_box(
        fig,
        (0.075, 0.075),
        (0.85, 0.105),
        "Como ler",
        "W2 e W3 positivos são certificados válidos, mas podem perder estados mágicos: por isso suas "
        "durações são menores. W1/2 é a escolha principal para este artigo porque, no átomo reduzido de "
        "um qubit, seu sinal coincide exatamente com a fronteira do octaedro. M2 não deve ser usado como "
        "certificado de magia mista.",
        width=100,
    )
    save_page(pdf, fig, page, mixed_run)


def format_scan_value(case: dict[str, object], value: float) -> str:
    if case["config"]["case_id"] == "magic_cos_omega":
        return f"{value:.4f} ({value / np.pi:.4f} pi)"
    return f"{value:.4f}"


def page_full_scan(pdf: PdfPages, page: int, mixed_run: Path, cases: list[dict[str, object]]) -> None:
    fig = new_page("10. Comparação das varreduras completas", "Qual parâmetro maximiza a duração da magia exata?")
    stats_list = [scan_statistics(case) for case in cases]
    for idx, (case, stats) in enumerate(zip(cases, stats_list)):
        ax = fig.add_axes([0.08 + idx * 0.305, 0.56, 0.27, 0.29])
        x = stats["scan"]
        if case["config"]["case_id"] == "magic_cos_omega":
            x = x / np.pi
            xlabel = r"$\omega/\pi$"
        else:
            xlabel = str(case["config"]["scan_short"])
        ax.plot(x, stats["durations"], color=GREEN, linewidth=1.8)
        best = stats["best_duration_index"]
        ax.scatter([x[best]], [stats["durations"][best]], color=RED, s=30, zorder=5)
        ax.set(xlabel=xlabel, ylabel="duração W1/2>0")
        ax.grid(alpha=0.18)
        ax.set_title(str(case["config"]["scan_label"]), fontsize=9.5)

    ax_table = fig.add_axes([0.075, 0.30, 0.85, 0.18])
    ax_table.axis("off")
    rows = []
    for case, stats in zip(cases, stats_list):
        best_d = stats["best_duration_index"]
        best_a = stats["best_area_index"]
        rows.append(
            [
                str(case["config"]["scan_label"]),
                format_scan_value(case, float(stats["scan"][best_d])),
                f"{stats['durations'][best_d]:.2f}",
                format_scan_value(case, float(stats["scan"][best_a])),
                f"{100 * stats['final_magic_fraction']:.1f}%",
            ]
        )
    columns = ["varredura", "melhor duração", "duração", "maior área positiva", "mágica no final"]
    table = ax_table.table(cellText=rows, colLabels=columns, loc="center", cellLoc="center", colLoc="center", colWidths=[0.23, 0.22, 0.14, 0.23, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(7.8)
    table.scale(1, 1.55)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#cbd2d6")
        if r == 0:
            cell.set_facecolor(BLUE)
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f5f7f8")

    add_box(
        fig,
        (0.075, 0.13),
        (0.85, 0.115),
        "Síntese física",
        "Nos pulsos gaussianos, a tendência é simples: pulso mais estreito ou mais tardio deixa o "
        "estado inicial mágico exposto à dissipação/interação por uma janela efetiva diferente. No "
        "cosseno, a dependência é não monótona e aparecem bandas de revivals. Dentro do intervalo do "
        "artigo, o ótimo discreto de duração fica perto de omega=0,6157, enquanto pi/5 maximiza a área "
        "positiva entre os pontos amostrados.",
        facecolor="#edf6f2",
        edgecolor="#afd0c5",
        title_color=GREEN,
        width=102,
    )
    fig.text(0.075, 0.082, wrapped("A fração de frequências com magia exata no tempo final é 38% no recorte do cosseno; ela é 0% nas duas varreduras gaussianas. W2 e W3 não permanecem positivos no tempo final em nenhum dos três scans.", 112), fontsize=8.7, color=INK)
    save_page(pdf, fig, page, mixed_run)


def page_recommendations(pdf: PdfPages, page: int, mixed_run: Path) -> None:
    fig = new_page("11. O que mudar no artigo", "Ordem sugerida: do indispensável ao refinamento editorial")
    items = [
        (
            "1 — Trocar a interpretação de M2",
            "Não apresentar M2 como quantificador válido da magia durante a dinâmica mista. Usar W1/2 como diagnóstico principal do átomo e explicar o critério |x|+|y|+|z|>1. M2 pode ficar em comparação histórica, claramente rotulado como não certificador.",
            RED,
        ),
        (
            "2 — Atualizar figuras e conclusões",
            "Substituir mapas/curvas antigos pelos resultados de W1/2. Reescrever qualquer afirmação de 'magia estacionária': nos pontos gaussianos selecionados, o estado final é livre; no cosseno, apenas parte das frequências termina mágica.",
            RED,
        ),
        (
            "3 — Corrigir a definição das taxas",
            "Manter no texto kappa=10^-1, gamma=0 e gamma_phi=10^-2. Se escrever operadores de colapso, usar sqrt(kappa)a, sqrt(gamma)sigma_- e sqrt(gamma_phi)sigma_z. Não substituir as taxas por suas raízes na equação de Lindblad.",
            ORANGE,
        ),
        (
            "4 — Tornar os omegas inequívocos",
            "Os dados selecionados são pi/20, pi/10 e pi/5. Pi/10 é metade de omega_max, mas não é a média aritmética de pi/20 e pi/5; essa média seria pi/8. Portanto chamar pi/10 de ponto intermediário selecionado ou definir explicitamente omega_av=omega_max/2.",
            ORANGE,
        ),
        (
            "5 — Limitar o alcance da referência de 2026",
            "O algoritmo eficiente para alpha ímpar e o teste assintótico de baixa/alta magia tratam estados de n qubits sob hipóteses de entropia [5]. Aqui n=1: calculamos W3, mas não aplicamos o teorema assintótico de property testing.",
            BLUE,
        ),
        (
            "6 — Manter estado inicial e amplitude consistentes",
            "Escrever alpha=sqrt(5) e <n>=5 em toda a seção. O qubit inicial tem x=y=z=1/sqrt(3) e é maximamente mágico para o SRE puro considerado.",
            BLUE,
        ),
    ]
    y = 0.855
    for title, body, color in items:
        fig.text(0.085, y, title, fontsize=10.2, fontweight="bold", color=color, va="top")
        fig.text(0.105, y - 0.035, wrapped(body, 104), fontsize=8.8, color=INK, va="top", linespacing=1.32)
        y -= 0.118

    add_box(
        fig,
        (0.075, 0.06),
        (0.85, 0.095),
        "Decisão recomendada para a narrativa",
        "Apresentar W1/2 como teste exato de não estabilizerness do átomo; W2 e W3 como witnesses "
        "suficientes e mais conservadores; LR e DF como monotones genuínos não avaliados. Isso torna as "
        "afirmações compatíveis com estados mistos sem esconder o resultado numérico anterior.",
        facecolor="#edf6f2",
        edgecolor="#afd0c5",
        title_color=GREEN,
        width=100,
    )
    save_page(pdf, fig, page, mixed_run)


def page_references(pdf: PdfPages, page: int, mixed_run: Path) -> None:
    fig = new_page("12. Referências e rastreabilidade", "Fontes primárias usadas para as definições e interpretações")
    references = [
        "[1] E. T. Jaynes and F. W. Cummings, Comparison of quantum and semiclassical radiation theories with application to the beam maser, Proc. IEEE 51, 89–109 (1963). DOI: 10.1109/PROC.1963.1664.",
        "[2] G. Lindblad, On the generators of quantum dynamical semigroups, Communications in Mathematical Physics 48, 119–130 (1976). DOI: 10.1007/BF01608499.",
        "[3] S. Bravyi and A. Kitaev, Universal quantum computation with ideal Clifford gates and noisy ancillas, Physical Review A 71, 022316 (2005). DOI: 10.1103/PhysRevA.71.022316. O octaedro estabilizador de um qubit aparece na discussão introdutória associada à Fig. 1.",
        "[4] L. Leone, S. F. E. Oliviero and A. Hamma, Stabilizer Rényi Entropy, Physical Review Letters 128, 050402 (2022). DOI: 10.1103/PhysRevLett.128.050402. Introduz o SRE como medida de magia para estados puros.",
        "[5] T. Haug and P. S. Tarabunga, Efficient witnessing and testing of magic in mixed quantum states, npj Quantum Information 12, 40 (2026). DOI: 10.1038/s41534-026-01189-z. Eqs. (1)–(5) definem W_alpha, A_alpha, M_alpha e a relação com a norma estabilizadora; Eqs. (7)–(9) discutem limites para monotones.",
        "[6] V. Veitch, S. A. H. Mousavian, D. Gottesman and J. Emerson, The resource theory of stabilizer quantum computation, New Journal of Physics 16, 013009 (2014). DOI: 10.1088/1367-2630/16/1/013009. Formaliza estados estabilizadores como objetos livres e monotonicidade do recurso.",
        "[7] S. F. E. Oliviero, L. Leone, A. Hamma and S. Lloyd, Measuring magic on a quantum processor, npj Quantum Information 8 (2022). DOI: 10.1038/s41534-022-00666-5. Discute acesso experimental ao SRE por medidas randomizadas.",
    ]
    y = 0.86
    for ref in references:
        fig.text(0.085, y, wrapped(ref, 110), fontsize=8.6, color=INK, va="top", linespacing=1.34)
        y -= 0.085 if len(ref) < 220 else 0.10

    add_box(
        fig,
        (0.075, 0.055),
        (0.85, 0.095),
        "Arquivos reprodutíveis",
        "Script: generate_mixed_state_magic_report.py. Dados: results/mixed_state_magic_witnesses "
        "e results/expected_values_all_cases_xy. Definições: quantum/non_classicality.py.",
        body_size=8.3,
        width=112,
    )
    save_page(pdf, fig, page, mixed_run)


def main() -> int:
    configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mixed_run, expected_run, cases = load_all_data()

    with PdfPages(OUTPUT_PDF) as pdf:
        page_cover(pdf, 1, mixed_run, expected_run)
        page_model(pdf, 2, mixed_run)
        page_geometry(pdf, 3, mixed_run)
        page_metrics(pdf, 4, mixed_run)
        page_counterexample(pdf, 5, mixed_run)
        page_pipeline(pdf, 6, mixed_run, expected_run)
        page_number = 7
        for case in cases:
            case_result_page(pdf, page_number, mixed_run, case)
            page_number += 1
        page_witness_comparison(pdf, page_number, mixed_run, cases)
        page_number += 1
        page_full_scan(pdf, page_number, mixed_run, cases)
        page_number += 1
        page_recommendations(pdf, page_number, mixed_run)
        page_number += 1
        page_references(pdf, page_number, mixed_run)

        metadata = pdf.infodict()
        metadata["Title"] = "Magia quântica do átomo: reanálise de estados mistos"
        metadata["Author"] = "Relatório numérico reprodutível"
        metadata["Subject"] = "TDJC, mixed-state magic witnesses and manuscript audit"
        metadata["Keywords"] = "quantum magic, mixed states, Jaynes-Cummings, stabilizer witnesses"

    print(f"Relatório: {OUTPUT_PDF}")
    print(f"Páginas: 13")
    print(f"Dados de witnesses: {mixed_run}")
    print(f"Curvas selecionadas: {expected_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
