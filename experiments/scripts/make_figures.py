"""Produce Fig. 1–9 as PNG + PDF under figures/."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments"))

from lpdp.config import load_config, resolve_path  # noqa: E402
from lpdp.eval.pareto import non_dominated  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "figure.dpi": 140,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

MECH_COLOR = {
    "M0": "#1d3557",
    "M1": "#457b9d",
    "M2": "#2a9d8f",
    "M3": "#e9c46a",
    "M4": "#e76f51",
    "M5": "#6d597a",
    "M6": "#b56576",
    "M7": "#4a4e69",
}


def _save(fig: plt.Figure, figures: Path, stem: str) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{stem}.png")
    fig.savefig(figures / f"{stem}.pdf")
    plt.close(fig)


def fig1_framework(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    boxes = [
        (0.04, "Raw check-ins"),
        (0.24, "Mechanisms\nM0–M7"),
        (0.44, "Tasks\nT1 next-POI\nT2 hotspots"),
        (0.64, "Attacks\nA1 re-id\nA2 MIA"),
        (0.84, "DPDP\nchecklist"),
    ]
    for x, label in boxes:
        ax.add_patch(
            plt.Rectangle((x, 0.35), 0.14, 0.4, fill=True, facecolor="#eef3f8", edgecolor="#1d3557", lw=1.4)
        )
        ax.text(x + 0.07, 0.55, label, ha="center", va="center", fontsize=8)
    for x in (0.18, 0.38, 0.58, 0.78):
        ax.annotate("", xy=(x + 0.06, 0.55), xytext=(x, 0.55), arrowprops=dict(arrowstyle="->", color="#1d3557"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Fig. 1  Framework overview")
    _save(fig, figures, "fig1_framework")


def fig2_threat(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.05, 0.25), 0.38, 0.55, fill=False, edgecolor="#e76f51", lw=1.6))
    ax.text(0.24, 0.72, "A1 Re-identification", ha="center", fontweight="bold")
    ax.text(0.24, 0.48, "Attacker holds p raw points\n+ released dataset\nwith pseudonymous IDs", ha="center", fontsize=8)
    ax.add_patch(plt.Rectangle((0.55, 0.25), 0.38, 0.55, fill=False, edgecolor="#6d597a", lw=1.6))
    ax.text(0.74, 0.72, "A2 Membership inference", ha="center", fontweight="bold")
    ax.text(0.74, 0.48, "Attacker queries the model\nmean CE loss on a user's\nwindows → member?", ha="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Fig. 2  Threat model")
    _save(fig, figures, "fig2_threat")


def fig3_placement(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    layers = [
        (0.08, "Client", "M1 cloaking\nM3 geo-ind"),
        (0.36, "Curator", "M2 k-anon\nM7 DP hist."),
        (0.64, "Model training", "M4 DP-SGD\nM5 FedAvg\nM6 DP-FedAvg"),
    ]
    for x, title, body in layers:
        ax.add_patch(plt.Rectangle((x, 0.28), 0.24, 0.5, facecolor="#f7f4ef", edgecolor="#1d3557"))
        ax.text(x + 0.12, 0.68, title, ha="center", fontweight="bold")
        ax.text(x + 0.12, 0.46, body, ha="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Fig. 3  Mechanism placement")
    _save(fig, figures, "fig3_placement")


def fig4_federated(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.axis("off")
    ax.add_patch(plt.Circle((0.5, 0.62), 0.12, fill=False, edgecolor="#1d3557", lw=1.6))
    ax.text(0.5, 0.62, "Server\n+ noise (M6)", ha="center", fontsize=8)
    for x, label in ((0.18, "Client i\nclip Δ"), (0.82, "Client j\nclip Δ"), (0.5, "Client k\nclip Δ")):
        y = 0.22 if label.startswith("Client k") else 0.22
        ax.add_patch(plt.Rectangle((x - 0.08, y), 0.16, 0.18, fill=False, edgecolor="#2a9d8f"))
        ax.text(x, y + 0.09, label, ha="center", fontsize=7)
        ax.annotate("", xy=(0.5, 0.50), xytext=(x, y + 0.18), arrowprops=dict(arrowstyle="->", color="#1d3557"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Fig. 4  Federated setup (DP-FedAvg)")
    _save(fig, figures, "fig4_federated")


def _load_runs(dataset: str) -> pd.DataFrame:
    cfg = load_config(dataset)
    path = resolve_path(cfg, "results") / "runs.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df["dataset"] == dataset].copy()


def fig5_utility(df: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    poi = df[df["task"] == "next_poi"]
    hot = df[df["task"] == "hotspot"]
    for mech, g in poi.groupby("mechanism"):
        g = g.dropna(subset=["acc10"]).groupby("param_value", as_index=False)["acc10"].mean()
        axes[0].plot(g["param_value"], g["acc10"], marker="o", label=mech, color=MECH_COLOR.get(mech))
    axes[0].set_xlabel("Mechanism parameter")
    axes[0].set_ylabel("Acc@10")
    axes[0].set_title("Next-POI")
    axes[0].legend(fontsize=7, ncol=2)
    for mech, g in hot.groupby("mechanism"):
        g = g.dropna(subset=["hotspot_f1"]).groupby("param_value", as_index=False)["hotspot_f1"].mean()
        axes[1].plot(g["param_value"], g["hotspot_f1"], marker="o", label=mech, color=MECH_COLOR.get(mech))
    axes[1].set_xlabel("Mechanism parameter")
    axes[1].set_ylabel("Hotspot F1")
    axes[1].set_title("Hotspots")
    axes[1].legend(fontsize=7, ncol=2)
    fig.suptitle("Fig. 5  Privacy–utility curves (parameter is mechanism-specific)")
    _save(fig, figures, "fig5_utility")


def fig6_attacks(df: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6))
    sub = df.groupby(["mechanism", "param_value"], as_index=False)[["reid_top1_p4", "mia_auc"]].mean()
    for mech, g in sub.groupby("mechanism"):
        if g["reid_top1_p4"].notna().any():
            axes[0].plot(g["param_value"], g["reid_top1_p4"], marker="o", label=mech, color=MECH_COLOR.get(mech))
        if g["mia_auc"].notna().any():
            axes[1].plot(g["param_value"], g["mia_auc"], marker="o", label=mech, color=MECH_COLOR.get(mech))
    axes[0].axhline(0.05, ls="--", color="#999", lw=0.8, label="policy 5%")
    axes[1].axhline(0.55, ls="--", color="#999", lw=0.8, label="policy 0.55")
    axes[0].set_title("A1 top-1 re-id (p=4)")
    axes[1].set_title("A2 MIA AUC")
    for ax in axes:
        ax.set_xlabel("Mechanism parameter")
        ax.legend(fontsize=7)
    fig.suptitle("Fig. 6  Attack success versus budget")
    _save(fig, figures, "fig6_attacks")


def fig7_pareto(df: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    poi = df[df["task"] == "next_poi"].copy()
    agg = (
        poi.groupby(["mechanism", "param_value"], as_index=False)
        .agg(acc10=("acc10", "mean"), mia_auc=("mia_auc", "mean"))
        .dropna()
    )
    if agg.empty:
        ax.text(0.5, 0.5, "No next-POI results yet", ha="center")
    else:
        front = non_dominated(agg, "acc10", "mia_auc")
        for mech, g in agg.groupby("mechanism"):
            ax.scatter(g["mia_auc"], g["acc10"], label=mech, color=MECH_COLOR.get(mech), s=40)
        if not front.empty:
            ax.scatter(front["mia_auc"], front["acc10"], facecolors="none", edgecolors="black", s=110, label="Pareto")
    ax.set_xlabel("Privacy risk (MIA AUC, lower is better)")
    ax.set_ylabel("Utility (Acc@10)")
    ax.legend(fontsize=7)
    ax.set_title("Fig. 7  Pareto front (empirical attack success as the common axis)")
    _save(fig, figures, "fig7_pareto")


def fig8_map(figures: Path, dataset: str = "synthetic") -> None:
    from lpdp.data.preprocess import load_processed
    from lpdp.mechanisms.geoind import geo_indistinguishability

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    try:
        checkins, pois, _ = load_processed(dataset)
    except FileNotFoundError:
        ax.text(0.5, 0.5, "Process a dataset first", ha="center")
        _save(fig, figures, "fig8_map")
        return
    sample = checkins.sample(n=min(80, len(checkins)), random_state=0)
    rng = np.random.default_rng(0)
    pert = geo_indistinguishability(sample, pois, eps_per_km=2.0, rng=rng)
    ax.scatter(sample["lon"], sample["lat"], s=18, c="#1d3557", label="raw", alpha=0.85)
    ax.scatter(pert["lon"], pert["lat"], s=18, c="#e76f51", label="M3 ε=2/km", alpha=0.85)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend()
    ax.set_title("Fig. 8  Spatial noise example")
    ax.set_aspect("equal", adjustable="datalim")
    _save(fig, figures, "fig8_map")


def fig9_dpdp(figures: Path, dataset: str) -> None:
    from lpdp.compliance.evaluate import evaluate_compliance

    df = evaluate_compliance(dataset)
    mat = df.pivot_table(index="obligation", columns="mechanism_id", values="status", aggfunc="first")
    order = ["pass", "partial", "fail", "na"]
    colors = {"pass": "#2a9d8f", "partial": "#e9c46a", "fail": "#e76f51", "na": "#ced4da"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    numeric = mat.replace({k: i for i, k in enumerate(order)}).astype(float)
    cmap = plt.matplotlib.colors.ListedColormap([colors[k] for k in order])
    ax.imshow(numeric.to_numpy(), cmap=cmap, vmin=0, vmax=3, aspect="auto")
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels([s[:42] for s in mat.index], fontsize=8)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, mat.iloc[i, j], ha="center", va="center", fontsize=7)
    ax.set_title("Fig. 9  DPDP obligation × mechanism (indicative, not legal advice)")
    _save(fig, figures, "fig9_dpdp")


def make_all(dataset: str = "synthetic") -> None:
    cfg = load_config(dataset)
    figures = resolve_path(cfg, "figures")
    fig1_framework(figures)
    fig2_threat(figures)
    fig3_placement(figures)
    fig4_federated(figures)
    df = _load_runs(dataset)
    fig5_utility(df, figures)
    fig6_attacks(df, figures)
    fig7_pareto(df, figures)
    fig8_map(figures, dataset)
    fig9_dpdp(figures, dataset)
    print(f"Figures written to {figures}")


if __name__ == "__main__":
    make_all()
