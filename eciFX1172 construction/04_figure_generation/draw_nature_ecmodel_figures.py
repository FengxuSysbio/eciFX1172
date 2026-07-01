import json
import math
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "ec_iFX1172_final_calibrated"
ANALYSIS = BASE / "analysis"
V2 = BASE / "advanced_analysis_v2"
TABLES = V2 / "tables"
OUT = BASE / "nature_figures"
FIG_DIR = OUT / "figures"
SOURCE_DIR = OUT / "source_data"


PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "green": "#2E9E44",
    "green_soft": "#AADCA9",
    "red": "#B64342",
    "red_soft": "#E9A6A1",
    "gold": "#D7A928",
    "violet": "#7A5195",
    "neutral_dark": "#272727",
    "neutral_mid": "#767676",
    "neutral_light": "#D7D7D7",
    "neutral_pale": "#F3F3F3",
}


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
        }
    )


def ensure_dirs():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def save_pub(fig, name):
    stem = FIG_DIR / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, label, x=-0.08, y=1.05):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_round_box(ax, xy, w, h, text, fc, ec=None, fontsize=6.8, text_color="#222", lw=0.9):
    box = patches.FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=fc,
        edgecolor=ec or fc,
        linewidth=lw,
    )
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=fontsize, color=text_color)
    return box


def arrow(ax, p1, p2, color=None, lw=1.2, rad=0.0):
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color or PALETTE["neutral_dark"], shrinkA=3, shrinkB=3, connectionstyle=f"arc3,rad={rad}"),
    )


def arrow_plain(ax, p1, p2, color=None, lw=1.0, rad=0.0):
    ax.annotate(
        "",
        xy=p2,
        xytext=p1,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color or PALETTE["neutral_dark"], shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}"),
    )


def reaction_class_table():
    detail = read_csv("pathway_detail")
    rows = []
    for _, row in detail.iterrows():
        rid = str(row["reaction"])
        if rid.startswith("EX_"):
            cls = "Exchange"
        elif rid.startswith(("DM_", "SK_", "sink_")):
            cls = "Demand/sink"
        elif re.match(r"r_000[8-9]|r_001[0-3]", rid):
            cls = "Gentamicin module"
        elif "transport" in rid.lower() or rid.endswith("t") or "abc" in rid.lower():
            cls = "Transport"
        elif bool(row.get("enzyme_constrained", False)) or str(row.get("enzyme_constrained", "")).lower() == "true":
            cls = "Enzyme-constrained metabolic"
        elif bool(row.get("has_gene", False)) or str(row.get("has_gene", "")).lower() == "true":
            cls = "Gene-associated metabolic"
        else:
            cls = "Other metabolic"
        rows.append({"model": row["model"], "reaction_class": cls, "reaction": rid})
    table = pd.DataFrame(rows)
    return table.groupby(["model", "reaction_class"], as_index=False).agg(reaction_count=("reaction", "count"))


def load_summary_numbers():
    summary = pd.read_csv(ANALYSIS / "model_summary.csv").iloc[0].to_dict()
    reaction_kcat = pd.read_csv(ANALYSIS / "reaction_kcat_MW.csv")
    gene_mass = pd.read_csv(ANALYSIS / "gene_protein_mass.csv")
    memote = read_csv("memote_qc")
    substrate = read_csv("substrate_panel")
    single = read_csv("single_gene_ko")
    targets = read_csv("metastrain_targets")
    numbers = {
        "gem_growth": float(memote.loc[memote["model"] == "iFX1172", "growth"].iloc[0]),
        "ec_growth": float(memote.loc[memote["model"] == "eciFX1172", "growth"].iloc[0]),
        "reactions": int(memote.loc[memote["model"] == "eciFX1172", "reactions"].iloc[0]),
        "metabolites": int(memote.loc[memote["model"] == "eciFX1172", "metabolites"].iloc[0]),
        "genes": int(memote.loc[memote["model"] == "eciFX1172", "genes"].iloc[0]),
        "enzyme_reactions": int(len(reaction_kcat)),
        "carbon_sources": substrate[substrate["panel"] == "carbon"]["substrate"].nunique(),
        "amino_acids": substrate[substrate["panel"] == "amino_acid"]["substrate"].nunique(),
        "essential_genes": int((single["phenotype"] == "essential").sum()),
        "targets": int(targets["gene"].nunique()) if "gene" in targets else 0,
    }
    return summary, reaction_kcat, gene_mass, numbers


def figure1_construction():
    summary, reaction_kcat, gene_mass, numbers = load_summary_numbers()
    fig = plt.figure(figsize=(7.2, 5.1))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], width_ratios=[1.45, 1, 1], hspace=0.38, wspace=0.32)
    ax_flow = fig.add_subplot(gs[0, :])
    ax_kcat = fig.add_subplot(gs[1, 0])
    ax_mass = fig.add_subplot(gs[1, 1])
    ax_metric = fig.add_subplot(gs[1, 2])

    clean_axis(ax_flow)
    panel_label(ax_flow, "a", -0.015, 1.02)
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, 1)
    ax_flow.text(0.02, 0.94, "Construction of enzyme-constrained iFX1172", fontsize=9, fontweight="bold")
    w, h = 0.135, 0.17
    boxes = {
        "gem": (0.035, 0.53, "iFX1172 GEM\n5,044 reactions\n1,172 genes", PALETTE["neutral_pale"]),
        "irr": (0.225, 0.66, "Irreversible\nconversion", "#EAF1FA"),
        "prot": (0.225, 0.34, "Protein\nannotation", "#EAF1FA"),
        "kcat": (0.430, 0.66, "kcat\nassignment", "#EFF7EF"),
        "mw": (0.430, 0.34, "Molecular\nmass", "#EFF7EF"),
        "usage": (0.650, 0.50, "Enzyme usage\nv/(kcat/MW)", "#FFF5D8"),
        "ec": (0.835, 0.50, "eciFX1172\nprotein-pool\nconstrained", "#F6E7EA"),
    }
    for x, y, text, fc in boxes.values():
        draw_round_box(ax_flow, (x, y), w, h, text, fc, ec=PALETTE["neutral_light"], fontsize=6.0)
    def mid_right(key):
        x, y, *_ = boxes[key]
        return (x + w, y + h / 2)
    def mid_left(key):
        x, y, *_ = boxes[key]
        return (x, y + h / 2)
    arrow_plain(ax_flow, mid_right("gem"), mid_left("irr"), lw=0.95)
    arrow_plain(ax_flow, mid_right("gem"), mid_left("prot"), lw=0.95)
    arrow_plain(ax_flow, mid_right("irr"), mid_left("kcat"), lw=0.95)
    arrow_plain(ax_flow, mid_right("prot"), mid_left("mw"), lw=0.95)
    arrow_plain(ax_flow, mid_right("kcat"), (boxes["usage"][0], boxes["usage"][1] + h * 0.72), lw=0.95)
    arrow_plain(ax_flow, mid_right("mw"), (boxes["usage"][0], boxes["usage"][1] + h * 0.28), lw=0.95)
    arrow_plain(ax_flow, mid_right("usage"), mid_left("ec"), lw=0.95)
    ax_flow.text(
        0.73,
        0.26,
        r"$\sum_i v_i/(k_{cat,i}/MW_i)\leq P_{pool}$",
        fontsize=8.2,
        color=PALETTE["neutral_dark"],
        ha="center",
        va="center",
    )
    ax_flow.text(0.84, 0.30, "calibrated enzyme pool", fontsize=6.2, ha="center", color=PALETTE["neutral_mid"])

    panel_label(ax_kcat, "b")
    source_counts = reaction_kcat["data_type"].fillna("unknown").value_counts().head(8)
    colors = [PALETTE["blue_main"], PALETTE["teal"], PALETTE["green_soft"], PALETTE["gold"], PALETTE["red_soft"], "#B4C0E4", "#E4CCD8", PALETTE["neutral_light"]]
    y = np.arange(len(source_counts))[::-1]
    ax_kcat.barh(y, source_counts.values[::-1], color=colors[: len(source_counts)][::-1])
    ax_kcat.set_yticks(y)
    ax_kcat.set_yticklabels([str(x).replace("_", " ")[:23] for x in source_counts.index[::-1]], fontsize=6)
    ax_kcat.set_xlabel("Reaction count")
    ax_kcat.set_title("kcat/MW evidence")

    panel_label(ax_mass, "c")
    mass_counts = gene_mass["mass_source"].fillna("unknown").value_counts()
    wedges, _ = ax_mass.pie(
        mass_counts.values,
        colors=[PALETTE["blue_secondary"], PALETTE["neutral_light"], PALETTE["gold"], PALETTE["red_soft"]][: len(mass_counts)],
        startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=0.8),
    )
    ax_mass.text(0, 0, f"{len(gene_mass):,}\ngenes", ha="center", va="center", fontsize=8, fontweight="bold")
    ax_mass.set_title("Protein-mass coverage")
    ax_mass.legend(wedges, [str(x).replace("_", " ")[:20] for x in mass_counts.index], loc="lower center", bbox_to_anchor=(0.5, -0.28), fontsize=5.8, ncol=1)

    panel_label(ax_metric, "d")
    clean_axis(ax_metric)
    metrics = [
        ("Reactions", f"{numbers['reactions']:,}"),
        ("Metabolites", f"{numbers['metabolites']:,}"),
        ("Genes", f"{numbers['genes']:,}"),
        ("Enzyme-constrained\nreactions", f"{numbers['enzyme_reactions']:,}"),
        ("Growth ratio\nec/GEM", f"{numbers['ec_growth'] / numbers['gem_growth']:.2f}"),
    ]
    ax_metric.set_xlim(0, 1)
    ax_metric.set_ylim(0, 1)
    for i, (label, value) in enumerate(metrics):
        y0 = 0.82 - i * 0.18
        draw_round_box(ax_metric, (0.03, y0), 0.43, 0.12, label, PALETTE["neutral_pale"], ec="white", fontsize=5.9)
        draw_round_box(ax_metric, (0.52, y0), 0.36, 0.12, value, "#EAF1FA", ec="white", fontsize=7.5, text_color=PALETTE["blue_main"])
    ax_metric.set_title("Model scale")

    fig.suptitle("Fig. 1 | Building an enzyme-constrained genome-scale model for gentamicin biosynthesis", x=0.01, y=1.01, ha="left", fontsize=10, fontweight="bold")
    source_counts.reset_index().rename(columns={"index": "kcat_source", "data_type": "reaction_count"}).to_csv(SOURCE_DIR / "figure1_kcat_sources.csv", index=False)
    mass_counts.reset_index().rename(columns={"index": "mass_source", "mass_source": "gene_count"}).to_csv(SOURCE_DIR / "figure1_mass_sources.csv", index=False)
    save_pub(fig, "figure1_ecmodel_construction")


def figure2_validation():
    memote = read_csv("memote_qc")
    substrate = read_csv("substrate_panel")
    robustness = read_csv("robustness")
    phase = read_csv("phase_plane")
    product_phase = read_csv("product_phase_plane")

    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, width_ratios=[0.9, 1.15, 1.15], height_ratios=[1, 1], hspace=0.48, wspace=0.45)
    ax_growth = fig.add_subplot(gs[0, 0])
    ax_qc = fig.add_subplot(gs[1, 0])
    ax_sub = fig.add_subplot(gs[:, 1])
    ax_rob = fig.add_subplot(gs[0, 2])
    ax_phase = fig.add_subplot(gs[1, 2])

    panel_label(ax_growth, "a")
    models = ["iFX1172", "eciFX1172"]
    growth = [float(memote.loc[memote["model"] == m, "growth"].iloc[0]) for m in models]
    ax_growth.bar([0, 1], growth, color=[PALETTE["neutral_mid"], PALETTE["blue_main"]], width=0.62)
    for i, v in enumerate(growth):
        ax_growth.text(i, v + 0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=6)
    ax_growth.set_xticks([0, 1])
    ax_growth.set_xticklabels(models, rotation=25, ha="right")
    ax_growth.set_ylabel("Growth rate (h$^{-1}$)")
    ax_growth.set_title("Growth calibration")
    ax_growth.set_ylim(0, max(growth) * 1.28)

    panel_label(ax_qc, "b")
    qc_metrics = ["formula_coverage", "charge_coverage", "gpr_coverage"]
    x = np.arange(len(qc_metrics))
    for off, model, color in [(-0.18, "iFX1172", PALETTE["neutral_mid"]), (0.18, "eciFX1172", PALETTE["blue_main"])]:
        vals = [float(memote.loc[memote["model"] == model, m].iloc[0]) for m in qc_metrics]
        ax_qc.bar(x + off, vals, width=0.32, color=color, label=model)
    ax_qc.set_ylim(0, 1.05)
    ax_qc.set_xticks(x)
    ax_qc.set_xticklabels(["Formula", "Charge", "GPR"], rotation=25, ha="right")
    ax_qc.set_ylabel("Coverage")
    ax_qc.set_title("MEMOTE-style QC")

    panel_label(ax_sub, "c")
    sub_pivot = substrate.pivot_table(index="substrate", columns=["panel", "model"], values="growth", aggfunc="mean")
    order = (
        substrate.groupby("substrate")["growth"]
        .max()
        .sort_values(ascending=False)
        .index.tolist()
    )
    order = order[:28] + [x for x in order[28:44]]
    heat = []
    labels = []
    for s in order:
        row = []
        for panel in ["carbon", "amino_acid"]:
            for model in ["iFX1172", "eciFX1172"]:
                try:
                    row.append(sub_pivot.loc[s, (panel, model)])
                except Exception:
                    row.append(np.nan)
        if not all(pd.isna(row)):
            heat.append(row)
            labels.append(s)
    heat = np.array(heat, dtype=float)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="white")
    im = ax_sub.imshow(np.ma.masked_invalid(heat), aspect="auto", cmap=cmap, norm=Normalize(vmin=0, vmax=np.nanmax(heat) if np.nanmax(heat) > 0 else 1))
    ax_sub.set_yticks(np.arange(len(labels)))
    ax_sub.set_yticklabels(labels, fontsize=4.8)
    ax_sub.set_xticks(range(4))
    ax_sub.set_xticklabels(["C\nGEM", "C\nec", "AA\nGEM", "AA\nec"], fontsize=6)
    ax_sub.tick_params(length=0)
    ax_sub.set_title("24 carbon sources and 20 amino acids")
    fig.colorbar(im, ax=ax_sub, fraction=0.035, pad=0.02, label="Growth")

    panel_label(ax_rob, "d")
    for model, color in [("iFX1172", PALETTE["neutral_mid"]), ("eciFX1172", PALETTE["blue_main"])]:
        d = robustness[robustness["model"] == model]
        ax_rob.plot(d["glucose_uptake"], d["growth"], color=color, lw=1.7, marker="o", markersize=2.2, label=model)
    ax_rob.set_xlabel("Glucose uptake bound")
    ax_rob.set_ylabel("Growth")
    ax_rob.set_title("Robustness")
    ax_rob.legend(fontsize=6, loc="lower right")

    panel_label(ax_phase, "e")
    pivot = phase.pivot(index="enzyme_pool_factor", columns="glucose_uptake", values="growth")
    im2 = ax_phase.imshow(pivot.values, origin="lower", aspect="auto", cmap="viridis")
    ax_phase.set_xticks([0, len(pivot.columns) - 1])
    ax_phase.set_xticklabels([f"{pivot.columns[0]:.1f}", f"{pivot.columns[-1]:.1f}"])
    ax_phase.set_yticks([0, len(pivot.index) - 1])
    ax_phase.set_yticklabels([f"{pivot.index[0]:.1f}", f"{pivot.index[-1]:.1f}"])
    ax_phase.set_xlabel("Glucose")
    ax_phase.set_ylabel("Enzyme pool")
    ax_phase.set_title("Phenotypic phase plane")
    fig.colorbar(im2, ax=ax_phase, fraction=0.045, pad=0.02, label="Growth")

    fig.suptitle("Fig. 2 | Validation shows enzyme-capacity restriction and preserved model quality", x=0.01, y=1.01, ha="left", fontsize=10, fontweight="bold")
    memote.to_csv(SOURCE_DIR / "figure2_memote_qc.csv", index=False)
    substrate.to_csv(SOURCE_DIR / "figure2_substrate_panel.csv", index=False)
    robustness.to_csv(SOURCE_DIR / "figure2_robustness.csv", index=False)
    phase.to_csv(SOURCE_DIR / "figure2_phase_plane.csv", index=False)
    save_pub(fig, "figure2_ecmodel_validation")


def figure3_analysis_prediction():
    pathway = reaction_class_table()
    single = read_csv("single_gene_ko")
    double = read_csv("double_gene_ko")
    dfba = read_csv("dfba")
    proxy = read_csv("dfba_intracellular_proxy")
    fseof = read_csv("fseof_gene_targets")
    targets = read_csv("target_algorithms")
    meta = read_csv("metastrain_targets")
    product_phase = read_csv("product_phase_plane")

    fig = plt.figure(figsize=(7.25, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.95, 1.05, 1.1], hspace=0.55, wspace=0.42)
    ax_path = fig.add_subplot(gs[0, 0])
    ax_pp = fig.add_subplot(gs[0, 1])
    ax_dfba = fig.add_subplot(gs[0, 2])
    ax_proxy = fig.add_subplot(gs[1, 0])
    ax_single = fig.add_subplot(gs[1, 1])
    ax_double = fig.add_subplot(gs[1, 2])
    ax_fseof = fig.add_subplot(gs[2, 0])
    ax_alg = fig.add_subplot(gs[2, 1])
    ax_meta = fig.add_subplot(gs[2, 2])

    panel_label(ax_path, "a")
    top = pathway.groupby("reaction_class")["reaction_count"].sum().sort_values(ascending=False).head(7).index
    p = pathway[pathway["reaction_class"].isin(top)].pivot(index="reaction_class", columns="model", values="reaction_count").fillna(0).loc[top[::-1]]
    y = np.arange(len(p))
    ax_path.barh(y - 0.16, p.get("iFX1172", pd.Series(0, index=p.index)), height=0.30, color=PALETTE["neutral_mid"], label="GEM")
    ax_path.barh(y + 0.16, p.get("eciFX1172", pd.Series(0, index=p.index)), height=0.30, color=PALETTE["blue_main"], label="ec")
    ax_path.set_yticks(y)
    ax_path.set_yticklabels([str(x)[:22] for x in p.index], fontsize=5.2)
    ax_path.set_xlabel("Reactions")
    ax_path.set_title("Pathway distribution")
    ax_path.legend(fontsize=5.5)

    panel_label(ax_pp, "b")
    pivot = product_phase.pivot(index="growth_fraction_constraint", columns="product_fraction_constraint", values="feasible_growth")
    im = ax_pp.imshow(pivot.values, origin="lower", aspect="auto", cmap="magma")
    ax_pp.set_xticks([0, len(pivot.columns) - 1])
    ax_pp.set_xticklabels([f"{pivot.columns[0]:.1f}", f"{pivot.columns[-1]:.1f}"])
    ax_pp.set_yticks([0, len(pivot.index) - 1])
    ax_pp.set_yticklabels([f"{pivot.index[0]:.1f}", f"{pivot.index[-1]:.1f}"])
    ax_pp.set_xlabel("Forced product")
    ax_pp.set_ylabel("Growth constraint")
    ax_pp.set_title("Growth-product plane")
    fig.colorbar(im, ax=ax_pp, fraction=0.045, pad=0.02)

    panel_label(ax_dfba, "c")
    ax_dfba.plot(dfba["time_h"], dfba["biomass_gDW_L"], color=PALETTE["blue_main"], lw=1.8, label="Biomass")
    ax2 = ax_dfba.twinx()
    for col, color in [("glucose_mmol_L", PALETTE["gold"]), ("nh4_mmol_L", PALETTE["green"]), ("oxygen_mmol_L", PALETTE["violet"]), ("phosphate_mmol_L", PALETTE["red"])]:
        ax2.plot(dfba["time_h"], dfba[col], color=color, lw=1.0, alpha=0.85)
    ax_dfba.set_xlabel("Time (h)")
    ax_dfba.set_ylabel("Biomass")
    ax2.set_ylabel("Substrates")
    ax_dfba.set_title("dFBA extracellular dynamics")

    panel_label(ax_proxy, "d")
    for met, d in proxy.groupby("metabolite"):
        ax_proxy.plot(d["time_h"], d["synthesis_capacity_proxy"], lw=1.4, label=met)
    ax_proxy.set_xlabel("Time (h)")
    ax_proxy.set_ylabel("Turnover proxy")
    ax_proxy.set_title("Intracellular dynamic proxies")
    ax_proxy.legend(fontsize=4.8, loc="upper right")

    panel_label(ax_single, "e")
    ax_single.hist(single["growth_ratio"].clip(0, 1.2), bins=28, color=PALETTE["blue_secondary"], edgecolor="white", linewidth=0.25)
    ax_single.axvline(0.01, color=PALETTE["red"], lw=1.1, ls="--")
    ax_single.text(0.03, ax_single.get_ylim()[1] * 0.86, f"{(single['phenotype'] == 'essential').sum()} essential", fontsize=6, color=PALETTE["red"])
    ax_single.set_xlabel("KO growth ratio")
    ax_single.set_ylabel("Genes")
    ax_single.set_title("Single-gene knockout")

    panel_label(ax_double, "f")
    dtop = double.sort_values("interaction_score", ascending=False).head(10).copy()
    dtop["pair"] = dtop["gene_a"].str.replace("GA0070618_", "G", regex=False) + "+" + dtop["gene_b"].str.replace("GA0070618_", "G", regex=False)
    ax_double.barh(np.arange(len(dtop))[::-1], dtop["interaction_score"], color=PALETTE["red_soft"])
    ax_double.set_yticks(np.arange(len(dtop))[::-1])
    ax_double.set_yticklabels(dtop["pair"], fontsize=5)
    ax_double.set_xlabel("Interaction")
    ax_double.set_title("Double-gene knockout")

    panel_label(ax_fseof, "g")
    ftop = fseof.head(12).iloc[::-1]
    colors = [PALETTE["blue_main"] if x == "overexpression" else PALETTE["red"] for x in ftop["recommendation"]]
    ax_fseof.barh(ftop["gene"].str.replace("GA0070618_", "G", regex=False), ftop["fseof_score"], color=colors)
    ax_fseof.set_xlabel("FSEOF score")
    ax_fseof.set_title("FSEOF targets")
    ax_fseof.tick_params(axis="y", labelsize=5.2)

    panel_label(ax_alg, "h")
    alg = targets.copy()
    alg = alg.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    alg = alg[alg["score"] > -100]
    top_alg = alg.sort_values("score", ascending=False).head(22)
    algorithms = list(top_alg["algorithm"].unique())
    genes = list(top_alg["gene"].astype(str).str.replace("GA0070618_", "G", regex=False).unique())
    x = [algorithms.index(a) for a in top_alg["algorithm"]]
    y = [genes.index(g.replace("GA0070618_", "G")) for g in top_alg["gene"].astype(str)]
    score = top_alg["score"].astype(float)
    sizes = 15 + 120 * (score - score.min()) / max(score.max() - score.min(), 1e-9)
    sc = ax_alg.scatter(x, y, s=sizes, c=score, cmap="viridis", alpha=0.85, edgecolor="white", linewidth=0.3)
    ax_alg.set_xticks(range(len(algorithms)))
    ax_alg.set_xticklabels([a.replace("-like", "") for a in algorithms], rotation=35, ha="right", fontsize=5.3)
    ax_alg.set_yticks(range(len(genes)))
    ax_alg.set_yticklabels(genes, fontsize=5)
    ax_alg.set_title("Opt/MOMA target screen")
    fig.colorbar(sc, ax=ax_alg, fraction=0.045, pad=0.02)

    panel_label(ax_meta, "i")
    mtop = meta.sort_values("metastrain_style_score", ascending=False).head(12).iloc[::-1]
    colors = [PALETTE["blue_main"] if op == "OE" else PALETTE["red"] for op in mtop["recommended_operation"]]
    ax_meta.barh(mtop["gene"].str.replace("GA0070618_", "G", regex=False), mtop["metastrain_style_score"], color=colors)
    ax_meta.set_xlabel("MetaStrain-style score")
    ax_meta.set_title("MetaStrain algorithm")
    ax_meta.tick_params(axis="y", labelsize=5.2)

    fig.suptitle("Fig. 3 | Multi-scale model analysis prioritizes metabolic engineering targets", x=0.01, y=1.005, ha="left", fontsize=10, fontweight="bold")
    for name, table in [
        ("figure3_reaction_class_distribution.csv", pathway),
        ("figure3_product_phase_plane.csv", product_phase),
        ("figure3_dfba.csv", dfba),
        ("figure3_dfba_intracellular_proxy.csv", proxy),
        ("figure3_single_gene_ko.csv", single),
        ("figure3_double_gene_ko.csv", double),
        ("figure3_fseof_gene_targets.csv", fseof),
        ("figure3_target_algorithms.csv", targets),
        ("figure3_metastrain_targets.csv", meta),
    ]:
        table.to_csv(SOURCE_DIR / name, index=False)
    save_pub(fig, "figure3_analysis_prediction")


def write_legend_file():
    text = """# Nature-style figure legends

## Figure 1 | Building an enzyme-constrained genome-scale model for gentamicin biosynthesis
Schematic overview of the eciFX1172 construction workflow. The iFX1172 stoichiometric model was converted to an irreversible, isoenzyme-resolved model and integrated with local UniProt protein annotations, molecular-mass estimates and reaction-level kcat assignments. Enzyme usage was encoded as v/(kcat/MW) and constrained by a calibrated global protein pool. Supporting panels summarize kcat/MW evidence classes, protein-mass coverage and final model scale.

## Figure 2 | Validation shows enzyme-capacity restriction and preserved model quality
Model validation compares iFX1172 and eciFX1172. The enzyme-constrained model predicts a lower default growth rate while retaining formula, charge and GPR annotation coverage. A substrate panel summarizes potential utilization across 24 carbon sources and 20 amino-acid supplements. Robustness and phase-plane analyses show that growth is jointly limited by glucose supply and total enzyme capacity.

## Figure 3 | Multi-scale model analysis prioritizes metabolic engineering targets
Integrated analysis of eciFX1172 includes pathway distribution, growth-product feasibility, dynamic FBA, intracellular turnover proxies, single- and double-gene knockout screens, FSEOF, OptKnock/OptForce/MOMA/OptGene-like analyses and MetaStrain-style target scoring. Together, these analyses prioritize genetic interventions linked to precursor supply, energy metabolism and Gentamicin A biosynthesis.
"""
    (OUT / "nature_figure_legends.md").write_text(text, encoding="utf-8")


def write_qa_notes():
    notes = """# Figure QA notes

- Backend: Python/matplotlib only.
- Export formats: SVG, PDF, TIFF 600 dpi and PNG preview.
- SVG font policy: `svg.fonttype = none`; PDF font policy: TrueType text.
- Source data: CSV copies are stored in `source_data`.
- Statistics: model-derived deterministic FBA/FVA/dFBA outputs; no experimental replicates or hypothesis tests are implied.
- Image integrity: no microscopy/raster manipulation; all panels are vector plots or schematic elements drawn from tabular model outputs.
- Limitations: OptKnock/OptGene panels use COBRApy-based reproducible approximations because cameo was unavailable in the current environment; MEMOTE-style QC is used because the MEMOTE command-line entry point has a NumPy/Cobra compatibility issue in this environment.
"""
    (OUT / "qa_notes.md").write_text(notes, encoding="utf-8")


def main():
    setup_style()
    ensure_dirs()
    figure1_construction()
    figure2_validation()
    figure3_analysis_prediction()
    write_legend_file()
    write_qa_notes()
    print(json.dumps({"out": str(OUT), "figures": [p.name for p in sorted(FIG_DIR.glob("*.svg"))]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
