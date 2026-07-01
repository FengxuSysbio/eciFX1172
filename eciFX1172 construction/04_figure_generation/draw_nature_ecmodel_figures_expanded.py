import json
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
TABLES = BASE / "advanced_analysis_v2" / "tables"
OUT = BASE / "nature_figures_expanded"
FIG_DIR = OUT / "figures"
PANEL_DIR = OUT / "individual_panels"
SOURCE_DIR = OUT / "source_data"

PALETTE = {
    "blue": "#0F4D92",
    "blue2": "#3775BA",
    "teal": "#42949E",
    "green": "#2E9E44",
    "green_soft": "#AADCA9",
    "red": "#B64342",
    "red_soft": "#E9A6A1",
    "gold": "#D7A928",
    "violet": "#7A5195",
    "grey": "#767676",
    "dark": "#272727",
    "light": "#D7D7D7",
    "pale": "#F3F3F3",
}


def setup_style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 6.4,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.65,
            "legend.frameon": False,
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
        }
    )


def ensure_dirs():
    for d in [FIG_DIR, PANEL_DIR, SOURCE_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def read_table(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def save_composite(fig, name):
    stem = FIG_DIR / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_panel(draw_func, name, size=(2.4, 2.0)):
    fig, ax = plt.subplots(figsize=size)
    draw_func(ax)
    fig.savefig((PANEL_DIR / name).with_suffix(".svg"), bbox_inches="tight")
    fig.savefig((PANEL_DIR / name).with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig((PANEL_DIR / name).with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def label(ax, letter):
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")


def no_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def box(ax, x, y, w, h, text, color, fs=5.3):
    p = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025", facecolor=color, edgecolor=PALETTE["light"], lw=0.8
    )
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=PALETTE["dark"])


def arrow(ax, p1, p2):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", lw=0.9, color=PALETTE["dark"], shrinkA=1, shrinkB=1))


def reaction_classes():
    detail = read_table("pathway_detail")
    rows = []
    for _, r in detail.iterrows():
        rid = str(r["reaction"])
        enzyme = str(r.get("enzyme_constrained", "")).lower() == "true" or bool(r.get("enzyme_constrained", False))
        has_gene = str(r.get("has_gene", "")).lower() == "true" or bool(r.get("has_gene", False))
        if rid.startswith("EX_"):
            cls = "Exchange"
        elif re.match(r"r_000[8-9]|r_001[0-3]", rid):
            cls = "Gentamicin"
        elif "abc" in rid.lower() or "transport" in rid.lower() or rid.endswith("t"):
            cls = "Transport"
        elif enzyme:
            cls = "Enzyme-constrained"
        elif has_gene:
            cls = "Gene-associated"
        else:
            cls = "Other"
        rows.append({"model": r["model"], "class": cls, "reaction": rid})
    return pd.DataFrame(rows).groupby(["model", "class"], as_index=False).agg(count=("reaction", "count"))


def load_data():
    data = {
        "summary": pd.read_csv(ANALYSIS / "model_summary.csv").iloc[0],
        "kcat": pd.read_csv(ANALYSIS / "reaction_kcat_MW.csv"),
        "mass": pd.read_csv(ANALYSIS / "gene_protein_mass.csv"),
        "memote": read_table("memote_qc"),
        "substrate": read_table("substrate_panel"),
        "robustness": read_table("robustness"),
        "phase": read_table("phase_plane"),
        "product_phase": read_table("product_phase_plane"),
        "single": read_table("single_gene_ko"),
        "double": read_table("double_gene_ko"),
        "dfba": read_table("dfba"),
        "proxy": read_table("dfba_intracellular_proxy"),
        "fseof_gene": read_table("fseof_gene_targets"),
        "fseof_rxn": read_table("fseof_reaction_targets"),
        "targets": read_table("target_algorithms"),
        "meta": read_table("metastrain_targets"),
        "classes": reaction_classes(),
    }
    return data


def panel_workflow(ax, d):
    no_axis(ax)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        (0.02, 0.58, "iFX1172\nGEM", PALETTE["pale"]),
        (0.25, 0.76, "Irreversible\nmodel", "#EAF1FA"),
        (0.25, 0.42, "Protein\nannotations", "#EAF1FA"),
        (0.50, 0.76, "kcat\nmapping", "#EFF7EF"),
        (0.50, 0.42, "MW\nmapping", "#EFF7EF"),
        (0.75, 0.58, "Enzyme\npool", "#FFF5D8"),
    ]
    for x, y, t, c in steps:
        box(ax, x, y, 0.16, 0.16, t, c, 5.0)
    pts = [(x + 0.16, y + 0.08) for x, y, _, _ in steps]
    lefts = [(x, y + 0.08) for x, y, _, _ in steps]
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)]:
        arrow(ax, pts[a], lefts[b])
    ax.set_title("Construction workflow")


def panel_model_scale(ax, d):
    s = d["summary"]
    vals = [int(s["irreversible_isoenzyme_reactions"]), int(s["original_metabolites"]), int(s["original_genes"]), int(s["reactions_with_enzyme_constraint"])]
    labs = ["Reactions", "Metabolites", "Genes", "Enzyme\nrxns"]
    ax.bar(range(4), vals, color=[PALETTE["blue"], PALETTE["teal"], PALETTE["green_soft"], PALETTE["gold"]])
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, rotation=25, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Model scale")


def panel_kcat_sources(ax, d):
    counts = d["kcat"]["data_type"].value_counts().head(8).iloc[::-1]
    ax.barh(range(len(counts)), counts.values, color=PALETTE["blue"])
    ax.set_yticks(range(len(counts))); ax.set_yticklabels([x.replace("_", " ")[:18] for x in counts.index], fontsize=4.8)
    ax.set_xlabel("Reactions")
    ax.set_title("kcat evidence")


def panel_kcat_distribution(ax, d):
    x = pd.to_numeric(d["kcat"]["kcat"], errors="coerce").dropna()
    x = x[x > 0]
    ax.hist(np.log10(x), bins=32, color=PALETTE["teal"], edgecolor="white", linewidth=0.2)
    ax.set_xlabel("log10 kcat")
    ax.set_ylabel("Reactions")
    ax.set_title("kcat distribution")


def panel_kcatmw_distribution(ax, d):
    x = pd.to_numeric(d["kcat"]["kcat_MW"], errors="coerce").dropna()
    x = x[x > 0]
    ax.hist(np.log10(x), bins=32, color=PALETTE["green"], edgecolor="white", linewidth=0.2)
    ax.set_xlabel("log10 kcat/MW")
    ax.set_ylabel("Reactions")
    ax.set_title("Enzyme efficiency")


def panel_mass_sources(ax, d):
    counts = d["mass"]["mass_source"].value_counts()
    wedges, _ = ax.pie(counts.values, colors=[PALETTE["blue2"], PALETTE["light"], PALETTE["gold"]], startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"))
    ax.text(0, 0, f"{len(d['mass'])}\ngenes", ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_title("Protein mass source")


def panel_mass_distribution(ax, d):
    x = pd.to_numeric(d["mass"]["mass_kda"], errors="coerce").dropna()
    ax.hist(x, bins=32, color=PALETTE["blue2"], edgecolor="white", linewidth=0.2)
    ax.set_xlabel("Protein MW (kDa)")
    ax.set_ylabel("Genes")
    ax.set_title("Protein MW distribution")


def panel_pool_calibration(ax, d):
    s = d["summary"]
    ax.bar([0, 1], [float(s["enzyme_pool_initial_upper_bound"]), float(s["enzyme_pool_upper_bound"])], color=[PALETTE["light"], PALETTE["red"]])
    ax.set_yscale("log")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Initial", "Calibrated"], rotation=25, ha="right")
    ax.set_ylabel("Pool upper bound")
    ax.set_title("Protein-pool calibration")


def panel_growth_calibration(ax, d):
    mem = d["memote"]
    vals = [float(mem.loc[mem.model == "iFX1172", "growth"].iloc[0]), float(mem.loc[mem.model == "eciFX1172", "growth"].iloc[0])]
    ax.bar([0, 1], vals, color=[PALETTE["grey"], PALETTE["blue"]])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["GEM", "ec"], rotation=25, ha="right")
    ax.set_ylabel("h$^{-1}$")
    ax.set_title("Growth after constraint")


def panel_export_formats(ax, d):
    no_axis(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, fmt in enumerate(["JSON", "SBML/XML", "YAML", "Excel"]):
        box(ax, 0.18, 0.78 - i * 0.20, 0.62, 0.12, fmt, "#EAF1FA", 6.4)
    ax.set_title("Model exports")


def panel_memote_qc(ax, d):
    mem = d["memote"]
    metrics = ["formula_coverage", "charge_coverage", "gpr_coverage"]
    x = np.arange(3)
    for off, model, c in [(-0.17, "iFX1172", PALETTE["grey"]), (0.17, "eciFX1172", PALETTE["blue"])]:
        vals = [float(mem.loc[mem.model == model, m].iloc[0]) for m in metrics]
        ax.bar(x + off, vals, width=0.32, color=c, label=model)
    ax.set_ylim(0, 1.05); ax.set_xticks(x); ax.set_xticklabels(["Formula", "Charge", "GPR"], rotation=25, ha="right")
    ax.set_ylabel("Coverage")
    ax.set_title("MEMOTE-style QC")


def panel_blocked_reactions(ax, d):
    mem = d["memote"]
    vals = pd.to_numeric(mem.get("blocked_reactions_fva0_first300", pd.Series([np.nan, np.nan])), errors="coerce").fillna(0).values
    ax.bar([0, 1], vals, color=[PALETTE["grey"], PALETTE["blue"]])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["GEM", "ec"], rotation=25, ha="right")
    ax.set_ylabel("Blocked in first 300")
    ax.set_title("FVA blocked screen")


def panel_substrate_heat(ax, d, panel="carbon", title="Carbon source use"):
    sub = d["substrate"][d["substrate"]["panel"] == panel]
    piv = sub.pivot_table(index="substrate", columns="model", values="growth", aggfunc="mean")
    order = piv.max(axis=1).sort_values(ascending=True).index
    piv = piv.loc[order]
    im = ax.imshow(piv.fillna(0).values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=4.6)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([x.replace("iFX1172", "GEM").replace("eciFX1172", "ec") for x in piv.columns], fontsize=5.4)
    ax.tick_params(length=0)
    ax.set_title(title)


def panel_robustness(ax, d):
    for model, c in [("iFX1172", PALETTE["grey"]), ("eciFX1172", PALETTE["blue"])]:
        r = d["robustness"][d["robustness"].model == model]
        ax.plot(r["glucose_uptake"], r["growth"], marker="o", ms=2, lw=1.4, color=c, label=model.replace("iFX1172", "GEM").replace("eciFX1172", "ec"))
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Growth")
    ax.set_title("Glucose robustness")


def panel_phase(ax, d):
    piv = d["phase"].pivot(index="enzyme_pool_factor", columns="glucose_uptake", values="growth")
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks([0, len(piv.columns)-1]); ax.set_xticklabels([f"{piv.columns[0]:.1f}", f"{piv.columns[-1]:.1f}"])
    ax.set_yticks([0, len(piv.index)-1]); ax.set_yticklabels([f"{piv.index[0]:.1f}", f"{piv.index[-1]:.1f}"])
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Enzyme pool")
    ax.set_title("Phase plane")


def panel_product_phase(ax, d):
    piv = d["product_phase"].pivot(index="growth_fraction_constraint", columns="product_fraction_constraint", values="feasible_growth")
    ax.imshow(piv.values, origin="lower", aspect="auto", cmap="magma")
    ax.set_xticks([0, len(piv.columns)-1]); ax.set_xticklabels([f"{piv.columns[0]:.1f}", f"{piv.columns[-1]:.1f}"])
    ax.set_yticks([0, len(piv.index)-1]); ax.set_yticklabels([f"{piv.index[0]:.1f}", f"{piv.index[-1]:.1f}"])
    ax.set_xlabel("Product")
    ax.set_ylabel("Growth")
    ax.set_title("Growth-product plane")


def panel_reaction_classes(ax, d):
    cls = d["classes"]
    top = cls.groupby("class")["count"].sum().sort_values(ascending=False).head(7).index
    piv = cls[cls["class"].isin(top)].pivot(index="class", columns="model", values="count").fillna(0).loc[top[::-1]]
    y = np.arange(len(piv))
    ax.barh(y - 0.16, piv.get("iFX1172", 0), height=0.3, color=PALETTE["grey"], label="GEM")
    ax.barh(y + 0.16, piv.get("eciFX1172", 0), height=0.3, color=PALETTE["blue"], label="ec")
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=5.0)
    ax.set_xlabel("Reactions")
    ax.set_title("Pathway distribution")


def panel_single_ko(ax, d):
    s = d["single"]
    ax.hist(s["growth_ratio"].clip(0, 1.2), bins=28, color=PALETTE["blue2"], edgecolor="white", linewidth=0.25)
    ax.axvline(0.01, color=PALETTE["red"], ls="--", lw=1)
    ax.text(0.05, ax.get_ylim()[1]*0.86, f"{(s['phenotype']=='essential').sum()} essential", color=PALETTE["red"], fontsize=5.8)
    ax.set_xlabel("Growth ratio")
    ax.set_ylabel("Genes")
    ax.set_title("Single-gene KO")


def panel_double_ko(ax, d):
    x = d["double"].sort_values("interaction_score", ascending=False).head(10).copy()
    x["pair"] = x["gene_a"].str.replace("GA0070618_", "G", regex=False) + "+" + x["gene_b"].str.replace("GA0070618_", "G", regex=False)
    ax.barh(np.arange(len(x))[::-1], x["interaction_score"], color=PALETTE["red_soft"])
    ax.set_yticks(np.arange(len(x))[::-1]); ax.set_yticklabels(x["pair"], fontsize=4.7)
    ax.set_xlabel("Interaction")
    ax.set_title("Double-gene KO")


def panel_dfba_external(ax, d):
    df = d["dfba"]
    ax.plot(df["time_h"], df["biomass_gDW_L"], color=PALETTE["blue"], lw=1.6, label="Biomass")
    ax2 = ax.twinx()
    for col, c in [("glucose_mmol_L", PALETTE["gold"]), ("nh4_mmol_L", PALETTE["green"]), ("oxygen_mmol_L", PALETTE["violet"]), ("phosphate_mmol_L", PALETTE["red"])]:
        ax2.plot(df["time_h"], df[col], color=c, lw=1.0, alpha=0.9)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Biomass")
    ax2.set_ylabel("Substrates")
    ax.set_title("dFBA extracellular")


def panel_dfba_proxy(ax, d):
    proxy = d["proxy"]
    for met, g in proxy.groupby("metabolite"):
        ax.plot(g["time_h"], g["synthesis_capacity_proxy"], lw=1.2, label=met)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Turnover")
    ax.set_title("Intracellular proxies")
    ax.legend(fontsize=4.5)


def panel_fseof_gene(ax, d):
    x = d["fseof_gene"].head(12).iloc[::-1]
    colors = [PALETTE["blue"] if r == "overexpression" else PALETTE["red"] for r in x["recommendation"]]
    ax.barh(x["gene"].str.replace("GA0070618_", "G", regex=False), x["fseof_score"], color=colors)
    ax.set_xlabel("Score")
    ax.set_title("FSEOF genes")
    ax.tick_params(axis="y", labelsize=4.8)


def panel_fseof_rxn(ax, d):
    x = d["fseof_rxn"].head(10).iloc[::-1]
    colors = [PALETTE["blue2"] if r == "overexpression" else PALETTE["red_soft"] for r in x["recommendation"]]
    ax.barh(x["reaction"], x["abs_slope"], color=colors)
    ax.set_xlabel("|slope|")
    ax.set_title("FSEOF reactions")
    ax.tick_params(axis="y", labelsize=4.8)


def panel_algorithm(ax, d, algorithm, title):
    t = d["targets"].copy()
    t = t[t["algorithm"].astype(str).str.contains(algorithm, regex=False)]
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"]).sort_values("score", ascending=False).head(10).iloc[::-1]
    if t.empty:
        no_axis(ax); ax.set_title(title); ax.text(0.5, 0.5, "No records", ha="center", va="center"); return
    ax.barh(t["gene"].astype(str).str.replace("GA0070618_", "G", regex=False), t["score"], color=PALETTE["teal"])
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=4.8)


def panel_alg_bubble(ax, d):
    t = d["targets"].replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    t = t[t["score"] > -100].sort_values("score", ascending=False).head(25)
    algs = list(t["algorithm"].unique())
    genes = list(t["gene"].astype(str).str.replace("GA0070618_", "G", regex=False).unique())
    x = [algs.index(a) for a in t["algorithm"]]
    y = [genes.index(g.replace("GA0070618_", "G")) for g in t["gene"].astype(str)]
    score = t["score"].astype(float)
    sizes = 15 + 100 * (score - score.min()) / max(score.max() - score.min(), 1e-9)
    ax.scatter(x, y, s=sizes, c=score, cmap="viridis", alpha=0.85, edgecolor="white", linewidth=0.2)
    ax.set_xticks(range(len(algs))); ax.set_xticklabels([a.replace("-like", "") for a in algs], rotation=35, ha="right", fontsize=4.8)
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=4.8)
    ax.set_title("Algorithm target map")


def panel_meta(ax, d):
    m = d["meta"].sort_values("metastrain_style_score", ascending=False).head(12).iloc[::-1]
    colors = [PALETTE["blue"] if op == "OE" else PALETTE["red"] for op in m["recommended_operation"]]
    ax.barh(m["gene"].str.replace("GA0070618_", "G", regex=False), m["metastrain_style_score"], color=colors)
    ax.set_xlabel("Score")
    ax.set_title("MetaStrain targets")
    ax.tick_params(axis="y", labelsize=4.8)


def panel_operation_counts(ax, d):
    m = d["meta"]
    counts = m["recommended_operation"].value_counts()
    ax.bar(counts.index, counts.values, color=[PALETTE["blue"], PALETTE["red"], PALETTE["grey"]][:len(counts)])
    ax.set_ylabel("Genes")
    ax.set_title("MetaStrain operations")
    ax.tick_params(axis="x", rotation=25)


def panel_target_overlap(ax, d):
    f = set(d["fseof_gene"].head(50)["gene"])
    m = set(d["meta"].head(50)["gene"])
    o = set(d["targets"].sort_values("score", ascending=False).head(80)["gene"].astype(str))
    vals = [len(f), len(m), len(o), len(f & m), len(f & o), len(m & o), len(f & m & o)]
    labs = ["FSEOF", "Meta", "Opt/MOMA", "F∩M", "F∩O", "M∩O", "All"]
    ax.bar(range(len(vals)), vals, color=[PALETTE["blue"], PALETTE["violet"], PALETTE["teal"], PALETTE["light"], PALETTE["light"], PALETTE["light"], PALETTE["red"]])
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, rotation=35, ha="right", fontsize=4.9)
    ax.set_ylabel("Genes")
    ax.set_title("Target overlap")


def panel_product_flux(ax, d):
    df = d["dfba"]
    ax.plot(df["time_h"], df["product_flux"], color=PALETTE["red"], lw=1.6)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Flux")
    ax.set_title("Gentamicin flux in dFBA")


def plot_grid(d, panels, nrows, ncols, title, name, figsize):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, (ax, (func, ptitle)) in enumerate(zip(axes, panels)):
        func(ax, d)
        label(ax, letters[i])
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, y=1.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_composite(fig, name)
    for i, (func, ptitle) in enumerate(panels):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", ptitle).strip("_").lower()
        save_panel(lambda ax, f=func: f(ax, d), f"{name}_panel_{letters[i]}_{safe}", size=(2.35, 1.95))


def write_legends():
    text = """# Expanded Nature-style figure legends

## Figure 1 | Construction of eciFX1172
Nine panels summarize model conversion, enzyme-parameter acquisition, protein mass mapping, enzyme pool calibration and export readiness.

## Figure 2 | Validation of eciFX1172
Nine panels compare original GEM and ecModel growth, annotation quality, substrate utilization, robustness and phase-plane behavior.

## Figure 3 | Analysis and prediction using eciFX1172
Eighteen panels summarize pathway distribution, substrate panels, robustness, phase planes, knockout analysis, dFBA, intracellular dynamic proxies, FSEOF, OptKnock-like, OptForce-like, MOMA, OptGene-like and MetaStrain-style target prioritization.
"""
    (OUT / "expanded_figure_legends.md").write_text(text, encoding="utf-8")


def write_manifest():
    files = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file())
    (OUT / "expanded_figure_manifest.json").write_text(json.dumps(files, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    setup_style()
    ensure_dirs()
    d = load_data()
    for key, table in d.items():
        if isinstance(table, pd.DataFrame):
            table.to_csv(SOURCE_DIR / f"{key}.csv", index=False)

    fig1_panels = [
        (panel_workflow, "workflow"),
        (panel_model_scale, "model scale"),
        (panel_kcat_sources, "kcat sources"),
        (panel_kcat_distribution, "kcat distribution"),
        (panel_kcatmw_distribution, "kcatmw distribution"),
        (panel_mass_sources, "mass sources"),
        (panel_mass_distribution, "mass distribution"),
        (panel_pool_calibration, "pool calibration"),
        (panel_export_formats, "export formats"),
    ]
    fig2_panels = [
        (panel_growth_calibration, "growth calibration"),
        (panel_memote_qc, "memote qc"),
        (panel_blocked_reactions, "blocked reactions"),
        (lambda ax, data: panel_substrate_heat(ax, data, "carbon", "Carbon sources"), "carbon sources"),
        (lambda ax, data: panel_substrate_heat(ax, data, "amino_acid", "Amino acids"), "amino acids"),
        (panel_robustness, "robustness"),
        (panel_phase, "phase plane"),
        (panel_product_phase, "product phase"),
        (panel_reaction_classes, "reaction classes"),
    ]
    fig3_panels = [
        (panel_reaction_classes, "pathway distribution"),
        (lambda ax, data: panel_substrate_heat(ax, data, "carbon", "Carbon sources"), "carbon utilization"),
        (lambda ax, data: panel_substrate_heat(ax, data, "amino_acid", "Amino acids"), "amino acid utilization"),
        (panel_robustness, "robustness"),
        (panel_phase, "phase plane"),
        (panel_product_phase, "growth product phase"),
        (panel_single_ko, "single knockout"),
        (panel_double_ko, "double knockout"),
        (panel_dfba_external, "dfba extracellular"),
        (panel_dfba_proxy, "dfba intracellular proxy"),
        (panel_product_flux, "product flux"),
        (panel_fseof_gene, "fseof genes"),
        (panel_fseof_rxn, "fseof reactions"),
        (lambda ax, data: panel_algorithm(ax, data, "OptKnock", "OptKnock-like"), "optknock"),
        (lambda ax, data: panel_algorithm(ax, data, "OptForce", "OptForce-like"), "optforce"),
        (lambda ax, data: panel_algorithm(ax, data, "MOMA", "MOMA"), "moma"),
        (lambda ax, data: panel_algorithm(ax, data, "OptGene", "OptGene-like"), "optgene"),
        (panel_meta, "metastrain"),
        (panel_alg_bubble, "algorithm bubble"),
        (panel_operation_counts, "operation counts"),
        (panel_target_overlap, "target overlap"),
    ]
    # Keep the requested minimum of 18 panels while preserving additional target-summary panels.
    fig3_panels = fig3_panels[:21]

    plot_grid(d, fig1_panels, 3, 3, "Fig. 1 | Construction of the enzyme-constrained iFX1172 model", "figure1_construction_9panels", (7.3, 6.1))
    plot_grid(d, fig2_panels, 3, 3, "Fig. 2 | Validation of the enzyme-constrained model", "figure2_validation_9panels", (7.3, 6.1))
    plot_grid(d, fig3_panels, 3, 7, "Fig. 3 | Analysis and prediction using iFX1172 and eciFX1172", "figure3_analysis_prediction_21panels", (13.6, 6.8))
    write_legends()
    write_manifest()
    print(json.dumps({"out": str(OUT), "composites": len(list(FIG_DIR.glob('*.svg'))), "individual_panels": len(list(PANEL_DIR.glob('*.svg')))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
