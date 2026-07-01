import json
import math
import re
from pathlib import Path

import numpy as np

if not hasattr(np, "object"):
    np.object = object

import cobra
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from cobra.flux_analysis import flux_variability_analysis
from matplotlib import patches
from matplotlib.colors import Normalize


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "ec_iFX1172_final_calibrated"
ANALYSIS = BASE / "analysis"
MODEL_DIR = BASE / "model"
ADV = BASE / "advanced_analysis"
V2 = BASE / "advanced_analysis_v2" / "tables"
OUT = BASE / "docx_requested_outputs"
FIG_DIR = OUT / "figures"
PANEL_DIR = OUT / "individual_panels"
TABLE_DIR = OUT / "tables"
SOURCE_DIR = OUT / "source_data"

EC_JSON = MODEL_DIR / "eciFX1172.json"
GEM_JSON = MODEL_DIR / "iFX1172_irreversible.json"

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
            "font.size": 6.6,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
        }
    )


def ensure_dirs():
    for folder in [FIG_DIR, PANEL_DIR, TABLE_DIR, SOURCE_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def read_csv(path):
    return pd.read_csv(path)


def read_v2(name):
    return read_csv(V2 / f"{name}.csv")


def save_fig(fig, name, outdir=FIG_DIR, tiff=True):
    stem = outdir / name
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=450, bbox_inches="tight")
    if tiff:
        fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_panel(draw_func, name, data, size=(2.5, 2.05)):
    fig, ax = plt.subplots(figsize=size)
    draw_func(ax, data)
    save_fig(fig, name, PANEL_DIR, tiff=False)


def label(ax, text):
    ax.text(-0.12, 1.08, text, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")


def no_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def round_box(ax, x, y, w, h, text, color, fs=5.7):
    patch = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=color, edgecolor=PALETTE["light"], linewidth=0.8
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=PALETTE["dark"])


def arr(ax, p1, p2):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", lw=0.9, color=PALETTE["dark"], shrinkA=1, shrinkB=1))


def load_ec_model():
    with open(EC_JSON, encoding="utf-8") as f:
        data = json.load(f)
    model = cobra.io.load_json_model(str(EC_JSON))
    model.solver = "glpk"
    coefs = {}
    by_id = {r["id"]: r.get("kcat_MW") for r in data["reactions"]}
    for r in model.reactions:
        val = by_id.get(r.id)
        if val not in ("", None):
            try:
                coefs[r.forward_variable] = 1.0 / float(val)
            except Exception:
                pass
    cons = model.problem.Constraint(
        0,
        lb=float(data["enzyme_constraint"]["lowerbound"]),
        ub=float(data["enzyme_constraint"]["upperbound"]),
        name="enzyme_pool",
    )
    model.add_cons_vars(cons)
    model.solver.update()
    cons.set_linear_coefficients(coefs)
    return model


def load_gem_model():
    model = cobra.io.load_json_model(str(GEM_JSON))
    model.solver = "glpk"
    return model


def reaction_class_table(pathway_detail):
    rows = []
    for _, r in pathway_detail.iterrows():
        rid = str(r["reaction"])
        enzyme = str(r.get("enzyme_constrained", "")).lower() == "true" or bool(r.get("enzyme_constrained", False))
        has_gene = str(r.get("has_gene", "")).lower() == "true" or bool(r.get("has_gene", False))
        if rid.startswith("EX_"):
            cls = "Exchange"
        elif re.match(r"r_000[8-9]|r_001[0-3]", rid):
            cls = "C1a/product module"
        elif "abc" in rid.lower() or "transport" in rid.lower() or rid.endswith("t"):
            cls = "Transport"
        elif enzyme:
            cls = "Enzyme-constrained"
        elif has_gene:
            cls = "Gene-associated"
        else:
            cls = "Other"
        rows.append({"model": r["model"], "class": cls, "reaction": rid})
    return pd.DataFrame(rows).groupby(["model", "class"], as_index=False).agg(reaction_count=("reaction", "count"))


def load_data():
    data = {
        "summary": read_csv(ANALYSIS / "model_summary.csv").iloc[0],
        "kcat": read_csv(ANALYSIS / "reaction_kcat_MW.csv"),
        "mass": read_csv(ANALYSIS / "gene_protein_mass.csv"),
        "memote": read_v2("memote_qc"),
        "substrate": read_v2("substrate_panel"),
        "robustness": read_v2("robustness"),
        "phase": read_v2("phase_plane"),
        "product_phase": read_v2("product_phase_plane"),
        "single": read_v2("single_gene_ko"),
        "double": read_v2("double_gene_ko"),
        "dfba": read_v2("dfba"),
        "proxy": read_v2("dfba_intracellular_proxy"),
        "fseof_rxn": read_v2("fseof_reaction_targets"),
        "fseof_gene": read_v2("fseof_gene_targets"),
        "targets": read_v2("target_algorithms"),
        "meta": read_v2("metastrain_targets"),
        "pool": read_csv(ADV / "tables" / "robustness_enzyme_pool.csv") if (ADV / "tables" / "robustness_enzyme_pool.csv").exists() else pd.DataFrame(),
    }
    data["classes"] = reaction_class_table(read_v2("pathway_detail"))
    return data


def p_workflow(ax, d):
    no_axis(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nodes = [
        (0.01, 0.50, "iFX1172", PALETTE["pale"]),
        (0.19, 0.68, "GPR\ncuration", "#EAF1FA"),
        (0.19, 0.32, "Reversible\nsplit", "#EAF1FA"),
        (0.39, 0.68, "Isoenzyme\nsplit", "#EAF1FA"),
        (0.39, 0.32, "kcat/MW\nmapping", "#EFF7EF"),
        (0.61, 0.50, "Protein-pool\nconstraint", "#FFF5D8"),
        (0.81, 0.50, "eciFX1172", "#F6E7EA"),
    ]
    for x, y, t, c in nodes:
        round_box(ax, x, y, 0.14, 0.16, t, c, 5.2)
    centers = [(x + 0.14, y + 0.08) for x, y, _, _ in nodes]
    left = [(x, y + 0.08) for x, y, _, _ in nodes]
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5), (5, 6)]:
        arr(ax, centers[a], left[b])
    ax.set_title("iFX1172 to eciFX1172")


def p_math(ax, d):
    no_axis(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    round_box(ax, 0.06, 0.56, 0.34, 0.22, "GEM\nS · v = 0\nlb ≤ v ≤ ub", PALETTE["pale"], 6.2)
    round_box(ax, 0.58, 0.56, 0.34, 0.22, "ecGEM\nS · v = 0\nΣv/(kcat/MW) ≤ Ppool", "#FFF5D8", 5.8)
    arr(ax, (0.42, 0.67), (0.56, 0.67))
    ax.text(0.50, 0.78, "+ enzyme capacity", ha="center", fontsize=6.0)
    ax.text(0.50, 0.28, r"$\sum_i v_i/(k_{cat,i}/MW_i)\leq P_{pool}$", ha="center", fontsize=8.2)
    ax.set_title("Mathematical logic")


def p_scale_coverage(ax, d):
    s = d["summary"]
    vals = [int(s["irreversible_isoenzyme_reactions"]), int(s["reactions_with_enzyme_constraint"])]
    ax.bar([0, 1], vals, color=[PALETTE["grey"], PALETTE["blue"]])
    cov = vals[1] / vals[0] * 100
    ax.text(1, vals[1] * 1.03, f"{cov:.1f}%", ha="center", fontsize=6.4, color=PALETTE["blue"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["All\nrxns", "Enzyme\nrxns"])
    ax.set_ylabel("Reaction count")
    ax.set_title("Scale and coverage")


def p_growth(ax, d):
    s = d["summary"]
    vals = [float(s["GEM_growth"]), float(s["ecGEM_growth"])]
    ax.bar([0, 1], vals, color=[PALETTE["grey"], PALETTE["blue"]])
    ax.text(0.5, max(vals) * 1.12, f"{vals[1] / vals[0] * 100:.2f}% retained", ha="center", fontsize=6.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["iFX1172", "eciFX1172"], rotation=18, ha="right")
    ax.set_ylabel("Objective value")
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_title("Objective retention")


def p_flux_space(ax, d):
    rob = d["robustness"]
    for model, c in [("iFX1172", PALETTE["grey"]), ("eciFX1172", PALETTE["blue"])]:
        r = rob[rob["model"] == model]
        ax.plot(r["glucose_uptake"], r["growth"], marker="o", ms=2, lw=1.3, color=c, label=model)
    ax.set_xlabel("Glucose uptake")
    ax.set_ylabel("Growth")
    ax.set_title("Flux-space compression")


def p_c1a_bottleneck(ax, d):
    no_axis(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    nodes = [
        (0.05, 0.67, "Acetyl-CoA", PALETTE["pale"]),
        (0.05, 0.42, "GlcNAc-1P", PALETTE["pale"]),
        (0.05, 0.17, "NADPH", PALETTE["pale"]),
        (0.36, 0.56, "UDP-sugars", "#EAF1FA"),
        (0.36, 0.30, "SAM cycle", "#EAF1FA"),
        (0.68, 0.43, "C1a/product\nmodule", "#F6E7EA"),
    ]
    for x, y, t, c in nodes:
        round_box(ax, x, y, 0.20, 0.13, t, c, 5.4)
    for p1, p2 in [((0.25, 0.735), (0.36, 0.625)), ((0.25, 0.485), (0.36, 0.625)), ((0.25, 0.235), (0.36, 0.365)), ((0.56, 0.625), (0.68, 0.495)), ((0.56, 0.365), (0.68, 0.495))]:
        arr(ax, p1, p2)
    ax.text(0.72, 0.20, "high enzyme-cost\ncandidates", color=PALETTE["red"], fontsize=5.7)
    ax.set_title("C1a bottleneck map")


def p_network_check(ax, d):
    s = d["summary"]
    vals = [int(s["irreversible_isoenzyme_reactions"]), int(s["original_metabolites"]), int(s["original_genes"]), int(s["reactions_with_enzyme_constraint"])]
    labs = ["Rxns", "Mets", "Genes", "ec rxns"]
    ax.bar(range(4), vals, color=[PALETTE["blue"], PALETTE["teal"], PALETTE["green_soft"], PALETTE["gold"]])
    ax.set_xticks(range(4)); ax.set_xticklabels(labs, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Network structure")


def p_split_example(ax, d):
    no_axis(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    round_box(ax, 0.05, 0.66, 0.25, 0.13, "A <-> B", PALETTE["pale"], 6)
    round_box(ax, 0.42, 0.72, 0.22, 0.11, "A → B", "#EAF1FA", 5.8)
    round_box(ax, 0.42, 0.56, 0.22, 0.11, "B → A", "#EAF1FA", 5.8)
    round_box(ax, 0.05, 0.28, 0.25, 0.13, "gene1 OR gene2", PALETTE["pale"], 5.5)
    round_box(ax, 0.42, 0.34, 0.22, 0.11, "isozyme 1", "#EFF7EF", 5.5)
    round_box(ax, 0.42, 0.18, 0.22, 0.11, "isozyme 2", "#EFF7EF", 5.5)
    round_box(ax, 0.73, 0.43, 0.18, 0.14, "enzyme\nvariables", "#FFF5D8", 5.4)
    for p1, p2 in [((0.30,0.72),(0.42,0.77)),((0.30,0.72),(0.42,0.61)),((0.30,0.34),(0.42,0.39)),((0.30,0.34),(0.42,0.23)),((0.64,0.77),(0.73,0.50)),((0.64,0.23),(0.73,0.50))]:
        arr(ax,p1,p2)
    ax.set_title("Reaction splitting")


def p_kcat_sources(ax, d):
    c = d["kcat"]["data_type"].value_counts().head(8).iloc[::-1]
    ax.barh(range(len(c)), c.values, color=PALETTE["blue"])
    ax.set_yticks(range(len(c))); ax.set_yticklabels([x.replace("_"," ")[:20] for x in c.index], fontsize=4.8)
    ax.set_xlabel("Reactions")
    ax.set_title("kcat sources")


def p_pool_sensitivity(ax, d):
    pool = d["pool"]
    if not pool.empty:
        ax.plot(pool["enzyme_pool_factor"], pool["ec_growth"], marker="o", ms=2, lw=1.2, color=PALETTE["red"])
        ax.set_xlabel("Protein-pool factor")
        ax.set_ylabel("Growth")
    else:
        no_axis(ax); ax.text(0.5,0.5,"Pool table not available",ha="center")
    ax.set_title("Protein-pool sensitivity")


def p_fva_key(ax, d):
    fva = d.get("fva", pd.DataFrame())
    if fva.empty or "range_compression" not in fva.columns:
        alt = d["fseof_rxn"].head(10).iloc[::-1]
        ax.barh(alt["reaction"], alt["abs_slope"], color=PALETTE["teal"])
        ax.set_xlabel("Flux-response proxy")
        ax.set_title("FVA comparison proxy")
        ax.tick_params(axis="y", labelsize=4.8)
        return
    top = fva.sort_values("range_compression", ascending=False).head(10).iloc[::-1]
    ax.barh(top["reaction"], top["range_compression"], color=PALETTE["teal"])
    ax.set_xlabel("Range compression")
    ax.set_title("FVA comparison")
    ax.tick_params(axis="y", labelsize=4.8)


def p_enzyme_allocation(ax, d):
    k = d["kcat"].copy()
    def mod(r):
        rid = str(r)
        if re.match(r"r_000[8-9]|r_001[0-3]", rid):
            return "C1a/product"
        if any(x in rid for x in ["GAPD", "G6PD", "PFK", "PYK", "HEX"]):
            return "Central carbon"
        if any(x in rid for x in ["AKG", "ICD", "SUCD", "FUM", "MDH"]):
            return "TCA"
        if any(x in rid for x in ["GLU", "GLN", "NH4"]):
            return "Nitrogen"
        if any(x in rid for x in ["NAD", "SAM", "MET", "AMET"]):
            return "Cofactor"
        return "Other"
    k["module"] = k["reaction"].map(mod)
    counts = k["module"].value_counts().iloc[::-1]
    ax.barh(range(len(counts)), counts.values, color=PALETTE["green_soft"])
    ax.set_yticks(range(len(counts))); ax.set_yticklabels(counts.index, fontsize=5.4)
    ax.set_xlabel("Enzyme-constrained rxns")
    ax.set_title("Module enzyme allocation")


def p_distribution(ax, d, col, title, color):
    x = pd.to_numeric(d["kcat"][col], errors="coerce").dropna()
    x = x[x > 0]
    ax.hist(np.log10(x), bins=35, color=color, edgecolor="white", linewidth=0.2)
    ax.set_xlabel(f"log10 {col}")
    ax.set_ylabel("Reactions")
    ax.set_title(title)


def p_mw_dist(ax, d):
    x = pd.to_numeric(d["mass"]["mass_kda"], errors="coerce").dropna()
    ax.hist(x, bins=35, color=PALETTE["blue2"], edgecolor="white", linewidth=0.2)
    ax.set_xlabel("MW (kDa)")
    ax.set_ylabel("Genes")
    ax.set_title("MW distribution")


def p_central_map(ax, d):
    no_axis(ax); ax.set_xlim(0,1); ax.set_ylim(0,1)
    nodes = [("Glucose",0.08,0.72),("G6P",0.31,0.72),("PPP\nNADPH",0.55,0.82),("Pyruvate",0.55,0.58),("Acetyl-CoA",0.78,0.58),("TCA",0.78,0.30),("NH4/\nGlu",0.31,0.30)]
    for t,x,y in nodes:
        round_box(ax,x,y,0.16,0.12,t,PALETTE["pale"],5.2)
    for p1,p2 in [((0.24,0.78),(0.31,0.78)),((0.47,0.78),(0.55,0.86)),((0.47,0.72),(0.55,0.64)),((0.71,0.64),(0.78,0.64)),((0.86,0.58),(0.86,0.42)),((0.47,0.36),(0.78,0.36))]:
        arr(ax,p1,p2)
    ax.set_title("Central carbon map")


def p_c1a_map(ax, d):
    p_c1a_bottleneck(ax, d)
    ax.set_title("Gentamicin C1a map")


def p_growth_feasibility(ax, d):
    rob = d["robustness"]
    for model, c in [("iFX1172", PALETTE["grey"]), ("eciFX1172", PALETTE["blue"])]:
        r = rob[rob["model"] == model]
        ax.plot(r["glucose_uptake"], r["growth"], color=c, lw=1.3, marker="o", ms=2, label=model)
    ax.set_xlabel("Constraint/uptake")
    ax.set_ylabel("Growth")
    ax.set_title("Feasibility under constraints")


def p_substrate_heat(ax, d, panel, title):
    sub = d["substrate"][d["substrate"]["panel"] == panel]
    piv = sub.pivot_table(index="substrate", columns="model", values="growth", aggfunc="mean")
    order = piv.max(axis=1).sort_values(ascending=True).index
    piv = piv.loc[order]
    ax.imshow(piv.fillna(0).values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(len(piv))); ax.set_yticklabels(piv.index, fontsize=4.6)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels([x.replace("iFX1172","GEM").replace("eciFX1172","ec") for x in piv.columns], fontsize=5.2)
    ax.tick_params(length=0)
    ax.set_title(title)


def p_phase(ax, d):
    piv = d["phase"].pivot(index="enzyme_pool_factor", columns="glucose_uptake", values="growth")
    ax.imshow(piv.values, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel("Glucose")
    ax.set_ylabel("Protein pool")
    ax.set_title("Phenotypic phase plane")


def p_single(ax, d):
    s = d["single"]
    ax.hist(s["growth_ratio"].clip(0,1.2), bins=28, color=PALETTE["blue2"], edgecolor="white", linewidth=0.2)
    ax.axvline(0.01, color=PALETTE["red"], ls="--", lw=1)
    ax.text(0.05, ax.get_ylim()[1]*0.85, f"{(s['phenotype']=='essential').sum()} essential", color=PALETTE["red"], fontsize=5.8)
    ax.set_xlabel("Growth ratio")
    ax.set_ylabel("Genes")
    ax.set_title("Single-gene KO")


def p_double(ax, d):
    x = d["double"].sort_values("interaction_score", ascending=False).head(10).iloc[::-1].copy()
    x["pair"] = x["gene_a"].str.replace("GA0070618_","G", regex=False) + "+" + x["gene_b"].str.replace("GA0070618_","G", regex=False)
    ax.barh(x["pair"], x["interaction_score"], color=PALETTE["red_soft"])
    ax.set_xlabel("Interaction")
    ax.set_title("Double-gene KO")
    ax.tick_params(axis="y", labelsize=4.8)


def p_dfba(ax, d):
    df = d["dfba"]
    ax.plot(df["time_h"], df["biomass_gDW_L"], color=PALETTE["blue"], lw=1.5)
    ax2 = ax.twinx()
    for col,c in [("glucose_mmol_L",PALETTE["gold"]),("nh4_mmol_L",PALETTE["green"]),("oxygen_mmol_L",PALETTE["violet"]),("phosphate_mmol_L",PALETTE["red"])]:
        ax2.plot(df["time_h"], df[col], color=c, lw=0.9, alpha=0.9)
    ax.set_xlabel("Time")
    ax.set_ylabel("Biomass")
    ax2.set_ylabel("Substrates")
    ax.set_title("dFBA dynamics")


def p_proxy(ax, d):
    for met, g in d["proxy"].groupby("metabolite"):
        ax.plot(g["time_h"], g["synthesis_capacity_proxy"], lw=1.1, label=met)
    ax.set_xlabel("Time")
    ax.set_ylabel("Turnover")
    ax.set_title("Intracellular dynamics")
    ax.legend(fontsize=4.2)


def p_fseof(ax, d):
    x = d["fseof_gene"].head(12).iloc[::-1]
    colors = [PALETTE["blue"] if r == "overexpression" else PALETTE["red"] for r in x["recommendation"]]
    ax.barh(x["gene"].str.replace("GA0070618_","G", regex=False), x["fseof_score"], color=colors)
    ax.set_xlabel("Score")
    ax.set_title("FSEOF")
    ax.tick_params(axis="y", labelsize=4.8)


def p_algorithm(ax, d, alg, title):
    t = d["targets"].replace([np.inf,-np.inf], np.nan).dropna(subset=["score"])
    t = t[t["algorithm"].astype(str).str.contains(alg, regex=False)].sort_values("score", ascending=False).head(10).iloc[::-1]
    if t.empty:
        no_axis(ax); ax.text(0.5,0.5,"No records",ha="center"); ax.set_title(title); return
    ax.barh(t["gene"].astype(str).str.replace("GA0070618_","G", regex=False), t["score"], color=PALETTE["teal"])
    ax.set_xlabel("Score")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=4.8)


def p_meta(ax, d):
    x = d["meta"].sort_values("metastrain_style_score", ascending=False).head(12).iloc[::-1]
    colors = [PALETTE["blue"] if op == "OE" else PALETTE["red"] for op in x["recommended_operation"]]
    ax.barh(x["gene"].str.replace("GA0070618_","G", regex=False), x["metastrain_style_score"], color=colors)
    ax.set_xlabel("Score")
    ax.set_title("MetaStrain")
    ax.tick_params(axis="y", labelsize=4.8)


def p_pathway(ax, d):
    cls = d["classes"]
    top = cls.groupby("class")["reaction_count"].sum().sort_values(ascending=False).head(7).index
    piv = cls[cls["class"].isin(top)].pivot(index="class", columns="model", values="reaction_count").fillna(0).loc[top[::-1]]
    y = np.arange(len(piv))
    ax.barh(y-0.16, piv.get("iFX1172",0), height=0.3, color=PALETTE["grey"], label="GEM")
    ax.barh(y+0.16, piv.get("eciFX1172",0), height=0.3, color=PALETTE["blue"], label="ec")
    ax.set_yticks(y); ax.set_yticklabels(piv.index, fontsize=5)
    ax.set_xlabel("Reactions")
    ax.set_title("Pathway distribution")


def make_composite(name, title, panels, nrows, ncols, figsize, data):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    letters = "abcdefghijklmnopqrstuvwxyz"
    for i, (func, ptitle) in enumerate(panels):
        ax = axes[i]
        func(ax, data)
        label(ax, letters[i])
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle(title, x=0.01, y=1.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, name, FIG_DIR, tiff=True)
    for i, (func, ptitle) in enumerate(panels):
        safe = re.sub(r"[^A-Za-z0-9]+", "_", ptitle).strip("_").lower()
        save_panel(func, f"{name}_panel_{letters[i]}_{safe}", data)


def compute_fva_table():
    selected = ["growth", "EX_glc__D_e_reverse", "r_0008", "r_0009", "r_0010", "r_0011", "r_0012", "r_0013", "GAPD", "G6PDH2r", "ICDHyr", "AKGDH", "ATPS4r"]
    probe_gem = load_gem_model()
    probe_ec = load_ec_model()
    rxns = [r for r in selected if r in probe_gem.reactions and r in probe_ec.reactions]
    rows = []
    gem = probe_gem
    ec = probe_ec
    def bound(model, rid, direction):
        model.objective = model.reactions.get_by_id(rid)
        model.objective_direction = direction
        try:
            value = model.slim_optimize(error_value=np.nan)
            return float(value) if value is not None else np.nan
        except Exception:
            return np.nan
    for r in rxns:
        gmin, gmax = bound(gem, r, "min"), bound(gem, r, "max")
        emin, emax = bound(ec, r, "min"), bound(ec, r, "max")
        grange, erange = gmax - gmin, emax - emin
        rows.append({
            "reaction": r,
            "iFX1172_min": gmin,
            "iFX1172_max": gmax,
            "iFX1172_range": grange,
            "eciFX1172_min": emin,
            "eciFX1172_max": emax,
            "eciFX1172_range": erange,
            "range_compression": 1 - erange / grange if pd.notna(grange) and pd.notna(erange) and abs(grange) > 1e-12 else np.nan,
        })
    return pd.DataFrame(rows)


def make_tables(data):
    s = data["summary"]
    summary = pd.DataFrame([
        {"metric": "model_id", "iFX1172": "iFX1172", "eciFX1172": "eciFX1172"},
        {"metric": "reactions", "iFX1172": int(s["irreversible_isoenzyme_reactions"]), "eciFX1172": int(s["irreversible_isoenzyme_reactions"])},
        {"metric": "metabolites", "iFX1172": int(s["original_metabolites"]), "eciFX1172": int(s["original_metabolites"])},
        {"metric": "genes", "iFX1172": int(s["original_genes"]), "eciFX1172": int(s["original_genes"])},
        {"metric": "enzyme_constrained_reactions", "iFX1172": 0, "eciFX1172": int(s["reactions_with_enzyme_constraint"])},
        {"metric": "enzyme_constraint_coverage", "iFX1172": 0, "eciFX1172": int(s["reactions_with_enzyme_constraint"]) / int(s["irreversible_isoenzyme_reactions"])},
        {"metric": "objective_value", "iFX1172": float(s["GEM_growth"]), "eciFX1172": float(s["ecGEM_growth"])},
        {"metric": "objective_retention", "iFX1172": 1.0, "eciFX1172": float(s["ecGEM_growth"]) / float(s["GEM_growth"])},
    ])
    enzyme = data["kcat"].copy()
    enzyme = enzyme.rename(columns={"reaction": "reaction_id", "data_type": "kcat_source", "MW": "protein_MW_kDa"})
    enzyme["enzyme_name"] = ""
    enzyme["gene_id"] = ""
    enzyme["evidence_level"] = enzyme["kcat_source"].map(lambda x: "direct/database" if "SABIO" in str(x) or "BRENDA" in str(x) else "inferred/default")
    enzyme = enzyme[["reaction_id", "enzyme_name", "gene_id", "ec_code", "kcat", "protein_MW_kDa", "kcat_MW", "kcat_source", "evidence_level"]]
    pool_rows = [
        {"parameter_set": "initial", "protein_pool": float(s["enzyme_pool_initial_upper_bound"]), "saturation_factor": "", "default_kcat": "", "objective_retention": "", "feasible": True},
        {"parameter_set": "calibrated", "protein_pool": float(s["enzyme_pool_upper_bound"]), "saturation_factor": "", "default_kcat": "", "objective_retention": float(s["ecGEM_growth"]) / float(s["GEM_growth"]), "feasible": True},
    ]
    if not data["pool"].empty:
        for _, r in data["pool"].iterrows():
            pool_rows.append({"parameter_set": f"factor_{r['enzyme_pool_factor']:.2f}", "protein_pool": r["enzyme_pool_upper_bound"], "saturation_factor": r["enzyme_pool_factor"], "default_kcat": "", "objective_retention": r["ec_growth"] / float(s["GEM_growth"]) if pd.notna(r["ec_growth"]) else "", "feasible": pd.notna(r["ec_growth"]) and r["ec_growth"] > 0})
    pool = pd.DataFrame(pool_rows)
    fva = compute_fva_table()
    data["fva"] = fva
    bottlenecks = data["meta"].copy().head(80)
    bottlenecks = bottlenecks.rename(columns={"gene": "gene_id", "representative_reactions": "high_priority_reactions", "metastrain_style_score": "priority_score"})
    bottlenecks["pathway_module"] = bottlenecks["high_priority_reactions"].map(lambda x: "C1a/product-linked" if any(t in str(x) for t in ["r_001", "NMN", "NT5C"]) else "central/cofactor/energy")
    bottlenecks["predicted_contribution_to_C1a"] = bottlenecks["recommendation"].map(lambda x: "positive forcing response" if x == "overexpression" else "negative forcing response")
    bottlenecks["engineering_strategy"] = bottlenecks["recommended_operation"]
    bottlenecks = bottlenecks[["gene_id", "high_priority_reactions", "pathway_module", "fseof_score", "single_KO_growth_ratio", "priority_score", "predicted_contribution_to_C1a", "engineering_strategy"]]
    tables = {
        "Supplementary Table 1": summary,
        "Supplementary Table 2": enzyme,
        "Supplementary Table 3": pool,
        "Supplementary Table 4": fva,
        "Supplementary Table 5": bottlenecks,
    }
    xlsx = TABLE_DIR / "docx_requested_supplementary_tables.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        for sheet, df in tables.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
            df.to_csv(TABLE_DIR / (sheet.replace(" ", "_").lower() + ".csv"), index=False, encoding="utf-8-sig")
    return tables, xlsx


def write_legends():
    text = """# Figure and table output generated from 图1相关.docx

Main Figure 1 contains panels Fig. 1a-f requested in the document.
Extended Data Figure 1 contains panels Extended Data Fig. 1a-f.
Supplementary Figures 1-6 are exported as separate figure files.
Supplementary Tables 1-5 are exported as individual CSV files and a combined Excel workbook.

Note: the current model explicitly contains product reaction r_0013 (GM-A production). Figures and tables use the document wording C1a/product module while retaining reaction IDs for traceability.
"""
    (OUT / "README_docx_requested_outputs.md").write_text(text, encoding="utf-8")


def main():
    setup_style()
    ensure_dirs()
    data = load_data()
    tables, xlsx = make_tables(data)
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(SOURCE_DIR / f"{key}.csv", index=False)

    fig1_panels = [
        (p_workflow, "Fig_1a_model_upgrade_pipeline"),
        (p_math, "Fig_1b_mathematical_logic"),
        (p_scale_coverage, "Fig_1c_model_scale_coverage"),
        (p_growth, "Fig_1d_objective_retention"),
        (p_flux_space, "Fig_1e_flux_space_compression"),
        (p_c1a_bottleneck, "Fig_1f_C1a_bottleneck_map"),
    ]
    ext_panels = [
        (p_network_check, "Extended_Data_Fig_1a_network_structure"),
        (p_split_example, "Extended_Data_Fig_1b_reaction_splitting"),
        (p_kcat_sources, "Extended_Data_Fig_1c_kcat_sources"),
        (p_pool_sensitivity, "Extended_Data_Fig_1d_pool_sensitivity"),
        (p_fva_key, "Extended_Data_Fig_1e_FVA_comparison"),
        (p_enzyme_allocation, "Extended_Data_Fig_1f_enzyme_allocation"),
    ]
    supp_figs = [
        (p_workflow, "Supplementary_Fig_1_eciFX1172_construction_pipeline"),
        (p_split_example, "Supplementary_Fig_2_GPR_curation_examples"),
        (lambda ax, d: p_distribution(ax, d, "kcat", "kcat distribution", PALETTE["teal"]), "Supplementary_Fig_3a_kcat_distribution"),
        (lambda ax, d: p_distribution(ax, d, "kcat_MW", "kcat/MW distribution", PALETTE["green"]), "Supplementary_Fig_3b_kcatMW_distribution"),
        (p_growth_feasibility, "Supplementary_Fig_4_model_feasibility"),
        (p_central_map, "Supplementary_Fig_5_central_carbon_map"),
        (p_c1a_map, "Supplementary_Fig_6_gentamicin_C1a_biosynthesis_map"),
    ]

    make_composite("Figure_1_ecModel_construction", "Figure 1 | Construction and calibration of eciFX1172", fig1_panels, 2, 3, (7.25, 4.9), data)
    make_composite("Extended_Data_Figure_1_ecModel_supporting_analysis", "Extended Data Figure 1 | Supporting model-construction analyses", ext_panels, 2, 3, (7.25, 4.9), data)
    for func, name in supp_figs:
        fig, ax = plt.subplots(figsize=(3.1, 2.35))
        func(ax, data)
        save_fig(fig, name, FIG_DIR, tiff=True)
        save_panel(func, name + "_panel", data, size=(2.7, 2.1))

    write_legends()
    manifest = {
        "out": str(OUT),
        "main_figures": sorted(p.name for p in FIG_DIR.glob("*.svg")),
        "individual_panels": sorted(p.name for p in PANEL_DIR.glob("*.svg")),
        "tables_excel": str(xlsx),
        "tables": sorted(p.name for p in TABLE_DIR.glob("*.csv")),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
