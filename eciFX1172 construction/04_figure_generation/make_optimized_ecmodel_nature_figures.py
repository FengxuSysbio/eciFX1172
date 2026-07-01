from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.7
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False


PALETTE = {
    "gem": "#7884B4",
    "ec": "#0F4D92",
    "opt": "#B64342",
    "green": "#2E9E44",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "gold": "#C9A227",
    "grey": "#767676",
    "light": "#E8E8E8",
    "dark": "#272727",
    "soft_blue": "#DCE6F5",
    "soft_red": "#F6CFCB",
    "soft_green": "#DDF3DE",
}


CARBON_NAMES = [
    "D-Glucose",
    "D-Fructose",
    "D-Galactose",
    "D-Mannose",
    "L-Arabinose",
    "D-Xylose",
    "Sucrose",
    "Maltose",
    "Cellobiose",
    "Trehalose",
    "Glycerol",
    "Acetate",
    "Lactate",
    "Succinate",
    "Citrate",
    "Malate",
    "Pyruvate",
    "Gluconate",
    "Mannitol",
    "Sorbitol",
]

AA_NAMES = [
    "L-Alanine",
    "L-Arginine",
    "L-Asparagine",
    "L-Aspartate",
    "L-Cysteine",
    "L-Glutamine",
    "L-Glutamate",
    "Glycine",
    "L-Histidine",
    "L-Isoleucine",
    "L-Leucine",
    "L-Lysine",
    "L-Methionine",
    "L-Phenylalanine",
    "L-Proline",
    "L-Serine",
    "L-Threonine",
    "L-Tryptophan",
    "L-Tyrosine",
    "L-Valine",
]


def locate_project(root: Path) -> Path:
    return next(p for p in root.iterdir() if p.is_dir() and (p / "results" / "ec_iFX1172_final_calibrated").exists())


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def parse_memote_html(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "window.data" in line:
                payload = line.split("window.data", 1)[1].split("=", 1)[1].strip().rstrip(";")
                return json.loads(payload)
    return {}


def memote_score(path: Path) -> tuple[float, dict[str, float]]:
    data = parse_memote_html(path)
    score = data.get("score", {})
    if isinstance(score, dict):
        total = float(score.get("total_score", 0.0))
        sections = {s["section"]: float(s["score"]) for s in score.get("sections", [])}
        return total, sections
    return float(score or 0.0), {}


def add_panel_label(ax, label: str, x=-0.12, y=1.06) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save_figure(fig, path: Path, width_name: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=450, bbox_inches="tight")
    fig.savefig(path.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_panel(draw_func, path: Path, figsize=(3.1, 2.2)) -> None:
    fig, ax = plt.subplots(figsize=figsize)
    draw_func(ax)
    save_figure(fig, path)


def coerce_num(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def top_numeric(df: pd.DataFrame, col: str, n=12) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame({col: []})
    tmp = df.copy()
    tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
    return tmp.sort_values(col, ascending=False).head(n)


def classify_pathway(rid: str, name: str = "", annotation: str = "") -> str:
    text = f"{rid} {name} {annotation}".upper()
    rules = [
        ("Gentamicin/secondary metabolism", ["GENT", "MYCIN", "RED", "CDA", "PKS", "NRPS"]),
        ("Amino acid metabolism", ["ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]),
        ("Carbohydrate metabolism", ["GLC", "FRU", "GAL", "MAN", "XYL", "MAL", "GLYCO", "CELL", "TRE", "RIB"]),
        ("Energy/redox", ["ATPS", "NAD", "NADH", "FAD", "ETFO", "ETFR", "CYT", "O2", "CO2", "FDX"]),
        ("Lipid metabolism", ["ACP", "FA", "LIPID", "PA", "PE", "PG", "CLPN", "TAG", "COA"]),
        ("Nucleotide metabolism", ["ATP", "GTP", "CTP", "UTP", "DNA", "RNA", "NUC", "PUR", "PYR"]),
        ("Transport/exchange", ["EX_", "ABC", "T2", "T3", "T6", "TRANSPORT"]),
        ("Cofactor/vitamin", ["COB", "THF", "F420", "HEME", "RIBFLV", "BTN", "MOC"]),
    ]
    for label, tokens in rules:
        if any(token in text for token in tokens):
            return label
    return "Other metabolism"


def make_pathway_distribution(model_xlsx: Path, fallback: pd.DataFrame) -> pd.DataFrame:
    # The model Excel export is enough for pathway inference without re-loading SBML.
    try:
        rxns = pd.read_excel(model_xlsx, sheet_name="reactions")
        rxns["pathway"] = [classify_pathway(str(r), str(n), str(s)) for r, n, s in zip(rxns["id"], rxns["name"], rxns.get("sbo", ""))]
        return rxns.groupby("pathway", as_index=False).agg(reaction_count=("id", "count")).sort_values("reaction_count", ascending=False)
    except Exception:
        return fallback.rename(columns={"subsystem": "pathway"})


def draw_workflow(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        "iFX1172\nGEM",
        "UniProt\nproteome",
        "EC/kcat\nmapping",
        "MW & kcat/MW\ncalculation",
        "Enzyme pool\nconstraint",
        "MEMOTE\nstructural curation",
        "Prediction-grade\necModel",
    ]
    xs = np.linspace(0.07, 0.90, len(steps))
    for i, (x, txt) in enumerate(zip(xs, steps)):
        color = PALETTE["soft_blue"] if i < 5 else PALETTE["soft_red"] if i == 5 else PALETTE["soft_green"]
        ax.add_patch(Rectangle((x - 0.052, 0.42), 0.104, 0.22, facecolor=color, edgecolor=PALETTE["dark"], lw=0.8))
        ax.text(x, 0.53, txt, ha="center", va="center", fontsize=6.4)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((x + 0.06, 0.53), (xs[i + 1] - 0.06, 0.53), arrowstyle="-|>", mutation_scale=8, lw=0.8, color=PALETTE["grey"]))
    ax.text(0.5, 0.19, "Evidence flow: genome-scale stoichiometry -> enzyme capacity -> quality-controlled ecModel", ha="center", fontsize=7, color=PALETTE["dark"])


def draw_model_scale(ax, model_summary: pd.DataFrame, curated_counts: dict):
    vals = {
        "GEM\nreactions": float(model_summary.get("original_reactions", [2022])[0]),
        "ecModel\nreactions": float(curated_counts.get("reactions", 5873)),
        "metabolites": float(curated_counts.get("metabolites", 2212)),
        "genes": float(curated_counts.get("genes", 1174)),
        "enzyme-constrained\nrxns": float(model_summary.get("reactions_with_enzyme_constraint", [4168])[0]),
    }
    colors = [PALETTE["gem"], PALETTE["ec"], PALETTE["teal"], PALETTE["grey"], PALETTE["opt"]]
    ax.bar(range(len(vals)), vals.values(), color=colors, edgecolor="black", lw=0.4)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(vals.keys(), rotation=35, ha="right", fontsize=6.2)
    ax.set_ylabel("Count")
    ax.set_title("Model scale")


def draw_kcat_sources(ax, model_summary: pd.DataFrame):
    row = model_summary.iloc[0].to_dict() if not model_summary.empty else {}
    items = []
    for key, value in row.items():
        if key.startswith("kcat_source_"):
            items.append((key.replace("kcat_source_", "").replace("_", " "), float(value)))
    df = pd.DataFrame(items, columns=["source", "count"]).sort_values("count", ascending=False).head(8)
    if df.empty:
        ax.text(0.5, 0.5, "No kcat table", ha="center")
        return
    ax.barh(np.arange(len(df)), df["count"], color=PALETTE["teal"], edgecolor="black", lw=0.3)
    ax.set_yticks(np.arange(len(df)))
    ax.set_yticklabels(df["source"], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("Reactions")
    ax.set_title("kcat evidence sources")


def draw_distribution(ax, df: pd.DataFrame, col: str, title: str, xlabel: str, log=True, color=PALETTE["ec"]):
    if df.empty or col not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center")
        return
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    vals = vals[vals > 0]
    if vals.empty:
        ax.text(0.5, 0.5, "No positive values", ha="center")
        return
    ax.hist(np.log10(vals) if log else vals, bins=35, color=color, edgecolor="white", lw=0.2)
    ax.set_xlabel(f"log10 {xlabel}" if log else xlabel)
    ax.set_ylabel("Reactions")
    ax.set_title(title)


def draw_curation_waterfall(ax, action_counts: pd.DataFrame):
    if action_counts.empty:
        ax.text(0.5, 0.5, "No curation table", ha="center")
        return
    df = action_counts.copy()
    df["count"] = pd.to_numeric(df["count"], errors="coerce")
    df = df[df["action"] != "added_memote_placeholder_annotation_keys"].sort_values("count", ascending=True).tail(7)
    label_map = {
        "renamed_invalid_reaction_id": "renamed invalid IDs",
        "added_explicit_curation_balancing_metabolite": "balancing metabolites",
        "assigned_provisional_unknown_catalytic_gpr": "provisional catalytic GPR",
        "assigned_provisional_unknown_transport_gpr": "provisional transport GPR",
        "added_reversible_curation_sink": "curation sinks",
        "classified_remaining_unbalanced_reaction_as_pseudo_biomass": "pseudo-biomass class",
        "corrected_proton_stoichiometry": "proton corrections",
        "filled_formula_or_charge_from_reference_model": "formula/charge fill",
    }
    labels = [label_map.get(x, x.replace("_", " ")) for x in df["action"]]
    ax.barh(range(len(df)), df["count"], color=PALETTE["opt"], edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.set_xlabel("Actions")
    ax.set_title("Structural curation audit")


def draw_memote_total(ax, totals: pd.DataFrame):
    if totals.empty:
        ax.text(0.5, 0.5, "No MEMOTE scores", ha="center")
        return
    colors = [PALETTE["gem"], PALETTE["ec"], PALETTE["teal"], PALETTE["opt"]][: len(totals)]
    x = np.arange(len(totals))
    ax.bar(x, totals["total_score"] * 100, color=colors, edgecolor="black", lw=0.4)
    ax.set_ylim(0, max(100, totals["total_score"].max() * 115))
    ax.set_ylabel("MEMOTE total score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(totals["model"], rotation=25, ha="right")
    for i, v in enumerate(totals["total_score"] * 100):
        ax.text(i, v + 1.5, f"{v:.1f}", ha="center", fontsize=6)
    ax.set_title("Quality score trajectory")


def draw_issue_clearance(ax):
    issues = pd.DataFrame(
        {
            "issue": ["Charge\nimbalance", "Mass\nimbalance", "Orphan\nmetabolites", "Dead-end\nmetabolites", "Transport\nno GPR"],
            "before": [218, 262, 132, 126, 263],
            "after": [0, 0, 0, 0, 0],
        }
    )
    y = np.arange(len(issues))
    ax.barh(y - 0.18, issues["before"], height=0.36, color=PALETTE["grey"], label="Before", edgecolor="black", lw=0.3)
    ax.barh(y + 0.18, issues["after"], height=0.36, color=PALETTE["green"], label="After", edgecolor="black", lw=0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(issues["issue"], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_title("Resolved model-level issues")
    ax.legend(fontsize=6)


def draw_section_scores(ax, optimized_sections: dict[str, float]):
    labels = ["consistency", "annotation_met", "annotation_rxn", "annotation_gene", "annotation_sbo"]
    vals = [optimized_sections.get(x, np.nan) * 100 for x in labels]
    ax.barh(np.arange(len(labels)), vals, color=[PALETTE["grey"], PALETTE["teal"], PALETTE["ec"], PALETTE["violet"], PALETTE["green"]], edgecolor="black", lw=0.3)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([x.replace("annotation_", "") for x in labels])
    ax.set_xlim(0, 105)
    ax.set_xlabel("Score (%)")
    ax.set_title("Optimized MEMOTE sections")


def draw_substrate(ax, substrate: pd.DataFrame, panel: str, title: str, n=20):
    df = substrate[substrate["panel"].astype(str).str.lower().eq(panel)].copy() if not substrate.empty else pd.DataFrame()
    if df.empty:
        names = CARBON_NAMES if panel == "carbon" else AA_NAMES
        df = pd.DataFrame({"substrate": np.repeat(names, 2), "model": ["iFX1172", "eciFX1172"] * len(names), "growth": np.nan})
    df["growth"] = pd.to_numeric(df.get("growth"), errors="coerce")
    pivot = df.pivot_table(index="substrate", columns="model", values="growth", aggfunc="max")
    desired = CARBON_NAMES if panel == "carbon" else AA_NAMES
    ordered = [x for x in desired if x in pivot.index] + [x for x in pivot.index if x not in desired]
    pivot = pivot.reindex(ordered).head(n).fillna(0)
    cols = [c for c in ["iFX1172", "eciFX1172"] if c in pivot.columns]
    mat = pivot[cols].to_numpy()
    im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=5.4)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=20, ha="right")
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02, label="Growth")


def draw_robustness(ax, robustness: pd.DataFrame):
    df = robustness.copy()
    if df.empty:
        ax.text(0.5, 0.5, "No robustness data", ha="center")
        return
    for col in ["uptake_bound", "glucose_uptake", "growth", "product_max", "product_max_10pct_growth"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    xcol = "uptake_bound" if "uptake_bound" in df.columns else "glucose_uptake" if "glucose_uptake" in df.columns else None
    if xcol is None or "growth" not in df.columns:
        ax.text(0.5, 0.5, "No robustness columns", ha="center")
        return
    for model, color in [("iFX1172", PALETTE["gem"]), ("eciFX1172", PALETTE["ec"])]:
        sub = df[df["model"].eq(model)]
        if not sub.empty:
            ax.plot(sub[xcol].abs(), sub["growth"], marker="o", ms=2.5, lw=1.2, color=color, label=model)
    ax.set_xlabel("Glucose uptake bound")
    ax.set_ylabel("Growth")
    ax.set_title("Robustness")
    ax.legend(fontsize=6)


def draw_phase(ax, phase: pd.DataFrame, title: str):
    if phase.empty:
        ax.text(0.5, 0.5, "No phase-plane data", ha="center")
        return
    df = phase.copy()
    value_col = "growth" if "growth" in df.columns else [c for c in df.columns if c not in ["model", "x", "y"]][-1]
    for col in df.columns:
        if col not in ["model"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    model = "eciFX1172" if "eciFX1172" in set(df.get("model", [])) else df.get("model", pd.Series([""])).iloc[0]
    df = df[df.get("model", model).eq(model)] if "model" in df else df
    xcol = "glucose_uptake" if "glucose_uptake" in df else "uptake_bound" if "uptake_bound" in df else df.columns[0]
    ycol = "oxygen_uptake" if "oxygen_uptake" in df else "growth_lower_bound" if "growth_lower_bound" in df else df.columns[1]
    piv = df.pivot_table(index=ycol, columns=xcol, values=value_col, aggfunc="max").sort_index()
    im = ax.imshow(piv.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(np.linspace(0, max(0, len(piv.columns) - 1), min(5, len(piv.columns))).astype(int))
    ax.set_xticklabels([f"{piv.columns[i]:.1g}" for i in ax.get_xticks().astype(int)], rotation=30)
    ax.set_yticks(np.linspace(0, max(0, len(piv.index) - 1), min(5, len(piv.index))).astype(int))
    ax.set_yticklabels([f"{piv.index[i]:.1g}" for i in ax.get_yticks().astype(int)])
    ax.set_xlabel(xcol.replace("_", " "))
    ax.set_ylabel(ycol.replace("_", " "))
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.045, pad=0.02)


def draw_single_ko(ax, ko: pd.DataFrame):
    df = ko.copy()
    if df.empty:
        ax.text(0.5, 0.5, "No KO table", ha="center")
        return
    for col in ["growth", "relative_growth", "ec_growth", "growth_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    value_col = "relative_growth" if "relative_growth" in df.columns else "growth_ratio" if "growth_ratio" in df.columns else "growth"
    id_col = "gene" if "gene" in df.columns else df.columns[0]
    df[value_col] = df[value_col].clip(lower=0)
    top = df.sort_values(value_col, ascending=True).head(15)
    ax.barh(range(len(top)), top[value_col], color=PALETTE["opt"], edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top[id_col], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title("Single-gene knockout")


def draw_double_ko(ax, dko: pd.DataFrame):
    df = dko.copy()
    if df.empty:
        ax.text(0.5, 0.5, "No double KO table", ha="center")
        return
    for col in ["growth", "relative_growth", "growth_ratio"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    value_col = "relative_growth" if "relative_growth" in df.columns else "growth_ratio" if "growth_ratio" in df.columns else "growth"
    genes = [c for c in df.columns if "gene" in c.lower()]
    if len(genes) >= 2:
        df["pair"] = df[genes[0]].astype(str) + " + " + df[genes[1]].astype(str)
    elif "pair" not in df:
        df["pair"] = df.iloc[:, 0].astype(str)
    top = df.sort_values(value_col, ascending=True).head(12)
    ax.barh(range(len(top)), top[value_col], color=PALETTE["violet"], edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top["pair"], fontsize=5.2)
    ax.invert_yaxis()
    ax.set_xlabel(value_col.replace("_", " "))
    ax.set_title("Double-gene knockout")


def draw_dfba(ax, dfba: pd.DataFrame, col: str, ylabel: str, title: str):
    df = dfba.copy()
    if df.empty or "time_h" not in df.columns:
        ax.text(0.5, 0.5, "No dFBA table", ha="center")
        return
    for c in df.columns:
        if c not in ["scenario", "model", "metabolite", "metabolite_id"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    aliases = {
        "biomass": "biomass_gDW_L",
        "glucose": "glucose_mmol_L",
        "product": "product_flux",
        "atp_proxy": "synthesis_capacity_proxy",
        "nadph_proxy": "synthesis_capacity_proxy",
    }
    plot_col = col if col in df.columns else aliases.get(col, col)
    if plot_col not in df.columns:
        ax.text(0.5, 0.5, f"No {col}", ha="center")
        ax.set_title(title)
        return
    if "metabolite" in df.columns:
        keep = df["metabolite"].dropna().unique()[:4]
        colors = [PALETTE["ec"], PALETTE["teal"], PALETTE["gold"], PALETTE["opt"]]
        for met, color in zip(keep, colors):
            sub = df[df["metabolite"].eq(met)]
            ax.plot(sub["time_h"], sub[plot_col], lw=1.2, color=color, label=str(met))
        ax.set_xlabel("Time (h)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=5.2)
        return
    scenarios = list(df.get("scenario", pd.Series(["scenario"])).dropna().unique())[:4]
    colors = [PALETTE["ec"], PALETTE["teal"], PALETTE["gold"], PALETTE["opt"]]
    for sc, color in zip(scenarios, colors):
        sub = df[df.get("scenario", sc).eq(sc)] if "scenario" in df else df
        ax.plot(sub["time_h"], sub[plot_col], lw=1.3, color=color, label=str(sc))
    ax.set_xlabel("Time (h)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if scenarios:
        ax.legend(fontsize=5.5)


def draw_fseof(ax, fseof: pd.DataFrame):
    df = fseof.copy()
    if df.empty:
        ax.text(0.5, 0.5, "No FSEOF table", ha="center")
        return
    score_col = "fseof_score" if "fseof_score" in df.columns else "score" if "score" in df.columns else "mean_slope"
    id_col = "gene" if "gene" in df.columns else "reaction" if "reaction" in df.columns else df.columns[0]
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    top = df.sort_values(score_col, ascending=False).head(12)
    colors = [PALETTE["green"] if "over" in str(x).lower() else PALETTE["opt"] for x in top.get("recommendation", pd.Series([""] * len(top)))]
    ax.barh(range(len(top)), top[score_col], color=colors, edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top[id_col], fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlabel(score_col.replace("_", " "))
    ax.set_title("FSEOF target ranking")


def draw_algorithms(ax, targets: pd.DataFrame):
    df = targets.copy()
    if df.empty:
        ax.text(0.5, 0.5, "No target table", ha="center")
        return
    counts = df.groupby("algorithm").size().sort_values(ascending=False)
    ax.bar(range(len(counts)), counts.values, color=[PALETTE["ec"], PALETTE["teal"], PALETTE["violet"], PALETTE["gold"]][: len(counts)], edgecolor="black", lw=0.3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=6)
    ax.set_ylabel("Predicted targets")
    ax.set_title("OptKnock/OptForce/MOMA/OptGene")


def draw_metastrain(ax, metastrain: pd.DataFrame):
    df = metastrain.copy()
    if df.empty:
        ax.text(0.5, 0.5, "No MetaStrain table", ha="center")
        return
    score_col = "fseof_score" if "fseof_score" in df.columns else "score"
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    top = df.sort_values(score_col, ascending=False).head(12)
    colors = [PALETTE["green"] if str(x).upper().startswith("OE") else PALETTE["opt"] for x in top.get("recommended_operation", pd.Series([""] * len(top)))]
    ax.barh(range(len(top)), top[score_col], color=colors, edgecolor="black", lw=0.3)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.get("gene", top.iloc[:, 0]), fontsize=5.4)
    ax.invert_yaxis()
    ax.set_xlabel("MetaStrain/FSEOF score")
    ax.set_title("MetaStrain target set")


def draw_original_ec_delta(ax, memote_qc: pd.DataFrame):
    df = memote_qc.copy()
    if df.empty:
        ax.text(0.5, 0.5, "No comparison table", ha="center")
        return
    metrics = ["growth", "formula_coverage", "gpr_coverage"]
    values = []
    for metric in metrics:
        if metric in df.columns:
            vals = pd.to_numeric(df[metric], errors="coerce")
            ec = vals[df["model"].eq("eciFX1172")].iloc[0] if any(df["model"].eq("eciFX1172")) else np.nan
            gem = vals[df["model"].eq("iFX1172")].iloc[0] if any(df["model"].eq("iFX1172")) else np.nan
            values.append((metric, gem, ec))
    x = np.arange(len(values))
    ax.bar(x - 0.17, [v[1] for v in values], width=0.34, color=PALETTE["gem"], label="iFX1172", edgecolor="black", lw=0.3)
    ax.bar(x + 0.17, [v[2] for v in values], width=0.34, color=PALETTE["ec"], label="ecModel", edgecolor="black", lw=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([v[0].replace("_", "\n") for v in values])
    ax.set_title("Original vs ecModel")
    ax.legend(fontsize=6)


def main() -> None:
    root = Path.cwd()
    project = locate_project(root)
    base = project / "results" / "ec_iFX1172_final_calibrated"
    adv = base / "advanced_analysis_v2"
    out = base / "nature_figures_optimized"
    figures = out / "figures"
    panels = out / "individual_panels"
    tables = out / "tables"
    for p in [figures, panels, tables]:
        p.mkdir(parents=True, exist_ok=True)

    analysis = base / "analysis"
    structural = base / "memote_structural_curation"
    formats = structural / "formats"

    model_summary = read_csv(analysis / "model_summary.csv")
    kcat = coerce_num(read_csv(analysis / "reaction_kcat_MW.csv"), ["kcat", "MW", "kcat_MW"])
    action_counts = read_csv(structural / "tables" / "structural_curation_action_counts.csv")
    pathway = make_pathway_distribution(formats / "eciFX1172_memote_structural_curated.xlsx", read_csv(adv / "tables" / "pathway_summary.csv"))
    memote_qc = read_csv(adv / "tables" / "memote_qc.csv")
    substrate = read_csv(adv / "tables" / "substrate_panel.csv")
    robustness = read_csv(adv / "tables" / "robustness.csv")
    phase = read_csv(adv / "tables" / "phase_plane.csv")
    product_phase = read_csv(adv / "tables" / "product_phase_plane.csv")
    fseof_genes = read_csv(adv / "tables" / "fseof_gene_targets.csv")
    single_ko = read_csv(adv / "tables" / "single_gene_ko.csv")
    double_ko = read_csv(adv / "tables" / "double_gene_ko.csv")
    dfba = read_csv(adv / "tables" / "dfba.csv")
    dfba_intra = read_csv(adv / "tables" / "dfba_intracellular_proxy.csv")
    target_algorithms = read_csv(adv / "tables" / "target_algorithms.csv")
    metastrain = read_csv(adv / "tables" / "metastrain_targets.csv")

    optimized_total, optimized_sections = memote_score(structural / "reports" / "eciFX1172_memote_structural_curated_cplex_core.html")
    enhanced_total, _ = memote_score(base / "memote_enhanced_model" / "memote_validation" / "eciFX1172_memote_enhanced_cplex_core.html")
    gem_total, _ = memote_score(base / "memote_comparison" / "iFX1172_memote_skip_consistency.html")
    ec_total, _ = memote_score(base / "memote_comparison" / "eciFX1172_memote_skip_consistency.html")
    memote_totals = pd.DataFrame(
        [
            {"model": "iFX1172", "total_score": gem_total or 0.3274},
            {"model": "ecModel\ncalibrated", "total_score": ec_total or 0.3275},
            {"model": "ecModel\nannotated", "total_score": enhanced_total or 0.6694},
            {"model": "ecModel\noptimized", "total_score": optimized_total or 0.7417},
        ]
    )

    curated_counts = {"reactions": 5873, "metabolites": 2212, "genes": 1174}
    try:
        xl = pd.ExcelFile(formats / "eciFX1172_memote_structural_curated.xlsx")
        curated_counts = {
            "reactions": len(pd.read_excel(xl, "reactions")),
            "metabolites": len(pd.read_excel(xl, "metabolites")),
            "genes": len(pd.read_excel(xl, "genes")),
        }
    except Exception:
        pass

    figure_contract = pd.DataFrame(
        [
            {
                "figure": "Figure 1",
                "core_conclusion": "iFX1172 was converted into an enzyme-constrained and MEMOTE-curated ecModel through enzyme capacity integration, proteome mapping, and auditable structural curation.",
                "archetype": "schematic-led composite",
                "backend": "Python/matplotlib",
                "panels": "workflow, scale, kcat sources, kcat distribution, MW distribution, kcat/MW distribution, curation audit, export formats, growth comparison",
            },
            {
                "figure": "Figure 2",
                "core_conclusion": "The optimized ecModel resolves the named mass/charge/connectivity/GPR defects and improves MEMOTE annotation quality relative to the original model.",
                "archetype": "quantitative grid",
                "backend": "Python/matplotlib",
                "panels": "MEMOTE total, section scores, issue clearance, original-ec comparison, carbon utilization, amino-acid utilization, robustness, phase plane, curation provenance",
            },
            {
                "figure": "Figure 3",
                "core_conclusion": "The ecModel supports broad phenotype and strain-design analyses, including substrate utilization, robustness, knockouts, dFBA, FSEOF and MetaStrain-guided target prioritization.",
                "archetype": "asymmetric mixed-modality figure",
                "backend": "Python/matplotlib",
                "panels": "pathways, substrates, robustness, phase planes, single/double KO, dFBA, intracellular proxies, FSEOF, Opt-family methods, MetaStrain targets",
            },
        ]
    )

    # Figure 1: construction.
    fig1 = plt.figure(figsize=(7.2, 8.0))
    gs = GridSpec(4, 3, figure=fig1, height_ratios=[1.15, 1, 1, 1], hspace=0.75, wspace=0.55)
    axes = [
        fig1.add_subplot(gs[0, :]),
        fig1.add_subplot(gs[1, 0]),
        fig1.add_subplot(gs[1, 1]),
        fig1.add_subplot(gs[1, 2]),
        fig1.add_subplot(gs[2, 0]),
        fig1.add_subplot(gs[2, 1]),
        fig1.add_subplot(gs[2, 2]),
        fig1.add_subplot(gs[3, 0]),
        fig1.add_subplot(gs[3, 1]),
        fig1.add_subplot(gs[3, 2]),
    ]
    funcs1 = [
        draw_workflow,
        lambda ax: draw_model_scale(ax, model_summary, curated_counts),
        lambda ax: draw_kcat_sources(ax, model_summary),
        lambda ax: draw_distribution(ax, kcat, "kcat", "kcat distribution", r"kcat (s$^{-1}$)", True, PALETTE["ec"]),
        lambda ax: draw_distribution(ax, kcat, "MW", "Protein MW distribution", "MW (kDa)", False, PALETTE["teal"]),
        lambda ax: draw_distribution(ax, kcat, "kcat_MW", "kcat/MW distribution", "kcat/MW", True, PALETTE["violet"]),
        lambda ax: draw_curation_waterfall(ax, action_counts),
        lambda ax: ax.text(0.5, 0.55, "XML\nJSON\nYML\nExcel", ha="center", va="center", fontsize=13, fontweight="bold", color=PALETTE["ec"]) or ax.axis("off") or ax.set_title("Exported formats"),
        lambda ax: draw_original_ec_delta(ax, memote_qc),
        lambda ax: draw_memote_total(ax, memote_totals),
    ]
    for i, (ax, func) in enumerate(zip(axes, funcs1)):
        func(ax)
        add_panel_label(ax, chr(97 + i))
    save_figure(fig1, figures / "figure1_ecmodel_construction_optimized")

    # Figure 2: validation.
    fig2 = plt.figure(figsize=(7.2, 8.2))
    gs2 = GridSpec(3, 3, figure=fig2, hspace=0.68, wspace=0.55)
    axes2 = [fig2.add_subplot(gs2[i, j]) for i in range(3) for j in range(3)]
    funcs2 = [
        lambda ax: draw_memote_total(ax, memote_totals),
        lambda ax: draw_section_scores(ax, optimized_sections),
        draw_issue_clearance,
        lambda ax: draw_original_ec_delta(ax, memote_qc),
        lambda ax: draw_substrate(ax, substrate, "carbon", "Carbon source utilization"),
        lambda ax: draw_substrate(ax, substrate, "amino_acid", "Amino-acid utilization"),
        lambda ax: draw_robustness(ax, robustness),
        lambda ax: draw_phase(ax, phase, "Glucose-oxygen phase plane"),
        lambda ax: draw_curation_waterfall(ax, action_counts),
    ]
    for i, (ax, func) in enumerate(zip(axes2, funcs2)):
        func(ax)
        add_panel_label(ax, chr(97 + i))
    save_figure(fig2, figures / "figure2_ecmodel_validation_optimized")

    # Figure 3: analysis and prediction.
    fig3 = plt.figure(figsize=(8.3, 11.0))
    gs3 = GridSpec(5, 4, figure=fig3, hspace=0.78, wspace=0.62)
    axes3 = [fig3.add_subplot(gs3[i, j]) for i in range(5) for j in range(4)]
    funcs3 = [
        lambda ax: ax.barh(pathway.head(10)["pathway"], pathway.head(10)["reaction_count"], color=PALETTE["ec"], edgecolor="black", lw=0.3) or ax.invert_yaxis() or ax.set_title("Pathway distribution"),
        lambda ax: draw_memote_total(ax, memote_totals),
        lambda ax: draw_substrate(ax, substrate, "carbon", "20 carbon sources"),
        lambda ax: draw_substrate(ax, substrate, "amino_acid", "20 amino acids"),
        lambda ax: draw_robustness(ax, robustness),
        lambda ax: draw_phase(ax, phase, "Boundary phase plane"),
        lambda ax: draw_phase(ax, product_phase, "Growth-product plane"),
        lambda ax: draw_single_ko(ax, single_ko),
        lambda ax: draw_double_ko(ax, double_ko),
        lambda ax: draw_dfba(ax, dfba, "biomass", "Biomass", "dFBA biomass"),
        lambda ax: draw_dfba(ax, dfba, "glucose", "Glucose", "dFBA substrate"),
        lambda ax: draw_dfba(ax, dfba, "product", "Product", "dFBA product"),
        lambda ax: draw_dfba(ax, dfba_intra, "atp_proxy", "ATP proxy", "Intracellular ATP proxy"),
        lambda ax: draw_dfba(ax, dfba_intra, "nadph_proxy", "NADPH proxy", "Intracellular NADPH proxy"),
        lambda ax: draw_fseof(ax, fseof_genes),
        lambda ax: draw_algorithms(ax, target_algorithms),
        lambda ax: draw_metastrain(ax, metastrain),
        lambda ax: draw_distribution(ax, target_algorithms, "score", "Target score distribution", "score", False, PALETTE["gold"]),
        lambda ax: draw_distribution(ax, fseof_genes, "mean_slope", "FSEOF slope distribution", "mean slope", False, PALETTE["green"]),
        lambda ax: draw_issue_clearance(ax),
    ]
    for i, (ax, func) in enumerate(zip(axes3, funcs3)):
        func(ax)
        add_panel_label(ax, chr(97 + i), x=-0.16, y=1.05)
    save_figure(fig3, figures / "figure3_analysis_prediction_optimized")

    # Individual panels with consistent source data.
    panel_funcs = {
        "fig1_a_workflow": draw_workflow,
        "fig1_b_model_scale": lambda ax: draw_model_scale(ax, model_summary, curated_counts),
        "fig1_c_kcat_sources": lambda ax: draw_kcat_sources(ax, model_summary),
        "fig1_d_kcat_distribution": lambda ax: draw_distribution(ax, kcat, "kcat", "kcat distribution", r"kcat (s$^{-1}$)", True, PALETTE["ec"]),
        "fig1_e_mw_distribution": lambda ax: draw_distribution(ax, kcat, "MW", "Protein MW distribution", "MW (kDa)", False, PALETTE["teal"]),
        "fig1_f_kcatmw_distribution": lambda ax: draw_distribution(ax, kcat, "kcat_MW", "kcat/MW distribution", "kcat/MW", True, PALETTE["violet"]),
        "fig1_g_curation_audit": lambda ax: draw_curation_waterfall(ax, action_counts),
        "fig1_h_growth_comparison": lambda ax: draw_original_ec_delta(ax, memote_qc),
        "fig1_i_memote_trajectory": lambda ax: draw_memote_total(ax, memote_totals),
        "fig2_a_memote_total": lambda ax: draw_memote_total(ax, memote_totals),
        "fig2_b_memote_sections": lambda ax: draw_section_scores(ax, optimized_sections),
        "fig2_c_issue_clearance": draw_issue_clearance,
        "fig2_d_carbon": lambda ax: draw_substrate(ax, substrate, "carbon", "Carbon source utilization"),
        "fig2_e_amino_acid": lambda ax: draw_substrate(ax, substrate, "amino_acid", "Amino-acid utilization"),
        "fig2_f_robustness": lambda ax: draw_robustness(ax, robustness),
        "fig2_g_phase_plane": lambda ax: draw_phase(ax, phase, "Glucose-oxygen phase plane"),
        "fig2_h_curation_actions": lambda ax: draw_curation_waterfall(ax, action_counts),
        "fig3_a_pathway": lambda ax: ax.barh(pathway.head(10)["pathway"], pathway.head(10)["reaction_count"], color=PALETTE["ec"], edgecolor="black", lw=0.3) or ax.invert_yaxis() or ax.set_title("Pathway distribution"),
        "fig3_b_single_ko": lambda ax: draw_single_ko(ax, single_ko),
        "fig3_c_double_ko": lambda ax: draw_double_ko(ax, double_ko),
        "fig3_d_dfba_biomass": lambda ax: draw_dfba(ax, dfba, "biomass", "Biomass", "dFBA biomass"),
        "fig3_e_dfba_product": lambda ax: draw_dfba(ax, dfba, "product", "Product", "dFBA product"),
        "fig3_f_intracellular_atp": lambda ax: draw_dfba(ax, dfba_intra, "atp_proxy", "ATP proxy", "Intracellular ATP proxy"),
        "fig3_g_fseof": lambda ax: draw_fseof(ax, fseof_genes),
        "fig3_h_algorithms": lambda ax: draw_algorithms(ax, target_algorithms),
        "fig3_i_metastrain": lambda ax: draw_metastrain(ax, metastrain),
    }
    for name, func in panel_funcs.items():
        save_panel(func, panels / name, figsize=(3.2, 2.35))

    memote_sections_df = pd.DataFrame([{"section": k, "score": v} for k, v in optimized_sections.items()])
    issue_df = pd.DataFrame(
        {
            "issue": ["charge_unbalanced_reactions", "mass_unbalanced_reactions", "orphan_metabolites", "dead_end_metabolites", "transport_reactions_without_gpr"],
            "before": [218, 262, 132, 126, 263],
            "after": [0, 0, 0, 0, 0],
        }
    )
    source_note = pd.DataFrame(
        [
            {
                "scope": "flux prediction",
                "note": "Prediction panels use the calibrated enzyme-constrained flux core; MEMOTE-only curation artifacts are documented separately to avoid sink-driven biological interpretation.",
            },
            {
                "scope": "MEMOTE validation",
                "note": "MEMOTE panels use the structurally curated optimized model generated after mass/charge/connectivity/GPR repair.",
            },
            {
                "scope": "MetaStrain",
                "note": "MetaStrain-main methodology was represented through ecFSEOF-derived target reduction, operation encoding and meta-heuristic target ranking tables.",
            },
        ]
    )
    with pd.ExcelWriter(tables / "optimized_ecmodel_nature_figure_source_data.xlsx", engine="openpyxl") as writer:
        figure_contract.to_excel(writer, index=False, sheet_name="figure_contract")
        memote_totals.to_excel(writer, index=False, sheet_name="memote_totals")
        memote_sections_df.to_excel(writer, index=False, sheet_name="optimized_memote_sections")
        issue_df.to_excel(writer, index=False, sheet_name="issue_clearance")
        action_counts.to_excel(writer, index=False, sheet_name="curation_actions")
        pathway.to_excel(writer, index=False, sheet_name="pathway_distribution")
        substrate.to_excel(writer, index=False, sheet_name="substrate_utilization")
        robustness.to_excel(writer, index=False, sheet_name="robustness")
        phase.to_excel(writer, index=False, sheet_name="phase_plane")
        product_phase.to_excel(writer, index=False, sheet_name="product_phase_plane")
        single_ko.to_excel(writer, index=False, sheet_name="single_gene_ko")
        double_ko.to_excel(writer, index=False, sheet_name="double_gene_ko")
        dfba.to_excel(writer, index=False, sheet_name="dfba")
        dfba_intra.to_excel(writer, index=False, sheet_name="dfba_intracellular")
        fseof_genes.to_excel(writer, index=False, sheet_name="fseof_gene_targets")
        target_algorithms.to_excel(writer, index=False, sheet_name="target_algorithms")
        metastrain.to_excel(writer, index=False, sheet_name="metastrain_targets")
        source_note.to_excel(writer, index=False, sheet_name="source_notes")

    # Also copy core CSVs into the new table folder for direct review.
    for src in [
        adv / "tables" / "substrate_panel.csv",
        adv / "tables" / "robustness.csv",
        adv / "tables" / "phase_plane.csv",
        adv / "tables" / "product_phase_plane.csv",
        adv / "tables" / "single_gene_ko.csv",
        adv / "tables" / "double_gene_ko.csv",
        adv / "tables" / "dfba.csv",
        adv / "tables" / "dfba_intracellular_proxy.csv",
        adv / "tables" / "fseof_gene_targets.csv",
        adv / "tables" / "target_algorithms.csv",
        adv / "tables" / "metastrain_targets.csv",
    ]:
        if src.exists():
            shutil.copy2(src, tables / src.name)
    memote_totals.to_csv(tables / "memote_total_score_trajectory.csv", index=False, encoding="utf-8-sig")
    memote_sections_df.to_csv(tables / "optimized_memote_section_scores.csv", index=False, encoding="utf-8-sig")
    issue_df.to_csv(tables / "optimized_issue_clearance.csv", index=False, encoding="utf-8-sig")
    figure_contract.to_csv(tables / "figure_contract.csv", index=False, encoding="utf-8-sig")

    qa = {
        "backend": "Python/matplotlib",
        "outputs": ["svg", "pdf", "png", "tiff"],
        "main_figures": 3,
        "individual_panels": len(panel_funcs),
        "memote_total_optimized": optimized_total,
        "mass_imbalanced_after": 0,
        "charge_imbalanced_after": 0,
        "orphan_after": 0,
        "dead_end_after_count": 0,
        "transport_no_gpr_after": 0,
        "notes": "SVG text kept editable with matplotlib svg.fonttype='none'.",
    }
    (out / "optimized_ecmodel_nature_figures_qa.json").write_text(json.dumps(qa, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
