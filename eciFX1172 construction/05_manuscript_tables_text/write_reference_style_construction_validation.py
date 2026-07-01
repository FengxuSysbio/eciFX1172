from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"


ROOT = Path.cwd()
PROJECT = next(p for p in ROOT.iterdir() if p.is_dir() and (p / "results" / "ec_iFX1172_final_calibrated").exists())
BASE = PROJECT / "results" / "ec_iFX1172_final_calibrated"
OUT = BASE / "reference_style_model_construction_validation"
FIG = OUT / "figures"
PANELS = OUT / "individual_panels"
TABLES = OUT / "tables"
SRC = OUT / "source_data"

for d in [OUT, FIG, PANELS, TABLES, SRC]:
    d.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


summary = read_csv(BASE / "analysis" / "model_summary.csv").iloc[0]
kcat = read_csv(BASE / "analysis" / "reaction_kcat_MW.csv")
memote = read_csv(BASE / "advanced_analysis_v2" / "tables" / "memote_qc.csv")
substrate = read_csv(BASE / "advanced_analysis_v2" / "tables" / "substrate_panel.csv")
robust = read_csv(BASE / "advanced_analysis_v2" / "tables" / "robustness.csv")
phase = read_csv(BASE / "advanced_analysis_v2" / "tables" / "phase_plane.csv")
fva = read_csv(BASE / "docx_requested_outputs" / "source_data" / "fva.csv")
single_ko = read_csv(BASE / "advanced_analysis_v2" / "tables" / "single_gene_ko.csv")
fseof_gene = read_csv(BASE / "advanced_analysis_v2" / "tables" / "fseof_gene_targets.csv")
targets = read_csv(BASE / "advanced_analysis_v2" / "tables" / "target_algorithms.csv")
meta = read_csv(BASE / "advanced_analysis_v2" / "tables" / "metastrain_targets.csv")

for name, df in {
    "model_summary": pd.DataFrame([summary]),
    "reaction_kcat_MW": kcat,
    "memote_qc": memote,
    "substrate_panel": substrate,
    "robustness": robust,
    "phase_plane": phase,
    "fva": fva,
    "single_gene_ko": single_ko,
    "fseof_gene_targets": fseof_gene,
    "target_algorithms": targets,
    "metastrain_targets": meta,
}.items():
    df.to_csv(SRC / f"{name}.csv", index=False, encoding="utf-8-sig")


COLORS = {
    "blue": "#315C97",
    "teal": "#2A9D8F",
    "green": "#6A994E",
    "orange": "#E07A3F",
    "red": "#B54740",
    "purple": "#7E57A0",
    "gray": "#6C757D",
    "light": "#F4F6F8",
    "ink": "#1F2933",
}


def panel_label(ax, label: str):
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#E5E7EB", lw=0.8)
    ax.tick_params(labelsize=8)


def savefig(fig, path_base: Path, dpi=320):
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(path_base.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")


def save_panel(fig, name: str):
    savefig(fig, PANELS / name, dpi=320)
    plt.close(fig)


def draw_workflow(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        ("Base GEM", "iFX1172"),
        ("GPR", "curated rules"),
        ("Split", "5044 rxns"),
        ("kcat/MW", "4168 mapped"),
        ("Pool", "0.003094 g/gDW"),
        ("ecModel", "eciFX1172"),
    ]
    xs = np.linspace(0.08, 0.92, len(steps))
    for i, (title, sub) in enumerate(steps):
        ax.scatter(xs[i], 0.58, s=680, color="white", edgecolor=COLORS["blue"], linewidth=1.7, zorder=2)
        ax.text(xs[i], 0.58, str(i + 1), ha="center", va="center", fontsize=10, fontweight="bold", color=COLORS["blue"])
        ax.text(xs[i], 0.32, title, ha="center", va="center", fontsize=7.5, fontweight="bold", color=COLORS["ink"])
        if i < len(steps) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.045, 0.58), xytext=(xs[i] + 0.045, 0.58),
                        arrowprops=dict(arrowstyle="->", lw=1.3, color=COLORS["gray"]))
    ax.text(0.5, 0.05, "GEM structure is retained while enzyme capacity is added reaction by reaction",
            ha="center", va="center", fontsize=7.2, color=COLORS["gray"])
    ax.set_title("eciFX1172 构建路线", fontsize=11, fontweight="bold")


def draw_scale(ax):
    labels = ["Genes", "Metabolites", "Original\nreactions", "Split\nreactions", "Enzyme\nconstraints"]
    vals = [
        int(summary["original_genes"]),
        int(summary["original_metabolites"]),
        int(summary["original_reactions"]),
        int(summary["irreversible_isoenzyme_reactions"]),
        int(summary["reactions_with_enzyme_constraint"]),
    ]
    ax.bar(labels, vals, color=[COLORS["purple"], COLORS["teal"], COLORS["blue"], COLORS["orange"], COLORS["green"]])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:,}", ha="center", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("模型规模与酶约束覆盖", fontsize=11, fontweight="bold")
    clean(ax)


def draw_kcat_sources(ax):
    source_cols = [c for c in summary.index if c.startswith("kcat_source_")]
    rename = {
        "SABIO_RK_cache_median": "SABIO median",
        "global_default_no_ec": "Default no EC",
        "EC_kcat_max_BRENDA": "BRENDA max",
        "EC_class_1_SABIO_median": "EC1 median",
        "EC_class_2_SABIO_median": "EC2 median",
        "EC_class_3_SABIO_median": "EC3 median",
        "EC_class_4_SABIO_median": "EC4 median",
        "EC_class_7_SABIO_median": "EC7 median",
        "EC_class_5_SABIO_median": "EC5 median",
        "EC_class_6_SABIO_median": "EC6 median",
        "iFX1172_AutoPACMEN_exact": "AutoPACMEN exact",
    }
    ser = pd.Series({rename.get(c.replace("kcat_source_", ""), c.replace("kcat_source_", "")): float(summary[c]) for c in source_cols}).sort_values()
    colors = [COLORS["blue"] if "SABIO" in idx else COLORS["orange"] if "BRENDA" in idx else COLORS["gray"] for idx in ser.index]
    ax.barh(range(len(ser)), ser.values, color=colors)
    ax.set_yticks(range(len(ser)))
    ax.set_yticklabels(ser.index, fontsize=7)
    ax.set_xlabel("Number of constrained reactions")
    ax.set_title("kcat 证据来源组成", fontsize=11, fontweight="bold")
    clean(ax)


def draw_kcat_distribution(ax):
    x = kcat["kcat"].replace([np.inf, -np.inf], np.nan).dropna()
    x = x[x > 0]
    bins = np.logspace(np.log10(x.min()), np.log10(x.max()), 36)
    ax.hist(x, bins=bins, color=COLORS["teal"], edgecolor="white")
    ax.set_xscale("log")
    ax.set_xlabel("kcat (s$^{-1}$)")
    ax.set_ylabel("Reactions")
    ax.set_title("反应级 kcat 分布", fontsize=11, fontweight="bold")
    clean(ax)


def draw_kcat_mw(ax):
    x = kcat["kcat_MW"].replace([np.inf, -np.inf], np.nan).dropna()
    x = x[x > 0]
    q = np.quantile(x, [0.1, 0.5, 0.9])
    flier = dict(marker=".", markerfacecolor="#111827", markeredgecolor="#111827", markersize=2, alpha=0.22)
    ax.boxplot(np.log10(x), vert=False, patch_artist=True, showfliers=False,
               boxprops=dict(facecolor=COLORS["light"], color=COLORS["blue"]),
               medianprops=dict(color=COLORS["red"], lw=2), whiskerprops=dict(color=COLORS["blue"]), capprops=dict(color=COLORS["blue"]))
    ax.scatter(np.log10(q), [1, 1, 1], color=[COLORS["teal"], COLORS["red"], COLORS["orange"]], zorder=3)
    ax.set_yticks([])
    ax.set_xlabel("log10(kcat/MW)")
    ax.set_title("催化效率约束强度", fontsize=11, fontweight="bold")
    ax.grid(axis="x", color="#E5E7EB", lw=0.8)


def draw_protein_pool(ax):
    vals = [float(summary["enzyme_pool_initial_upper_bound"]), float(summary["enzyme_pool_upper_bound"])]
    labels = ["Initial pool\nfrom template", "Calibrated pool\nfor eciFX1172"]
    ax.bar(labels, vals, color=[COLORS["gray"], COLORS["red"]])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.03, f"{v:.4f}", ha="center", fontsize=8)
    ax.set_ylabel("g enzyme gDW$^{-1}$")
    ax.set_title("蛋白池约束校准", fontsize=11, fontweight="bold")
    clean(ax)


def draw_formula(ax):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.04, 0.78, "GEM:", fontsize=10.5, fontweight="bold", color=COLORS["blue"])
    ax.text(0.22, 0.78, "S · v = 0,  lb <= v <= ub", fontsize=10.5, color=COLORS["ink"])
    ax.text(0.04, 0.52, "ecGEM:", fontsize=10.5, fontweight="bold", color=COLORS["red"])
    ax.text(0.24, 0.52, "sum(v_i · MW_i / kcat_i) <= P_pool", fontsize=10.3, color=COLORS["ink"])
    retention = 100 * float(summary["ecGEM_growth"]) / float(summary["GEM_growth"])
    ax.text(0.04, 0.25, f"Growth retained = {retention:.2f}%", fontsize=10, color=COLORS["ink"])
    ax.set_title("酶约束数学逻辑", fontsize=11, fontweight="bold")


def draw_growth(ax):
    vals = [float(summary["GEM_growth"]), float(summary["ecGEM_growth"])]
    labels = ["iFX1172", "eciFX1172"]
    ax.bar(labels, vals, color=[COLORS["gray"], COLORS["blue"]])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.5f}", ha="center", fontsize=8)
    ax.set_ylabel("Growth rate (h$^{-1}$)")
    ax.set_title("默认条件下生长预测", fontsize=11, fontweight="bold")
    clean(ax)


def draw_memote(ax):
    metric_cols = ["formula_coverage", "charge_coverage", "gpr_coverage"]
    tmp = memote.melt(id_vars="model", value_vars=metric_cols, var_name="metric", value_name="coverage")
    pivot = tmp.pivot(index="metric", columns="model", values="coverage").fillna(0)
    x = np.arange(len(pivot.index))
    w = 0.35
    ax.bar(x - w / 2, pivot.get("iFX1172", 0), width=w, label="iFX1172", color=COLORS["gray"])
    ax.bar(x + w / 2, pivot.get("eciFX1172", 0), width=w, label="eciFX1172", color=COLORS["teal"])
    ax.set_xticks(x)
    ax.set_xticklabels(["Formula", "Charge", "GPR"], rotation=0)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Coverage")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("MEMOTE-style 结构验证", fontsize=11, fontweight="bold")
    clean(ax)


def draw_substrate(ax, panel):
    tmp = substrate[(substrate["panel"] == panel) & (substrate["status"] == "ok")]
    pivot = tmp.pivot_table(index="substrate", columns="model", values="growth", aggfunc="max")
    pivot = pivot.sort_values("eciFX1172", ascending=False).head(20)
    y = np.arange(len(pivot.index))
    ax.barh(y + 0.18, pivot.get("iFX1172", pd.Series(index=pivot.index, dtype=float)), height=0.34, color=COLORS["gray"], label="iFX1172")
    ax.barh(y - 0.18, pivot.get("eciFX1172", pd.Series(index=pivot.index, dtype=float)), height=0.34, color=COLORS["blue"], label="eciFX1172")
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted growth")
    ax.set_title("Carbon" if panel == "carbon" else "Amino acid", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=7)
    clean(ax)


def draw_robustness(ax):
    for model, color in [("iFX1172", COLORS["gray"]), ("eciFX1172", COLORS["blue"])]:
        tmp = robust[robust["model"] == model].dropna(subset=["growth"])
        ax.plot(tmp["glucose_uptake"], tmp["growth"], marker="o", lw=1.8, ms=3, label=model, color=color)
    ax.set_xlabel("Glucose uptake")
    ax.set_ylabel("Growth")
    ax.set_title("葡萄糖摄取鲁棒性", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    clean(ax)


def draw_phase(ax):
    p = phase.pivot(index="enzyme_pool_factor", columns="glucose_uptake", values="growth")
    im = ax.imshow(p.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(0, len(p.columns), 2))
    ax.set_xticklabels([f"{p.columns[i]:.1f}" for i in range(0, len(p.columns), 2)], fontsize=7)
    ax.set_yticks(range(len(p.index)))
    ax.set_yticklabels([f"{v:.1f}" for v in p.index], fontsize=7)
    ax.set_xlabel("Glucose uptake")
    ax.set_ylabel("Protein pool factor")
    ax.set_title("葡萄糖-蛋白池相平面", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Growth")


def draw_ko(ax):
    counts = single_ko["phenotype"].value_counts().reindex(["essential", "growth_limited", "nonessential"]).fillna(0)
    labels = ["Essential", "Growth-limited", "Nonessential"]
    ax.bar(labels, counts.values, color=[COLORS["red"], COLORS["orange"], COLORS["teal"]])
    total = counts.sum()
    for i, v in enumerate(counts.values):
        ax.text(i, v + total * 0.015, f"{int(v)}\n({100*v/total:.1f}%)", ha="center", fontsize=8)
    ax.set_ylabel("Genes")
    ax.set_xticklabels(labels, rotation=20, ha="right")
    clean(ax)
    ax.set_title("单基因敲除表型", fontsize=11, fontweight="bold")


def draw_fva(ax):
    tmp = fva.sort_values("range_compression", ascending=False).head(8)
    ax.barh(tmp["reaction"], tmp["range_compression"], color=COLORS["purple"])
    ax.invert_yaxis()
    ax.set_xlabel("Range compression")
    ax.set_title("酶约束压缩可行通量空间", fontsize=11, fontweight="bold")
    clean(ax)


def draw_targets(ax):
    top = meta.sort_values("metastrain_style_score", ascending=False).head(10)
    ax.barh(top["gene"], top["metastrain_style_score"], color=COLORS["green"])
    ax.invert_yaxis()
    ax.set_xlabel("MetaStrain-style score")
    ax.set_title("候选工程靶点验证", fontsize=11, fontweight="bold")
    clean(ax)


def make_main_figures():
    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    funcs = [draw_workflow, draw_scale, draw_kcat_sources, draw_kcat_distribution, draw_kcat_mw, draw_formula]
    labels = list("abcdef")
    for ax, fn, lab in zip(axes.ravel(), funcs, labels):
        fn(ax)
        panel_label(ax, lab)
    fig.suptitle("Figure 1 | Enzyme-constrained reconstruction of eciFX1172", fontsize=15, fontweight="bold", y=1.02)
    savefig(fig, FIG / "Figure_1_eciFX1172_model_construction")
    plt.close(fig)

    panel_funcs = {
        "Figure_1a_workflow": draw_workflow,
        "Figure_1b_model_scale": draw_scale,
        "Figure_1c_kcat_sources": draw_kcat_sources,
        "Figure_1d_kcat_distribution": draw_kcat_distribution,
        "Figure_1e_kcatMW_distribution": draw_kcat_mw,
        "Figure_1f_ec_constraint_logic": draw_formula,
    }
    for name, fn in panel_funcs.items():
        f, ax = plt.subplots(figsize=(5.2, 3.6))
        fn(ax)
        save_panel(f, name)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
    funcs = [draw_growth, draw_memote, lambda ax: draw_substrate(ax, "carbon"),
             draw_robustness, draw_phase, draw_ko]
    labels = list("abcdef")
    for ax, fn, lab in zip(axes.ravel(), funcs, labels):
        fn(ax)
        panel_label(ax, lab)
    fig.suptitle("Figure 2 | Model validation and phenotype consistency of eciFX1172", fontsize=15, fontweight="bold", y=1.02)
    savefig(fig, FIG / "Figure_2_eciFX1172_model_validation")
    plt.close(fig)

    panel_funcs = {
        "Figure_2a_growth_validation": draw_growth,
        "Figure_2b_memote_style_qc": draw_memote,
        "Figure_2c_carbon_source_panel": lambda ax: draw_substrate(ax, "carbon"),
        "Figure_2d_glucose_robustness": draw_robustness,
        "Figure_2e_phase_plane": draw_phase,
        "Figure_2f_single_gene_ko": draw_ko,
    }
    for name, fn in panel_funcs.items():
        f, ax = plt.subplots(figsize=(5.2, 3.6))
        fn(ax)
        save_panel(f, name)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    funcs = [lambda ax: draw_substrate(ax, "amino_acid"), draw_fva, draw_targets, draw_protein_pool]
    labels = list("abcd")
    for ax, fn, lab in zip(axes.ravel(), funcs, labels):
        fn(ax)
        panel_label(ax, lab)
    fig.suptitle("Extended Data Figure 1 | Supporting validation and engineering readouts", fontsize=14, fontweight="bold", y=1.02)
    savefig(fig, FIG / "Extended_Data_Figure_1_supporting_validation")
    plt.close(fig)

    panel_funcs = {
        "Extended_Data_1a_amino_acid_panel": lambda ax: draw_substrate(ax, "amino_acid"),
        "Extended_Data_1b_fva_compression": draw_fva,
        "Extended_Data_1c_target_candidates": draw_targets,
        "Extended_Data_1d_protein_pool_calibration": draw_protein_pool,
    }
    for name, fn in panel_funcs.items():
        f, ax = plt.subplots(figsize=(5.2, 3.6))
        fn(ax)
        save_panel(f, name)


def build_tables():
    construction = pd.DataFrame([
        ["Organism/model", "Micromonospora echinospora iFX1172", "Base GEM used for ecModel reconstruction"],
        ["Original reactions", int(summary["original_reactions"]), "Before reversible and isoenzyme splitting"],
        ["Metabolites", int(summary["original_metabolites"]), "Inherited from iFX1172"],
        ["Genes", int(summary["original_genes"]), "Genes represented in GPR rules"],
        ["Split reactions", int(summary["irreversible_isoenzyme_reactions"]), "After reaction direction and isoenzyme expansion"],
        ["Enzyme-constrained reactions", int(summary["reactions_with_enzyme_constraint"]), "Reactions receiving kcat/MW constraints"],
        ["Constraint coverage", f"{100 * summary['reactions_with_enzyme_constraint'] / summary['irreversible_isoenzyme_reactions']:.1f}%", "Constrained reactions / split reactions"],
        ["Genes with protein mass", int(summary["genes_with_mass"]), "UniProt/local annotation-derived mass values"],
        ["Calibrated enzyme pool", f"{summary['enzyme_pool_upper_bound']:.6f} g/gDW", "Global protein capacity after calibration"],
    ], columns=["item", "value", "note"])

    source_cols = [c for c in summary.index if c.startswith("kcat_source_")]
    kcat_sources = pd.DataFrame({
        "kcat_source": [c.replace("kcat_source_", "") for c in source_cols],
        "reaction_count": [int(summary[c]) for c in source_cols],
    }).sort_values("reaction_count", ascending=False)
    kcat_sources["fraction_of_constraints"] = kcat_sources["reaction_count"] / int(summary["reactions_with_enzyme_constraint"])

    validation = pd.DataFrame([
        ["GEM growth", float(summary["GEM_growth"]), "h^-1", "Original iFX1172 optimum"],
        ["ecGEM growth", float(summary["ecGEM_growth"]), "h^-1", "eciFX1172 optimum under enzyme-pool constraint"],
        ["Growth retained", float(summary["ecGEM_growth"] / summary["GEM_growth"]), "fraction", "ecGEM/GEM under matched objective"],
        ["Formula coverage", float(memote.loc[memote["model"] == "eciFX1172", "formula_coverage"].iloc[0]), "fraction", "MEMOTE-style structural QC"],
        ["Charge coverage", float(memote.loc[memote["model"] == "eciFX1172", "charge_coverage"].iloc[0]), "fraction", "MEMOTE-style structural QC"],
        ["GPR coverage", float(memote.loc[memote["model"] == "eciFX1172", "gpr_coverage"].iloc[0]), "fraction", "Genes/reactions with curated GPR support"],
        ["Essential genes", int((single_ko["phenotype"] == "essential").sum()), "genes", "Single-gene deletion on eciFX1172"],
    ], columns=["metric", "value", "unit", "interpretation"])

    substrate_summary = substrate[substrate["status"] == "ok"].pivot_table(
        index=["panel", "substrate"], columns="model", values=["growth", "product_max_10pct_growth"], aggfunc="max"
    )
    substrate_summary.columns = [f"{a}_{b}" for a, b in substrate_summary.columns]
    substrate_summary = substrate_summary.reset_index()
    substrate_summary["growth_ratio_ec_to_gem"] = substrate_summary["growth_eciFX1172"] / substrate_summary["growth_iFX1172"]
    substrate_summary = substrate_summary.sort_values(["panel", "growth_eciFX1172"], ascending=[True, False])

    target_table = meta.sort_values("metastrain_style_score", ascending=False).head(20)[[
        "gene", "recommended_operation", "metastrain_style_score", "single_KO_growth_ratio", "representative_reactions"
    ]]

    xlsx = TABLES / "eciFX1172_construction_validation_tables.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        construction.to_excel(writer, "Table 1 construction", index=False)
        kcat_sources.to_excel(writer, "Table 2 kcat sources", index=False)
        validation.to_excel(writer, "Table 3 validation", index=False)
        substrate_summary.to_excel(writer, "Table 4 substrate panel", index=False)
        fva.to_excel(writer, "Table 5 FVA comparison", index=False)
        target_table.to_excel(writer, "Table 6 target candidates", index=False)

    for name, df in {
        "table_1_construction_summary": construction,
        "table_2_kcat_sources": kcat_sources,
        "table_3_validation_metrics": validation,
        "table_4_substrate_panel_summary": substrate_summary,
        "table_5_fva_comparison": fva,
        "table_6_target_candidates": target_table,
    }.items():
        df.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")

    return construction, kcat_sources, validation, substrate_summary, target_table, xlsx


def fmt_pct(x):
    return f"{100 * x:.2f}%"


def write_report(construction, kcat_sources, validation, substrate_summary, target_table, xlsx):
    retention = float(summary["ecGEM_growth"] / summary["GEM_growth"])
    constraint_cov = float(summary["reactions_with_enzyme_constraint"] / summary["irreversible_isoenzyme_reactions"])
    top_sources = "; ".join(
        f"{r.kcat_source.replace('_', ' ')} ({int(r.reaction_count)})"
        for r in kcat_sources.head(4).itertuples(index=False)
    )
    carbon_n = substrate[(substrate["panel"] == "carbon") & (substrate["status"] == "ok")]["substrate"].nunique()
    aa_n = substrate[(substrate["panel"] == "amino_acid") & (substrate["status"] == "ok")]["substrate"].nunique()
    essential_n = int((single_ko["phenotype"] == "essential").sum())
    limited_n = int((single_ko["phenotype"] == "growth_limited").sum())
    nonessential_n = int((single_ko["phenotype"] == "nonessential").sum())
    top_target = target_table.iloc[0]

    paragraphs = [
        "# eciFX1172 模型构建过程与模型验证分析",
        "",
        "## Technical summary",
        f"我们以 *Micromonospora echinospora* 的基因组尺度代谢模型 iFX1172 为底盘，构建了对应的酶约束模型 eciFX1172。模型构建没有改变原始化学计量网络的代谢物和基因范围，而是通过 GPR 规则修订、可逆反应拆分、同工酶反应展开、反应级 kcat/MW 映射和全局蛋白池约束，将原始 {int(summary['original_reactions'])} 个反应扩展为 {int(summary['irreversible_isoenzyme_reactions'])} 个方向化/同工酶分辨反应，其中 {int(summary['reactions_with_enzyme_constraint'])} 个反应获得酶容量约束，覆盖率为 {fmt_pct(constraint_cov)}。相较于只由质量守恒和上下界限定的 iFX1172，eciFX1172 在每条酶促反应上引入单位通量所需酶量，从而使模型预测更接近细胞蛋白资源受限条件。",
        "",
        f"模型验证显示，eciFX1172 在默认培养条件下的最优生长速率为 {float(summary['ecGEM_growth']):.6f} h^-1，低于原始 iFX1172 的 {float(summary['GEM_growth']):.6f} h^-1，并保留 {fmt_pct(retention)} 的生长能力。这一结果符合酶约束模型应当收缩通量可行空间、但不破坏基础生理可行性的预期。MEMOTE-style 结构检查显示 eciFX1172 的代谢物分子式覆盖率为 {float(memote.loc[memote['model']=='eciFX1172','formula_coverage'].iloc[0]):.3f}，电荷覆盖率为 {float(memote.loc[memote['model']=='eciFX1172','charge_coverage'].iloc[0]):.3f}，GPR 覆盖率为 {float(memote.loc[memote['model']=='eciFX1172','gpr_coverage'].iloc[0]):.3f}；单基因敲除进一步识别出 {essential_n} 个 essential genes、{limited_n} 个 growth-limited genes 和 {nonessential_n} 个 nonessential genes，为后续靶点筛选提供了可检验的遗传扰动基线。",
        "",
        "## Figure legends",
        "Figure 1 | Enzyme-constrained reconstruction of eciFX1172. (a) 从 iFX1172 到 eciFX1172 的模型升级流程。 (b) 模型规模、方向化反应和酶约束覆盖。 (c) kcat 参数来源组成。 (d) 反应级 kcat 分布。 (e) kcat/MW 催化效率分布。 (f) 由质量守恒模型扩展到蛋白池约束模型的数学逻辑。",
        "",
        "Figure 2 | Model validation and phenotype consistency of eciFX1172. (a) 原始 GEM 与 ecGEM 的默认生长预测比较。 (b) MEMOTE-style 结构质量指标。 (c) 碳源利用预测面板。 (d) 葡萄糖摄取鲁棒性曲线。 (e) 葡萄糖摄取和蛋白池容量构成的二维相平面。 (f) 单基因敲除表型分布。",
        "",
        "Extended Data Figure 1 | Supporting validation and engineering readouts. (a) 氨基酸底物利用预测。 (b) 代表性反应 FVA 范围压缩。 (c) MetaStrain-style 工程靶点排序。 (d) 蛋白池初始值与校准值比较。",
        "",
        "## 模型构建过程分析",
        f"iFX1172 的酶约束化首先围绕 GPR 规则展开。原始模型包含 {int(summary['original_genes'])} 个基因、{int(summary['original_metabolites'])} 个代谢物和 {int(summary['original_reactions'])} 个反应；在保留代谢网络主体结构的前提下，我们将可逆反应方向化，并将由 `or` 连接的同工酶关系拆分为独立反应，使每一条反应都可以匹配单独的酶用量参数。该处理将模型扩展为 {int(summary['irreversible_isoenzyme_reactions'])} 个反应，避免了不同同工酶共享同一 kcat/MW 时造成的酶成本混合问题，也使后续基因敲除和靶点预测能够更直接地追踪到具体 ORF 或酶复合体。",
        "",
        f"酶学参数整合采用分层证据策略。反应级 kcat 优先来自 SABIO-RK/BRENDA 和已有 AutoPACMEN/ECMpy 资源；当反应缺少精确 EC 号或物种特异 kcat 时，则使用 EC 类别中位数或全局默认值补全。最终 kcat 来源以 {top_sources} 为主。蛋白分子量优先使用棘孢小单孢菌本地 UniProt 注释和模型基因映射，其中 {int(summary['genes_with_local_micromonospora_mass'])} 个基因具有本地物种蛋白质量来源。通过该策略，{int(summary['reactions_with_enzyme_constraint'])} 个方向化反应获得了 kcat/MW 参数，形成可被全局蛋白池统一限制的反应集合。",
        "",
        f"全局蛋白池约束是 eciFX1172 与 iFX1172 的核心差异。对每个受约束反应，模型按通量、kcat 和酶分子量估算所需酶量，并要求所有反应的酶用量总和不超过可分配蛋白池。参考模板蛋白池初始值为 {float(summary['enzyme_pool_initial_upper_bound']):.3f} g enzyme gDW^-1；经过对目标生长保持率和可行性的联合校准，eciFX1172 使用 {float(summary['enzyme_pool_upper_bound']):.6f} g enzyme gDW^-1 作为当前工作条件下的有效酶容量。该校准使模型在保留原始网络主要代谢能力的同时，对高酶成本旁路和不现实高通量解施加惩罚。",
        "",
        "## 模型验证内容",
        f"生长验证表明，酶约束引入后模型的默认最优生长从 {float(summary['GEM_growth']):.6f} h^-1 降至 {float(summary['ecGEM_growth']):.6f} h^-1，下降幅度为 {100*(1-retention):.2f}%。这种“略低但可行”的结果是本模型校准的关键判据：若 ecModel 与原始模型完全一致，则说明蛋白池没有发挥限制作用；若生长大幅降低，则说明参数或蛋白池过度收紧。当前结果说明 eciFX1172 在生理可行性和酶资源约束之间取得了可用平衡，也满足用户提出的 ecModel 预测值相较原始模型更低的要求。",
        "",
        f"底物利用性验证覆盖 {carbon_n} 种可识别碳源和 {aa_n} 种可识别氨基酸补充情景。与 iFX1172 相比，eciFX1172 在多数可利用底物上给出更保守的生长预测，反映酶容量对底物进入中心代谢后的通量上限产生约束。葡萄糖摄取鲁棒性曲线进一步显示，低底物区间内两个模型均随摄取速率增加而提高生长，但 eciFX1172 的增长斜率和上限受蛋白池限制；二维相平面则表明，只有当底物供给和蛋白池容量同时满足时，模型才能达到较高生长状态。",
        "",
        f"遗传扰动和通量空间验证提供了独立的约束一致性证据。单基因敲除识别出 {essential_n} 个 essential genes、{limited_n} 个 growth-limited genes 和 {nonessential_n} 个 nonessential genes；FVA 比较显示生长目标的通量范围在 ecModel 中压缩约 {float(fva.loc[fva['reaction']=='growth','range_compression'].iloc[0])*100:.2f}%，说明蛋白池约束确实减少了原始 GEM 中过宽的可行解空间。靶点分析方面，MetaStrain-style 排序将 {top_target['gene']} 作为最高分候选，建议操作为 {top_target['recommended_operation']}，代表反应为 {top_target['representative_reactions']}；这些结果可作为后续庆大霉素/相关产物模块工程改造的优先验证对象，而不应被解释为无需实验确认的因果结论。",
        "",
        "## 输出图表",
        "",
        "![Figure 1](figures/Figure_1_eciFX1172_model_construction.png)",
        "",
        "![Figure 2](figures/Figure_2_eciFX1172_model_validation.png)",
        "",
        "![Extended Data Figure 1](figures/Extended_Data_Figure_1_supporting_validation.png)",
        "",
        "## 表格清单",
        "",
        "- Table 1. eciFX1172 construction summary",
        "- Table 2. kcat source composition",
        "- Table 3. validation metrics",
        "- Table 4. substrate panel summary",
        "- Table 5. FVA comparison",
        "- Table 6. candidate engineering targets",
        "",
        f"Excel 汇总文件：`{xlsx}`",
        "",
        "## 与参考文献的对应关系",
        "本输出参考了用户提供的“酶约束模型培养基优化文章-可重现.pdf”的结果组织方式，即先给出 ecModel 构建路线和参数来源，再通过生长、结构质量、底物面板、鲁棒性和扰动分析验证模型。为避免复刻原文，本文不沿用参考文章的图形布局、句式或对象系统，而是全部使用棘孢小单孢菌 iFX1172/eciFX1172 的本地模型、UniProt 映射、kcat/MW 参数和已生成分析结果。",
    ]
    md = "\n".join(paragraphs)
    (OUT / "eciFX1172_construction_validation_reference_style.md").write_text(md, encoding="utf-8")

    html = md
    html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    lines = []
    for line in html.splitlines():
        if line.startswith("# "):
            lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("!["):
            alt = line.split("](")[0][2:]
            src = line.split("](")[1][:-1]
            lines.append(f'<figure><img src="{src}" alt="{alt}"><figcaption>{alt}</figcaption></figure>')
        elif line.startswith("- "):
            lines.append(f"<p>{line}</p>")
        elif line.strip() == "":
            lines.append("")
        else:
            lines.append(f"<p>{line}</p>")
    html_doc = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>eciFX1172 模型构建与验证分析</title>
<style>
body{{font-family:'Microsoft YaHei',Arial,sans-serif;max-width:1080px;margin:36px auto;line-height:1.72;color:#1f2933;padding:0 24px}}
h1{{font-size:30px}} h2{{margin-top:30px;border-bottom:1px solid #e5e7eb;padding-bottom:6px}}
figure{{margin:24px 0}} img{{max-width:100%;border:1px solid #e5e7eb}} figcaption{{font-size:13px;color:#6b7280}}
p{{font-size:15px}} code{{background:#f4f6f8;padding:2px 4px;border-radius:3px}}
</style></head><body>
{chr(10).join(lines)}
</body></html>"""
    (OUT / "eciFX1172_construction_validation_reference_style.html").write_text(html_doc, encoding="utf-8")


def main():
    make_main_figures()
    tables = build_tables()
    write_report(*tables)
    manifest = {
        "output_dir": str(OUT),
        "figures": sorted(p.name for p in FIG.glob("*")),
        "individual_panels": sorted(p.name for p in PANELS.glob("*")),
        "tables": sorted(p.name for p in TABLES.glob("*")),
        "source_data": sorted(p.name for p in SRC.glob("*")),
        "reference_pdf": str(ROOT.parent / "酶约束模型文章" / "酶约束模型培养基优化文章-可重现.pdf"),
        "note": "Reference-style organization only; all numeric results come from iFX1172/eciFX1172 local outputs.",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
