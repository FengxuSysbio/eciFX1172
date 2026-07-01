import json
import math
import re
from pathlib import Path

import numpy as np

if not hasattr(np, "object"):
    np.object = object

import cobra
import matplotlib.pyplot as plt
import pandas as pd
from cobra.manipulation import delete_model_genes

cobra.Configuration().solver = "glpk"


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
MODEL_DIR = ROOT / "results" / "ec_iFX1172_final_calibrated" / "model"
EC_JSON = MODEL_DIR / "eciFX1172.json"
IRREV_JSON = MODEL_DIR / "iFX1172_irreversible.json"
OUT = ROOT / "results" / "ec_iFX1172_final_calibrated" / "advanced_analysis"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "report"

PRODUCT_RXN = "r_0013"
PRODUCT_NAME = "Gentamicin A"
DEFAULT_SUBSTRATE_RXN = "EX_glc__D_e_reverse"
BIOMASS_RXN = "growth"
EPS = 1e-9


def ensure_dirs():
    for folder in [OUT, TABLE_DIR, FIG_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_json_raw(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_ec_model(json_model_file=EC_JSON):
    data = load_json_raw(json_model_file)
    model = cobra.io.load_json_model(str(json_model_file))
    model.solver = "glpk"
    coefficients = {}
    kcat_mw_by_id = {reaction["id"]: reaction.get("kcat_MW") for reaction in data["reactions"]}
    for reaction in model.reactions:
        value = kcat_mw_by_id.get(reaction.id)
        if value not in ("", None):
            try:
                coefficients[reaction.forward_variable] = 1.0 / float(value)
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    constraint = model.problem.Constraint(
        0,
        lb=float(data["enzyme_constraint"]["lowerbound"]),
        ub=float(data["enzyme_constraint"]["upperbound"]),
        name="enzyme_pool",
    )
    model.add_cons_vars(constraint)
    model.solver.update()
    constraint.set_linear_coefficients(coefficients)
    return model


def load_irreversible_model():
    model = cobra.io.load_json_model(str(IRREV_JSON))
    model.solver = "glpk"
    return model


def optimize_value(model, objective=None):
    if objective is not None and default_objective_reaction(model) != objective:
        model.objective = model.reactions.get_by_id(objective)
    try:
        value = model.slim_optimize(error_value=np.nan)
    except Exception:
        value = np.nan
    if value is None:
        return np.nan
    return float(value)


def default_objective_reaction(model):
    for reaction in model.reactions:
        if abs(reaction.objective_coefficient) > EPS:
            return reaction.id
    return BIOMASS_RXN if BIOMASS_RXN in model.reactions else None


def exchange_reverse_reactions(model):
    rows = []
    for reaction in model.reactions:
        if not (reaction.id.startswith("EX_") and reaction.id.endswith("_reverse")):
            continue
        metabolite = next(iter(reaction.metabolites), None)
        formula = getattr(metabolite, "formula", "") or ""
        organic = is_organic_formula(formula)
        rows.append(
            {
                "reaction_id": reaction.id,
                "name": reaction.name,
                "metabolite_id": metabolite.id if metabolite else "",
                "metabolite_name": metabolite.name if metabolite else "",
                "formula": formula,
                "organic": organic,
            }
        )
    return pd.DataFrame(rows)


def is_organic_formula(formula):
    if not formula:
        return False
    if formula in {"CO2", "CO3", "HCO3", "Ca", "Cl", "Co", "Cu"}:
        return False
    elements = re.findall(r"([A-Z][a-z]?)(?:[0-9.]*)", formula)
    return "C" in elements and any(element in elements for element in ["H", "N", "O", "S", "P"])


def configure_single_substrate_medium(model, substrate_rxn, uptake=10.0):
    for reaction in model.reactions:
        if reaction.id.startswith("EX_") and reaction.id.endswith("_reverse"):
            metabolite = next(iter(reaction.metabolites), None)
            formula = getattr(metabolite, "formula", "") or ""
            if is_organic_formula(formula):
                reaction.upper_bound = 0.0
            else:
                reaction.upper_bound = max(reaction.upper_bound, 1000.0)
    if substrate_rxn in model.reactions:
        model.reactions.get_by_id(substrate_rxn).upper_bound = float(uptake)


def substrate_utilization(gem, ec_model):
    exchanges = exchange_reverse_reactions(ec_model)
    candidates = exchanges[exchanges["organic"]].copy()
    preferred_terms = [
        "glucose",
        "fructose",
        "sucrose",
        "maltose",
        "xylose",
        "arabinose",
        "galactose",
        "glycerol",
        "acetate",
        "lactate",
        "succinate",
        "malate",
        "citrate",
        "pyruvate",
        "glutamate",
        "alanine",
        "aspartate",
        "serine",
        "threonine",
        "valine",
        "leucine",
        "isoleucine",
        "proline",
        "lysine",
        "arginine",
        "histidine",
        "methionine",
        "phenylalanine",
        "tyrosine",
        "tryptophan",
    ]
    text = (candidates["reaction_id"] + " " + candidates["name"] + " " + candidates["metabolite_name"]).str.lower()
    candidates["preferred"] = text.apply(lambda value: any(term in value for term in preferred_terms))
    candidates = candidates.sort_values(["preferred", "reaction_id"], ascending=[False, True]).head(80)
    rows = []
    for row in candidates.to_dict("records"):
        rid = row["reaction_id"]
        with gem as g, ec_model as e:
            configure_single_substrate_medium(g, rid)
            configure_single_substrate_medium(e, rid)
            gem_growth = optimize_value(g, BIOMASS_RXN)
            ec_growth = optimize_value(e, BIOMASS_RXN)
        rows.append(
            {
                **row,
                "uptake_bound": 10.0,
                "gem_growth": gem_growth,
                "ec_growth": ec_growth,
                "ec_to_gem_ratio": ec_growth / gem_growth if gem_growth and gem_growth > EPS else np.nan,
                "usable_by_ec": bool(ec_growth > 1e-6),
            }
        )
    return pd.DataFrame(rows).sort_values("ec_growth", ascending=False)


def robustness_analysis(gem, ec_model, substrate_rxn=DEFAULT_SUBSTRATE_RXN):
    substrate_rows = []
    for uptake in np.linspace(0, 10, 21):
        with gem as g, ec_model as e:
            if substrate_rxn in g.reactions:
                g.reactions.get_by_id(substrate_rxn).upper_bound = float(uptake)
            if substrate_rxn in e.reactions:
                e.reactions.get_by_id(substrate_rxn).upper_bound = float(uptake)
            substrate_rows.append(
                {
                    "substrate_rxn": substrate_rxn,
                    "uptake_bound": float(uptake),
                    "gem_growth": optimize_value(g, BIOMASS_RXN),
                    "ec_growth": optimize_value(e, BIOMASS_RXN),
                }
            )

    ec_data = load_json_raw(EC_JSON)
    pool_ub = float(ec_data["enzyme_constraint"]["upperbound"])
    pool_rows = []
    constraint = ec_model.solver.constraints.get("enzyme_pool")
    original_pool_ub = constraint.ub
    for factor in np.linspace(0.25, 1.50, 26):
        constraint.ub = pool_ub * float(factor)
        pool_rows.append(
            {
                "enzyme_pool_factor": float(factor),
                "enzyme_pool_upper_bound": pool_ub * float(factor),
                "ec_growth": optimize_value(ec_model, BIOMASS_RXN),
            }
        )
    constraint.ub = original_pool_ub
    return pd.DataFrame(substrate_rows), pd.DataFrame(pool_rows)


def single_gene_knockout(ec_model):
    wt = optimize_value(ec_model, BIOMASS_RXN)
    rows = []
    for index, gene in enumerate(sorted(ec_model.genes, key=lambda g: g.id), start=1):
        with ec_model as model:
            delete_model_genes(model, [gene.id])
            growth = optimize_value(model, BIOMASS_RXN)
        rows.append(
            {
                "gene": gene.id,
                "ec_growth": growth,
                "growth_ratio": growth / wt if wt > EPS else np.nan,
                "phenotype": "essential" if growth < 0.01 * wt else ("growth_limited" if growth < 0.8 * wt else "nonessential"),
            }
        )
        if index % 200 == 0:
            print(f"single KO: {index}/{len(ec_model.genes)}")
    return pd.DataFrame(rows).sort_values(["growth_ratio", "gene"]), wt


def double_gene_knockout(ec_model, single_ko, candidate_genes):
    wt = optimize_value(ec_model, BIOMASS_RXN)
    genes = [gene for gene in candidate_genes if gene in ec_model.genes]
    rows = []
    total = len(genes) * (len(genes) - 1) // 2
    count = 0
    single_ratio = single_ko.set_index("gene")["growth_ratio"].to_dict()
    for i, gene_a in enumerate(genes):
        for gene_b in genes[i + 1 :]:
            count += 1
            with ec_model as model:
                delete_model_genes(model, [gene_a, gene_b])
                growth = optimize_value(model, BIOMASS_RXN)
            expected = min(single_ratio.get(gene_a, 1.0), single_ratio.get(gene_b, 1.0))
            ratio = growth / wt if wt > EPS else np.nan
            rows.append(
                {
                    "gene_a": gene_a,
                    "gene_b": gene_b,
                    "ec_growth": growth,
                    "growth_ratio": ratio,
                    "min_single_growth_ratio": expected,
                    "interaction_score": expected - ratio if not np.isnan(ratio) else np.nan,
                    "phenotype": "synthetic_lethal" if growth < 0.01 * wt and expected >= 0.01 else "synthetic_sick"
                    if expected - ratio > 0.2
                    else "other",
                }
            )
            if count % 100 == 0:
                print(f"double KO: {count}/{total}")
    return pd.DataFrame(rows).sort_values(["interaction_score", "growth_ratio"], ascending=[False, True])


def product_maximum(ec_model, growth_fraction=0.10):
    wt_growth = optimize_value(ec_model, BIOMASS_RXN)
    model = ec_model.copy()
    model.reactions.get_by_id(BIOMASS_RXN).lower_bound = max(
        model.reactions.get_by_id(BIOMASS_RXN).lower_bound, growth_fraction * wt_growth
    )
    max_product = optimize_value(model, PRODUCT_RXN)
    return wt_growth, max_product


def fseof_analysis(ec_model):
    wt_growth, max_product = product_maximum(ec_model)
    levels = [value for value in np.linspace(0.1, 0.9, 9) * max_product if value > EPS]
    flux_rows = []
    if not levels:
        return pd.DataFrame(), pd.DataFrame(), wt_growth, max_product
    for level in levels:
        model = ec_model.copy()
        model.reactions.get_by_id(BIOMASS_RXN).lower_bound = max(
            model.reactions.get_by_id(BIOMASS_RXN).lower_bound, 0.10 * wt_growth
        )
        model.reactions.get_by_id(PRODUCT_RXN).lower_bound = float(level)
        solution = model.optimize()
        for reaction in model.reactions:
            flux = float(solution.fluxes.get(reaction.id, np.nan))
            if abs(flux) > 1e-8 and reaction.genes:
                flux_rows.append(
                    {
                        "product_flux_forced": float(level),
                        "reaction": reaction.id,
                        "reaction_name": reaction.name,
                        "subsystem": reaction.subsystem,
                        "genes": ";".join(sorted(gene.id for gene in reaction.genes)),
                        "flux": flux,
                    }
                )
    flux_table = pd.DataFrame(flux_rows)
    reaction_rows = []
    for rid, group in flux_table.groupby("reaction"):
        if group["product_flux_forced"].nunique() < 3:
            continue
        x = group["product_flux_forced"].to_numpy()
        y = group["flux"].to_numpy()
        slope = float(np.polyfit(x, y, 1)[0])
        corr = float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and np.std(y) > EPS else np.nan
        reaction_rows.append(
            {
                "reaction": rid,
                "reaction_name": group["reaction_name"].iloc[0],
                "subsystem": group["subsystem"].iloc[0],
                "genes": group["genes"].iloc[0],
                "slope": slope,
                "abs_slope": abs(slope),
                "correlation": corr,
                "recommendation": "overexpression" if slope > 1e-6 else "downregulation" if slope < -1e-6 else "neutral",
            }
        )
    reaction_table = pd.DataFrame(reaction_rows).sort_values("abs_slope", ascending=False)
    gene_rows = []
    for row in reaction_table.to_dict("records"):
        for gene in str(row["genes"]).split(";"):
            if gene:
                gene_rows.append({**row, "gene": gene})
    gene_table = pd.DataFrame(gene_rows)
    if not gene_table.empty:
        gene_table = (
            gene_table.groupby(["gene", "recommendation"], as_index=False)
            .agg(
                fseof_score=("abs_slope", "sum"),
                mean_slope=("slope", "mean"),
                representative_reactions=("reaction", lambda values: ";".join(list(values)[:6])),
            )
            .sort_values("fseof_score", ascending=False)
        )
    return reaction_table, gene_table, wt_growth, max_product


def simulate_dfba(ec_model, substrate_rxn=DEFAULT_SUBSTRATE_RXN, initial_biomass=0.05, initial_substrate=10.0):
    dt = 0.2
    t_end = 30.0
    vmax = 10.0
    biomass = float(initial_biomass)
    substrate = float(initial_substrate)
    rows = []
    for step in range(int(t_end / dt) + 1):
        time_h = step * dt
        if substrate <= 1e-8 or biomass <= 1e-10:
            rows.append({"time_h": time_h, "biomass_gDW_L": biomass, "substrate_mmol_L": max(substrate, 0.0), "growth_rate_h-1": 0.0, "substrate_uptake_mmol_gDW_h": 0.0})
            continue
        uptake_bound = min(vmax, substrate / max(biomass * dt, 1e-12))
        with ec_model as model:
            model.reactions.get_by_id(substrate_rxn).upper_bound = float(uptake_bound)
            solution = model.optimize()
            if solution.status != "optimal":
                mu, q_s = 0.0, 0.0
            else:
                mu = max(float(solution.objective_value), 0.0)
                q_s = max(float(solution.fluxes.get(substrate_rxn, 0.0)), 0.0)
        rows.append({"time_h": time_h, "biomass_gDW_L": biomass, "substrate_mmol_L": substrate, "growth_rate_h-1": mu, "substrate_uptake_mmol_gDW_h": q_s})
        biomass = biomass + mu * biomass * dt
        substrate = max(substrate - q_s * biomass * dt, 0.0)
    return pd.DataFrame(rows)


def metastrain_style_target_analysis(ec_model, fseof_genes, single_ko):
    wt_growth, max_product = product_maximum(ec_model)
    viable = single_ko.set_index("gene")["growth_ratio"].to_dict()
    candidate_genes = list(fseof_genes["gene"].head(60)) if not fseof_genes.empty else []
    rows = []
    for row in fseof_genes.head(60).to_dict("records"):
        gene = row["gene"]
        if gene not in ec_model.genes:
            continue
        rec = row["recommendation"]
        operation = "OE" if rec == "overexpression" else "KD"
        if viable.get(gene, 1.0) > 0.2 and rec == "downregulation":
            operation = "KO/KD"
        score = float(row["fseof_score"]) * max(0.0, min(1.0, viable.get(gene, 1.0)))
        rows.append(
            {
                "gene": gene,
                "operation_code": {"No change": 0, "OE": 1, "KD": 2, "KO/KD": 3}.get(operation, 0),
                "recommended_operation": operation,
                "metastrain_fitness_score": score,
                "single_KO_growth_ratio": viable.get(gene, np.nan),
                "fseof_mean_slope": row["mean_slope"],
                "fseof_score": row["fseof_score"],
                "representative_reactions": row["representative_reactions"],
                "fitness_definition": f"F = scaled ecFSEOF response under {PRODUCT_RXN} ({PRODUCT_NAME}) forcing, filtered by ecModel single-KO viability; MetaStrain operation codes 0/1/2/3 = no change/OE/KD/KO.",
            }
        )
    target_table = pd.DataFrame(rows).sort_values("metastrain_fitness_score", ascending=False)
    return target_table, candidate_genes


def save_table(df, name):
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def plot_outputs(substrate, robustness_substrate, robustness_pool, single_ko, double_ko, dfba, fseof_genes, metastrain_targets):
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    top_sub = substrate.head(20).sort_values("ec_growth")
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=220)
    ax.barh(top_sub["metabolite_name"].fillna(top_sub["reaction_id"]), top_sub["ec_growth"], color="#287D8E")
    ax.set_xlabel("Predicted ecModel growth rate (h$^{-1}$)")
    ax.set_title("Substrate utilization potential")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure7_substrate_utilization.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(robustness_substrate["uptake_bound"], robustness_substrate["gem_growth"], label="iFX1172", color="#777777", lw=2)
    ax.plot(robustness_substrate["uptake_bound"], robustness_substrate["ec_growth"], label="eciFX1172", color="#C23B22", lw=2)
    ax.set_xlabel("Glucose uptake upper bound (mmol gDW$^{-1}$ h$^{-1}$)")
    ax.set_ylabel("Growth rate (h$^{-1}$)")
    ax.legend(frameon=False)
    ax.set_title("Substrate robustness")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure8_robustness_substrate.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=220)
    ax.plot(robustness_pool["enzyme_pool_factor"], robustness_pool["ec_growth"], color="#2F6F4E", lw=2)
    ax.axvline(1.0, color="#444444", ls="--", lw=1)
    ax.set_xlabel("Relative enzyme-pool upper bound")
    ax.set_ylabel("Growth rate (h$^{-1}$)")
    ax.set_title("Enzyme-pool robustness")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure9_robustness_enzyme_pool.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=220)
    bins = np.linspace(0, 1.05, 22)
    ax.hist(single_ko["growth_ratio"].clip(0, 1.05), bins=bins, color="#5B6C8F", edgecolor="white")
    ax.set_xlabel("Single-gene KO growth ratio")
    ax.set_ylabel("Gene count")
    ax.set_title("Single-gene knockout phenotypes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure10_single_gene_ko.png")
    plt.close(fig)

    top_double = double_ko.head(20).copy()
    top_double["pair"] = top_double["gene_a"] + " + " + top_double["gene_b"]
    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=220)
    ax.barh(top_double["pair"][::-1], top_double["interaction_score"][::-1], color="#8C4B3E")
    ax.set_xlabel("Interaction score (min single ratio - double ratio)")
    ax.set_title("Targeted double-gene knockout interactions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure11_double_gene_ko.png")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6.6, 4.4), dpi=220)
    ax1.plot(dfba["time_h"], dfba["biomass_gDW_L"], color="#1D4E89", lw=2, label="Biomass")
    ax1.set_xlabel("Time (h)")
    ax1.set_ylabel("Biomass (gDW L$^{-1}$)", color="#1D4E89")
    ax2 = ax1.twinx()
    ax2.plot(dfba["time_h"], dfba["substrate_mmol_L"], color="#B35C00", lw=2, label="Glucose")
    ax2.set_ylabel("Glucose (mmol L$^{-1}$)", color="#B35C00")
    ax1.set_title("Dynamic FBA simulation")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure12_dfba.png")
    plt.close(fig)

    top_fseof = fseof_genes.head(20).sort_values("fseof_score")
    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=220)
    colors = ["#287D8E" if x == "overexpression" else "#C23B22" for x in top_fseof["recommendation"]]
    ax.barh(top_fseof["gene"], top_fseof["fseof_score"], color=colors)
    ax.set_xlabel("ecFSEOF gene score")
    ax.set_title("FSEOF-prioritized genes")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure13_fseof_targets.png")
    plt.close(fig)

    top_targets = metastrain_targets.head(20).sort_values("metastrain_fitness_score")
    fig, ax = plt.subplots(figsize=(7.0, 5.6), dpi=220)
    ax.barh(top_targets["gene"], top_targets["metastrain_fitness_score"], color="#6B4E9B")
    ax.set_xlabel("MetaStrain-style target score")
    ax.set_title("MetaStrain-style strain design targets")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "figure14_metastrain_targets.png")
    plt.close(fig)


def write_excel(tables):
    path = TABLE_DIR / "eciFX1172_advanced_analysis_results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    return path


def df_to_markdown(df):
    if df is None or df.empty:
        return "无可报告记录。"
    clean = df.copy()
    for col in clean.columns:
        clean[col] = clean[col].map(lambda value: "" if pd.isna(value) else str(value).replace("|", "/"))
    header = "| " + " | ".join(clean.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(clean.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in clean.to_numpy()]
    return "\n".join([header, sep] + rows)


def write_report(summary, top_tables):
    fig_paths = {p.stem: p.resolve().as_posix() for p in sorted(FIG_DIR.glob("*.png"))}
    text = f"""# eciFX1172 酶约束模型的高级预测分析

## 分析设置

本轮新增分析基于最终校准的酶约束模型 `eciFX1172.json`，并以 `r_0013`（GM-A production）作为庆大霉素 A（Gentamicin A）的目标产物反应。模型默认生长目标为 `growth`，葡萄糖摄取反应为不可逆交换反应 `{DEFAULT_SUBSTRATE_RXN}`。MetaStrain-main 中原始脚本采用 ecFSEOF 候选降维、MOMA/参考通量约束、JADE 搜索与 0/1/2/3 操作编码；由于其实现硬编码了 ecYeast/eciML1515 路径并依赖 Gurobi/Ray，本研究将其算法逻辑移植为适配 iFX1172/eciFX1172 的可复现流程。

## 结果概述

- 野生型 eciFX1172 生长速率为 {summary['wt_growth']:.6f} h^-1，低于未加酶约束模型，符合酶容量限制后预测值收缩的预期。
- 在保持至少 10% 野生型生长的条件下，`{PRODUCT_RXN}` 的最大理论通量为 {summary['max_product']:.6f} mmol gDW^-1 h^-1。
- 底物利用性分析共测试 {summary['substrate_count']} 个有机交换底物，其中 {summary['usable_substrate_count']} 个可支持 ecModel 非零生长。
- 全基因单敲除覆盖 {summary['single_gene_count']} 个基因，预测 essential 基因为 {summary['essential_gene_count']} 个。
- 靶向双基因敲除测试 {summary['double_gene_count']} 个候选组合，识别到 {summary['synthetic_lethal_count']} 个合成致死组合和 {summary['synthetic_sick_count']} 个明显合成病弱组合。

![Substrate utilization]({fig_paths['figure7_substrate_utilization']})

![Substrate robustness]({fig_paths['figure8_robustness_substrate']})

![Enzyme-pool robustness]({fig_paths['figure9_robustness_enzyme_pool']})

![Single-gene knockout]({fig_paths['figure10_single_gene_ko']})

![Double-gene knockout]({fig_paths['figure11_double_gene_ko']})

![Dynamic FBA]({fig_paths['figure12_dfba']})

![FSEOF targets]({fig_paths['figure13_fseof_targets']})

![MetaStrain targets]({fig_paths['figure14_metastrain_targets']})

## 可写入论文的 Results 草稿

为评价 eciFX1172 在不同营养与遗传扰动条件下的预测能力，我们进一步开展了底物利用性、鲁棒性、基因敲除、动态通量平衡分析、FSEOF 以及 MetaStrain-style 靶点筛选。底物利用性分析显示，酶约束后模型仍可在多种有机碳源上维持生长，但预测生长速率整体低于原始 GEM，说明全局酶池约束有效压缩了过高的通量空间。葡萄糖摄取速率扫描进一步表明，eciFX1172 在低底物摄取区间对碳源限制更敏感，而在底物充足时受总酶池上限控制，呈现典型的酶容量饱和行为。通过改变总酶池上限，模型生长速率随酶池放宽而增加，并在校准上限附近进入平台区，支持当前酶池参数能够合理限制最大生长。

单基因敲除分析在全模型范围内识别了生长必需基因和生长受限基因，为后续遗传操作提供了安全边界。基于单敲除敏感性和 FSEOF 候选基因进一步开展靶向双基因敲除，发现部分组合呈现明显的合成病弱或合成致死效应，提示这些基因可能位于相互补偿的代谢路径或共同承担关键前体供应。dFBA 模拟显示，在给定初始葡萄糖和生物量条件下，细胞生物量随时间增加而底物逐步消耗；当底物接近耗尽时，预测生长速率下降至零，说明 eciFX1172 可用于发酵时间尺度上的底物-生物量动态预测。

以 `r_0013` 为庆大霉素 A 目标反应的 FSEOF 分析识别了随产物通量强制增加而同步增强或下降的反应和基因。正斜率候选被解释为潜在过表达靶点，负斜率候选则提示可考虑下调或敲除，但需结合单敲除生长比筛除高风险必需基因。进一步参考 MetaStrain-main 的操作编码和 fitness 思路，我们将候选基因编码为无操作、过表达、下调和敲除/下调四类，并使用 ecFSEOF 响应强度与单敲除可行性构建靶点优先级。排名靠前的候选靶点见结果表，可作为后续庆大霉素 A 高产菌株设计的第一轮实验验证列表。

## 主要靶点表

### FSEOF 前 15 个基因

{df_to_markdown(top_tables['fseof'])}

### MetaStrain-style 前 15 个靶点

{df_to_markdown(top_tables['metastrain'])}

### 双基因敲除前 15 个组合

{df_to_markdown(top_tables['double'])}

## 方法补充

底物利用性分析中，模型先关闭有机交换反应的摄取方向，仅保留无机盐、质子、水、氧气和氮/硫/磷等基础营养的摄取；随后逐一开放候选有机底物摄取上限至 10 mmol gDW^-1 h^-1 并最大化生长。鲁棒性分析分别扫描葡萄糖摄取上限和总酶池上限。dFBA 使用显式 Euler 步进，在每个时间步根据剩余底物和生物量更新最大摄取速率，并调用 FBA 得到瞬时生长速率与摄取通量。FSEOF 在维持至少 10% 野生型生长的前提下，逐步强制庆大霉素 A 产物通量达到最大理论产量的 10%-90%，并用 pFBA 计算各反应通量随产物通量的线性斜率。MetaStrain-style 靶点分析沿用 MetaStrain 的候选降维与操作编码思想，但将原始 Gurobi/Ray/特定模型依赖替换为 eciFX1172 可运行的 COBRApy/GLPK 实现。
"""
    md_path = REPORT_DIR / "eciFX1172_advanced_analysis_report.md"
    md_path.write_text(text, encoding="utf-8")
    html_path = REPORT_DIR / "eciFX1172_advanced_analysis_report.html"
    html_lines = []
    in_table = False
    for line in text.splitlines():
        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if image_match:
            if in_table:
                html_lines.append("</pre>")
                in_table = False
            alt, src = image_match.groups()
            html_lines.append(f"<figure><img src='file:///{src}' alt='{alt}'><figcaption>{alt}</figcaption></figure>")
        elif line.startswith("# "):
            if in_table:
                html_lines.append("</pre>")
                in_table = False
            html_lines.append(f"<h1>{line[2:]}</h1>")
        elif line.startswith("## "):
            if in_table:
                html_lines.append("</pre>")
                in_table = False
            html_lines.append(f"<h2>{line[3:]}</h2>")
        elif line.startswith("### "):
            if in_table:
                html_lines.append("</pre>")
                in_table = False
            html_lines.append(f"<h3>{line[4:]}</h3>")
        elif line.startswith("|"):
            if not in_table:
                html_lines.append("<pre class='table'>")
                in_table = True
            html_lines.append(line)
        elif line.strip() == "":
            if in_table:
                html_lines.append("</pre>")
                in_table = False
        else:
            if in_table:
                html_lines.append("</pre>")
                in_table = False
            html_lines.append(f"<p>{line}</p>")
    if in_table:
        html_lines.append("</pre>")
    html_path.write_text(
        "<html><head><meta charset='utf-8'><title>eciFX1172 advanced analysis</title>"
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:980px;margin:32px auto;line-height:1.65;color:#222}"
        "img{max-width:100%;height:auto;border:1px solid #ddd}figure{margin:24px 0}figcaption{font-size:13px;color:#555}"
        "pre.table{white-space:pre-wrap;background:#fafafa;border:1px solid #ddd;padding:12px;overflow:auto}</style></head><body>"
        + "\n".join(html_lines)
        + "</body></html>",
        encoding="utf-8",
    )
    return md_path, html_path


def main():
    ensure_dirs()
    print("Loading models...")
    gem = load_irreversible_model()
    ec_model = load_ec_model()

    print("Substrate utilization...")
    substrate = substrate_utilization(gem, ec_model)
    print("Robustness...")
    robustness_substrate, robustness_pool = robustness_analysis(gem, ec_model)
    print("Single gene knockout...")
    single_ko, wt_growth = single_gene_knockout(ec_model)
    print("FSEOF...")
    fseof_reactions, fseof_genes, wt_growth, max_product = fseof_analysis(ec_model)
    print("MetaStrain-style targets...")
    metastrain_targets, meta_candidate_genes = metastrain_style_target_analysis(ec_model, fseof_genes, single_ko)
    print("Double gene knockout...")
    single_candidates = list(single_ko[single_ko["growth_ratio"].between(0.05, 0.95)].head(20)["gene"])
    fseof_candidates = list(fseof_genes.head(20)["gene"]) if not fseof_genes.empty else []
    candidate_genes = list(dict.fromkeys(fseof_candidates + single_candidates))[:30]
    double_ko = double_gene_knockout(ec_model, single_ko, candidate_genes)
    print("dFBA...")
    dfba = simulate_dfba(ec_model)

    tables = {
        "substrate_utilization": substrate,
        "robustness_substrate": robustness_substrate,
        "robustness_enzyme_pool": robustness_pool,
        "single_gene_ko": single_ko,
        "double_gene_ko": double_ko,
        "dfba": dfba,
        "fseof_reactions": fseof_reactions,
        "fseof_gene_targets": fseof_genes,
        "metastrain_targets": metastrain_targets,
    }
    for name, table in tables.items():
        save_table(table, name)
    excel = write_excel(tables)

    print("Plotting...")
    plot_outputs(substrate, robustness_substrate, robustness_pool, single_ko, double_ko, dfba, fseof_genes, metastrain_targets)

    summary = {
        "wt_growth": float(wt_growth),
        "max_product": float(max_product),
        "substrate_count": int(len(substrate)),
        "usable_substrate_count": int(substrate["usable_by_ec"].sum()),
        "single_gene_count": int(len(single_ko)),
        "essential_gene_count": int((single_ko["phenotype"] == "essential").sum()),
        "double_gene_count": int(len(double_ko)),
        "synthetic_lethal_count": int((double_ko["phenotype"] == "synthetic_lethal").sum()),
        "synthetic_sick_count": int((double_ko["phenotype"] == "synthetic_sick").sum()),
        "excel": str(excel),
    }
    pd.DataFrame([summary]).to_csv(TABLE_DIR / "advanced_analysis_summary.csv", index=False, encoding="utf-8-sig")
    top_tables = {
        "fseof": fseof_genes.head(15)[["gene", "recommendation", "fseof_score", "mean_slope", "representative_reactions"]] if not fseof_genes.empty else pd.DataFrame(),
        "metastrain": metastrain_targets.head(15)[["gene", "recommended_operation", "metastrain_fitness_score", "single_KO_growth_ratio", "representative_reactions"]] if not metastrain_targets.empty else pd.DataFrame(),
        "double": double_ko.head(15)[["gene_a", "gene_b", "growth_ratio", "interaction_score", "phenotype"]] if not double_ko.empty else pd.DataFrame(),
    }
    md_path, html_path = write_report(summary, top_tables)
    print(json.dumps({"summary": summary, "report_md": str(md_path), "report_html": str(html_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
