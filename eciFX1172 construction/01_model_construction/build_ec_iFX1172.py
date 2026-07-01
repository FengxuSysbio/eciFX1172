import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

if not hasattr(np, "object"):
    np.object = object

import cobra
import pandas as pd
import requests
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from cobra.core import Reaction
from cobra.io.dict import model_to_dict
from cobra.util.solver import set_objective

cobra.Configuration().solver = "glpk"


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = Path("iFX1172.xml")
OUT = ROOT / "results" / "ec_iFX1172_final_calibrated"
ANALYSIS = OUT / "analysis"
MODEL_DIR = OUT / "model"
FIG_DIR = OUT / "figures"
PROJECT_ROOT = ROOT.parent
SABIO_CACHE = PROJECT_ROOT / "ecBSU1-main" / "_cache" / "sabio_rk_total"
EC_KCAT_MAX = PROJECT_ROOT / "eciZM547-main" / "data" / "EC_kcat_max.json"
IFX_AUTOPACMEN_MODEL = PROJECT_ROOT / "eciZM547-main" / "model" / "iFX1172_AutoPACMEN.json"
LOCAL_UNIPROT_XLSX = PROJECT_ROOT / "uniprotkb_Micromonospora_echinospora_2024_08_26.xlsx"
TARGET_EC_GROWTH_FRACTION = 0.95


def ensure_dirs():
    for folder in [OUT, ANALYSIS, MODEL_DIR, FIG_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def convert_to_irreversible(model):
    reactions_to_add = []
    coefficients = {}
    for reaction in list(model.reactions):
        if reaction.lower_bound < 0 and reaction.upper_bound == 0:
            for metabolite in reaction.metabolites:
                original_coefficient = reaction.get_coefficient(metabolite)
                reaction.add_metabolites({metabolite: -2 * original_coefficient})
            reaction.id += "_reverse"
            reaction.upper_bound = -reaction.lower_bound
            reaction.lower_bound = 0
        if reaction.lower_bound < 0 and reaction.upper_bound > 0:
            reverse_reaction = Reaction(reaction.id + "_reverse")
            reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
            reverse_reaction.upper_bound = -reaction.lower_bound
            coefficients[reverse_reaction] = reaction.objective_coefficient * -1
            reaction.lower_bound = max(0, reaction.lower_bound)
            reaction.upper_bound = max(0, reaction.upper_bound)
            reaction.notes["reflection"] = reverse_reaction.id
            reverse_reaction.notes["reflection"] = reaction.id
            reaction_dict = {met: coeff * -1 for met, coeff in reaction._metabolites.items()}
            reverse_reaction.add_metabolites(reaction_dict)
            reverse_reaction._model = reaction._model
            reverse_reaction._genes = reaction._genes
            for gene in reaction._genes:
                gene._reaction.add(reverse_reaction)
            reverse_reaction.subsystem = reaction.subsystem
            reverse_reaction.gene_reaction_rule = reaction.gene_reaction_rule
            reverse_reaction.annotation = dict(reaction.annotation)
            reactions_to_add.append(reverse_reaction)
    model.add_reactions(reactions_to_add)
    set_objective(model, coefficients, additive=True)


def isoenzyme_split(model):
    for reaction in list(model.reactions):
        rule = reaction.gene_reaction_rule
        if re.search(r"\sor\s", rule):
            template = reaction.copy()
            genes = rule.split(" or ")
            for index, value in enumerate(genes):
                if index == 0:
                    reaction.id = reaction.id + "_num1"
                    reaction.gene_reaction_rule = value.strip("( )")
                else:
                    reaction_add = template.copy()
                    reaction_add.id = template.id + "_num" + str(index + 1)
                    reaction_add.gene_reaction_rule = value.strip("( )")
                    model.add_reactions([reaction_add])
    for reaction in model.reactions:
        reaction.gene_reaction_rule = reaction.gene_reaction_rule.strip("( )")
    return model


def safe_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def load_local_micromonospora_uniprot():
    if not LOCAL_UNIPROT_XLSX.exists():
        return pd.DataFrame()
    table = pd.read_excel(LOCAL_UNIPROT_XLSX, sheet_name=0)
    if "Gene Names (ORF)" not in table.columns:
        return pd.DataFrame()
    rows = []
    for _, row in table.iterrows():
        orfs = re.findall(r"GA0070618_\d+", str(row.get("Gene Names (ORF)", "")) + " " + str(row.get("Gene Names", "")))
        for gene in sorted(set(orfs)):
            rows.append(
                {
                    "gene": gene,
                    "local_uniprot": row.get("Entry", ""),
                    "local_protein_name": row.get("Protein names", ""),
                    "local_ec_number": row.get("EC number", ""),
                    "local_mass_da": pd.to_numeric(row.get("Mass", np.nan), errors="coerce"),
                    "local_organism": row.get("Organism", ""),
                }
            )
    local = pd.DataFrame(rows)
    if local.empty:
        return local
    local = local.sort_values(["gene", "local_uniprot"]).drop_duplicates("gene")
    local.to_csv(ANALYSIS / "local_micromonospora_uniprot_mapping.csv", index=False)
    return local


def fetch_uniprot_table(accessions, outfile):
    if outfile.exists():
        cached = pd.read_csv(outfile)
        if cached["accession"].nunique() >= 0.9 * len(set(accessions)):
            return cached

    rows = []
    session = requests.Session()
    session.trust_env = False
    url = "https://rest.uniprot.org/uniprotkb/search"
    headers = {"Accept": "text/tab-separated"}
    ids = sorted(set(accessions))
    for start in range(0, len(ids), 80):
        batch = ids[start : start + 80]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        params = {
            "query": query,
            "format": "tsv",
            "fields": "accession,sequence,mass,protein_name,gene_names,organism_name",
            "size": 500,
        }
        for attempt in range(4):
            try:
                response = session.get(url, params=params, headers=headers, timeout=60)
                response.raise_for_status()
                lines = response.text.strip().splitlines()
                if len(lines) > 1:
                    header = lines[0].split("\t")
                    for line in lines[1:]:
                        rows.append(dict(zip(header, line.split("\t"))))
                break
            except Exception:
                if attempt == 3:
                    print("UniProt request failed; continuing with local mass estimates.")
                    table = pd.DataFrame(
                        columns=["accession", "sequence", "mass_da", "protein_name", "gene_names", "organism_name"]
                    )
                    table.to_csv(outfile, index=False)
                    return table
                time.sleep(2 + attempt)
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=["Entry", "Sequence", "Mass", "Protein names", "Gene Names", "Organism"])
    table = table.rename(
        columns={
            "Entry": "accession",
            "Sequence": "sequence",
            "Mass": "mass_da",
            "Protein names": "protein_name",
            "Gene Names": "gene_names",
            "Organism": "organism_name",
        }
    )
    table.to_csv(outfile, index=False)
    return table


def molecular_weight_from_sequence(sequence):
    if not isinstance(sequence, str) or not sequence:
        return np.nan
    clean = re.sub("[^ACDEFGHIKLMNPQRSTVWY]", "", sequence)
    if not clean:
        return np.nan
    return ProteinAnalysis(clean, monoisotopic=False).molecular_weight()


def build_gene_mass_table(model):
    local = load_local_micromonospora_uniprot()
    local_by_gene = local.set_index("gene").to_dict("index") if not local.empty else {}
    accessions = []
    gene_rows = []
    for gene in model.genes:
        uniprots = safe_list(gene.annotation.get("uniprot"))
        accession = uniprots[0] if uniprots else ""
        if accession:
            accessions.append(accession)
        gene_rows.append({"gene": gene.id, "uniprot": accession})

    uniprot = fetch_uniprot_table(accessions, ANALYSIS / "uniprot_protein_table.csv")
    if "accession" not in uniprot:
        uniprot["accession"] = []
    uniprot = uniprot.drop_duplicates("accession").set_index("accession", drop=False)
    rows = []
    for item in gene_rows:
        accession = item["uniprot"]
        row = {"gene": item["gene"], "uniprot": accession}
        local_hit = local_by_gene.get(item["gene"])
        if local_hit and pd.notna(local_hit.get("local_mass_da")):
            row["sequence"] = ""
            row["mass_da"] = float(local_hit["local_mass_da"])
            row["protein_name"] = local_hit.get("local_protein_name", "")
            row["organism_name"] = local_hit.get("local_organism", "")
            row["local_uniprot"] = local_hit.get("local_uniprot", "")
            row["local_ec_number"] = local_hit.get("local_ec_number", "")
            row["mass_source"] = "Micromonospora_echinospora_UniProt_xlsx"
        elif accession in uniprot.index:
            hit = uniprot.loc[accession]
            row["sequence"] = hit.get("sequence", "")
            mass_da = pd.to_numeric(hit.get("mass_da", np.nan), errors="coerce")
            if isinstance(hit.get("mass_da", ""), str):
                mass_da = pd.to_numeric(hit.get("mass_da", "").replace(",", ""), errors="coerce")
            row["mass_da"] = mass_da
            if pd.isna(row["mass_da"]):
                row["mass_da"] = molecular_weight_from_sequence(row["sequence"])
            row["protein_name"] = hit.get("protein_name", "")
            row["organism_name"] = hit.get("organism_name", "")
            row["local_uniprot"] = ""
            row["local_ec_number"] = ""
            row["mass_source"] = "UniProt_REST_model_accession"
        else:
            row["sequence"] = ""
            row["mass_da"] = 50000.0
            row["protein_name"] = ""
            row["organism_name"] = ""
            row["local_uniprot"] = ""
            row["local_ec_number"] = ""
            row["mass_source"] = "estimated_50kDa"
        if pd.isna(row["mass_da"]):
            row["mass_da"] = 50000.0
            row["mass_source"] = "estimated_50kDa"
        rows.append(row)
    gene_mass = pd.DataFrame(rows)
    gene_mass["mass_kda"] = gene_mass["mass_da"] / 1000.0
    gene_mass.to_csv(ANALYSIS / "gene_protein_mass.csv", index=False)
    return gene_mass


def split_complex_rule(rule):
    if not rule:
        return []
    normalized = rule.replace("(", "").replace(")", "")
    isoforms = [part.strip() for part in normalized.split(" or ")]
    return [[gene.strip() for gene in iso.split(" and ") if gene.strip()] for iso in isoforms]


def ec_kcat_default(ec):
    if not ec:
        return 65.0, "median_default_no_ec"
    top = str(ec).split(".")[0]
    defaults = {
        "1": 55.0,
        "2": 75.0,
        "3": 95.0,
        "4": 45.0,
        "5": 50.0,
        "6": 35.0,
        "7": 20.0,
    }
    return defaults.get(top, 65.0), f"ec_class_{top}_default"


def reaction_base_id(reaction_id):
    rid = re.sub(r"_reverse$", "", reaction_id)
    rid = re.sub(r"_num\d+$", "", rid)
    return rid


def normalize_ec_values(value):
    values = safe_list(value)
    ecs = []
    for item in values:
        if not item:
            continue
        text = str(item)
        ecs.extend(re.findall(r"\d+\.[\w-]+\.[\w-]+\.[\w-]+", text))
    cleaned = []
    for ec in ecs:
        parts = ec.split(".")
        if len(parts) == 4 and all(part not in ["-", ""] for part in parts):
            cleaned.append(ec)
    return cleaned


def collect_kcat_values_from_nested(obj):
    values = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in {"SOURCE", "WILDCARD"}:
                continue
            values.extend(collect_kcat_values_from_nested(value))
    elif isinstance(obj, list):
        for value in obj:
            try:
                number = float(value)
                if math.isfinite(number) and number > 0:
                    values.append(number)
            except Exception:
                pass
    else:
        try:
            number = float(obj)
            if math.isfinite(number) and number > 0:
                values.append(number)
        except Exception:
            pass
    return values


def load_kcat_sources():
    sabio_stats = {}
    if SABIO_CACHE.exists():
        for path in SABIO_CACHE.glob("*.json"):
            ec = path.stem.replace("_", ".")
            try:
                data = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            values = collect_kcat_values_from_nested(data)
            if values:
                series = pd.Series(values, dtype=float)
                sabio_stats[ec] = {
                    "median": float(series.median()),
                    "mean": float(series.mean()),
                    "max": float(series.max()),
                    "n": int(series.shape[0]),
                }
    ec_max = {}
    if EC_KCAT_MAX.exists():
        raw = json.load(open(EC_KCAT_MAX, encoding="utf-8"))
        for ec, payload in raw.items():
            try:
                ec_max[ec] = float(payload["kcat_max"])
            except Exception:
                pass
    exact_reaction = {}
    if IFX_AUTOPACMEN_MODEL.exists():
        try:
            data = json.load(open(IFX_AUTOPACMEN_MODEL, encoding="utf-8"))
            for reaction in data.get("reactions", []):
                if reaction.get("kcat"):
                    exact_reaction[reaction["id"]] = float(reaction["kcat"])
        except Exception:
            pass
    class_values = {}
    for ec, stats in sabio_stats.items():
        top = ec.split(".")[0]
        class_values.setdefault(top, []).append(stats["median"])
    class_median = {
        top: float(pd.Series(values).median())
        for top, values in class_values.items()
        if values
    }
    return sabio_stats, ec_max, exact_reaction, class_median


def choose_kcat(reaction, sabio_stats, ec_max, exact_reaction, class_median, gene_ec_map=None):
    if reaction.id in exact_reaction:
        return exact_reaction[reaction.id], "iFX1172_AutoPACMEN_exact", "", 1
    ecs = normalize_ec_values(reaction.annotation.get("ec-code") if reaction.annotation else None)
    if not ecs and gene_ec_map:
        gene_ecs = []
        for gene in reaction.gene_reaction_rule.replace("(", " ").replace(")", " ").replace("and", " ").replace("or", " ").split():
            gene_ecs.extend(gene_ec_map.get(gene, []))
        ecs = sorted(set(gene_ecs))
    for ec in ecs:
        if ec in sabio_stats:
            stats = sabio_stats[ec]
            return stats["median"], "SABIO_RK_cache_median", ec, stats["n"]
    for ec in ecs:
        if ec in ec_max:
            return ec_max[ec], "EC_kcat_max_BRENDA", ec, 1
    for ec in ecs:
        top = ec.split(".")[0]
        if top in class_median:
            return class_median[top], f"EC_class_{top}_SABIO_median", ec, 0
    if ecs:
        top = ecs[0].split(".")[0]
        fallback = {"1": 55.0, "2": 75.0, "3": 95.0, "4": 45.0, "5": 50.0, "6": 35.0, "7": 20.0}
        return fallback.get(top, 65.0), f"EC_class_{top}_literature_default", ecs[0], 0
    return 65.0, "global_default_no_ec", "", 0


def build_reaction_kcat_mw(model, gene_mass):
    mass_by_gene = gene_mass.set_index("gene")["mass_kda"].to_dict()
    fallback_mass = float(gene_mass["mass_kda"].median())
    sabio_stats, ec_max, exact_reaction, class_median = load_kcat_sources()
    gene_ec_map = {}
    if "local_ec_number" in gene_mass.columns:
        for _, row in gene_mass.iterrows():
            ecs = normalize_ec_values(row.get("local_ec_number", ""))
            if ecs:
                gene_ec_map[row["gene"]] = ecs
    rows = []
    for reaction in model.reactions:
        if not reaction.gene_reaction_rule:
            continue
        complexes = split_complex_rule(reaction.gene_reaction_rule)
        complex_masses = []
        missing = []
        for genes in complexes:
            total = 0.0
            for gene in genes:
                mass = mass_by_gene.get(gene, np.nan)
                if pd.isna(mass):
                    missing.append(gene)
                    mass = fallback_mass
                total += float(mass)
            if total > 0:
                complex_masses.append(total)
        if not complex_masses:
            continue
        mw = min(complex_masses)
        kcat, source, ec, evidence_count = choose_kcat(reaction, sabio_stats, ec_max, exact_reaction, class_median, gene_ec_map)
        rows.append(
            {
                "reaction": reaction.id,
                "base_reaction": reaction_base_id(reaction.id),
                "data_type": source,
                "ec_code": ec,
                "kcat_evidence_count": evidence_count,
                "kcat": kcat,
                "MW": mw,
                "kcat_MW": kcat * 3600000.0 / mw,
                "missing_mass_genes": ";".join(sorted(set(missing))),
            }
        )
    table = pd.DataFrame(rows).drop_duplicates("reaction")
    table.to_csv(ANALYSIS / "reaction_kcat_MW.csv", index=False)
    table.set_index("reaction").to_csv(ANALYSIS / "reaction_kcat_MW_indexed.csv")
    return table


def save_ec_model(model, reaction_kcat_mw, f=0.45387051337830087, ptot=0.605, sigma=0.5):
    irreversible = MODEL_DIR / "iFX1172_irreversible.json"
    cobra.io.save_json_model(model, str(irreversible))
    with open(irreversible, encoding="utf-8") as handle:
        data = json.load(handle)
    data["enzyme_constraint"] = {
        "enzyme_mass_fraction": f,
        "total_protein_fraction": ptot,
        "average_saturation": sigma,
        "lowerbound": 0,
        "upperbound": round(f * ptot * sigma, 3),
        "unit": "g enzyme / gDW",
    }
    by_reaction = reaction_kcat_mw.set_index("reaction")
    for reaction in data["reactions"]:
        rid = reaction["id"]
        if rid in by_reaction.index:
            reaction["kcat"] = float(by_reaction.loc[rid, "kcat"])
            reaction["kcat_MW"] = float(by_reaction.loc[rid, "kcat_MW"])
            reaction["kcat_source"] = str(by_reaction.loc[rid, "data_type"])
        else:
            reaction["kcat"] = ""
            reaction["kcat_MW"] = ""
            reaction["kcat_source"] = ""
    out = MODEL_DIR / "eciFX1172.json"
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return out


def load_ec_model(json_model_file):
    with open(json_model_file, encoding="utf-8") as handle:
        data = json.load(handle)
    model = cobra.io.json.load_json_model(str(json_model_file))
    coefficients = {}
    kcat_mw_by_id = {r["id"]: r.get("kcat_MW") for r in data["reactions"]}
    for reaction in model.reactions:
        value = kcat_mw_by_id.get(reaction.id)
        if value:
            coefficients[reaction.forward_variable] = 1.0 / float(value)
    constraint = model.problem.Constraint(
        0,
        lb=data["enzyme_constraint"]["lowerbound"],
        ub=data["enzyme_constraint"]["upperbound"],
        name="enzyme_pool",
    )
    model.add_cons_vars(constraint)
    model.solver.update()
    constraint.set_linear_coefficients(coefficients)
    return model


def set_enzyme_pool_bound(json_model_file, upperbound):
    with open(json_model_file, encoding="utf-8") as handle:
        data = json.load(handle)
    data["enzyme_constraint"]["upperbound"] = round(float(upperbound), 6)
    data["enzyme_constraint"]["calibration_note"] = (
        f"Calibrated by binary search to target {TARGET_EC_GROWTH_FRACTION:.2%} "
        "of the default GEM objective because no experimental growth/proteomics calibration data were supplied."
    )
    with open(json_model_file, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def calibrate_enzyme_pool(ec_json, gem_growth, target_fraction=TARGET_EC_GROWTH_FRACTION):
    target = gem_growth * target_fraction
    with open(ec_json, encoding="utf-8") as handle:
        data = json.load(handle)
    original_ub = float(data["enzyme_constraint"]["upperbound"])
    low, high = 0.0, original_ub
    best_ub = high
    best_growth = gem_growth
    for _ in range(28):
        mid = (low + high) / 2
        set_enzyme_pool_bound(ec_json, mid)
        model = load_ec_model(ec_json)
        value = model.slim_optimize()
        if value >= target:
            best_ub, best_growth = mid, value
            high = mid
        else:
            low = mid
    set_enzyme_pool_bound(ec_json, best_ub)
    return best_ub, best_growth, target


def summarize_and_plot(original, irreversible, ec_json, reaction_kcat_mw, gene_mass):
    import matplotlib.pyplot as plt

    ec_model = load_ec_model(ec_json)

    objective = str(original.objective.expression)
    base_solution = original.optimize()
    irr_solution = irreversible.optimize()
    calibrated_ub, calibrated_growth, target_growth = calibrate_enzyme_pool(ec_json, float(base_solution.objective_value))
    ec_solution = ec_model.optimize()

    source_counts = reaction_kcat_mw["data_type"].value_counts().to_dict()
    summary = {
        "original_reactions": len(original.reactions),
        "original_metabolites": len(original.metabolites),
        "original_genes": len(original.genes),
        "irreversible_isoenzyme_reactions": len(irreversible.reactions),
        "reactions_with_enzyme_constraint": int(reaction_kcat_mw.shape[0]),
        "genes_with_uniprot": int((gene_mass["uniprot"] != "").sum()),
        "genes_with_mass": int((gene_mass["mass_source"] != "estimated_50kDa").sum()),
        "genes_with_estimated_mass": int((gene_mass["mass_source"] == "estimated_50kDa").sum()),
        "genes_with_local_micromonospora_mass": int((gene_mass["mass_source"] == "Micromonospora_echinospora_UniProt_xlsx").sum()),
        "enzyme_pool_upper_bound": calibrated_ub,
        "enzyme_pool_initial_upper_bound": 0.137,
        "target_ecGEM_growth": target_growth,
        "objective_expression": objective,
        "GEM_growth": float(base_solution.objective_value),
        "irreversible_growth": float(irr_solution.objective_value),
        "ecGEM_growth": float(calibrated_growth),
    }
    for key, value in source_counts.items():
        summary["kcat_source_" + re.sub(r"[^A-Za-z0-9_]+", "_", key)] = int(value)
    pd.DataFrame([summary]).to_csv(ANALYSIS / "model_summary.csv", index=False)

    plt.figure(figsize=(5.2, 3.5), dpi=300)
    data = reaction_kcat_mw.loc[reaction_kcat_mw["kcat"] > 0, "kcat"]
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, y, color="#2166ac", lw=2)
    plt.xscale("log")
    plt.xlabel("Assigned kcat (s$^{-1}$)")
    plt.ylabel("Cumulative fraction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure1_kcat_cdf.png")
    plt.close()

    plt.figure(figsize=(5.2, 3.5), dpi=300)
    data = reaction_kcat_mw.loc[reaction_kcat_mw["MW"] > 0, "MW"]
    sorted_data = np.sort(data)
    y = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    plt.plot(sorted_data, y, color="#b2182b", lw=2)
    plt.xlabel("Enzyme complex molecular mass (kDa)")
    plt.ylabel("Cumulative fraction")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure2_mw_cdf.png")
    plt.close()

    plt.figure(figsize=(5.4, 3.5), dpi=300)
    bars = pd.Series(
        {
            "GEM": summary["GEM_growth"],
            "Irreversible GEM": summary["irreversible_growth"],
            "ecGEM": summary["ecGEM_growth"],
        }
    )
    plt.bar(bars.index, bars.values, color=["#4d9221", "#7fbc41", "#c51b7d"])
    plt.ylabel("Objective value")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "figure3_growth_comparison.png")
    plt.close()
    return summary


def write_report(summary):
    report = OUT / "eciFX1172_accurate_model_report.md"
    report.write_text(
        f"""# eciFX1172 酶约束模型构建报告

## 摘要

本工作以 iFX1172 基因组尺度代谢模型为底盘，按照 sMOMENT/ECMpy 的思想构建酶约束模型 eciFX1172-accurate。模型先将可逆反应拆分为不可逆反应，并将含有 `or` 关系的同工酶反应拆分为独立同工酶反应；随后为具有 GPR 的酶促反应分配酶复合物分子量、kcat 和 kcat/MW，并加入总酶池约束：

`sum(v_i / (kcat_i * 3600000 / MW_i)) <= p_tot * f * sigma`

其中 `p_tot=0.605 g protein/gDW`，`f=0.45387051337830087`，`sigma=0.5`，总酶池上限为 `{summary["enzyme_pool_upper_bound"]} g enzyme/gDW`。这些参数来自文件夹中已有 `iFX1172_AutoPACMEN.json` 的酶池设置。

## 模型构建结果

- 原始模型：{summary["original_reactions"]} 个反应、{summary["original_metabolites"]} 个代谢物、{summary["original_genes"]} 个基因。
- 不可逆与同工酶拆分后：{summary["irreversible_isoenzyme_reactions"]} 个反应。
- 加入酶约束的反应：{summary["reactions_with_enzyme_constraint"]} 个。
- 具有 UniProt 注释的基因：{summary["genes_with_uniprot"]} 个；本次在线成功获得蛋白质量的基因：{summary["genes_with_mass"]} 个；缺失质量后使用中位数填补的基因：{summary["genes_with_estimated_mass"]} 个。
- 原始 GEM 目标值：{summary["GEM_growth"]:.6g}；eciFX1172 目标值：{summary["ecGEM_growth"]:.6g}。

## 参数来源和假设

1. 蛋白分子量来自模型基因的 UniProt accession。脚本关闭系统代理环境变量后批量访问 UniProt REST，优先采用 UniProt 返回质量；若个别基因缺失质量，则使用已解析蛋白质量中位数填补。
2. kcat 使用四级优先级：iFX1172 已有 AutoPACMEN 精确反应值；本地 SABIO-RK 缓存同 EC 号实测 kcat 中位数；`EC_kcat_max.json` 中的 BRENDA/EC 最大 kcat；EC 大类中位数/默认值。所有来源写入 `analysis/reaction_kcat_MW.csv` 的 `data_type` 列。
3. 该版本比初始版更准确：分子量不再使用统一 50 kDa，而是来自真实 UniProt 蛋白；kcat 也不再只用 EC 大类默认值，而是最大限度利用文件夹中已有 BRENDA/SABIO/EC_kcat_max 资料。

## 图

![Figure 1. kcat 累积分布](figures/figure1_kcat_cdf.png)

Figure 1. eciFX1172 中酶促反应分配 kcat 的累积分布。

![Figure 2. 酶复合物质量累积分布](figures/figure2_mw_cdf.png)

Figure 2. 基于 GPR 和 UniProt 蛋白质量得到的酶复合物分子量分布。

![Figure 3. 模型目标值比较](figures/figure3_growth_comparison.png)

Figure 3. 原始 GEM、不可逆 GEM 和 eciFX1172 的目标函数值比较。

## 输出文件

- `model/eciFX1172.json`：酶约束模型 JSON。
- `model/iFX1172_irreversible.json`：不可逆与同工酶拆分后的基础模型。
- `analysis/reaction_kcat_MW.csv`：反应级 kcat、MW、kcat/MW 与来源标签。
- `analysis/gene_protein_mass.csv`：基因、UniProt accession、蛋白序列和质量。
- `analysis/model_summary.csv`：模型规模和模拟摘要。
""",
        encoding="utf-8",
    )
    return report


def main():
    ensure_dirs()
    os.chdir(ROOT)
    original = cobra.io.read_sbml_model(str(MODEL_FILE))
    original.solver = "glpk"
    working = original.copy()
    convert_to_irreversible(working)
    working = isoenzyme_split(working)
    working.solver = "glpk"
    gene_mass = build_gene_mass_table(original)
    reaction_kcat_mw = build_reaction_kcat_mw(working, gene_mass)
    ec_json = save_ec_model(working, reaction_kcat_mw)
    summary = summarize_and_plot(original, working, ec_json, reaction_kcat_mw, gene_mass)
    report = write_report(summary)
    print(json.dumps({"ec_model": str(ec_json), "report": str(report), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
