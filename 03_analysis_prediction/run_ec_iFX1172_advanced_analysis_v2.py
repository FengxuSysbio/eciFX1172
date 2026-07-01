import json
import math
import random
import re
from pathlib import Path

import numpy as np

if not hasattr(np, "object"):
    np.object = object

import cobra
import matplotlib.pyplot as plt
import pandas as pd
from cobra.flux_analysis import flux_variability_analysis, moma
from cobra.manipulation import delete_model_genes

cobra.Configuration().solver = "glpk"

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "ec_iFX1172_final_calibrated"
MODEL_DIR = BASE / "model"
EC_JSON = MODEL_DIR / "eciFX1172.json"
GEM_JSON = MODEL_DIR / "iFX1172_irreversible.json"
OLD_ADV = BASE / "advanced_analysis" / "tables"
OUT = BASE / "advanced_analysis_v2"
TABLE_DIR = OUT / "tables"
FIG_DIR = OUT / "figures"
REPORT_DIR = OUT / "report"

BIOMASS_RXN = "growth"
PRODUCT_RXN = "r_0013"
PRODUCT_NAME = "Gentamicin A"
GLUCOSE_RXN = "EX_glc__D_e_reverse"
EPS = 1e-9

CARBON_PANEL = [
    ("D-Glucose", ["glc__D_c", "glc__D_e", "D-Glucose"]),
    ("D-Fructose", ["fru_c", "fru_e", "D-Fructose"]),
    ("D-Galactose", ["gal_c", "gal_e", "D-Galactose"]),
    ("D-Mannose", ["man_c", "man_e", "D-Mannose"]),
    ("D-Xylose", ["xyl__D_c", "xyl__D_e", "D-Xylose"]),
    ("L-Arabinose", ["arab__L_c", "arab__L_e", "L-Arabinose"]),
    ("Sucrose", ["sucr_c", "sucr_e", "Sucrose"]),
    ("Maltose", ["malt_c", "malt_e", "Maltose"]),
    ("Lactose", ["lcts_c", "lcts_e", "Lactose"]),
    ("Glycerol", ["glyc_c", "glyc_e", "Glycerol"]),
    ("Acetate", ["ac_c", "ac_e", "Acetate"]),
    ("Pyruvate", ["pyr_c", "pyr_e", "Pyruvate"]),
    ("Succinate", ["succ_c", "succ_e", "Succinate"]),
    ("Fumarate", ["fum_c", "fum_e", "Fumarate"]),
    ("L-Malate", ["mal__L_c", "mal__L_e", "L-Malate"]),
    ("Citrate", ["cit_c", "cit_e", "Citrate"]),
    ("2-Oxoglutarate", ["akg_c", "akg_e", "2-Oxoglutarate"]),
    ("Lactate", ["lac__D_c", "lac__L_c", "lac__D_e", "lac__L_e", "Lactate"]),
    ("Ethanol", ["etoh_c", "etoh_e", "Ethanol"]),
    ("Formate", ["for_c", "for_e", "Formate"]),
    ("Gluconate", ["glcn_c", "glcn_e", "Gluconate"]),
    ("Ribose", ["rib__D_c", "rib__D_e", "Ribose"]),
    ("Mannitol", ["mnl_c", "mnl_e", "Mannitol"]),
    ("Sorbitol", ["sbt__D_c", "sbt__D_e", "Sorbitol"]),
]

AMINO_PANEL = [
    ("L-Alanine", ["ala__L_c", "ala__L_e", "L-Alanine"]),
    ("L-Arginine", ["arg__L_c", "arg__L_e", "L-Arginine"]),
    ("L-Asparagine", ["asn__L_c", "asn__L_e", "L-Asparagine"]),
    ("L-Aspartate", ["asp__L_c", "asp__L_e", "L-Aspartate"]),
    ("L-Cysteine", ["cys__L_c", "cys__L_e", "L-Cysteine"]),
    ("L-Glutamate", ["glu__L_c", "glu__L_e", "L-Glutamate"]),
    ("L-Glutamine", ["gln__L_c", "gln__L_e", "L-Glutamine"]),
    ("Glycine", ["gly_c", "gly_e", "Glycine"]),
    ("L-Histidine", ["his__L_c", "his__L_e", "L-Histidine"]),
    ("L-Isoleucine", ["ile__L_c", "ile__L_e", "L-Isoleucine"]),
    ("L-Leucine", ["leu__L_c", "leu__L_e", "L-Leucine"]),
    ("L-Lysine", ["lys__L_c", "lys__L_e", "L-Lysine"]),
    ("L-Methionine", ["met__L_c", "met__L_e", "L-Methionine"]),
    ("L-Phenylalanine", ["phe__L_c", "phe__L_e", "L-Phenylalanine"]),
    ("L-Proline", ["pro__L_c", "pro__L_e", "L-Proline"]),
    ("L-Serine", ["ser__L_c", "ser__L_e", "L-Serine"]),
    ("L-Threonine", ["thr__L_c", "thr__L_e", "L-Threonine"]),
    ("L-Tryptophan", ["trp__L_c", "trp__L_e", "L-Tryptophan"]),
    ("L-Tyrosine", ["tyr__L_c", "tyr__L_e", "L-Tyrosine"]),
    ("L-Valine", ["val__L_c", "val__L_e", "L-Valine"]),
]

KEY_INTRACELLULAR = [
    ("akg_c", "2-oxoglutarate"),
    ("glu__L_c", "L-glutamate"),
    ("amet_c", "S-adenosyl-L-methionine"),
    ("m_0008", "Gentamicin A2"),
    ("m_0011", "Gentamicin A"),
]


def ensure_dirs():
    for folder in [OUT, TABLE_DIR, FIG_DIR, REPORT_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def load_raw(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_gem():
    model = cobra.io.load_json_model(str(GEM_JSON))
    model.solver = "glpk"
    return model


def load_ec():
    data = load_raw(EC_JSON)
    model = cobra.io.load_json_model(str(EC_JSON))
    model.solver = "glpk"
    coefficients = {}
    by_id = {r["id"]: r.get("kcat_MW") for r in data["reactions"]}
    for reaction in model.reactions:
        value = by_id.get(reaction.id)
        if value not in ("", None):
            try:
                coefficients[reaction.forward_variable] = 1.0 / float(value)
            except Exception:
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


def obj_value(model, objective=None):
    if objective is not None:
        model.objective = model.reactions.get_by_id(objective)
    try:
        value = model.slim_optimize(error_value=np.nan)
    except Exception:
        value = np.nan
    return float(value) if value is not None else np.nan


def solution(model, objective=None):
    if objective is not None:
        model.objective = model.reactions.get_by_id(objective)
    try:
        sol = model.optimize()
        return sol if sol.status == "optimal" else None
    except Exception:
        return None


def find_metabolite(model, tokens):
    lower = [(m.id.lower(), (m.name or "").lower(), m) for m in model.metabolites]
    for token in tokens:
        t = token.lower()
        for mid, _, met in lower:
            if mid == t:
                return met
    for token in tokens:
        t = token.lower()
        for mid, name, met in lower:
            if t == name or t in name:
                return met
    return None


def add_source(model, met, uptake=10.0, prefix="SRC"):
    rid = f"{prefix}_{re.sub('[^A-Za-z0-9_]', '_', met.id)}"
    if rid in model.reactions:
        reaction = model.reactions.get_by_id(rid)
    else:
        reaction = cobra.Reaction(rid)
        reaction.name = f"Temporary uptake/source for {met.name or met.id}"
        reaction.lower_bound = 0.0
        reaction.upper_bound = float(uptake)
        reaction.add_metabolites({met: 1.0})
        model.add_reactions([reaction])
    reaction.upper_bound = float(uptake)
    return reaction


def set_glucose(model, uptake):
    if GLUCOSE_RXN in model.reactions:
        model.reactions.get_by_id(GLUCOSE_RXN).upper_bound = float(uptake)


def product_max(model, growth_fraction=0.1):
    wt = obj_value(model, BIOMASS_RXN)
    m = model.copy()
    m.reactions.get_by_id(BIOMASS_RXN).lower_bound = max(m.reactions.get_by_id(BIOMASS_RXN).lower_bound, growth_fraction * wt)
    return wt, obj_value(m, PRODUCT_RXN)


def pathway_distribution(gem, ec):
    rows = []
    raw = load_raw(EC_JSON)
    constrained = {r["id"] for r in raw["reactions"] if r.get("kcat_MW") not in ("", None)}
    for label, model in [("iFX1172", gem), ("eciFX1172", ec)]:
        for reaction in model.reactions:
            subsystem = str(reaction.subsystem or "Unassigned")
            rows.append(
                {
                    "model": label,
                    "subsystem": subsystem,
                    "reaction": reaction.id,
                    "has_gene": bool(reaction.genes),
                    "enzyme_constrained": reaction.id in constrained if label == "eciFX1172" else False,
                }
            )
    table = pd.DataFrame(rows)
    summary = (
        table.groupby(["model", "subsystem"], as_index=False)
        .agg(reaction_count=("reaction", "count"), gene_associated_count=("has_gene", "sum"), enzyme_constrained_count=("enzyme_constrained", "sum"))
        .sort_values(["model", "reaction_count"], ascending=[True, False])
    )
    return table, summary


def memote_like_qc(gem, ec):
    rows = []
    for label, model in [("iFX1172", gem), ("eciFX1172", ec)]:
        mets_with_formula = sum(bool(m.formula) for m in model.metabolites)
        mets_with_charge = sum(m.charge is not None for m in model.metabolites)
        rxns_with_gpr = sum(bool(r.gene_reaction_rule) for r in model.reactions)
        blocked_sample = np.nan
        sample_ids = [r.id for r in list(model.reactions)[:300]]
        try:
            fva = flux_variability_analysis(model, reaction_list=sample_ids, fraction_of_optimum=0.0, processes=1)
            blocked_sample = int(((fva["minimum"].abs() < 1e-9) & (fva["maximum"].abs() < 1e-9)).sum())
        except Exception:
            pass
        rows.append(
            {
                "model": label,
                "reactions": len(model.reactions),
                "metabolites": len(model.metabolites),
                "genes": len(model.genes),
                "growth": obj_value(model, BIOMASS_RXN),
                "formula_coverage": mets_with_formula / max(len(model.metabolites), 1),
                "charge_coverage": mets_with_charge / max(len(model.metabolites), 1),
                "gpr_coverage": rxns_with_gpr / max(len(model.reactions), 1),
                "blocked_reactions_fva0_first300": blocked_sample,
                "memote_note": "MEMOTE package is installed; CLI requires numpy.object monkey patch in this environment. These are MEMOTE-style structural QC metrics plus FVA blocked-reaction screening.",
            }
        )
    return pd.DataFrame(rows)


def substrate_panel(gem, ec):
    rows = []
    for panel_name, panel, mode in [("carbon", CARBON_PANEL, "sole_carbon"), ("amino_acid", AMINO_PANEL, "glucose_supplement")]:
        for label, tokens in panel:
            met_ec = find_metabolite(ec, tokens)
            met_gem = find_metabolite(gem, tokens)
            for model_label, model0, met in [("iFX1172", gem, met_gem), ("eciFX1172", ec, met_ec)]:
                if met is None:
                    rows.append({"panel": panel_name, "substrate": label, "model": model_label, "metabolite_id": "", "mode": mode, "growth": np.nan, "product_max_10pct_growth": np.nan, "status": "metabolite_not_found"})
                    continue
                model = model0.copy()
                if panel_name == "carbon":
                    set_glucose(model, 0.0)
                    add_source(model, met, uptake=10.0, prefix="SRC_C")
                else:
                    set_glucose(model, 0.8)
                    add_source(model, met, uptake=5.0, prefix="SRC_AA")
                growth = obj_value(model, BIOMASS_RXN)
                _, pmax = product_max(model, growth_fraction=0.1) if growth > EPS else (growth, np.nan)
                rows.append({"panel": panel_name, "substrate": label, "model": model_label, "metabolite_id": met.id, "mode": mode, "growth": growth, "product_max_10pct_growth": pmax, "status": "ok" if growth > EPS else "no_growth"})
    return pd.DataFrame(rows)


def robustness_and_phase_plane(gem, ec):
    rob_rows = []
    for uptake in np.linspace(0, 2.0, 21):
        for label, model0 in [("iFX1172", gem), ("eciFX1172", ec)]:
            model = model0.copy()
            set_glucose(model, uptake)
            growth = obj_value(model, BIOMASS_RXN)
            _, pmax = product_max(model, 0.1) if growth > EPS else (growth, np.nan)
            rob_rows.append({"model": label, "glucose_uptake": uptake, "growth": growth, "product_max_10pct_growth": pmax})
    phase_rows = []
    ec_raw = load_raw(EC_JSON)
    pool_ub = float(ec_raw["enzyme_constraint"]["upperbound"])
    for uptake in np.linspace(0, 2.0, 13):
        for pool_factor in np.linspace(0.4, 1.4, 11):
            model = ec.copy()
            set_glucose(model, uptake)
            model.solver.constraints["enzyme_pool"].ub = pool_ub * pool_factor
            phase_rows.append({"glucose_uptake": uptake, "enzyme_pool_factor": pool_factor, "growth": obj_value(model, BIOMASS_RXN)})
    product_phase = []
    wt, pmax = product_max(ec, 0.1)
    for growth_frac in np.linspace(0.05, 0.9, 12):
        for product_frac in np.linspace(0.0, 0.9, 10):
            model = ec.copy()
            model.reactions.get_by_id(BIOMASS_RXN).lower_bound = wt * growth_frac
            model.reactions.get_by_id(PRODUCT_RXN).lower_bound = pmax * product_frac
            value = obj_value(model, BIOMASS_RXN)
            product_phase.append({"growth_fraction_constraint": growth_frac, "product_fraction_constraint": product_frac, "feasible_growth": value, "feasible": not np.isnan(value)})
    return pd.DataFrame(rob_rows), pd.DataFrame(phase_rows), pd.DataFrame(product_phase)


def fseof(ec):
    wt, pmax = product_max(ec, 0.1)
    rows = []
    for level in np.linspace(0.1, 0.9, 9) * pmax:
        model = ec.copy()
        model.reactions.get_by_id(BIOMASS_RXN).lower_bound = 0.1 * wt
        model.reactions.get_by_id(PRODUCT_RXN).lower_bound = level
        sol = solution(model, BIOMASS_RXN)
        if sol is None:
            continue
        for reaction in model.reactions:
            flux = float(sol.fluxes.get(reaction.id, 0.0))
            if reaction.genes and abs(flux) > 1e-8:
                rows.append({"product_forced": level, "reaction": reaction.id, "reaction_name": reaction.name, "genes": ";".join(sorted(g.id for g in reaction.genes)), "flux": flux})
    flux = pd.DataFrame(rows)
    scored = []
    for rid, group in flux.groupby("reaction"):
        if group["product_forced"].nunique() < 3:
            continue
        x, y = group["product_forced"].to_numpy(), group["flux"].to_numpy()
        slope = float(np.polyfit(x, y, 1)[0])
        scored.append({"reaction": rid, "reaction_name": group["reaction_name"].iloc[0], "genes": group["genes"].iloc[0], "slope": slope, "abs_slope": abs(slope), "recommendation": "overexpression" if slope > 1e-6 else "downregulation" if slope < -1e-6 else "neutral"})
    rxn_score = pd.DataFrame(scored).sort_values("abs_slope", ascending=False)
    gene_rows = []
    for row in rxn_score.to_dict("records"):
        for gene in str(row["genes"]).split(";"):
            if gene:
                gene_rows.append({"gene": gene, "recommendation": row["recommendation"], "score": row["abs_slope"], "mean_slope": row["slope"], "reaction": row["reaction"]})
    gene_score = pd.DataFrame(gene_rows)
    if not gene_score.empty:
        gene_score = (
            gene_score.groupby(["gene", "recommendation"], as_index=False)
            .agg(fseof_score=("score", "sum"), mean_slope=("mean_slope", "mean"), representative_reactions=("reaction", lambda x: ";".join(list(x)[:8])))
            .sort_values("fseof_score", ascending=False)
        )
    return flux, rxn_score, gene_score


def load_or_single_ko(ec):
    old = OLD_ADV / "single_gene_ko.csv"
    if old.exists():
        return pd.read_csv(old)
    wt = obj_value(ec, BIOMASS_RXN)
    rows = []
    for gene in sorted(ec.genes, key=lambda g: g.id):
        model = ec.copy()
        delete_model_genes(model, [gene.id])
        growth = obj_value(model, BIOMASS_RXN)
        rows.append({"gene": gene.id, "ec_growth": growth, "growth_ratio": growth / wt if wt else np.nan, "phenotype": "essential" if growth < 0.01 * wt else "growth_limited" if growth < 0.8 * wt else "nonessential"})
    return pd.DataFrame(rows).sort_values("growth_ratio")


def double_ko(ec, single, fseof_genes):
    candidates = list(dict.fromkeys(list(fseof_genes.head(18)["gene"]) + list(single[single["growth_ratio"].between(0.05, 0.98)].head(18)["gene"])))[:28]
    wt = obj_value(ec, BIOMASS_RXN)
    sratio = single.set_index("gene")["growth_ratio"].to_dict()
    rows = []
    for i, a in enumerate(candidates):
        for b in candidates[i + 1 :]:
            if a not in ec.genes or b not in ec.genes:
                continue
            model = ec.copy()
            delete_model_genes(model, [a, b])
            growth = obj_value(model, BIOMASS_RXN)
            ratio = growth / wt if wt else np.nan
            expected = min(sratio.get(a, 1.0), sratio.get(b, 1.0))
            rows.append({"gene_a": a, "gene_b": b, "growth": growth, "growth_ratio": ratio, "min_single_ratio": expected, "interaction_score": expected - ratio, "phenotype": "synthetic_lethal" if ratio < 0.01 and expected > 0.1 else "synthetic_sick" if expected - ratio > 0.2 else "other"})
    return pd.DataFrame(rows).sort_values(["interaction_score", "growth_ratio"], ascending=[False, True])


def dfba_multi(ec):
    dt, t_end = 0.5, 30.0
    biomass, glucose, nh4, o2, pi = 0.05, 12.0, 30.0, 80.0, 20.0
    rows, proxy_rows = [], []
    for step in range(int(t_end / dt) + 1):
        t = step * dt
        model = ec.copy()
        set_glucose(model, min(10.0, glucose / max(biomass * dt, 1e-12)))
        for rid, amount, vmax in [("EX_nh4_e_reverse", nh4, 20.0), ("EX_o2_e_reverse", o2, 30.0), ("EX_pi_e_reverse", pi, 10.0)]:
            if rid in model.reactions:
                model.reactions.get_by_id(rid).upper_bound = min(vmax, amount / max(biomass * dt, 1e-12))
        sol = solution(model, BIOMASS_RXN)
        if sol is None:
            mu = qg = qn = qo = qp = prod = 0.0
        else:
            mu = max(float(sol.objective_value), 0.0)
            qg = max(float(sol.fluxes.get(GLUCOSE_RXN, 0.0)), 0.0)
            qn = max(float(sol.fluxes.get("EX_nh4_e_reverse", 0.0)), 0.0)
            qo = max(float(sol.fluxes.get("EX_o2_e_reverse", 0.0)), 0.0)
            qp = max(float(sol.fluxes.get("EX_pi_e_reverse", 0.0)), 0.0)
            prod = max(float(sol.fluxes.get(PRODUCT_RXN, 0.0)), 0.0)
        rows.append({"time_h": t, "biomass_gDW_L": biomass, "glucose_mmol_L": glucose, "nh4_mmol_L": nh4, "oxygen_mmol_L": o2, "phosphate_mmol_L": pi, "growth_rate_h-1": mu, "glucose_uptake": qg, "product_flux": prod})
        for met_id, met_name in KEY_INTRACELLULAR:
            if met_id not in model.metabolites:
                continue
            turnover = 0.0
            if sol is not None:
                for reaction in model.metabolites.get_by_id(met_id).reactions:
                    turnover += abs(float(reaction.get_coefficient(met_id)) * float(sol.fluxes.get(reaction.id, 0.0)))
            proxy_rows.append({"time_h": t, "metabolite": met_name, "metabolite_id": met_id, "synthesis_capacity_proxy": turnover})
        biomass = biomass + mu * biomass * dt
        glucose = max(glucose - qg * biomass * dt, 0.0)
        nh4 = max(nh4 - qn * biomass * dt, 0.0)
        o2 = max(o2 - qo * biomass * dt, 0.0)
        pi = max(pi - qp * biomass * dt, 0.0)
    return pd.DataFrame(rows), pd.DataFrame(proxy_rows)


def target_algorithms(ec, single, fseof_genes):
    wt_growth, wt_pmax = product_max(ec, 0.1)
    ref = solution(ec, BIOMASS_RXN)
    candidates = list(dict.fromkeys(list(fseof_genes.head(35)["gene"]) + list(single[single["growth_ratio"] > 0.2].head(25)["gene"])))[:45]
    rows = []
    for gene in candidates:
        if gene not in ec.genes:
            continue
        model = ec.copy()
        delete_model_genes(model, [gene])
        growth = obj_value(model, BIOMASS_RXN)
        pmax = np.nan
        if growth > 0.05 * wt_growth:
            _, pmax = product_max(model, 0.1)
        moma_growth = np.nan
        if ref is not None and len(rows) < 24:
            m2 = ec.copy()
            delete_model_genes(m2, [gene])
            try:
                moma_sol = moma(m2, solution=ref, linear=True)
                moma_growth = float(moma_sol.fluxes.get(BIOMASS_RXN, np.nan))
            except Exception:
                pass
        rows.append({"gene": gene, "algorithm": "OptKnock-like", "operation": "KO", "growth": growth, "product_max": pmax, "score": (pmax - wt_pmax) if not np.isnan(pmax) else -999})
        moma_algorithm = "MOMA" if not np.isnan(moma_growth) else "MOMA-fallback"
        moma_growth_value = moma_growth if not np.isnan(moma_growth) else growth
        moma_score = (pmax / max(wt_pmax, EPS)) * max(moma_growth_value, 0) / max(wt_growth, EPS) if not np.isnan(moma_growth_value) and not np.isnan(pmax) else np.nan
        rows.append({"gene": gene, "algorithm": moma_algorithm, "operation": "KO", "growth": moma_growth_value, "product_max": pmax, "score": moma_score})
    # OptForce-like: compare WT flux and product-forced flux.
    forced = ec.copy()
    forced.reactions.get_by_id(BIOMASS_RXN).lower_bound = 0.1 * wt_growth
    forced.reactions.get_by_id(PRODUCT_RXN).lower_bound = 0.75 * wt_pmax
    sol_wt, sol_forced = solution(ec, BIOMASS_RXN), solution(forced, BIOMASS_RXN)
    if sol_wt is not None and sol_forced is not None:
        for reaction in ec.reactions:
            if not reaction.genes:
                continue
            delta = float(sol_forced.fluxes.get(reaction.id, 0.0) - sol_wt.fluxes.get(reaction.id, 0.0))
            if abs(delta) > 1e-6:
                for gene in reaction.genes:
                    rows.append({"gene": gene.id, "algorithm": "OptForce-like", "operation": "UP" if delta > 0 else "DOWN", "growth": wt_growth, "product_max": wt_pmax, "score": abs(delta), "reaction": reaction.id})
    # OptGene-like random one/two gene edit scoring from candidate pool.
    rng = random.Random(1172)
    pool = [g for g in candidates if g in ec.genes]
    for _ in range(min(80, max(0, len(pool) * 3))):
        genes = rng.sample(pool, k=1 if len(pool) < 2 or rng.random() < 0.65 else 2)
        model = ec.copy()
        delete_model_genes(model, genes)
        growth = obj_value(model, BIOMASS_RXN)
        pmax = np.nan
        if growth > 0.05 * wt_growth:
            _, pmax = product_max(model, 0.1)
        rows.append({"gene": ";".join(genes), "algorithm": "OptGene-like", "operation": "KO_set", "growth": growth, "product_max": pmax, "score": (pmax / max(wt_pmax, EPS)) * (growth / max(wt_growth, EPS)) if not np.isnan(pmax) else -999})
    table = pd.DataFrame(rows).sort_values(["algorithm", "score"], ascending=[True, False])
    meta = fseof_genes.copy()
    if not meta.empty:
        viability = single.set_index("gene")["growth_ratio"].to_dict()
        meta["recommended_operation"] = meta["recommendation"].map({"overexpression": "OE", "downregulation": "KD/KO"}).fillna("neutral")
        meta["operation_code"] = meta["recommended_operation"].map({"neutral": 0, "OE": 1, "KD/KO": 3}).fillna(0).astype(int)
        meta["single_KO_growth_ratio"] = meta["gene"].map(viability)
        meta["metastrain_style_score"] = meta["fseof_score"] * meta["single_KO_growth_ratio"].fillna(1.0).clip(0, 1.0)
        meta = meta.sort_values("metastrain_style_score", ascending=False)
    return table, meta


def save(name, table):
    path = TABLE_DIR / f"{name}.csv"
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def plot_all(tables):
    plt.style.use("default")
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["axes.unicode_minus"] = False

    path = tables["pathway_summary"].copy()
    top_subsystems = path.groupby("subsystem")["reaction_count"].sum().sort_values(ascending=False).head(18).index
    fig, ax = plt.subplots(figsize=(9.2, 6.2), dpi=220)
    pivot = path[path["subsystem"].isin(top_subsystems)].pivot(index="subsystem", columns="model", values="reaction_count").fillna(0).loc[top_subsystems[::-1]]
    y = np.arange(len(pivot))
    ax.barh(y - 0.18, pivot.get("iFX1172", pd.Series(0, index=pivot.index)), height=0.35, color="#666666", label="iFX1172")
    ax.barh(y + 0.18, pivot.get("eciFX1172", pd.Series(0, index=pivot.index)), height=0.35, color="#287D8E", label="eciFX1172")
    ax.set_yticks(y); ax.set_yticklabels(pivot.index)
    ax.set_title("Pathway distribution")
    ax.set_xlabel("Reaction count")
    ax.set_ylabel("")
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure1_pathway_distribution.png"); plt.close(fig)

    qc = tables["memote_qc"]
    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=220)
    qc_m = qc.melt(id_vars="model", value_vars=["formula_coverage", "charge_coverage", "gpr_coverage"], var_name="metric", value_name="coverage")
    metrics = ["formula_coverage", "charge_coverage", "gpr_coverage"]
    x = np.arange(len(metrics))
    for offset, (model_label, color) in zip([-0.18, 0.18], [("iFX1172", "#777777"), ("eciFX1172", "#C23B22")]):
        vals = [float(qc_m[(qc_m["model"] == model_label) & (qc_m["metric"] == m)]["coverage"].iloc[0]) for m in metrics]
        ax.bar(x + offset, vals, width=0.35, label=model_label, color=color)
    ax.set_xticks(x); ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylim(0, 1); ax.set_title("MEMOTE-style annotation coverage"); fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure2_memote_qc.png"); plt.close(fig)

    sub = tables["substrate_panel"]
    for panel, filename in [("carbon", "v2_figure3_carbon_sources.png"), ("amino_acid", "v2_figure4_amino_acids.png")]:
        data = sub[sub["panel"] == panel].copy()
        fig, ax = plt.subplots(figsize=(9.4, 6.6), dpi=220)
        pivot = data.pivot(index="substrate", columns="model", values="growth").fillna(0)
        order = pivot.max(axis=1).sort_values().index
        pivot = pivot.loc[order]
        y = np.arange(len(pivot))
        ax.barh(y - 0.18, pivot.get("iFX1172", pd.Series(0, index=pivot.index)), height=0.35, color="#777777", label="iFX1172")
        ax.barh(y + 0.18, pivot.get("eciFX1172", pd.Series(0, index=pivot.index)), height=0.35, color="#287D8E", label="eciFX1172")
        ax.set_yticks(y); ax.set_yticklabels(pivot.index, fontsize=8); ax.legend(frameon=False)
        ax.set_title("Carbon source utilization" if panel == "carbon" else "Amino acid supplementation")
        ax.set_xlabel("Predicted growth rate (h$^{-1}$)")
        ax.set_ylabel("")
        fig.tight_layout(); fig.savefig(FIG_DIR / filename); plt.close(fig)

    rob = tables["robustness"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4), dpi=220)
    for label, color in [("iFX1172", "#777777"), ("eciFX1172", "#287D8E")]:
        d = rob[rob["model"] == label]
        ax.plot(d["glucose_uptake"], d["growth"], label=label, lw=2, color=color)
    ax.legend(frameon=False)
    ax.set_title("Glucose robustness: GEM vs ecGEM"); ax.set_xlabel("Glucose uptake bound"); ax.set_ylabel("Growth")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure5_robustness_growth.png"); plt.close(fig)

    phase = tables["phase_plane"]
    pivot = phase.pivot(index="enzyme_pool_factor", columns="glucose_uptake", values="growth")
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=220)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f"{x:.2f}" for x in pivot.index], fontsize=7)
    fig.colorbar(im, ax=ax, label="Growth")
    ax.set_title("Phenotypic phase plane: glucose x enzyme pool")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure6_phase_plane_heatmap.png"); plt.close(fig)

    pphase = tables["product_phase_plane"]
    pivot = pphase.pivot(index="growth_fraction_constraint", columns="product_fraction_constraint", values="feasible_growth")
    fig, ax = plt.subplots(figsize=(7.2, 5.6), dpi=220)
    im = ax.imshow(pivot.values, aspect="auto", origin="lower", cmap="magma")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels([f"{x:.2f}" for x in pivot.columns], rotation=90, fontsize=7)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels([f"{x:.2f}" for x in pivot.index], fontsize=7)
    fig.colorbar(im, ax=ax, label="Feasible growth")
    ax.set_title("Growth-product phase plane")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure7_growth_product_phase.png"); plt.close(fig)

    single = tables["single_gene_ko"]
    fig, ax = plt.subplots(figsize=(6.5, 4.3), dpi=220)
    ax.hist(single["growth_ratio"].clip(0, 1.2), bins=35, color="#5B6C8F")
    ax.set_title("Single-gene knockout distribution"); ax.set_xlabel("Growth ratio"); ax.set_ylabel("Gene count")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure8_single_ko.png"); plt.close(fig)

    double = tables["double_gene_ko"].head(20).copy()
    if not double.empty:
        double["pair"] = double["gene_a"] + " + " + double["gene_b"]
        fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=220)
        d = double.sort_values("interaction_score")
        ax.barh(d["pair"], d["interaction_score"], color="#8C4B3E")
        ax.set_title("Double-gene knockout interactions")
        fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure9_double_ko.png"); plt.close(fig)

    dfba = tables["dfba"]
    fig, ax1 = plt.subplots(figsize=(7.5, 4.8), dpi=220)
    ax1.plot(dfba["time_h"], dfba["biomass_gDW_L"], color="#1D4E89", label="Biomass", lw=2)
    ax1.set_ylabel("Biomass")
    ax2 = ax1.twinx()
    for col, color in [("glucose_mmol_L", "#B35C00"), ("nh4_mmol_L", "#2F6F4E"), ("oxygen_mmol_L", "#6B4E9B"), ("phosphate_mmol_L", "#C23B22")]:
        ax2.plot(dfba["time_h"], dfba[col], label=col.replace("_mmol_L", ""), lw=1.6, color=color)
    ax1.set_xlabel("Time (h)"); ax2.set_ylabel("Extracellular substrate (mmol/L)")
    ax2.legend(frameon=False, fontsize=8); ax1.set_title("dFBA multi-substrate dynamics")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure10_dfba_multi.png"); plt.close(fig)

    proxy = tables["dfba_intracellular_proxy"]
    fig, ax = plt.subplots(figsize=(7.5, 5.0), dpi=220)
    for met, group in proxy.groupby("metabolite"):
        ax.plot(group["time_h"], group["synthesis_capacity_proxy"], label=met, lw=2)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Intracellular key-metabolite synthesis-capacity proxies")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure11_intracellular_proxy.png"); plt.close(fig)

    fsg = tables["fseof_gene_targets"].head(20).copy()
    fig, ax = plt.subplots(figsize=(7.5, 6.2), dpi=220)
    d = fsg.sort_values("fseof_score")
    colors = ["#287D8E" if r == "overexpression" else "#C23B22" for r in d["recommendation"]]
    ax.barh(d["gene"], d["fseof_score"], color=colors)
    ax.set_title("FSEOF targets"); fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure12_fseof.png"); plt.close(fig)

    target = tables["target_algorithms"].groupby(["algorithm", "gene"], as_index=False)["score"].max()
    top = target.sort_values("score", ascending=False).head(30)
    fig, ax = plt.subplots(figsize=(8.2, 6.6), dpi=220)
    algs = list(top["algorithm"].unique())
    genes = list(top["gene"].unique())
    x = [algs.index(a) for a in top["algorithm"]]
    y = [genes.index(g) for g in top["gene"]]
    sizes = 40 + 210 * (top["score"] - top["score"].min()) / max(top["score"].max() - top["score"].min(), EPS)
    sc = ax.scatter(x, y, s=sizes, c=top["score"], cmap="rocket_r" if "rocket_r" in plt.colormaps() else "plasma", alpha=0.85)
    ax.set_xticks(range(len(algs))); ax.set_xticklabels(algs, rotation=20, ha="right")
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=8)
    fig.colorbar(sc, ax=ax, label="Score")
    ax.set_title("OptKnock/OptForce/MOMA/OptGene-like target scores")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure13_algorithm_targets.png"); plt.close(fig)

    meta = tables["metastrain_targets"].head(20).copy()
    fig, ax = plt.subplots(figsize=(7.6, 6.0), dpi=220)
    d = meta.sort_values("metastrain_style_score")
    colors = ["#287D8E" if op == "OE" else "#C23B22" for op in d["recommended_operation"]]
    ax.barh(d["gene"], d["metastrain_style_score"], color=colors)
    ax.set_title("MetaStrain-style targets")
    fig.tight_layout(); fig.savefig(FIG_DIR / "v2_figure14_metastrain.png"); plt.close(fig)


def write_excel(tables):
    path = TABLE_DIR / "eciFX1172_advanced_analysis_v2_results.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in tables.items():
            table.to_excel(writer, sheet_name=name[:31], index=False)
    return path


def md_table(df, n=12):
    if df is None or df.empty:
        return "无记录。"
    x = df.head(n).copy()
    for col in x.columns:
        x[col] = x[col].map(lambda v: "" if pd.isna(v) else str(v).replace("|", "/"))
    return "\n".join(["| " + " | ".join(x.columns) + " |", "| " + " | ".join(["---"] * len(x.columns)) + " |"] + ["| " + " | ".join(row) + " |" for row in x.to_numpy()])


def write_report(tables, excel_path):
    figs = {p.stem: p.resolve().as_posix() for p in sorted(FIG_DIR.glob("*.png"))}
    qc = tables["memote_qc"]
    sub = tables["substrate_panel"]
    single = tables["single_gene_ko"]
    text = f"""# eciFX1172 高级分析与预测 v2

## 分析内容

本报告在最终校准的 eciFX1172 基础上，补充模型途径分布、MEMOTE-style 质量检查、20+ 碳源面板、20 种氨基酸补充面板、鲁棒性分析、二维相平面、单/双基因敲除、dFBA 多物质动态、FSEOF、OptKnock-like、OptForce-like、MOMA、OptGene-like 与 MetaStrain-style 靶点分析。原始模型使用不可逆化的 iFX1172，ec 模型使用总酶池约束后的 eciFX1172。

## 关键结论

- eciFX1172 默认生长速率为 {float(qc.loc[qc['model']=='eciFX1172','growth'].iloc[0]):.6f} h^-1，低于 iFX1172 的 {float(qc.loc[qc['model']=='iFX1172','growth'].iloc[0]):.6f} h^-1。
- 底物面板包含 {sub[sub['panel']=='carbon']['substrate'].nunique()} 种碳源和 {sub[sub['panel']=='amino_acid']['substrate'].nunique()} 种氨基酸；不存在于模型中的底物在表中标记为 `metabolite_not_found`，存在者通过临时 source 反应评估潜在利用能力。
- 单基因敲除覆盖 {len(single)} 个基因，其中 essential 基因 {int((single['phenotype']=='essential').sum())} 个。
- dFBA 同时追踪葡萄糖、铵、氧、磷酸盐、生物量，并以合成能力代理指标展示 2-oxoglutarate、glutamate、SAM 和庆大霉素中间体等胞内关键物质的动态变化。
- 由于当前环境未安装 cameo，OptKnock/OptGene 使用可复现的 COBRApy 近似实现；MEMOTE 命令行入口存在 NumPy/Cobra 兼容问题，因此报告给出 MEMOTE-style 结构质量指标和 FVA 阻塞反应筛查。

## 图件

![Pathway distribution]({figs['v2_figure1_pathway_distribution']})
![MEMOTE QC]({figs['v2_figure2_memote_qc']})
![Carbon sources]({figs['v2_figure3_carbon_sources']})
![Amino acids]({figs['v2_figure4_amino_acids']})
![Robustness]({figs['v2_figure5_robustness_growth']})
![Phase plane]({figs['v2_figure6_phase_plane_heatmap']})
![Growth product phase]({figs['v2_figure7_growth_product_phase']})
![Single KO]({figs['v2_figure8_single_ko']})
![Double KO]({figs.get('v2_figure9_double_ko','')})
![dFBA multi]({figs['v2_figure10_dfba_multi']})
![Intracellular proxy]({figs['v2_figure11_intracellular_proxy']})
![FSEOF]({figs['v2_figure12_fseof']})
![Algorithm targets]({figs['v2_figure13_algorithm_targets']})
![MetaStrain]({figs['v2_figure14_metastrain']})

## 主要结果表

### MEMOTE-style QC
{md_table(tables['memote_qc'], 10)}

### FSEOF 靶点
{md_table(tables['fseof_gene_targets'][['gene','recommendation','fseof_score','mean_slope','representative_reactions']], 15)}

### OptKnock/OptForce/MOMA/OptGene-like 靶点
{md_table(tables['target_algorithms'][['algorithm','gene','operation','growth','product_max','score']].sort_values('score', ascending=False), 20)}

### MetaStrain-style 靶点
{md_table(tables['metastrain_targets'][['gene','recommended_operation','operation_code','single_KO_growth_ratio','metastrain_style_score','representative_reactions']], 20)}

## 输出文件

Excel 汇总：`{excel_path}`
"""
    md = REPORT_DIR / "eciFX1172_advanced_analysis_v2_report.md"
    md.write_text(text, encoding="utf-8")
    html = REPORT_DIR / "eciFX1172_advanced_analysis_v2_report.html"
    html_lines = []
    in_table = False
    for line in text.splitlines():
        image = re.match(r"!\[(.*?)\]\((.*?)\)", line)
        if image:
            if in_table:
                html_lines.append("</pre>")
                in_table = False
            alt, src = image.groups()
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
        elif not line.strip():
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
    html.write_text(
        "<html><head><meta charset='utf-8'><style>body{font-family:Arial,'Microsoft YaHei',sans-serif;max-width:1100px;margin:30px auto;line-height:1.65;color:#222}"
        "img{max-width:100%;border:1px solid #ddd}figure{margin:24px 0}figcaption{font-size:13px;color:#555}.table{white-space:pre-wrap;background:#fafafa;padding:12px;border:1px solid #ddd;overflow:auto}</style></head><body>"
        + "\n".join(html_lines)
        + "</body></html>",
        encoding="utf-8",
    )
    return md, html


def main():
    ensure_dirs()
    print("Loading models")
    gem, ec = load_gem(), load_ec()
    print("Pathway and QC")
    pathway_detail, pathway_summary = pathway_distribution(gem, ec)
    qc = memote_like_qc(gem, ec)
    save("pathway_detail", pathway_detail); save("pathway_summary", pathway_summary); save("memote_qc", qc)
    gem, ec = load_gem(), load_ec()
    print("Substrate panel")
    sub = substrate_panel(gem, ec)
    save("substrate_panel", sub)
    print("Robustness and phase plane")
    rob, phase, product_phase = robustness_and_phase_plane(gem, ec)
    save("robustness", rob); save("phase_plane", phase); save("product_phase_plane", product_phase)
    print("FSEOF")
    f_flux, f_rxn, f_gene = fseof(ec)
    save("fseof_flux", f_flux); save("fseof_reaction_targets", f_rxn); save("fseof_gene_targets", f_gene)
    print("Single and double KO")
    single = load_or_single_ko(ec)
    double = double_ko(ec, single, f_gene)
    save("single_gene_ko", single); save("double_gene_ko", double)
    print("dFBA")
    dfba, proxy = dfba_multi(ec)
    save("dfba", dfba); save("dfba_intracellular_proxy", proxy)
    print("Targets")
    targets, meta = target_algorithms(ec, single, f_gene)
    save("target_algorithms", targets); save("metastrain_targets", meta)
    tables = {
        "pathway_detail": pathway_detail,
        "pathway_summary": pathway_summary,
        "memote_qc": qc,
        "substrate_panel": sub,
        "robustness": rob,
        "phase_plane": phase,
        "product_phase_plane": product_phase,
        "single_gene_ko": single,
        "double_gene_ko": double,
        "dfba": dfba,
        "dfba_intracellular_proxy": proxy,
        "fseof_flux": f_flux,
        "fseof_reaction_targets": f_rxn,
        "fseof_gene_targets": f_gene,
        "target_algorithms": targets,
        "metastrain_targets": meta,
    }
    for name, table in tables.items():
        save(name, table)
    excel = write_excel(tables)
    print("Plotting")
    plot_all(tables)
    md, html = write_report(tables, excel)
    print(json.dumps({"excel": str(excel), "report_md": str(md), "report_html": str(html), "figures": len(list(FIG_DIR.glob('*.png')))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
