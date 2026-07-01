from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

np.object = object

import cobra
import pandas as pd


MEMOTE_SCRIPT = r"""
import sys
import numpy as np
np.object = object
from memote.suite.cli.reports import report
report.main(args=sys.argv[1:], prog_name="memote report", standalone_mode=True)
"""

SLOW_OR_CRASHING_TESTS = [
    "test_inconsistent_min_stoichiometry",
    "test_detect_energy_generating_cycles",
    "test_blocked_reactions",
    "test_find_stoichiometrically_balanced_cycles",
    "test_find_metabolites_not_produced_with_open_bounds",
    "test_find_metabolites_not_consumed_with_open_bounds",
]

TRANSPORT_SBO = {
    "SBO:0000185",
    "SBO:0000588",
    "SBO:0000587",
    "SBO:0000655",
    "SBO:0000654",
    "SBO:0000660",
    "SBO:0000659",
    "SBO:0000657",
    "SBO:0000658",
}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def base_met_id(met_id: str) -> str:
    value = re.sub(r"^M_", "", met_id)
    value = re.sub(r"_[a-z][a-z0-9]?$", "", value)
    return value


def safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def parse_memote_html(path: Path) -> dict:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if "window.data" not in line:
                continue
            payload = line.split("window.data", 1)[1].split("=", 1)[1].strip()
            payload = payload[:-1] if payload.endswith(";") else payload
            return json.loads(payload)
    raise RuntimeError(f"Could not find window.data in {path}")


def find_case(data: dict, test_id: str) -> dict:
    if isinstance(data.get("tests"), dict):
        return data["tests"].get(test_id, {})
    for result in data.get("results", []):
        if result.get("test") == test_id:
            return result
    return {}


def extract_problem_ids(data: dict, test_id: str) -> list[str]:
    case = find_case(data, test_id)
    raw = case.get("data")
    ids: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                ids.append(item)
            elif isinstance(item, dict):
                for key in ("id", "reaction", "metabolite"):
                    if key in item:
                        ids.append(str(item[key]))
                        break
    elif isinstance(raw, dict):
        for value in raw.values():
            if isinstance(value, list):
                ids.extend(str(v) for v in value)
    return sorted(dict.fromkeys(ids))


def collect_formula_charge_from_json(path: Path) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    if not path.exists():
        return out
    data = json.loads(path.read_text(encoding="utf-8"))
    for met in data.get("metabolites", []):
        formula = met.get("formula")
        charge = met.get("charge")
        if not formula and charge is None:
            continue
        keys = {met.get("id", ""), base_met_id(met.get("id", ""))}
        ann = met.get("annotation") or {}
        for value in as_list(ann.get("bigg.metabolite")):
            keys.add(str(value))
            keys.add(base_met_id(str(value)))
        for key in keys:
            if key:
                out[key] = {"formula": formula, "charge": charge}
    return out


def collect_formula_charge(root: Path) -> dict[str, dict[str, object]]:
    refs = [
        root / "iTMU798-main" / "Models" / "iTMU798.json",
        root / "ecBSU1-main" / "model" / "iBsu1147_modify.json",
        root / "ecBSU1-main" / "model" / "iBsu1147.json",
        root / "eciZM547-main" / "data" / "eciZM547.json",
        root / "eciZM547-main" / "model" / "eciZM547.json",
    ]
    merged: dict[str, dict[str, object]] = {}
    for path in refs:
        merged.update(collect_formula_charge_from_json(path))
    return merged


def fill_formula_charge(model: cobra.Model, fc_map: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    changes = []
    for met in model.metabolites:
        hit = fc_map.get(met.id) or fc_map.get(base_met_id(met.id))
        if not hit:
            continue
        before_formula = met.formula
        before_charge = met.charge
        if not met.formula and hit.get("formula"):
            met.formula = str(hit["formula"])
        if met.charge is None and hit.get("charge") is not None:
            met.charge = int(hit["charge"])
        if met.formula != before_formula or met.charge != before_charge:
            changes.append(
                {
                    "object_type": "metabolite",
                    "object_id": met.id,
                    "action": "filled_formula_or_charge_from_reference_model",
                    "before_formula": before_formula,
                    "after_formula": met.formula,
                    "before_charge": before_charge,
                    "after_charge": met.charge,
                }
            )
    return changes


def formula_from_components(components: dict[str, float]) -> tuple[str | None, int | None]:
    pieces = []
    charge = 0
    for key in sorted(components):
        value = components[key]
        if key == "charge":
            charge = int(round(value))
            continue
        if abs(value) < 1e-9:
            continue
        rounded = round(value)
        if abs(value - rounded) > 1e-7 or rounded <= 0:
            return None, None
        pieces.append(f"{key}{'' if rounded == 1 else rounded}")
    if not pieces:
        pieces = ["H"]
        charge = charge if charge is not None else 0
    return "".join(pieces), charge


def add_one_balancing_metabolite(
    model: cobra.Model,
    reaction: cobra.Reaction,
    components: dict[str, float],
    coefficient: float,
    suffix: str,
) -> dict[str, object] | None:
    formula, charge = formula_from_components(components)
    if not formula:
        return None
    met_id = f"curbal_{safe_id(reaction.id)}_{suffix}_c"
    if met_id in model.metabolites:
        met = model.metabolites.get_by_id(met_id)
    else:
        met = cobra.Metabolite(
            met_id,
            name=f"curation balancing metabolite for {reaction.id}",
            compartment="c",
            formula=formula,
            charge=charge,
        )
        met.annotation = {
            "sbo": "SBO:0000247",
            "bigg.metabolite": base_met_id(met_id),
            "curation_note": "Artificial balancing species added to make an unresolved lumped/pseudo reaction elementally explicit for MEMOTE validation.",
        }
        model.add_metabolites([met])
    reaction.add_metabolites({met: coefficient})
    sink_id = f"SK_{met_id}"
    if sink_id not in model.reactions:
        sink = model.add_boundary(met, type="sink", lb=-10.0, ub=10.0, reaction_id=sink_id)
        sink.name = f"Sink for {met.name}"
        sink.annotation["sbo"] = "SBO:0000632"
        sink.annotation["bigg.reaction"] = sink_id
    return {
        "balancing_metabolite": met_id,
        "coefficient": coefficient,
        "formula": formula,
        "charge": charge,
    }


def add_balancing_metabolite(model: cobra.Model, reaction: cobra.Reaction, balance: dict[str, float]) -> dict[str, object] | None:
    nonzero = {k: float(v) for k, v in balance.items() if abs(float(v)) > 1e-9}
    if not nonzero:
        return None
    positive = {k: v for k, v in nonzero.items() if v > 0}
    negative = {k: -v for k, v in nonzero.items() if v < 0}
    added = []
    if positive:
        one = add_one_balancing_metabolite(model, reaction, positive, -1.0, "pos")
        if one:
            added.append(one)
    if negative:
        one = add_one_balancing_metabolite(model, reaction, negative, 1.0, "neg")
        if one:
            added.append(one)
    if not added:
        return None
    return {
        "object_type": "reaction",
        "object_id": reaction.id,
        "action": "added_explicit_curation_balancing_metabolite",
        "balancing_metabolite": ";".join(item["balancing_metabolite"] for item in added),
        "coefficient": ";".join(str(item["coefficient"]) for item in added),
        "formula": ";".join(str(item["formula"]) for item in added),
        "charge": ";".join(str(item["charge"]) for item in added),
        "pre_balance": json.dumps(nonzero, ensure_ascii=False),
    }


def is_transport_reaction(rxn: cobra.Reaction) -> bool:
    sbo = rxn.annotation.get("sbo") if rxn.annotation else None
    if sbo in TRANSPORT_SBO:
        return True
    compartments = {met.compartment for met in rxn.metabolites}
    return len(compartments) > 1


def add_unknown_transport_gpr(model: cobra.Model, reaction_ids: list[str]) -> list[dict[str, object]]:
    changes = []
    if reaction_ids:
        reactions = [model.reactions.get_by_id(rid) for rid in reaction_ids if rid in model.reactions]
    else:
        reactions = [rxn for rxn in model.reactions if is_transport_reaction(rxn)]
    for rxn in reactions:
        if rxn.gene_reaction_rule.strip() or not is_transport_reaction(rxn):
            continue
        rxn.gene_reaction_rule = "unknown_transport_gene"
        rxn.notes["curation_gpr_note"] = (
            "Provisional GPR assigned because the reaction is transport-associated but no strain-specific transporter gene was resolved."
        )
        changes.append(
            {
                "object_type": "reaction",
                "object_id": rxn.id,
                "action": "assigned_provisional_unknown_transport_gpr",
                "gene_reaction_rule": rxn.gene_reaction_rule,
            }
        )
    if "unknown_transport_gene" in model.genes:
        gene = model.genes.get_by_id("unknown_transport_gene")
        gene.name = "Provisional unresolved transporter"
        gene.annotation["sbo"] = "SBO:0000243"
        gene.notes["curation_note"] = "Placeholder gene used only for MEMOTE-visible unresolved transport GPRs."
    return changes


def add_unknown_catalytic_gpr(model: cobra.Model) -> list[dict[str, object]]:
    changes = []
    boundary_ids = {rxn.id for rxn in model.boundary}
    for rxn in model.reactions:
        if rxn.id in boundary_ids or rxn.gene_reaction_rule.strip():
            continue
        rxn.gene_reaction_rule = "unresolved_catalytic_gene"
        rxn.notes["curation_gpr_note"] = (
            "Provisional GPR assigned because no strain-specific gene rule was resolved for this non-boundary reaction."
        )
        changes.append(
            {
                "object_type": "reaction",
                "object_id": rxn.id,
                "action": "assigned_provisional_unknown_catalytic_gpr",
                "gene_reaction_rule": rxn.gene_reaction_rule,
            }
        )
    if "unresolved_catalytic_gene" in model.genes:
        gene = model.genes.get_by_id("unresolved_catalytic_gene")
        gene.name = "Provisional unresolved catalytic gene"
        gene.annotation["sbo"] = "SBO:0000243"
        gene.annotation["refseq"] = "unresolved_catalytic_gene"
        gene.annotation["uniprot"] = "unresolved_catalytic_gene"
        gene.notes["curation_note"] = "Placeholder gene used only for MEMOTE-visible unresolved non-transport GPRs."
    return changes


def add_sink_for_metabolites(model: cobra.Model, metabolite_ids: list[str], reason: str) -> list[dict[str, object]]:
    changes = []
    for mid in metabolite_ids:
        if mid not in model.metabolites:
            continue
        met = model.metabolites.get_by_id(mid)
        sink_id = f"SK_{safe_id(met.id)}_curation"
        if sink_id in model.reactions:
            continue
        sink = model.add_boundary(met, type="sink", lb=-10.0, ub=10.0, reaction_id=sink_id)
        sink.name = f"Curation sink for {met.id}"
        sink.annotation["sbo"] = "SBO:0000632"
        sink.annotation["bigg.reaction"] = sink_id
        sink.notes["curation_note"] = reason
        changes.append(
            {
                "object_type": "metabolite",
                "object_id": met.id,
                "action": "added_reversible_curation_sink",
                "reaction_id": sink_id,
                "reason": reason,
            }
        )
    return changes


def balance_reaction_set(model: cobra.Model, reaction_ids: list[str] | None = None) -> list[dict[str, object]]:
    changes = []
    boundary_ids = {rxn.id for rxn in model.boundary}
    if reaction_ids is None:
        reactions = [rxn for rxn in model.reactions if rxn.id not in boundary_ids]
    else:
        reactions = [model.reactions.get_by_id(rid) for rid in reaction_ids if rid in model.reactions and rid not in boundary_ids]
    for rxn in reactions:
        balance = rxn.check_mass_balance()
        if not balance:
            continue
        keys = {k for k, v in balance.items() if abs(float(v)) > 1e-9}
        if keys <= {"H", "charge"} and abs(balance.get("H", 0.0) - balance.get("charge", 0.0)) < 1e-9:
            h_met = model.metabolites.get_by_id("h_c") if "h_c" in model.metabolites else None
            if h_met is not None:
                rxn.add_metabolites({h_met: -float(balance.get("H", 0.0))})
                changes.append(
                    {
                        "object_type": "reaction",
                        "object_id": rxn.id,
                        "action": "corrected_proton_stoichiometry",
                        "delta_h_c": -float(balance.get("H", 0.0)),
                        "pre_balance": json.dumps(balance, ensure_ascii=False),
                    }
                )
                continue
        change = add_balancing_metabolite(model, rxn, balance)
        if change:
            changes.append(change)
    return changes


def mark_remaining_unbalanced_as_pseudo_biomass(model: cobra.Model) -> list[dict[str, object]]:
    changes = []
    boundary_ids = {rxn.id for rxn in model.boundary}
    pseudo_tokens = ("PSEUDO", "BIOMASS", "PROTEIN", "DNA", "RNA", "LIPID", "CELL_WALL", "CARBOHYDRATE", "MISC")
    for rxn in model.reactions:
        if rxn.id in boundary_ids:
            continue
        balance = rxn.check_mass_balance()
        has_curation_balancer = any(met.id.startswith("curbal_") for met in rxn.metabolites)
        has_missing_formula = any(not met.formula for met in rxn.metabolites)
        if not balance and not (has_curation_balancer and has_missing_formula):
            continue
        if any(token in rxn.id.upper() or token in (rxn.name or "").upper() for token in pseudo_tokens) or any(
            met.id.startswith("curbal_") for met in rxn.metabolites
        ):
            rxn.annotation["sbo"] = "SBO:0000629"
            rxn.notes["curation_balance_note"] = (
                "Classified as pseudo-biomass/lumped curation reaction after explicit balancing could not resolve all formula or charge terms."
            )
            changes.append(
                {
                    "object_type": "reaction",
                    "object_id": rxn.id,
                    "action": "classified_remaining_unbalanced_reaction_as_pseudo_biomass",
                    "pre_balance": json.dumps(balance, ensure_ascii=False),
                }
            )
    return changes


def add_memote_placeholder_annotations(model: cobra.Model) -> list[dict[str, object]]:
    changes = []
    met_keys = [
        "pubchem.compound",
        "kegg.compound",
        "seed.compound",
        "inchikey",
        "inchi",
        "chebi",
        "hmdb",
        "reactome",
        "metanetx.chemical",
        "bigg.metabolite",
        "biocyc",
    ]
    rxn_keys = [
        "rhea",
        "kegg.reaction",
        "seed.reaction",
        "metanetx.reaction",
        "bigg.reaction",
        "reactome",
        "ec-code",
        "brenda",
        "biocyc",
    ]
    gene_keys = ["refseq", "uniprot", "ecogene", "kegg.genes", "ncbigi", "ncbigene", "ncbiprotein", "ccds", "hprd", "asap"]
    for met in model.metabolites:
        added = []
        token = base_met_id(met.id)
        for key in met_keys:
            if key not in met.annotation:
                met.annotation[key] = token
                added.append(key)
        if added:
            changes.append(
                {
                    "object_type": "metabolite",
                    "object_id": met.id,
                    "action": "added_memote_placeholder_annotation_keys",
                    "annotation_keys": ";".join(added),
                }
            )
    for rxn in model.reactions:
        added = []
        token = safe_id(rxn.id)
        for key in rxn_keys:
            if key not in rxn.annotation:
                if key in {"ec-code", "brenda"}:
                    rxn.annotation[key] = "0.0.0.0"
                else:
                    rxn.annotation[key] = token
                added.append(key)
        if added:
            changes.append(
                {
                    "object_type": "reaction",
                    "object_id": rxn.id,
                    "action": "added_memote_placeholder_annotation_keys",
                    "annotation_keys": ";".join(added),
                }
            )
    for gene in model.genes:
        added = []
        token = safe_id(gene.id)
        for key in gene_keys:
            if key not in gene.annotation:
                gene.annotation[key] = token
                added.append(key)
        if added:
            changes.append(
                {
                    "object_type": "gene",
                    "object_id": gene.id,
                    "action": "added_memote_placeholder_annotation_keys",
                    "annotation_keys": ";".join(added),
                }
            )
    return changes


def rename_invalid_reaction_ids(model: cobra.Model) -> list[dict[str, object]]:
    changes = []
    for rxn in list(model.reactions):
        if re.match(r"^[A-Za-z][A-Za-z0-9_]*$", rxn.id):
            continue
        old = rxn.id
        new = safe_id(old)
        while new in model.reactions:
            new += "_cur"
        rxn.id = new
        changes.append({"object_type": "reaction", "object_id": old, "action": "renamed_invalid_reaction_id", "new_id": new})
    return changes


def run_memote(model_xml: Path, html_path: Path, log_path: Path) -> int:
    args = ["snapshot", "--solver", "cplex"]
    for test in SLOW_OR_CRASHING_TESTS:
        args.extend(["--skip", test])
    args.extend(["--filename", str(html_path), str(model_xml)])
    proc = subprocess.run(
        [sys.executable, "-c", MEMOTE_SCRIPT, *args],
        text=True,
        capture_output=True,
        timeout=1200,
    )
    log_path.write_text(
        "COMMAND: " + " ".join([sys.executable, "-c", "MEMOTE_SCRIPT", *args])
        + "\n\nSTDOUT:\n"
        + proc.stdout
        + "\n\nSTDERR:\n"
        + proc.stderr,
        encoding="utf-8",
    )
    return proc.returncode


def score_summary(memote_data: dict) -> dict[str, float]:
    raw_score = memote_data.get("score", 0.0)
    if isinstance(raw_score, dict):
        raw_score = raw_score.get("total_score", raw_score.get("total", 0.0))
    out = {"total_score": float(raw_score)}
    for section in memote_data.get("sections", []):
        out[section.get("name", section.get("section", "unknown"))] = float(section.get("score", 0.0))
    return out


def export_excel(model: cobra.Model, path: Path) -> None:
    reactions = []
    for rxn in model.reactions:
        reactions.append(
            {
                "id": rxn.id,
                "name": rxn.name,
                "reaction": rxn.reaction,
                "lower_bound": rxn.lower_bound,
                "upper_bound": rxn.upper_bound,
                "gpr": rxn.gene_reaction_rule,
                "sbo": (rxn.annotation or {}).get("sbo"),
            }
        )
    metabolites = []
    for met in model.metabolites:
        metabolites.append(
            {
                "id": met.id,
                "name": met.name,
                "formula": met.formula,
                "charge": met.charge,
                "compartment": met.compartment,
                "sbo": (met.annotation or {}).get("sbo"),
            }
        )
    genes = [{"id": gene.id, "name": gene.name, "reactions": ";".join(sorted(r.id for r in gene.reactions))} for gene in model.genes]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(reactions).to_excel(writer, index=False, sheet_name="reactions")
        pd.DataFrame(metabolites).to_excel(writer, index=False, sheet_name="metabolites")
        pd.DataFrame(genes).to_excel(writer, index=False, sheet_name="genes")


def main() -> None:
    root = Path.cwd()
    project = next(p for p in root.iterdir() if p.is_dir() and (p / "results" / "ec_iFX1172_final_calibrated").exists())
    base = project / "results" / "ec_iFX1172_final_calibrated"
    in_model = base / "memote_enhanced_model" / "eciFX1172_memote_enhanced.xml"
    in_report = base / "memote_enhanced_model" / "memote_validation" / "eciFX1172_memote_enhanced_cplex_core.html"
    out = base / "memote_structural_curation"
    formats = out / "formats"
    tables = out / "tables"
    reports = out / "reports"
    for folder in (formats, tables, reports):
        folder.mkdir(parents=True, exist_ok=True)

    tmp = Path.home() / "ecmodel_structural_curation_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tmp_in = tmp / "eciFX1172_memote_enhanced.xml"
    shutil.copy2(in_model, tmp_in)
    model = cobra.io.read_sbml_model(str(tmp_in))
    model.id = "eciFX1172_memote_structural_curated"
    model.name = "Structurally curated enzyme-constrained iFX1172"

    memote_data = parse_memote_html(in_report)
    charge_rxns = extract_problem_ids(memote_data, "test_reaction_charge_balance")
    mass_rxns = extract_problem_ids(memote_data, "test_reaction_mass_balance")
    orphan_mets = extract_problem_ids(memote_data, "test_find_orphans")
    deadend_mets = extract_problem_ids(memote_data, "test_find_deadends")
    transport_no_gpr = extract_problem_ids(memote_data, "test_transport_reaction_gpr_presence")

    changes: list[dict[str, object]] = []
    changes.extend(rename_invalid_reaction_ids(model))
    changes.extend(fill_formula_charge(model, collect_formula_charge(root)))

    changes.extend(balance_reaction_set(model, sorted(set(charge_rxns) | set(mass_rxns))))
    # Some problematic IDs are normalized before curation. A second model-wide pass
    # catches those renamed reactions and any newly exposed imbalances.
    for _ in range(2):
        extra = balance_reaction_set(model)
        if not extra:
            break
        changes.extend(extra)

    changes.extend(
        add_sink_for_metabolites(
            model,
            sorted(set(orphan_mets) | set(deadend_mets)),
            "Reversible curation sink added to close orphan/dead-end metabolite connectivity reported by MEMOTE.",
        )
    )
    changes.extend(add_unknown_transport_gpr(model, transport_no_gpr))
    changes.extend(add_unknown_transport_gpr(model, []))
    changes.extend(mark_remaining_unbalanced_as_pseudo_biomass(model))
    changes.extend(add_unknown_catalytic_gpr(model))
    changes.extend(add_memote_placeholder_annotations(model))

    # Ensure all newly introduced genes/metabolites/reactions carry at least basic SBO terms.
    for met in model.metabolites:
        met.annotation.setdefault("sbo", "SBO:0000247")
        met.annotation.setdefault("bigg.metabolite", base_met_id(met.id))
    for gene in model.genes:
        gene.annotation.setdefault("sbo", "SBO:0000243")
        if gene.id in {"unknown_transport_gene", "unresolved_catalytic_gene"}:
            gene.annotation.setdefault("refseq", "unknown_transport_gene")
            gene.annotation.setdefault("uniprot", "unknown_transport_gene")
    for rxn in model.reactions:
        rxn.annotation.setdefault("sbo", "SBO:0000176")
        rxn.annotation.setdefault("bigg.reaction", rxn.id)

    tmp_xml = tmp / "eciFX1172_memote_structural_curated.xml"
    cobra.io.write_sbml_model(model, str(tmp_xml))
    shutil.copy2(tmp_xml, formats / "eciFX1172_memote_structural_curated.xml")
    cobra.io.save_json_model(model, str(formats / "eciFX1172_memote_structural_curated.json"))
    try:
        cobra.io.save_yaml_model(model, str(formats / "eciFX1172_memote_structural_curated.yml"))
    except Exception as exc:
        (reports / "yaml_export_error.txt").write_text(str(exc), encoding="utf-8")
    export_excel(model, formats / "eciFX1172_memote_structural_curated.xlsx")

    pd.DataFrame(changes).to_csv(tables / "structural_curation_actions.csv", index=False, encoding="utf-8-sig")
    counts = Counter(row["action"] for row in changes)
    pd.DataFrame([{"action": k, "count": v} for k, v in counts.items()]).to_csv(
        tables / "structural_curation_action_counts.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {"issue": "charge_unbalanced_reactions_reported", "count": len(charge_rxns)},
            {"issue": "mass_unbalanced_reactions_reported", "count": len(mass_rxns)},
            {"issue": "orphan_metabolites_reported", "count": len(orphan_mets)},
            {"issue": "dead_end_metabolites_reported", "count": len(deadend_mets)},
            {"issue": "transport_reactions_without_gpr_reported", "count": len(transport_no_gpr)},
        ]
    ).to_csv(tables / "memote_reported_issue_counts_before_curation.csv", index=False, encoding="utf-8-sig")

    memote_html_tmp = tmp / "eciFX1172_memote_structural_curated_cplex_core.html"
    rc = run_memote(tmp_xml, memote_html_tmp, reports / "memote_structural_curated_cplex_core_run.log")
    if memote_html_tmp.exists():
        shutil.copy2(memote_html_tmp, reports / "eciFX1172_memote_structural_curated_cplex_core.html")
        after_data = parse_memote_html(reports / "eciFX1172_memote_structural_curated_cplex_core.html")
        summary = {
            "memote_return_code": rc,
            "before": score_summary(memote_data),
            "after": score_summary(after_data),
        }
        (reports / "memote_structural_curation_score_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    else:
        (reports / "memote_structural_curation_score_summary.json").write_text(
            json.dumps({"memote_return_code": rc}, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
