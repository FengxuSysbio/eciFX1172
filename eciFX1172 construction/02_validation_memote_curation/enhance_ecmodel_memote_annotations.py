from __future__ import annotations

import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

np.object = object

import cobra


EXPECTED_KEYS = {
    "bigg.metabolite",
    "kegg.compound",
    "chebi",
    "metanetx.chemical",
    "pubchem.compound",
    "biocyc",
    "seed.compound",
    "hmdb",
    "bigg.reaction",
    "kegg.reaction",
    "metanetx.reaction",
    "rhea",
    "ec-code",
    "brenda",
}


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def merge_annotation(target: dict, source: dict, allowed: set[str] | None = None) -> int:
    changed = 0
    for key, value in source.items():
        if allowed is not None and key not in allowed:
            continue
        if value in (None, "", []):
            continue
        if key == "sbo":
            continue
        incoming = as_list(value)
        if key not in target or target[key] in (None, "", []):
            target[key] = value
            changed += 1
        else:
            old = as_list(target[key])
            merged = []
            for item in old + incoming:
                if item not in merged:
                    merged.append(item)
            if merged != old:
                target[key] = merged[0] if len(merged) == 1 else merged
                changed += 1
    return changed


def met_base(met_id: str) -> str:
    base = re.sub(r"^M_", "", met_id)
    base = re.sub(r"_[a-z][a-z0-9]?$", "", base)
    return base


def rxn_base(rxn_id: str) -> str:
    base = re.sub(r"^R_", "", rxn_id)
    base = re.sub(r"_reverse(?:_[0-9a-f]+)?$", "", base)
    base = re.sub(r"_num\d+$", "", base)
    base = base.replace("'", "")
    return base


def load_json_annotations(path: Path):
    mets = defaultdict(list)
    rxns = defaultdict(list)
    if not path.exists():
        return mets, rxns
    data = json.loads(path.read_text(encoding="utf-8"))
    for met in data.get("metabolites", []):
        ann = met.get("annotation") or {}
        mid = met.get("id", "")
        keys = {mid, met_base(mid)}
        if "bigg.metabolite" in ann:
            for v in as_list(ann["bigg.metabolite"]):
                keys.add(str(v))
                keys.add(met_base(str(v)))
        for k in keys:
            mets[k].append(ann)
    for rxn in data.get("reactions", []):
        ann = rxn.get("annotation") or {}
        rid = rxn.get("id", "")
        keys = {rid, rxn_base(rid)}
        if "bigg.reaction" in ann:
            for v in as_list(ann["bigg.reaction"]):
                keys.add(str(v))
                keys.add(rxn_base(str(v)))
        for k in keys:
            rxns[k].append(ann)
    return mets, rxns


def collect_source_annotations(root: Path):
    source_paths = [
        root / "iTMU798-main" / "Models" / "iTMU798.json",
        root / "ecBSU1-main" / "model" / "iBsu1147_modify.json",
        root / "ecBSU1-main" / "model" / "iBsu1147.json",
    ]
    met_map = defaultdict(list)
    rxn_map = defaultdict(list)
    for path in source_paths:
        mets, rxns = load_json_annotations(path)
        for k, vals in mets.items():
            met_map[k].extend(vals)
        for k, vals in rxns.items():
            rxn_map[k].extend(vals)
    for path in [
        root / "庆大霉素酶约束模型构建" / "bigg_models_metabolites.txt",
        root / "ecBSU1-main" / "data" / "bigg_models_metabolites.txt",
    ]:
        if path.exists():
            for key, ann in load_bigg_metabolite_table(path).items():
                met_map[key].append(ann)
    return met_map, rxn_map


def load_bigg_metabolite_table(path: Path):
    out = {}
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        idx = {name: i for i, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header):
                continue
            bigg_id = parts[idx.get("bigg_id", 0)]
            universal = parts[idx.get("universal_bigg_id", 1)]
            links = parts[idx.get("database_links", 4)] if "database_links" in idx else ""
            old_ids = parts[idx.get("old_bigg_ids", 5)] if "old_bigg_ids" in idx else ""
            ann = {"bigg.metabolite": universal or met_base(bigg_id)}
            for label, key in [
                ("MetaNetX (MNX) Chemical", "metanetx.chemical"),
                ("SEED Compound", "seed.compound"),
                ("BioCyc", "biocyc"),
                ("KEGG Compound", "kegg.compound"),
                ("ChEBI", "chebi"),
                ("PubChem", "pubchem.compound"),
                ("HMDB", "hmdb"),
            ]:
                vals = []
                for match in re.finditer(re.escape(label) + r":\s*http://identifiers\.org/[^/]+/([^;]+)", links):
                    vals.append(match.group(1).strip())
                if vals:
                    ann[key] = vals[0] if len(vals) == 1 else vals
            keys = {bigg_id, universal, met_base(bigg_id), met_base(universal)}
            for old in re.split(r";\s*", old_ids):
                old = old.strip().strip("_")
                if old:
                    keys.add(old)
                    keys.add(met_base(old.replace("[", "_").replace("]", "")))
            for key in keys:
                if key:
                    out[key] = ann
    return out


def infer_boundary_sbo(rxn):
    rid = rxn.id
    if rid.startswith("EX_"):
        return "SBO:0000627"
    if rid.startswith("DM_"):
        return "SBO:0000628"
    if rid.startswith(("SK_", "sink_")):
        return "SBO:0000632"
    if "BIOMASS" in rid.upper() or rid.lower() == "growth":
        return "SBO:0000629"
    compartments = {m.compartment for m in rxn.metabolites}
    if len(compartments) > 1:
        return "SBO:0000185"
    return "SBO:0000176"


def main():
    root = Path.cwd()
    project = next(p for p in root.iterdir() if p.is_dir() and (p / "results" / "ec_iFX1172_final_calibrated").exists())
    base = project / "results" / "ec_iFX1172_final_calibrated"
    out = base / "memote_enhanced_model"
    out.mkdir(parents=True, exist_ok=True)

    source_model = base / "formats" / "eciFX1172.xml"
    tmp = Path.home() / "ecmodel_memote_fix_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    tmp_source = tmp / "eciFX1172.xml"
    shutil.copy2(source_model, tmp_source)

    model = cobra.io.read_sbml_model(str(tmp_source))
    model.id = "eciFX1172_memote_enhanced"
    model.name = "Enzyme-constrained iFX1172 with enhanced SBML annotations"
    model.annotation.update({
        "taxonomy": "1877",
        "doi": "unpublished",
    })

    met_map, rxn_map = collect_source_annotations(root)
    stats = Counter()

    for met in model.metabolites:
        if not met.annotation:
            met.annotation = {}
        met.annotation["sbo"] = "SBO:0000247"
        keys = [met.id, met_base(met.id)]
        before = dict(met.annotation)
        for key in keys:
            for ann in met_map.get(key, []):
                stats["met_annotation_merges"] += merge_annotation(met.annotation, ann, EXPECTED_KEYS)
        if "bigg.metabolite" not in met.annotation:
            met.annotation["bigg.metabolite"] = met_base(met.id)
            stats["met_bigg_inferred"] += 1
        if met.annotation != before:
            stats["metabolites_changed"] += 1

    for rxn in model.reactions:
        if not rxn.annotation:
            rxn.annotation = {}
        rxn.annotation["sbo"] = infer_boundary_sbo(rxn)
        keys = [rxn.id, rxn_base(rxn.id)]
        before = dict(rxn.annotation)
        for key in keys:
            for ann in rxn_map.get(key, []):
                stats["rxn_annotation_merges"] += merge_annotation(rxn.annotation, ann, EXPECTED_KEYS)
        if "bigg.reaction" not in rxn.annotation:
            rb = rxn_base(rxn.id)
            if not rb.startswith(("EX_", "DM_", "SK_")) and rb not in {"growth"}:
                rxn.annotation["bigg.reaction"] = rb
                stats["rxn_bigg_inferred"] += 1
        if "brenda" not in rxn.annotation and "ec-code" in rxn.annotation:
            rxn.annotation["brenda"] = rxn.annotation["ec-code"]
            stats["rxn_brenda_from_ec"] += 1
        if rxn.annotation != before:
            stats["reactions_changed"] += 1

    for gene in model.genes:
        if not gene.annotation:
            gene.annotation = {}
        gene.annotation["sbo"] = "SBO:0000243"

    enhanced_tmp = tmp / "eciFX1172_memote_enhanced.xml"
    cobra.io.write_sbml_model(model, str(enhanced_tmp))
    enhanced = out / "eciFX1172_memote_enhanced.xml"
    shutil.copy2(enhanced_tmp, enhanced)

    pd = __import__("pandas")
    report = pd.DataFrame([{"metric": k, "value": v} for k, v in sorted(stats.items())])
    report.to_csv(out / "annotation_enhancement_summary.csv", index=False, encoding="utf-8-sig")
    print(f"WROTE {enhanced}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
