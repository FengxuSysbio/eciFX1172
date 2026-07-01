from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_score(path: Path):
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "window.data" in line:
            data = json.loads(re.search(r"window\.data\s*=\s*(\{.*\})", line).group(1))
            return data
    raise RuntimeError(path)


def main():
    root = Path.cwd()
    project = next(p for p in root.iterdir() if p.is_dir() and (p / "results" / "ec_iFX1172_final_calibrated").exists())
    base = project / "results" / "ec_iFX1172_final_calibrated"
    old_html = base / "memote_comparison" / "eciFX1172_memote_skip_consistency.html"
    enhanced_dir = base / "memote_enhanced_model"
    val = enhanced_dir / "memote_validation"
    enhanced_html = val / "eciFX1172_memote_enhanced_cplex_core.html"
    out_fig = enhanced_dir / "figures"
    out_tab = enhanced_dir / "tables"
    out_fig.mkdir(exist_ok=True)
    out_tab.mkdir(exist_ok=True)

    rows = []
    for label, path in [("eciFX1172 original", old_html), ("eciFX1172 enhanced", enhanced_html)]:
        data = load_score(path)
        rows.append({
            "model": label,
            "section": "total",
            "score_percent": data["score"]["total_score"] * 100,
            "html_report": str(path),
        })
        for sec in data["score"]["sections"]:
            rows.append({
                "model": label,
                "section": sec["section"],
                "score_percent": sec["score"] * 100,
                "html_report": str(path),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_tab / "memote_enhancement_scores.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(out_tab / "memote_enhancement_summary.xlsx", engine="openpyxl") as writer:
        df.to_excel(writer, "scores", index=False)
        pd.read_csv(enhanced_dir / "annotation_enhancement_summary.csv").to_excel(writer, "annotation_changes", index=False)

    pivot = df.pivot(index="section", columns="model", values="score_percent").loc[
        ["total", "consistency", "annotation_met", "annotation_rxn", "annotation_gene", "annotation_sbo"]
    ]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    x = range(len(pivot.index))
    w = 0.36
    ax.bar([i - w / 2 for i in x], pivot["eciFX1172 original"], width=w, color="#6C757D", label="original ecModel")
    ax.bar([i + w / 2 for i in x], pivot["eciFX1172 enhanced"], width=w, color="#315C97", label="enhanced ecModel")
    ax.axhline(80, color="#B54740", linestyle="--", lw=1.2, label="80% target")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=25, ha="right")
    ax.set_ylabel("memote score (%)")
    ax.set_title("memote score improvement after annotation enhancement")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_fig / "memote_enhancement_scores.png", dpi=320, bbox_inches="tight")
    fig.savefig(out_fig / "memote_enhancement_scores.pdf", bbox_inches="tight")
    fig.savefig(out_fig / "memote_enhancement_scores.svg", bbox_inches="tight")
    plt.close(fig)

    enhanced_total = pivot.loc["total", "eciFX1172 enhanced"]
    old_total = pivot.loc["total", "eciFX1172 original"]
    report = f"""# eciFX1172 memote score enhancement report

## Result
The ecModel memote total score increased from {old_total:.2f}% to {enhanced_total:.2f}% after annotation enhancement.

This does not yet reach the requested >80% target. The main remaining blockers are not file-format issues but biological/model-curation issues:

- `consistency` remains {pivot.loc['consistency', 'eciFX1172 enhanced']:.2f}% because stoichiometric consistency, unconserved metabolites, mass balance, charge balance, orphan metabolites and dead-end metabolites still fail or partially fail.
- `annotation_gene` remains {pivot.loc['annotation_gene', 'eciFX1172 enhanced']:.2f}% because memote 0.13.0 expects several organism-specific or human/E. coli-centric gene databases such as EcoGene, ASAP, HPRD and CCDS. Filling those namespaces for *Micromonospora echinospora* without real identifiers would artificially inflate the score.
- `annotation_met` improved to {pivot.loc['annotation_met', 'eciFX1172 enhanced']:.2f}% by transferring local BiGG/KEGG/ChEBI/MetaNetX/BioCyc annotations where they could be matched.
- `annotation_rxn` improved to {pivot.loc['annotation_rxn', 'eciFX1172 enhanced']:.2f}% by transferring reaction annotations and adding EC/Brenda-compatible entries.
- `annotation_sbo` improved to {pivot.loc['annotation_sbo', 'eciFX1172 enhanced']:.2f}% after assigning specific SBO terms for metabolites, genes, biochemical reactions, transport reactions, exchange reactions and biomass-like reactions.

## Files
- Enhanced ecModel: `memote_enhanced_model/eciFX1172_memote_enhanced.xml`
- Best memote HTML report: `memote_enhanced_model/memote_validation/eciFX1172_memote_enhanced_cplex_core.html`
- Score figure: `memote_enhanced_model/figures/memote_enhancement_scores.png`
- Score table: `memote_enhanced_model/tables/memote_enhancement_summary.xlsx`

## Recommendation
To reach a defensible >80% memote score, the next work should focus on true model curation rather than score inflation:

1. Resolve stoichiometric consistency and unconserved metabolite failures.
2. Fix mass- and charge-imbalanced reactions, especially the 218 charge-imbalanced and 262 mass-imbalanced reactions reported by memote.
3. Curate orphan/dead-end metabolites using gap-filling or reaction pruning.
4. Add experimentally or database-supported GPRs for transport reactions where possible.
5. Add organism-appropriate gene database cross-references only when real identifiers exist.
"""
    (enhanced_dir / "memote_enhancement_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
