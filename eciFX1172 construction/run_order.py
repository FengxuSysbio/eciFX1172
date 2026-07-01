from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


STAGES = [
    (
        "01_model_construction",
        [
            "build_ec_iFX1172.py",
            "export_ec_iFX1172_formats_and_article.py",
        ],
    ),
    (
        "02_validation_memote_curation",
        [
            "run_memote_compare.py",
            "summarize_memote_comparison.py",
            "enhance_ecmodel_memote_annotations.py",
            "summarize_memote_enhancement.py",
            "curate_ecmodel_memote_structural_issues.py",
        ],
    ),
    (
        "03_analysis_prediction",
        [
            "run_ec_iFX1172_advanced_analysis.py",
            "run_ec_iFX1172_advanced_analysis_v2.py",
        ],
    ),
    (
        "04_figure_generation",
        [
            "draw_nature_ecmodel_figures.py",
            "draw_nature_ecmodel_figures_expanded.py",
            "generate_docx_requested_outputs.py",
            "make_optimized_ecmodel_nature_figures.py",
        ],
    ),
    (
        "05_manuscript_tables_text",
        [
            "write_reference_style_construction_validation.py",
        ],
    ),
]


def main() -> None:
    print("Recommended iFX1172 -> final eciFX1172 code execution order")
    print("Run from the project root with D:\\python\\python.exe")
    print()
    for folder, scripts in STAGES:
        print(f"[{folder}]")
        for script in scripts:
            path = ROOT / folder / script
            print(f"  D:\\python\\python.exe {path}")
        print()


if __name__ == "__main__":
    main()

