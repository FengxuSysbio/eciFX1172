# iFX1172 to final eciFX1172 reproducible code package

This package collects the Python code used to build, validate, curate, analyze,
predict and plot the enzyme-constrained model derived from the original iFX1172
genome-scale metabolic model.

Run all scripts from the project root:

```powershell
cd F:\主机备份\华理博后相关\4.项目\2026\GH文章\酶约束模型构建
D:\python\python.exe <script_path>
```

## Folder layout

### 00_shared_ecmodel_tools

Shared ECMpy/AutoPACMEN support code and helper modules.

- `AutoPACMEN_function.py`: kcat and enzyme-constraint helper functions.
- `ECMpy_function.py`: ecGEM construction and conversion utilities.
- `get_ecGEM_onestop.py`: one-step ecGEM helper workflow.
- `model.py`: model object helpers used by ECMpy-derived scripts.
- `prediction_for_input.py`: kcat prediction input utilities.
- `uniprot_id_mapping.py`: UniProt mapping helper.

### 01_model_construction

Scripts for converting the original model to the calibrated ecModel.

- `build_ec_iFX1172.py`: main construction script. Builds irreversible reactions,
  maps genes to protein mass/kcat data, adds enzyme constraints and calibrates the
  enzyme pool.
- `export_ec_iFX1172_formats_and_article.py`: exports JSON/XML/YML/Excel and
  construction report artifacts.

Recommended order:

```powershell
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\01_model_construction\build_ec_iFX1172.py
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\01_model_construction\export_ec_iFX1172_formats_and_article.py
```

### 02_validation_memote_curation

Scripts for MEMOTE validation, annotation enrichment and final structural curation.

- `run_memote_compare.py`: compares original iFX1172 and calibrated eciFX1172.
- `summarize_memote_comparison.py`: extracts MEMOTE scores into tables/figures.
- `enhance_ecmodel_memote_annotations.py`: enriches SBML annotations using local
  reference models and BiGG/KEGG/MetaNetX-style fields.
- `summarize_memote_enhancement.py`: summarizes annotation-focused score changes.
- `curate_ecmodel_memote_structural_issues.py`: repairs the required model-level
  issues: mass/charge imbalance, orphan/dead-end metabolites, missing transport
  GPRs, and exports the optimized model.

Recommended order:

```powershell
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\02_validation_memote_curation\run_memote_compare.py
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\02_validation_memote_curation\summarize_memote_comparison.py
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\02_validation_memote_curation\enhance_ecmodel_memote_annotations.py
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\02_validation_memote_curation\summarize_memote_enhancement.py
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\02_validation_memote_curation\curate_ecmodel_memote_structural_issues.py
```

### 03_analysis_prediction

Scripts for phenotypic analysis, strain-design prediction and target ranking.

- `run_ec_iFX1172_advanced_analysis.py`: first-pass advanced analysis.
- `run_ec_iFX1172_advanced_analysis_v2.py`: extended analysis including pathway
  distribution, MEMOTE metrics, substrate utilization, robustness, phase-plane
  analysis, single/double gene knockout, dFBA, FSEOF, OptKnock-like, OptForce-like,
  MOMA-like, OptGene-like and MetaStrain-style target aggregation.
- `metastrain_algorithms/`: copied MetaStrain reference scripts for JADE,
  MOMA/MPMA, FVA, phase-plane and target-selection logic.

Recommended order:

```powershell
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\03_analysis_prediction\run_ec_iFX1172_advanced_analysis.py
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\03_analysis_prediction\run_ec_iFX1172_advanced_analysis_v2.py
```

### 04_figure_generation

All plotting scripts. Outputs are saved under the result folders.

- `draw_nature_ecmodel_figures.py`: initial Nature-style figure set.
- `draw_nature_ecmodel_figures_expanded.py`: expanded multi-panel figure set.
- `generate_docx_requested_outputs.py`: figure/table outputs requested from the
  document-based specification.
- `make_optimized_ecmodel_nature_figures.py`: final optimized ecModel figures
  using the structurally curated MEMOTE model plus prediction tables.

Recommended final plotting command:

```powershell
D:\python\python.exe 庆大霉素酶约束模型构建\code_package_iFX1172_to_eciFX1172\04_figure_generation\make_optimized_ecmodel_nature_figures.py
```

### 05_manuscript_tables_text

Scripts for paper-style text and table outputs.

- `write_reference_style_construction_validation.py`: writes construction and
  validation narrative/tables in the style of the reference enzyme-constrained
  model article.

## Important interpretation notes

- Prediction panels should be interpreted from the calibrated enzyme-constrained
  flux core.
- MEMOTE repair panels use the structurally curated optimized model.
- Provisional GPRs and placeholder annotations are explicitly recorded in
  `structural_curation_actions.csv`; they are quality-control annotations, not
  experimentally validated genes.
- CPLEX Community Edition may fail on direct optimization because of problem-size
  limits. Use GLPK for basic model optimization checks, and CPLEX only for MEMOTE
  runs where available.

