import base64
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np

if not hasattr(np, "object"):
    np.object = object

import cobra
import markdown
import pandas as pd


cobra.Configuration().solver = "glpk"

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "ec_iFX1172_final_calibrated"
MODEL_JSON = RESULT / "model" / "eciFX1172.json"
FORMATS = RESULT / "formats"
ARTICLE = RESULT / "article"
FIGURES = RESULT / "figures"
ANALYSIS = RESULT / "analysis"


def load_model_with_enzyme_notes():
    with open(MODEL_JSON, encoding="utf-8") as handle:
        data = json.load(handle)
    model = cobra.io.load_json_model(str(MODEL_JSON))
    model.notes["enzyme_constraint"] = json.dumps(data.get("enzyme_constraint", {}), ensure_ascii=False)

    by_reaction = {item["id"]: item for item in data.get("reactions", [])}
    for reaction in model.reactions:
        payload = by_reaction.get(reaction.id, {})
        if payload.get("kcat_MW"):
            reaction.notes["kcat_per_s"] = str(payload.get("kcat", ""))
            reaction.notes["kcat_MW"] = str(payload.get("kcat_MW", ""))
            reaction.notes["kcat_source"] = str(payload.get("kcat_source", payload.get("data_type", "")))
            reaction.annotation["ec-model:kcat_per_s"] = str(payload.get("kcat", ""))
            reaction.annotation["ec-model:kcat_MW"] = str(payload.get("kcat_MW", ""))
    return model, data


def export_formats():
    FORMATS.mkdir(parents=True, exist_ok=True)
    model, data = load_model_with_enzyme_notes()

    xml_path = FORMATS / "eciFX1172.xml"
    yml_path = FORMATS / "eciFX1172.yml"
    xlsx_path = FORMATS / "eciFX1172.xlsx"

    tmp_xml = Path(tempfile.gettempdir()) / "eciFX1172.xml"
    cobra.io.write_sbml_model(model, str(tmp_xml))
    shutil.copy2(tmp_xml, xml_path)
    cobra.io.save_yaml_model(model, str(yml_path))

    reactions_raw = pd.DataFrame(data["reactions"])
    metabolites_raw = pd.DataFrame(data["metabolites"])
    genes_raw = pd.DataFrame(data["genes"])
    reaction_params = pd.read_csv(ANALYSIS / "reaction_kcat_MW.csv")
    gene_mass = pd.read_csv(ANALYSIS / "gene_protein_mass.csv")
    summary = pd.read_csv(ANALYSIS / "model_summary.csv")
    enzyme_constraint = pd.DataFrame([data["enzyme_constraint"]])

    reaction_export = reactions_raw.copy()
    if "metabolites" in reaction_export.columns:
        reaction_export["stoichiometry_json"] = reaction_export["metabolites"].apply(
            lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
        )
        reaction_export = reaction_export.drop(columns=["metabolites"])
    for col in ["annotation", "notes"]:
        if col in reaction_export.columns:
            reaction_export[col] = reaction_export[col].apply(
                lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
            )

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="summary", index=False)
        enzyme_constraint.to_excel(writer, sheet_name="enzyme_constraint", index=False)
        reaction_params.to_excel(writer, sheet_name="reaction_kcat_MW", index=False)
        gene_mass.to_excel(writer, sheet_name="gene_protein_mass", index=False)
        reaction_export.to_excel(writer, sheet_name="reactions", index=False)
        metabolites_raw.to_excel(writer, sheet_name="metabolites", index=False)
        genes_raw.to_excel(writer, sheet_name="genes", index=False)

    return xml_path, yml_path, xlsx_path


def make_html(markdown_text, html_path):
    body = markdown.markdown(markdown_text, extensions=["tables"])
    for image in sorted((ARTICLE / "figures").glob("*.png")):
        rel = "figures/" + image.name
        data64 = base64.b64encode(image.read_bytes()).decode("ascii")
        body = body.replace(f'src="{rel}"', f'src="data:image/png;base64,{data64}"')
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>eciFX1172 酶约束模型文章草稿</title>
<style>
body{{font-family:"Microsoft YaHei",Arial,sans-serif;max-width:980px;margin:36px auto;line-height:1.75;color:#222}}
h1{{font-size:28px;border-bottom:2px solid #333;padding-bottom:8px}}
h2{{font-size:21px;margin-top:30px}}
h3{{font-size:17px;margin-top:24px}}
code{{background:#f5f5f5;padding:2px 4px;border-radius:3px}}
img{{max-width:850px;width:100%;display:block;margin:18px auto 6px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
td,th{{border:1px solid #ddd;padding:6px}}
th{{background:#f5f5f5}}
</style>
</head>
<body>{body}</body>
</html>"""
    html_path.write_text(html, encoding="utf-8")


def write_article():
    ARTICLE.mkdir(parents=True, exist_ok=True)
    (ARTICLE / "figures").mkdir(exist_ok=True)
    for image in FIGURES.glob("*.png"):
        shutil.copy2(image, ARTICLE / "figures" / image.name)

    summary = pd.read_csv(ANALYSIS / "model_summary.csv").iloc[0]
    reaction_params = pd.read_csv(ANALYSIS / "reaction_kcat_MW.csv")
    gene_mass = pd.read_csv(ANALYSIS / "gene_protein_mass.csv")
    local_map = pd.read_csv(ANALYSIS / "local_micromonospora_uniprot_mapping.csv")
    source_counts = reaction_params["data_type"].value_counts()
    mass_counts = gene_mass["mass_source"].value_counts()

    source_lines = "\n".join([f"| {idx} | {int(val)} |" for idx, val in source_counts.items()])
    mass_lines = "\n".join([f"| {idx} | {int(val)} |" for idx, val in mass_counts.items()])

    manuscript = f"""# eciFX1172：Micromonospora echinospora 基因组尺度代谢模型的酶约束模型构建与资源分配分析

## 摘要

酶约束基因组尺度代谢模型通过在通量平衡分析中加入酶容量限制，可以缓解传统 GEM 常见的过高通量预测问题，并提高模型对细胞资源分配的解释能力。本研究以 iFX1172 为底盘模型，整合 *Micromonospora echinospora* 本地 UniProt 蛋白注释、BRENDA/SABIO-RK 动力学数据、EC 最大 kcat 数据和 ECMpy/sMOMENT 酶池约束方法，构建了酶约束模型 eciFX1172。原始 iFX1172 包含 {int(summary['original_reactions'])} 个反应、{int(summary['original_metabolites'])} 个代谢物和 {int(summary['original_genes'])} 个基因；经可逆反应和同工酶拆分后，eciFX1172 包含 {int(summary['irreversible_isoenzyme_reactions'])} 个反应，其中 {int(summary['reactions_with_enzyme_constraint'])} 个反应具有 kcat/MW 约束参数。用户提供的 *M. echinospora* UniProt 表覆盖 {int(summary['genes_with_local_micromonospora_mass'])} 个模型基因，所有 {int(summary['genes_with_uniprot'])} 个基因均获得蛋白质量。通过校准总酶池上限，eciFX1172 的默认目标值由原始 GEM 的 {summary['GEM_growth']:.6f} 降至 {summary['ecGEM_growth']:.6f}，约为原始模型的 {summary['ecGEM_growth']/summary['GEM_growth']:.2%}。该模型为庆大霉素产生菌的代谢资源分配、培养基优化和代谢工程靶点解析提供了可复现的酶约束建模框架。

**关键词**：酶约束模型；iFX1172；Micromonospora echinospora；kcat；ECMpy；基因组尺度代谢模型

## 引言

基因组尺度代谢模型能够系统描述细胞代谢网络，并通过 FBA 等方法预测生长、底物利用和产物合成能力。然而，传统 GEM 仅由质量守恒和反应边界约束通量，通常缺少酶资源限制，因此在某些条件下会给出偏高的通量或生长预测。酶约束模型通过引入 kcat、酶分子量和总酶池上限，将代谢通量与酶用量联系起来，使模型更接近真实细胞资源分配状态。

链霉菌等放线菌常具有复杂的次级代谢和较高的基因组冗余度，酶资源分配约束对于解释生长和产物合成之间的权衡尤其重要。参考链霉菌酶约束模型文章的组织方式，本研究围绕模型构建、参数分配、模型校准和结果可视化四个层面，建立 iFX1172 对应的 eciFX1172，并输出 JSON、SBML/XML、YAML 和 Excel 等多种格式，便于后续仿真和论文复现。

## 结果

### eciFX1172 的构建流程

![Figure 1. eciFX1172 construction workflow](figures/figure0_workflow.png)

**图1. eciFX1172 的构建流程。** 以 iFX1172 为底盘，经可逆反应拆分、同工酶拆分、目标物种蛋白质量映射、kcat 分配和总酶池校准，得到最终酶约束模型。

原始 iFX1172 包含 {int(summary['original_reactions'])} 个反应和 {int(summary['original_genes'])} 个基因。为了使每个反应方向和同工酶形式拥有独立酶参数，模型首先转换为不可逆形式，并将 GPR 中含有 `or` 的同工酶反应拆分。处理后模型包含 {int(summary['irreversible_isoenzyme_reactions'])} 个反应。

### 蛋白质量映射和目标物种注释整合

用户提供的 `uniprotkb_Micromonospora_echinospora_2024_08_26.xlsx` 包含 Entry、Protein names、Gene Names、Organism、EC number、Gene Names (ORF)、Catalytic activity 和 Mass 等字段。通过 `GA0070618_xxxx` ORF 与 iFX1172 基因匹配，共获得 {local_map['gene'].nunique()} 个目标物种基因映射。

| 蛋白质量来源 | 基因数 |
|---|---:|
{mass_lines}

![Figure 2. Protein mass source](figures/figure6_mass_source.png)

**图2. 蛋白质量来源构成。** 大部分模型基因由目标物种 *M. echinospora* UniProt 表直接提供蛋白质量，其余基因由模型原 UniProt accession 补充。

![Figure 3. Enzyme molecular weight distribution](figures/figure2_mw_cdf.png)

**图3. 酶复合物分子量累积分布。** 复合物质量根据 GPR 计算，`and` 关系按亚基质量求和，`or` 关系经同工酶拆分后分别赋值。

### kcat 参数分配

kcat 参数按证据等级分配：iFX1172 已有 AutoPACMEN 精确值优先，其次使用 SABIO-RK/BRENDA 同 EC 号 kcat 中位数，再使用 EC_kcat_max 中的 EC 最大 kcat；仍缺失时使用 EC 大类中位数或全局默认值。最终 {int(summary['reactions_with_enzyme_constraint'])} 个反应获得 kcat/MW 参数。

| kcat 来源 | 反应数 |
|---|---:|
{source_lines}

![Figure 4. kcat source composition](figures/figure4_kcat_source_composition.png)

**图4. kcat 参数来源构成。** SABIO-RK/BRENDA 和 EC_kcat_max 是主要来源，无 EC 注释反应仍是后续人工修订重点。

![Figure 5. kcat distribution](figures/figure1_kcat_cdf.png)

**图5. kcat 累积分布。**

![Figure 6. kcat/MW distribution](figures/figure5_kcatmw_cdf.png)

**图6. kcat/MW 系数累积分布。** kcat/MW 是酶约束中将通量转换为酶用量的核心系数。

### 酶池约束使模型预测低于原始 GEM

模型采用 ECMpy/sMOMENT 形式的酶池约束：

`sum_i v_i / (kcat_i * 3600000 / MW_i) <= P_total * f * sigma`

初始参数参考已有 iFX1172 AutoPACMEN 模型：`P_total=0.605`、`f=0.45387051337830087`、`sigma=0.5`。由于当前没有实测生长速率和蛋白组丰度用于严格校准，本文使用二分搜索将总酶池上限校准为 {summary['enzyme_pool_upper_bound']:.6f} g enzyme/gDW，使默认目标值为原始 GEM 的约 95%。

![Figure 7. Growth prediction comparison](figures/figure3_growth_comparison.png)

**图7. 原始 GEM、不可逆 GEM 和 eciFX1172 的默认目标值比较。** eciFX1172 默认目标值为 {summary['ecGEM_growth']:.6f}，低于原始 GEM 的 {summary['GEM_growth']:.6f}。

## 讨论

eciFX1172 在模型结构、蛋白质量来源和 kcat 分配方面均较初始版本更接近目标物种。尤其是用户提供的 *M. echinospora* UniProt 表显著提升了蛋白质量和基因级 EC 证据的可靠性。校准后的酶池约束使模型预测低于原始 GEM，说明酶容量限制已对通量上限产生作用。

不过，模型仍存在三类局限。第一，部分反应缺少 EC 注释，导致仍需使用全局默认 kcat。第二，iFX1172 中若干复杂 GPR 在 COBRApy 解析时提示括号不规范，正式发表前建议回到 Excel 模型逐条修正。第三，当前酶池上限是根据“低于原始模型”的建模目标校准，而不是由实验蛋白组或生长速率反推；未来应结合培养基、底物摄取、产物分泌和蛋白组数据进一步定量校准。

## 材料与方法

### 模型预处理

使用 COBRApy 读取 iFX1172.xml，并将可逆反应拆分为不可逆反应。含有同工酶关系的反应根据 GPR 中的 `or` 逻辑拆分为多个反应副本，以便分别赋予酶参数。

### 蛋白质量计算

优先使用 `uniprotkb_Micromonospora_echinospora_2024_08_26.xlsx` 中的 `Gene Names (ORF)` 和 `Mass` 字段。无法由该表覆盖的基因，则根据模型自带 UniProt accession 查询 UniProt REST。反应复合物分子量由 GPR 计算。

### kcat 分配

反应 EC 号来自模型注释和目标物种 UniProt 表。kcat 来源按优先级依次为 iFX1172 AutoPACMEN 精确反应值、SABIO-RK/BRENDA 同 EC 中位数、EC_kcat_max 最大值、EC 大类中位数和全局默认值。所有反应级参数写入 `reaction_kcat_MW.csv`。

### 酶约束模型格式输出

最终模型以 JSON、SBML/XML、YAML 和 Excel 四种格式输出。XML 和 YAML 中通过 reaction notes/annotation 保留 kcat 与 kcat/MW 信息；Excel 中单独提供 enzyme_constraint、reaction_kcat_MW 和 gene_protein_mass 表。

## 数据与模型可用性

最终模型和参数表位于 `results/ec_iFX1172_final_calibrated/`。其中 `model/eciFX1172.json` 为主要可计算模型，`formats/` 中提供 XML、YML 和 Excel 格式。

## 参考文献

1. ECMpy2.0.pdf。
2. 基于机器学习的自动化酶约束模型构建.pdf。
3. 酶约束模型培养基优化文章-可重现.pdf。
4. Enzyme-constrained_genome-scale_model_of_Yarrowia_.pdf。
5. 链霉菌酶约束模型文章.pdf。
"""

    md_path = ARTICLE / "eciFX1172_manuscript.md"
    html_path = ARTICLE / "eciFX1172_manuscript.html"
    md_path.write_text(manuscript, encoding="utf-8")
    make_html(manuscript, html_path)
    return md_path, html_path


def main():
    xml_path, yml_path, xlsx_path = export_formats()
    md_path, html_path = write_article()
    print("XML", xml_path)
    print("YML", yml_path)
    print("Excel", xlsx_path)
    print("Article MD", md_path)
    print("Article HTML", html_path)


if __name__ == "__main__":
    main()
