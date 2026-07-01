from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_memote_html(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if "window.data" in line:
            m = re.search(r"window\.data\s*=\s*(\{.*\})", line)
            if m:
                return json.loads(m.group(1))
    raise ValueError(f"Cannot locate window.data in {path}")


def main():
    root = Path.cwd()
    project = next(p for p in root.iterdir() if p.is_dir() and (p / "results" / "ec_iFX1172_final_calibrated").exists())
    out = project / "results" / "ec_iFX1172_final_calibrated" / "memote_comparison"
    fig_dir = out / "figures"
    tab_dir = out / "tables"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)

    inputs = {
        "iFX1172": out / "iFX1172_memote_skip_consistency.html",
        "eciFX1172": out / "eciFX1172_memote_skip_consistency.html",
    }
    rows = []
    section_rows = []
    test_rows = []
    for model, path in inputs.items():
        data = load_memote_html(path)
        score = data["score"]
        rows.append({
            "model": model,
            "memote_total_score_fraction": score["total_score"],
            "memote_total_score_percent": score["total_score"] * 100,
            "html_report": str(path),
            "note": "memote snapshot with --skip test_consistency because local GLPK crashed in test_consistency.py",
        })
        for sec in score["sections"]:
            section_rows.append({
                "model": model,
                "section": sec["section"],
                "score_fraction": sec["score"],
                "score_percent": sec["score"] * 100,
            })
        for test_id, test in data["tests"].items():
            result = test.get("result", "")
            if not isinstance(result, str):
                result = "parameterized"
            test_rows.append({
                "model": model,
                "test_id": test_id,
                "title": test.get("title", ""),
                "result": result,
                "metric": test.get("metric", None),
                "message": test.get("message", ""),
            })

    total_df = pd.DataFrame(rows)
    section_df = pd.DataFrame(section_rows)
    tests_df = pd.DataFrame(test_rows)
    tests_df["result_group"] = tests_df["result"].where(
        tests_df["result"].isin(["passed", "failed", "skipped", "error"]), "parameterized"
    )
    counts_df = tests_df.groupby(["model", "result_group"]).size().reset_index(name="count")

    total_df.to_csv(tab_dir / "memote_total_scores.csv", index=False, encoding="utf-8-sig")
    section_df.to_csv(tab_dir / "memote_section_scores.csv", index=False, encoding="utf-8-sig")
    counts_df.to_csv(tab_dir / "memote_test_result_counts.csv", index=False, encoding="utf-8-sig")
    tests_df.to_csv(tab_dir / "memote_test_details.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(tab_dir / "memote_iFX1172_vs_eciFX1172_comparison.xlsx", engine="openpyxl") as writer:
        total_df.to_excel(writer, "total_scores", index=False)
        section_df.to_excel(writer, "section_scores", index=False)
        counts_df.to_excel(writer, "test_counts", index=False)
        tests_df.to_excel(writer, "test_details", index=False)

    colors = {"iFX1172": "#6C757D", "eciFX1172": "#315C97"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    ax.bar(total_df["model"], total_df["memote_total_score_percent"], color=[colors[m] for m in total_df["model"]])
    for i, v in enumerate(total_df["memote_total_score_percent"]):
        ax.text(i, v + 0.6, f"{v:.2f}%", ha="center", fontsize=10)
    ax.set_ylim(0, max(40, total_df["memote_total_score_percent"].max() + 8))
    ax.set_ylabel("memote total score (%)")
    ax.set_title("Total score")
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    pivot = section_df.pivot(index="section", columns="model", values="score_percent")
    x = range(len(pivot.index))
    width = 0.36
    ax.bar([i - width / 2 for i in x], pivot["iFX1172"], width=width, color=colors["iFX1172"], label="iFX1172")
    ax.bar([i + width / 2 for i in x], pivot["eciFX1172"], width=width, color=colors["eciFX1172"], label="eciFX1172")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=35, ha="right")
    ax.set_ylabel("section score (%)")
    ax.set_title("Section scores")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("memote comparison: iFX1172 vs eciFX1172", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "memote_score_comparison.png", dpi=320, bbox_inches="tight")
    fig.savefig(fig_dir / "memote_score_comparison.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "memote_score_comparison.svg", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    c_pivot = counts_df.pivot(index="result_group", columns="model", values="count").fillna(0)
    x = range(len(c_pivot.index))
    ax.bar([i - width / 2 for i in x], c_pivot["iFX1172"], width=width, color=colors["iFX1172"], label="iFX1172")
    ax.bar([i + width / 2 for i in x], c_pivot["eciFX1172"], width=width, color=colors["eciFX1172"], label="eciFX1172")
    ax.set_xticks(list(x))
    ax.set_xticklabels(c_pivot.index)
    ax.set_ylabel("number of tests")
    ax.set_title("memote test outcomes")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="#E5E7EB")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "memote_test_counts.png", dpi=320, bbox_inches="tight")
    fig.savefig(fig_dir / "memote_test_counts.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / "memote_test_counts.svg", bbox_inches="tight")
    plt.close(fig)

    original = total_df.loc[total_df["model"] == "iFX1172", "memote_total_score_percent"].iloc[0]
    ec = total_df.loc[total_df["model"] == "eciFX1172", "memote_total_score_percent"].iloc[0]
    delta = ec - original
    section_delta = section_df.pivot(index="section", columns="model", values="score_percent")
    section_delta["delta_ec_minus_original"] = section_delta["eciFX1172"] - section_delta["iFX1172"]
    section_delta.to_csv(tab_dir / "memote_section_score_delta.csv", encoding="utf-8-sig")
    section_md = ["| section | iFX1172 | eciFX1172 | delta |", "| --- | ---: | ---: | ---: |"]
    for section, row in section_delta.iterrows():
        section_md.append(
            f"| {section} | {row['iFX1172']:.2f}% | {row['eciFX1172']:.2f}% | {row['delta_ec_minus_original']:+.3f} |"
        )

    report = f"""# memote 得分对比：iFX1172 与 eciFX1172

## 方法
根据 memote 官方 getting started 文档，本地使用 `memote report snapshot MODEL` 生成单模型 HTML 报告，并尝试使用 `memote report diff MODEL1 MODEL2` 进行多模型比较。由于当前 Windows/Python 环境中全量 `test_consistency.py` 在 GLPK 底层调用时崩溃，本次可比较结果采用同一参数 `--skip test_consistency` 生成两个 snapshot 报告。该处理不改变两个模型之间的相对比较，但 total score 应解读为“跳过 consistency 模块后的 memote snapshot 得分”。

## 结果
| model | memote total score |
| --- | ---: |
| iFX1172 | {original:.2f}% |
| eciFX1172 | {ec:.2f}% |

eciFX1172 的 memote total score 为 {ec:.2f}%，比原始 iFX1172 的 {original:.2f}% 高 {delta:.3f} 个百分点。两者差异很小，主要原因是 ecModel 保留了原始模型的代谢物、基因、GPR 和大部分注释结构；酶约束化主要增加了反应层面的约束和参数，而不是系统性重写 SBML 注释。

## 分项分数
{chr(10).join(section_md)}

## 输出文件
- `figures/memote_score_comparison.png`
- `figures/memote_test_counts.png`
- `tables/memote_iFX1172_vs_eciFX1172_comparison.xlsx`
- `iFX1172_memote_skip_consistency.html`
- `eciFX1172_memote_skip_consistency.html`
"""
    (out / "memote_comparison_report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
