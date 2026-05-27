#!/usr/bin/env python3
"""Generate a LaTeX performance table from model_performance_analysis.csv.

Layouts:
  compact (default) — model blocks, problems as columns (~4 cols, fits page width)
  wide              — model rows, problem blocks as column groups (original sketch)

Run from survey_work/common/:

  python plot_model_performance_table.py --compile
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

COMMON_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = COMMON_DIR / "model_performance_analysis.csv"
DEFAULT_TEX = COMMON_DIR / "tex_and_pdf/model_performance_table.tex"

MODEL_ORDER = ("True", "DeepONet", "PCANet", "FNO")
METRICS = ("NumParams", "IDErr", "OODErr", "BayesErr")
METRIC_LABELS = {
    "NumParams": r"Trainable $n_{\mathrm{params}}$",
    "IDErr": "In-distribution error",
    "OODErr": "Out-of-distribution error",
    "BayesErr": "Bayesian inference error",
}
PROBLEM_LABELS = {
    "Poisson": "Poisson",
    "LinearElasticity": "Linear Elasticity",
    "Hyperelasticity": "Hyperelasticity",
}
TRUE_METRICS = ("BayesErr",)
NA = {"", "na", "n/a", "none", "null", "nan"}
# Horizontal padding around vertical rules between metric and problem columns.
DATA_COL_PAD = "1.5em"
METRIC_COL_WIDTH = "4cm"
PROBLEM_COL_WIDTH = "2.35cm"
MODEL_METRIC_PAD = "0.8em"
TABLE_EDGE_PAD = "0.4em"


def compact_col_spec(n_problems: int) -> str:
    """Model + metric columns, then |padded| problem columns with vertical rules."""
    problem_col = r">{\centering\arraybackslash}p{" + PROBLEM_COL_WIDTH + "}"
    metric_col = r">{\raggedright\arraybackslash}p{" + METRIC_COL_WIDTH + "}"
    pad = lambda w: r"@{\hspace{" + w + "}}"
    spec = ["@{}", pad(TABLE_EDGE_PAD), "l", pad(MODEL_METRIC_PAD), metric_col]
    for _ in range(n_problems):
        spec.append(pad(DATA_COL_PAD) + "|" + pad(DATA_COL_PAD) + problem_col)
    spec.extend([pad(TABLE_EDGE_PAD), "@{}"])
    return " ".join(spec)


def parse_args():
    p = argparse.ArgumentParser(description="Build LaTeX performance table")
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--output", type=Path, default=DEFAULT_TEX)
    p.add_argument(
        "--problems",
        nargs="+",
        default=["Poisson", "LinearElasticity", "Hyperelasticity"],
        help="CSV Problem values to include",
    )
    p.add_argument(
        "--layout",
        choices=("compact", "wide"),
        default="compact",
        help="compact: problems as columns (recommended); wide: original horizontal layout",
    )
    p.add_argument("--compile", action="store_true", help="Run pdflatex on the .tex file")
    return p.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def missing(val) -> bool:
    if val is None:
        return True
    s = str(val).strip().lower()
    return s in NA


def fmt_metric(val, key: str) -> str:
    if missing(val):
        return r"\textemdash{}"
    if key == "NumParams":
        return f"{int(round(float(val))):,}".replace(",", "{,}")
    x = float(val)
    if x.is_integer():
        return f"{int(x)}\\%"
    return f"{x:.1f}\\%"


def model_metrics(model: str) -> tuple[str, ...]:
    return TRUE_METRICS if model == "True" else METRICS


def build_tex_wide(rows: list[dict], problems: list[str]) -> str:
    lookup = {(r["Model"], r["Problem"]): r for r in rows}
    n_metrics = len(METRICS)
    col_spec = "l" + "c" * (n_metrics * len(problems))

    header_groups = " & ".join(
        rf"\multicolumn{{{n_metrics}}}{{c}}{{{PROBLEM_LABELS.get(p, p)}}}"
        for p in problems
    )
    cmidrules = " ".join(
        rf"\cmidrule(lr){{{2 + i * n_metrics}-{1 + (i + 1) * n_metrics}}}"
        for i in range(len(problems))
    )
    subheader = " & ".join(
        [r"\textbf{Model}"]
        + [METRIC_LABELS[m] for _ in problems for m in METRICS]
    )

    body_lines = []
    for model in MODEL_ORDER:
        cells = [rf"\textbf{{{model}}}"]
        for problem in problems:
            rec = lookup.get((model, problem), {})
            for m in METRICS:
                cells.append(fmt_metric(rec.get(m), m))
        body_lines.append(" & ".join(cells) + r" \\")

    body = "\n".join(body_lines)
    return rf"""\documentclass{{article}}
\usepackage{{booktabs}}
\begin{{document}}

\begin{{table}}[ht]
\centering
\small
\caption{{Performance table}}
\begin{{tabular}}{{{col_spec}}}
\toprule
& {header_groups} \\
{cmidrules}
{subheader} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}

\end{{document}}
"""


def build_tex_compact(rows: list[dict], problems: list[str]) -> str:
    lookup = {(r["Model"], r["Problem"]): r for r in rows}
    col_spec = compact_col_spec(len(problems))
    problem_header = " & ".join(PROBLEM_LABELS.get(p, p) for p in problems)
    n_cols = 2 + len(problems)
    inner_rule = rf"\cline{{{2}-{n_cols}}}"
    header_sep = r"\hline" + "\n" + r"\hline"
    model_sep = r"\hline" + "\n" + r"\hline"

    body_lines = []
    for model in MODEL_ORDER:
        metrics = model_metrics(model)
        n_rows = len(metrics)
        for i, metric in enumerate(metrics):
            model_cell = rf"\multirow{{{n_rows}}}{{*}}{{\textbf{{{model}}}}}" if i == 0 else ""
            metric_cell = METRIC_LABELS[metric]
            vals = [
                fmt_metric(lookup.get((model, problem), {}).get(metric), metric)
                for problem in problems
            ]
            row = " & ".join([model_cell, metric_cell, *vals]) + r" \\"
            body_lines.append(row)
            if i < len(metrics) - 1:
                body_lines.append(inner_rule)
        if model != MODEL_ORDER[-1]:
            body_lines.append(model_sep)

    body = "\n".join(body_lines)
    return rf"""\documentclass{{article}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{array}}
\usepackage{{multirow}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{document}}

\begin{{center}}
\vspace*{{1em}}
{{\large\bfseries Performance table\par}}
\vspace{{0.75em}}
\begin{{tabular}}{{{col_spec}}}
\hline
\hline
\textbf{{Model}} & & {problem_header} \\
{header_sep}
{body}
\hline
\hline
\end{{tabular}}
\end{{center}}

\end{{document}}
"""


def build_tex(rows: list[dict], problems: list[str], layout: str) -> str:
    if layout == "wide":
        return build_tex_wide(rows, problems)
    return build_tex_compact(rows, problems)


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    tex = build_tex(rows, args.problems, args.layout)
    args.output.write_text(tex, encoding="utf-8")
    print(f"Saved {args.output} (layout={args.layout})")

    if args.compile:
        compile_tex(args.output)
        print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
