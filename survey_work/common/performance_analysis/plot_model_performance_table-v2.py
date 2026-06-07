#!/usr/bin/env python3
"""Generate a LaTeX performance table from model_performance_analysis.csv.

Layout: metric + problem columns; each model block starts with a centered
model-name row spanning the table width, followed by metric rows.

Run from survey_work/common/performance_analysis/:

  python plot_model_performance_table-v2.py --compile
"""
from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

PERF_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = PERF_DIR / "model_performance_analysis.csv"
DEFAULT_TEX = PERF_DIR / "model_performance_table-v2.tex"

MODEL_ORDER = ("TRUE", "DeepONet", "PCANet", "FNO")
METRICS = ("NumParams", "IDErr", "OODErr1", "OODErr2", "BayesErr")
METRIC_LABELS = {
    "NumParams": r"Trainable Params $p_{\Theta}$",
    "IDErr": "In-distribution error",
    "OODErr1": "Out-of-distribution error (Case 1)",
    "OODErr2": "Out-of-distribution error (Case 2)",
    "BayesErr": "Bayesian inference error",
}
PROBLEM_LABELS = {
    "Poisson": "Poisson",
    "LinearElasticity": "Linear Elasticity",
    "Hyperelasticity": "Hyperelasticity",
}
MODEL_LABELS = {
    "TRUE": "True",
}
TRUE_METRICS = ("BayesErr",)
ERROR_METRICS = ("IDErr", "OODErr1", "OODErr2")
NA = {"", "na", "n/a", "none", "null", "nan"}
# Horizontal padding around vertical rules between metric and problem columns.
DATA_COL_PAD = "1em"
METRIC_COL_WIDTH = "4.2cm"
PROBLEM_COL_WIDTH = "2.4cm"
TABLE_EDGE_PAD = "0.4em"


def compact_col_spec(n_problems: int) -> str:
    """Metric column, then |padded| problem columns with vertical rules."""
    problem_col = r">{\centering\arraybackslash}p{" + PROBLEM_COL_WIDTH + "}"
    metric_col = r">{\raggedright\arraybackslash}p{" + METRIC_COL_WIDTH + "}"
    pad = lambda w: r"@{\hspace{" + w + "}}"
    spec = ["|", pad(TABLE_EDGE_PAD), metric_col]
    for _ in range(n_problems):
        spec.append(pad(DATA_COL_PAD) + "|" + pad(DATA_COL_PAD) + problem_col)
    spec.extend([pad(TABLE_EDGE_PAD), "|"])
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
    p.add_argument("--compile", action="store_true", help="Run pdflatex on the .tex file")
    return p.parse_args()


def load_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        return [
            {k.strip(): v for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


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
    if key in ERROR_METRICS:
        s = f"{x:.3f}".rstrip("0").rstrip(".")
        return f"{s}\\%"
    if x.is_integer():
        return f"{int(x)}\\%"
    return f"{x:.1f}\\%"


def model_metrics(model: str) -> tuple[str, ...]:
    return TRUE_METRICS if model == "TRUE" else METRICS


def model_label(model: str) -> str:
    return MODEL_LABELS.get(model, model)


def build_tex(rows: list[dict], problems: list[str]) -> str:
    lookup = {(r["Model"], r["Problem"]): r for r in rows}
    col_spec = compact_col_spec(len(problems))
    n_cols = 1 + len(problems)
    problem_header = " & ".join(PROBLEM_LABELS.get(p, p) for p in problems)
    inner_rule = rf"\cline{{1-{n_cols}}}"
    header_sep = r"\hline"
    model_sep = r"\hline" + "\n" + r"\hline"

    body_lines = []
    for model in MODEL_ORDER:
        label = model_label(model)
        body_lines.append(rf"\multicolumn{{{n_cols}}}{{|c|}}{{\textbf{{{label}}}}} \\")
        body_lines.append(r"\hline")
        metrics = model_metrics(model)
        for i, metric in enumerate(metrics):
            vals = [
                fmt_metric(lookup.get((model, problem), {}).get(metric), metric)
                for problem in problems
            ]
            row = " & ".join([METRIC_LABELS[metric], *vals]) + r" \\"
            body_lines.append(row)
            if i < len(metrics) - 1:
                body_lines.append(inner_rule)
        if model != MODEL_ORDER[-1]:
            body_lines.append(model_sep)

    body = "\n".join(body_lines)
    return rf"""\documentclass{{article}}
\usepackage[margin=2.5cm]{{geometry}}
\usepackage{{array}}
\renewcommand{{\arraystretch}}{{1.2}}
\begin{{document}}

\begin{{center}}
\vspace*{{1em}}
{{\large\bfseries Performance table\par}}
\vspace{{0.75em}}
\begin{{tabular}}{{{col_spec}}}
\hline
\textbf{{Metric}} & {problem_header} \\
{header_sep}
{body}
\hline
\end{{tabular}}
\end{{center}}

\end{{document}}
"""


def compile_tex(tex_path: Path) -> None:
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        check=True,
    )


def main():
    args = parse_args()
    rows = load_rows(args.csv)
    tex = build_tex(rows, args.problems)
    args.output.write_text(tex, encoding="utf-8")
    print(f"Saved {args.output}")

    if args.compile:
        compile_tex(args.output)
        print(f"Saved {args.output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
