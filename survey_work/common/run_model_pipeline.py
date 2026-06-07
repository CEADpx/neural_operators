#!/usr/bin/env python3
"""Run NOP training notebooks in parallel, then Bayesian inversion notebook.

Run from survey_work/ or survey_work/common/:

  python run_model_pipeline.py poisson --skip-training
  python run_model_pipeline.py linear_elasticity --skip-training
"""
import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor
from traitlets.config import Config

SURVEY_WORK_DIR = Path(__file__).resolve().parent.parent

MODELS = {
    "poisson": {
        "problem_dir": "problems/poisson",
        "bayes_dir": "applications/bayesian_inverse_problem_poisson_ood",
    },
    "linear_elasticity": {
        "problem_dir": "problems/linear_elasticity",
        "bayes_dir": "applications/bayesian_inverse_problem_linear_elasticity_ood",
    },
    "hyperelasticity": {
        "problem_dir": "problems/hyperelasticity",
        "bayes_dir": "applications/bayesian_inverse_problem_hyperelasticity",
    },
}

NOP_NAMES = ("DeepONet", "PCANet", "FNO")
TRAIN_NOTEBOOK = "training_and_testing.ipynb"
BAYES_NOTEBOOK = "BayesianInversion.ipynb"
MODEL_PKL = "Results/model.pkl"


def resolve_survey_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = SURVEY_WORK_DIR / path
    return path.resolve()


def run_notebook(notebook_path):
    notebook_path = Path(notebook_path)
    cwd = notebook_path.parent

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    print("\n" + "=" * 70)
    print("Running notebook:", notebook_path)
    print("  cwd:", cwd)
    print("=" * 70)

    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    config = Config()
    config.ExecutePreprocessor.timeout = -1
    config.ExecutePreprocessor.kernel_name = "python3"

    ep = ExecutePreprocessor(config=config)
    ep.preprocess(nb, {"metadata": {"path": str(cwd)}})

    with open(notebook_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def run_training_parallel(prob_dir):
    notebooks = [
        prob_dir / nop / TRAIN_NOTEBOOK for nop in NOP_NAMES
    ]
    missing = [nb for nb in notebooks if not nb.exists()]
    if missing:
        print("Missing training notebook(s):")
        for nb in missing:
            print(f"  {nb}")
        sys.exit(1)

    failures = []
    with ThreadPoolExecutor(max_workers=len(NOP_NAMES)) as pool:
        futures = {pool.submit(run_notebook, nb): nb for nb in notebooks}
        for future in as_completed(futures):
            nb = futures[future]
            try:
                future.result()
            except CellExecutionError as exc:
                failures.append((nb, exc))
            except Exception as exc:
                failures.append((nb, exc))

    if failures:
        print("\nTraining failed for:")
        for nb, err in failures:
            print(f"  {nb}")
            print(f"    {err}")
        sys.exit(1)

    missing_models = []
    for nop in NOP_NAMES:
        model_file = prob_dir / nop / MODEL_PKL
        if not model_file.exists():
            missing_models.append(model_file)
    if missing_models:
        print("\nTraining finished but model.pkl is missing for:")
        for path in missing_models:
            print(f"  {path}")
        sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NOPs in parallel from notebooks, then run Bayesian inversion"
    )
    parser.add_argument(
        "model",
        choices=sorted(MODELS),
        help="Problem name: poisson, linear_elasticity, or hyperelasticity",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip NOP training (assume model.pkl files exist)",
    )
    parser.add_argument(
        "--skip-bayesian",
        action="store_true",
        help="Skip Bayesian inversion notebook",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = MODELS[args.model]
    prob_dir = resolve_survey_path(cfg["problem_dir"])
    bayes_dir = resolve_survey_path(cfg["bayes_dir"])
    bayes_notebook = bayes_dir / BAYES_NOTEBOOK

    print(f"survey_work: {SURVEY_WORK_DIR}")
    print(f"model:       {args.model}")
    print(f"prob_dir:    {prob_dir}")
    print(f"bayes_dir:   {bayes_dir}")

    if not args.skip_training:
        run_training_parallel(prob_dir)
    else:
        missing_models = [
            prob_dir / nop / MODEL_PKL
            for nop in NOP_NAMES
            if not (prob_dir / nop / MODEL_PKL).exists()
        ]
        if missing_models:
            print("Missing trained model(s):")
            for path in missing_models:
                print(f"  {path}")
            sys.exit(1)

    if not args.skip_bayesian:
        if not bayes_notebook.exists():
            print(f"Missing Bayesian notebook: {bayes_notebook}")
            sys.exit(1)
        run_notebook(bayes_notebook)

    print("\nPipeline finished.")


if __name__ == "__main__":
    main()
