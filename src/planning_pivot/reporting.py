from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_validity_by_model_rep(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "model_id" not in df.columns:
        _empty_plot(out_path, "Validity by Model/Representation\n(no data)")
        return

    pivot = df.groupby(["model_id", "representation"])["val_valid"].mean().unstack(fill_value=0)
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("VAL-Valid Rate")
    ax.set_title("VAL Validity by Model and Representation")
    ax.set_ylim(0, 1)
    ax.legend(title="Representation")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_failure_taxonomy(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "failure_type" not in df.columns:
        _empty_plot(out_path, "Failure Taxonomy\n(no data)")
        return

    counts = df["failure_type"].value_counts()
    fig, ax = plt.subplots(figsize=(10, 6))
    counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Count")
    ax.set_title("Failure Type Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_prefix_lengths(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    col = "executable_prefix_len"
    if df.empty or col not in df.columns:
        _empty_plot(out_path, "Executable Prefix Lengths\n(no data)")
        return

    data = df[col].dropna()
    if data.empty:
        _empty_plot(out_path, "Executable Prefix Lengths\n(all NaN)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    data.hist(bins=30, ax=ax)
    ax.set_xlabel("Executable Prefix Length")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Executable Prefix Lengths")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_constrained_search_solved_rate(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "ranker" not in df.columns:
        _empty_plot(out_path, "Constrained Search Solved Rate\n(no data)")
        return

    pivot = df.groupby(["ranker", "beam_width"])["solved_sas"].mean().unstack(fill_value=0)
    ax = pivot.plot(kind="bar", figsize=(10, 6))
    ax.set_ylabel("SAS-Solved Rate")
    ax.set_title("Constrained Search Solved Rate by Ranker and Beam Width")
    ax.set_ylim(0, 1)
    ax.legend(title="Beam Width")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_repair_success_rate(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "ranker" not in df.columns:
        _empty_plot(out_path, "Repair Success Rate\n(no data)")
        return

    rates = df.groupby("ranker")["repaired"].mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    rates.plot(kind="bar", ax=ax)
    ax.set_ylabel("Repair Success Rate")
    ax.set_title("Prefix-Salvage Repair Success Rate by Ranker")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_portfolio_regret(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty or "regret" not in df.columns:
        _empty_plot(out_path, "Portfolio Regret\n(no data)")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    df["regret"].hist(bins=20, ax=ax)
    ax.set_xlabel("Regret (Oracle - Selected)")
    ax.set_ylabel("Count")
    ax.set_title("Portfolio Selector Regret Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _empty_plot(out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=14)
    ax.set_axis_off()
    plt.savefig(out_path, dpi=100)
    plt.close()


def generate_report_tables(output_root: Path) -> None:
    tables_dir = output_root / "report_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    for src, dst in [
        ("diagnostics/summary_by_model_representation.csv", "table_diagnostics_summary.csv"),
        ("constrained_search/summary.csv", "table_constrained_search_summary.csv"),
        ("repair/summary.csv", "table_repair_summary.csv"),
        ("portfolio/summary.csv", "table_portfolio_summary.csv"),
    ]:
        src_path = output_root / src
        dst_path = tables_dir / dst
        df = _safe_read_csv(src_path)
        df.to_csv(dst_path, index=False)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def generate_all_figures(output_root: Path) -> None:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    diag = _safe_read_csv(output_root / "diagnostics" / "diagnostics.csv")
    plot_validity_by_model_rep(diag, fig_dir / "validity_by_model_representation.png")
    plot_failure_taxonomy(diag, fig_dir / "failure_taxonomy.png")
    plot_prefix_lengths(diag, fig_dir / "executable_prefix_lengths.png")

    cs = _safe_read_csv(output_root / "constrained_search" / "mode_results.csv")
    plot_constrained_search_solved_rate(cs, fig_dir / "constrained_search_solved_rate.png")

    repair = _safe_read_csv(output_root / "repair" / "repair_results.csv")
    plot_repair_success_rate(repair, fig_dir / "repair_success_rate.png")

    port = _safe_read_csv(output_root / "portfolio" / "heldout_results.csv")
    plot_portfolio_regret(port, fig_dir / "portfolio_regret.png")


def generate_report_stub(output_root: Path) -> None:
    report = output_root / "REPORT_STUB.md"
    report.write_text("""\
# Verifier-Grounded LLM Planning: From SFT Failure to Structural Recovery

## 1. Original Hypothesis and Failed SFT Assumption

We hypothesized that supervised fine-tuning of Qwen3-1.7B on teacher-generated PDDL plans
would produce a capable autonomous planner. Training loss decreased normally, but
VAL-verified plan validity remained at 0%.

## 2. Diagnostic Evidence

Low training loss did not imply VAL validity. See `results/diagnostics/` for full breakdown.

## 3. Failure Taxonomy

Failures categorized by representation, model, and domain. See figures in `results/figures/`.

## 4. Legality-Constrained Search

Using Fast Downward SAS representation to enumerate only legal actions, we evaluated
beam search with multiple rankers. See `results/constrained_search/`.

## 5. Prefix-Salvage Repair

Invalid LLM-generated plans are truncated at the first illegal action, then completed
using constrained search. See `results/repair/`.

## 6. Portfolio Selector

A lightweight classifier predicts which planning mode (ranker + beam width) to use
per instance, based on structural features. See `results/portfolio/`.

## 7. Heldout-Domain Results

Results on heldout domains (miconic, sokoban, transport, satellite) test generalization.

## 8. Limitations

- SAS simulator does not support axioms.
- Repair only helps when plans have nonzero executable prefixes.
- Portfolio selector trained on small feature set.
- No comparison against state-of-the-art LLM planners.
""")
