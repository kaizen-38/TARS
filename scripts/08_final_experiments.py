#!/usr/bin/env python3
"""Final experiment additions: repair ablation, beam-width curve,
per-domain breakdown, first-invalid-step histogram, portfolio leakage check."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import random as stdlib_random
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from planning_pivot.sas import (
    parse_output_sas, SasTask, SasOperator,
    applicable_operators, apply_operator, is_goal,
    simulate_plan, operator_lookup, plan_lines_from_ops,
)
from planning_pivot.rankers import GoalCountRanker, RandomRanker
from planning_pivot.constrained_search import beam_search
from planning_pivot.plan_io import parse_plan_actions, extract_plan_text

RESULTS = Path("results")
FIG_DIR = RESULTS / "figures" / "detailed"
ANALYSIS_DIR = RESULTS / "analysis"
FIG_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
DPI = 200
SEED = 574

DOMAIN_COLORS = {
    "miconic": "#1f77b4", "transport": "#ff7f0e",
    "sokoban": "#2ca02c", "satellite": "#d62728",
    "blocksworld": "#9467bd", "gripper": "#8c564b",
    "ferry": "#e377c2", "delivery": "#7f7f7f",
    "childsnack": "#bcbd22", "floortile": "#17becf",
    "rovers": "#aec7e8", "spanner": "#ffbb78",
}


def load_sas_task(problem_id: str) -> SasTask | None:
    sas_dir = RESULTS / "sas_cache"
    stripped = problem_id.rsplit("_", 1)[0] if "_" in problem_id else problem_id
    for candidate in [problem_id, stripped]:
        p = sas_dir / candidate / "output.sas"
        if p.exists():
            try:
                return parse_output_sas(p)
            except Exception:
                pass
    return None


# ---------------------------------------------------------------------------
# 1. REPAIR VALUE ABLATION
# ---------------------------------------------------------------------------
def run_repair_ablation():
    print("=== 1. Repair Value Ablation ===")
    diag = pd.read_csv(RESULTS / "diagnostics" / "diagnostics.csv")
    gen = pd.read_csv(RESULTS / "diagnostics" / "generation_records.csv")
    instances = pd.read_csv(RESULTS / "instances.csv")

    invalid = diag[diag["val_valid"] == False].copy()
    if "raw_output_path" not in invalid.columns:
        invalid = invalid.merge(
            gen[["instance_id", "raw_output_path"]].drop_duplicates(),
            on="instance_id", how="left",
        )

    ranker = GoalCountRanker()
    beam_width = 4
    rng = stdlib_random.Random(SEED)

    rows = []
    seen = set()
    for _, drow in invalid.iterrows():
        instance_id = str(drow.get("instance_id", ""))
        problem_id = str(drow.get("problem_id", ""))
        if not problem_id or problem_id in seen:
            continue

        task = load_sas_task(problem_id)
        if task is None:
            continue

        raw_path_str = drow.get("raw_output_path", "")
        if pd.isna(raw_path_str) or not str(raw_path_str):
            continue
        raw_path = Path(str(raw_path_str))
        if not raw_path.exists():
            continue

        raw_text = raw_path.read_text()
        plan = parse_plan_actions(extract_plan_text(raw_text))
        if not plan.actions:
            continue

        seen.add(problem_id)
        domain = str(drow.get("domain_name", ""))
        split = str(drow.get("split", ""))
        rep = str(drow.get("representation", ""))
        teacher_len_val = drow.get("teacher_plan_len")
        teacher_len = int(teacher_len_val) if pd.notna(teacher_len_val) else 20
        max_steps = min(int(teacher_len * 3.0), 200)

        lines = [f"({a.name} {' '.join(a.args)})" for a in plan.actions]
        trace = simulate_plan(task, lines)
        llm_prefix_len = trace.valid_prefix_len

        lookup = operator_lookup(task)
        llm_prefix_ops = []
        state = task.init
        for i, line in enumerate(lines):
            if i >= llm_prefix_len:
                break
            line_c = line.strip()
            op = lookup.get(line_c) or lookup.get(line_c.lower())
            if op is None:
                inner = line_c.strip("() ").lower()
                op = lookup.get(inner) or lookup.get("(" + inner + ")")
            if op is None:
                break
            state = apply_operator(task, state, op)
            llm_prefix_ops.append(op)
        llm_prefix_state = state

        teacher_plan_path = None
        inst_match = instances[instances["instance_id"] == problem_id]
        if not inst_match.empty:
            tp = inst_match.iloc[0].get("teacher_plan_path", "")
            if tp and Path(str(tp)).exists():
                teacher_plan_path = Path(str(tp))

        teacher_prefix_ops = []
        teacher_prefix_state = task.init
        if teacher_plan_path:
            teacher_lines = [
                l.strip() for l in teacher_plan_path.read_text().splitlines()
                if l.strip() and not l.strip().startswith(";")
            ]
            teacher_trace = simulate_plan(task, teacher_lines)
            t_state = task.init
            for i, tl in enumerate(teacher_lines):
                if i >= min(llm_prefix_len, teacher_trace.valid_prefix_len):
                    break
                tl_c = tl.strip()
                t_op = lookup.get(tl_c) or lookup.get(tl_c.lower())
                if t_op is None:
                    inner = tl_c.strip("() ").lower()
                    t_op = lookup.get(inner) or lookup.get("(" + inner + ")")
                if t_op is None:
                    break
                t_state = apply_operator(task, t_state, t_op)
                teacher_prefix_ops.append(t_op)
            teacher_prefix_state = t_state

        random_prefix_ops = []
        r_state = task.init
        for _ in range(llm_prefix_len):
            apps = applicable_operators(task, r_state)
            if not apps:
                break
            op = rng.choice(apps)
            r_state = apply_operator(task, r_state, op)
            random_prefix_ops.append(op)
        random_prefix_state = r_state

        conditions = {
            "empty_prefix": ([], task.init),
            "llm_prefix": (llm_prefix_ops, llm_prefix_state),
            "random_prefix": (random_prefix_ops, random_prefix_state),
            "teacher_prefix": (teacher_prefix_ops, teacher_prefix_state),
        }

        for cond_name, (prefix_ops, start_state) in conditions.items():
            if is_goal(task, start_state):
                rows.append({
                    "problem_id": problem_id, "domain_name": domain,
                    "split": split, "representation": rep,
                    "condition": cond_name,
                    "prefix_len": len(prefix_ops),
                    "solved": True, "completion_len": 0,
                    "total_plan_len": len(prefix_ops),
                    "seconds": 0.0, "failure_type": None,
                })
                continue

            ctx = {"start_state": start_state, "prefix_ops": prefix_ops}
            result = beam_search(
                task, ranker, beam_width=beam_width,
                max_steps=max_steps, max_expansions=max_steps * beam_width * 2,
                context=ctx,
            )
            comp_len = len(result.plan_lines) - len(prefix_ops) if result.solved else None
            rows.append({
                "problem_id": problem_id, "domain_name": domain,
                "split": split, "representation": rep,
                "condition": cond_name,
                "prefix_len": len(prefix_ops),
                "solved": result.solved,
                "completion_len": comp_len,
                "total_plan_len": len(result.plan_lines) if result.solved else None,
                "seconds": result.seconds,
                "failure_type": result.failure_type,
            })

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(ANALYSIS_DIR / "repair_ablation.csv", index=False)

    if ablation_df.empty:
        print("  No ablation data generated.")
        return ablation_df

    summary = ablation_df.groupby("condition").agg(
        count=("solved", "count"),
        success_rate=("solved", "mean"),
        mean_prefix_len=("prefix_len", "mean"),
        mean_completion_len=("completion_len", "mean"),
        mean_total_plan_len=("total_plan_len", "mean"),
        mean_seconds=("seconds", "mean"),
    ).reindex(["empty_prefix", "llm_prefix", "random_prefix", "teacher_prefix"])
    summary.to_csv(ANALYSIS_DIR / "repair_ablation_summary.csv")
    print(summary.to_string())

    by_domain = ablation_df.groupby(["domain_name", "condition"])["solved"].mean().unstack(fill_value=0)
    by_domain = by_domain.reindex(columns=["empty_prefix", "llm_prefix", "random_prefix", "teacher_prefix"])
    by_domain.to_csv(ANALYSIS_DIR / "repair_ablation_by_domain.csv")

    # Figure: ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    cond_order = ["empty_prefix", "llm_prefix", "random_prefix", "teacher_prefix"]
    cond_labels = ["Empty\n(from init)", "LLM\nPrefix", "Random Legal\nPrefix", "Teacher\nPrefix"]
    rates = [summary.loc[c, "success_rate"] if c in summary.index else 0 for c in cond_order]
    colors = ["#95a5a6", "#e74c3c", "#f39c12", "#27ae60"]
    bars = ax.bar(cond_labels, rates, color=colors, edgecolor="black", linewidth=0.5)
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rate:.1%}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_ylabel("Completion Success Rate", fontsize=13)
    ax.set_title("Repair Ablation: Does the LLM Prefix Help?", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(rates) * 1.2 + 0.05)
    ax.axhline(y=rates[0], color="#95a5a6", linestyle="--", alpha=0.5, label="Empty baseline")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig22_repair_ablation.png", dpi=DPI)
    plt.close()

    # Figure: ablation by domain
    if not by_domain.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(by_domain))
        w = 0.2
        for i, (cond, label, color) in enumerate(zip(cond_order, cond_labels, colors)):
            if cond in by_domain.columns:
                vals = by_domain[cond].values
                ax.bar(x + i * w, vals, w, label=label.replace("\n", " "), color=color, edgecolor="black", linewidth=0.3)
        ax.set_xticks(x + 1.5 * w)
        ax.set_xticklabels(by_domain.index, rotation=30, ha="right")
        ax.set_ylabel("Completion Success Rate")
        ax.set_title("Repair Ablation by Domain", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.05)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig23_repair_ablation_by_domain.png", dpi=DPI)
        plt.close()

    return ablation_df


# ---------------------------------------------------------------------------
# 2. BEAM-WIDTH CURVE (already have data, just need the table + figure)
# ---------------------------------------------------------------------------
def beam_width_curve():
    print("\n=== 2. Beam-Width Curve ===")
    cs = pd.read_csv(RESULTS / "constrained_search" / "mode_results.csv")

    table = cs.groupby(["ranker", "beam_width", "split"])["solved_sas"].mean().reset_index()
    table_wide = table.pivot_table(
        index=["ranker", "split"], columns="beam_width", values="solved_sas"
    )
    table_wide.to_csv(ANALYSIS_DIR / "beam_width_curve.csv")
    print(table_wide.to_string(float_format="%.3f"))

    # Figure: line plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    ranker_colors = {"random": "#e74c3c", "goal_count": "#2ecc71", "teacher_frequency": "#3498db"}
    ranker_markers = {"random": "o", "goal_count": "s", "teacher_frequency": "^"}

    for ax, split in zip(axes, ["train", "heldout"]):
        split_data = table[table["split"] == split]
        for ranker in ["random", "goal_count", "teacher_frequency"]:
            rdata = split_data[split_data["ranker"] == ranker].sort_values("beam_width")
            if not rdata.empty:
                ax.plot(rdata["beam_width"], rdata["solved_sas"],
                        marker=ranker_markers.get(ranker, "o"), linewidth=2.5,
                        markersize=10, label=ranker, color=ranker_colors.get(ranker, "gray"))
        ax.set_xlabel("Beam Width", fontsize=12)
        ax.set_ylabel("SAS-Solved Rate" if split == "train" else "", fontsize=12)
        ax.set_title(f"{split.capitalize()} Domains", fontsize=13, fontweight="bold")
        ax.set_xticks([1, 4, 8])
        ax.set_ylim(0, 0.7)
        ax.legend(fontsize=10)

    fig.suptitle("Beam Width Scaling: Ranking vs Search Breadth", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig24_beam_width_scaling.png", dpi=DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 3. PER-DOMAIN BREAKDOWN TABLE
# ---------------------------------------------------------------------------
def per_domain_breakdown():
    print("\n=== 3. Per-Domain Breakdown ===")
    diag = pd.read_csv(RESULTS / "diagnostics" / "diagnostics.csv")
    cs = pd.read_csv(RESULTS / "constrained_search" / "mode_results.csv")
    repair = pd.read_csv(RESULTS / "repair" / "repair_results.csv")

    domains = sorted(diag["domain_name"].unique())
    rows = []
    for domain in domains:
        d_diag = diag[diag["domain_name"] == domain]
        llm_valid = d_diag["val_valid"].mean()
        n_instances = d_diag["problem_id"].nunique() if "problem_id" in d_diag.columns else len(d_diag)

        d_cs = cs[cs["domain_name"] == domain]
        random_bw8 = d_cs[(d_cs["ranker"] == "random") & (d_cs["beam_width"] == 8)]["solved_sas"].mean() if not d_cs.empty else 0
        gc_bw8 = d_cs[(d_cs["ranker"] == "goal_count") & (d_cs["beam_width"] == 8)]["solved_sas"].mean() if not d_cs.empty else 0
        tf_bw8 = d_cs[(d_cs["ranker"] == "teacher_frequency") & (d_cs["beam_width"] == 8)]["solved_sas"].mean() if not d_cs.empty else 0

        d_repair = repair[(repair["domain_name"] == domain) & (repair["ranker"] == "goal_count")]
        repair_gc = d_repair["repaired"].mean() if not d_repair.empty else 0

        prefix = d_diag["executable_prefix_len"].dropna()
        mean_prefix = prefix.mean() if not prefix.empty else 0
        nonzero_prefix_frac = (prefix > 0).mean() if not prefix.empty else 0

        notes = []
        if llm_valid > 0.1:
            notes.append("LLM solves trivial instances")
        if nonzero_prefix_frac > 0.8:
            notes.append(f"{nonzero_prefix_frac:.0%} nonzero prefix")
        if mean_prefix < 0.5:
            notes.append("near-zero prefix")
        mean_plan = d_diag["plan_len"].mean()
        if mean_plan > 100:
            notes.append(f"degenerate looping (mean {mean_plan:.0f} actions)")

        rows.append({
            "domain": domain,
            "n_instances": n_instances,
            "split": d_diag["split"].iloc[0] if not d_diag.empty else "",
            "LLM_valid": llm_valid,
            "random_bw8": random_bw8,
            "goal_count_bw8": gc_bw8,
            "teacher_freq_bw8": tf_bw8,
            "repair_goal_count": repair_gc,
            "mean_prefix_len": mean_prefix,
            "nonzero_prefix_frac": nonzero_prefix_frac,
            "notes": "; ".join(notes) if notes else "",
        })

    breakdown = pd.DataFrame(rows)
    breakdown.to_csv(ANALYSIS_DIR / "per_domain_breakdown.csv", index=False)
    print(breakdown.to_string(index=False, float_format="%.3f"))

    # Figure: comprehensive domain comparison
    heldout = breakdown[breakdown["split"] == "heldout"]
    if heldout.empty:
        heldout = breakdown

    fig, ax = plt.subplots(figsize=(14, 7))
    domains_plot = heldout["domain"].values
    x = np.arange(len(domains_plot))
    w = 0.15

    methods = [
        ("LLM_valid", "LLM Direct", "#e74c3c"),
        ("random_bw8", "Random bw=8", "#f39c12"),
        ("goal_count_bw8", "Goal-Count bw=8", "#27ae60"),
        ("teacher_freq_bw8", "Teacher-Freq bw=8", "#3498db"),
        ("repair_goal_count", "Repair (GC)", "#9b59b6"),
    ]

    for i, (col, label, color) in enumerate(methods):
        vals = heldout[col].values
        bars = ax.bar(x + i * w, vals, w, label=label, color=color, edgecolor="black", linewidth=0.3)
        for bar, v in zip(bars, vals):
            if v > 0.01:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + 2 * w)
    ax.set_xticklabels(domains_plot, rotation=30, ha="right", fontsize=11)
    ax.set_ylabel("Solved / Valid Rate", fontsize=13)
    ax.set_title("Per-Domain Method Comparison", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig25_per_domain_breakdown.png", dpi=DPI)
    plt.close()


# ---------------------------------------------------------------------------
# 4. FIRST-INVALID-STEP HISTOGRAM
# ---------------------------------------------------------------------------
def first_invalid_step_histogram():
    print("\n=== 4. First-Invalid-Step Histogram ===")
    diag = pd.read_csv(RESULTS / "diagnostics" / "diagnostics.csv")

    valid_mask = diag["val_valid"] == True
    has_fis = diag["first_invalid_step"].notna()
    no_actions = diag["plan_len"] == 0

    diag["fis_category"] = "other"
    diag.loc[valid_mask, "fis_category"] = "valid"
    diag.loc[no_actions & ~valid_mask, "fis_category"] = "no_actions"
    diag.loc[has_fis & (diag["first_invalid_step"] == 0), "fis_category"] = "step_0"
    diag.loc[has_fis & (diag["first_invalid_step"] == 1), "fis_category"] = "step_1"
    diag.loc[has_fis & (diag["first_invalid_step"] == 2), "fis_category"] = "step_2"
    diag.loc[has_fis & (diag["first_invalid_step"] >= 3), "fis_category"] = "step_3+"

    fis = diag["first_invalid_step"].dropna()
    print(f"  Total with first_invalid_step: {len(fis)}")
    print(f"  Step 0: {(fis == 0).sum()} ({(fis == 0).mean():.1%})")
    print(f"  Step 1: {(fis == 1).sum()} ({(fis == 1).mean():.1%})")
    print(f"  Step 2+: {(fis >= 2).sum()} ({(fis >= 2).mean():.1%})")
    print(f"  Valid: {valid_mask.sum()}")

    fis_stats = diag.groupby(["representation", "fis_category"]).size().unstack(fill_value=0)
    fis_stats.to_csv(ANALYSIS_DIR / "first_invalid_step_stats.csv")

    # Figure A: Overall histogram colored by representation
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    rep_colors = {"standard": "#e74c3c", "anonymized": "#3498db", "compact": "#2ecc71"}

    for ax, rep in zip(axes, ["standard", "anonymized", "compact"]):
        rep_data = diag[(diag["representation"] == rep) & diag["first_invalid_step"].notna()]
        fis_rep = rep_data["first_invalid_step"]
        if fis_rep.empty:
            ax.set_title(f"{rep} (no data)")
            continue

        bins = list(range(0, min(int(fis_rep.max()) + 2, 15))) + [fis_rep.max() + 1]
        ax.hist(fis_rep, bins=bins, color=rep_colors.get(rep, "gray"),
                edgecolor="black", linewidth=0.5, alpha=0.85)
        ax.axvline(fis_rep.mean(), color="black", linestyle="--", linewidth=2,
                   label=f"Mean: {fis_rep.mean():.1f}")
        ax.set_xlabel("First Invalid Step", fontsize=12)
        if ax == axes[0]:
            ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{rep.capitalize()}", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10)

    fig.suptitle("First Invalid Step by Representation", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig26_first_invalid_step_by_rep.png", dpi=DPI, bbox_inches="tight")
    plt.close()

    # Figure B: Stacked bar showing step-0 / step-1 / step-2+ / valid proportions by domain
    cat_order = ["no_actions", "step_0", "step_1", "step_2", "step_3+", "valid"]
    cat_colors = {"no_actions": "#bdc3c7", "step_0": "#e74c3c", "step_1": "#f39c12",
                  "step_2": "#e67e22", "step_3+": "#3498db", "valid": "#27ae60", "other": "#95a5a6"}

    domain_fis = diag.groupby(["domain_name", "fis_category"]).size().unstack(fill_value=0)
    domain_fis_frac = domain_fis.div(domain_fis.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    present_cats = [c for c in cat_order if c in domain_fis_frac.columns]
    if "other" in domain_fis_frac.columns and "other" not in present_cats:
        present_cats.append("other")
    bottom = np.zeros(len(domain_fis_frac))
    for cat in present_cats:
        if cat in domain_fis_frac.columns:
            vals = domain_fis_frac[cat].values
            ax.barh(domain_fis_frac.index, vals, left=bottom,
                    color=cat_colors.get(cat, "gray"), label=cat, edgecolor="white", linewidth=0.3)
            bottom += vals

    ax.set_xlabel("Fraction of Generations", fontsize=12)
    ax.set_title("Where Plans Fail: First Invalid Step by Domain", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig27_first_invalid_step_by_domain.png", dpi=DPI, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. PORTFOLIO LEAKAGE CHECK (GroupKFold by domain)
# ---------------------------------------------------------------------------
def portfolio_leakage_check():
    print("\n=== 5. Portfolio Leakage Check ===")
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    features_path = RESULTS / "portfolio" / "features.csv"
    labels_path = RESULTS / "portfolio" / "labels.csv"
    if not features_path.exists() or not labels_path.exists():
        print("  Portfolio data not found, skipping.")
        return

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    merged = features.merge(labels, on="instance_id", how="inner")
    if merged.empty or "best_mode" not in merged.columns:
        print("  No merged data, skipping.")
        return

    numeric_cols = [c for c in features.columns
                    if c not in ("instance_id", "domain_name", "split")
                    and features[c].dtype in ("float64", "int64", "float32", "int32")]
    X = merged[numeric_cols].fillna(0)
    y = merged["best_mode"]
    groups = merged["domain_name"]

    unique_groups = groups.nunique()
    print(f"  Unique domains for grouping: {unique_groups}")
    print(f"  Instances: {len(merged)}")
    print(f"  Labels: {y.value_counts().to_dict()}")

    models = {
        "dummy": Pipeline([("scaler", StandardScaler()), ("clf", DummyClassifier(strategy="most_frequent"))]),
        "logreg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))]),
        "rf": Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=SEED, class_weight="balanced"))]),
    }

    results_rows = []

    # GroupKFold by domain
    n_splits = min(5, unique_groups)
    if unique_groups >= 2 and n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for name, pipeline in models.items():
            fold_scores = []
            for train_idx, val_idx in gkf.split(X, y, groups):
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = pipeline.predict(X.iloc[val_idx])
                fold_scores.append(accuracy_score(y.iloc[val_idx], preds))
            results_rows.append({
                "method": f"GroupKFold(domain)_{name}",
                "mean_accuracy": np.mean(fold_scores),
                "std_accuracy": np.std(fold_scores),
                "n_folds": n_splits,
                "cv_type": "GroupKFold_by_domain",
            })
            print(f"  GroupKFold {name}: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

    # Leave-one-domain-out (most rigorous)
    if unique_groups >= 3:
        logo = LeaveOneGroupOut()
        for name, pipeline in models.items():
            fold_scores = []
            fold_details = []
            for train_idx, val_idx in logo.split(X, y, groups):
                holdout_domain = groups.iloc[val_idx[0]]
                pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
                preds = pipeline.predict(X.iloc[val_idx])
                score = accuracy_score(y.iloc[val_idx], preds)
                fold_scores.append(score)
                fold_details.append({"domain": holdout_domain, "accuracy": score, "n": len(val_idx)})
            results_rows.append({
                "method": f"LOGO_{name}",
                "mean_accuracy": np.mean(fold_scores),
                "std_accuracy": np.std(fold_scores),
                "n_folds": unique_groups,
                "cv_type": "LeaveOneDomainOut",
            })
            print(f"  LOGO {name}: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")

            if name == "rf":
                logo_details = pd.DataFrame(fold_details)
                logo_details.to_csv(ANALYSIS_DIR / "portfolio_logo_rf_details.csv", index=False)

    cv_df = pd.DataFrame(results_rows)
    cv_df.to_csv(ANALYSIS_DIR / "portfolio_leakage_check.csv", index=False)

    # Figure: comparison of CV methods
    if not cv_df.empty:
        fig, ax = plt.subplots(figsize=(12, 6))
        cv_types = cv_df["cv_type"].unique()
        type_colors = {"GroupKFold_by_domain": "#3498db", "LeaveOneDomainOut": "#e74c3c"}
        x_pos = 0
        tick_positions = []
        tick_labels = []
        for cv_type in cv_types:
            subset = cv_df[cv_df["cv_type"] == cv_type]
            for _, row in subset.iterrows():
                label = row["method"].split("_", 1)[-1] if "_" in row["method"] else row["method"]
                ax.bar(x_pos, row["mean_accuracy"],
                       yerr=row.get("std_accuracy", 0),
                       color=type_colors.get(cv_type, "gray"),
                       edgecolor="black", linewidth=0.5, capsize=4)
                ax.text(x_pos, row["mean_accuracy"] + row.get("std_accuracy", 0) + 0.02,
                        f"{row['mean_accuracy']:.1%}", ha="center", fontsize=10, fontweight="bold")
                tick_positions.append(x_pos)
                tick_labels.append(f"{label}\n({cv_type.replace('_', ' ')})")
                x_pos += 1
            x_pos += 0.5

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title("Portfolio Selector: Leakage-Aware Cross-Validation", fontsize=14, fontweight="bold")
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig28_portfolio_leakage_check.png", dpi=DPI)
        plt.close()


# ---------------------------------------------------------------------------
# BONUS: Updated money figure with all results
# ---------------------------------------------------------------------------
def updated_money_figure():
    print("\n=== Bonus: Updated Money Figure ===")
    diag = pd.read_csv(RESULTS / "diagnostics" / "diagnostics.csv")
    cs = pd.read_csv(RESULTS / "constrained_search" / "mode_results.csv")
    repair = pd.read_csv(RESULTS / "repair" / "repair_results.csv")

    heldout_diag = diag[diag["split"] == "heldout"]
    heldout_cs = cs[cs["split"] == "heldout"]
    heldout_repair = repair[repair["split"] == "heldout"] if not repair.empty else pd.DataFrame()

    ablation_path = ANALYSIS_DIR / "repair_ablation_summary.csv"
    ablation_rates = {}
    if ablation_path.exists():
        abl = pd.read_csv(ablation_path, index_col=0)
        for cond in abl.index:
            ablation_rates[cond] = abl.loc[cond, "success_rate"]

    methods = []

    methods.append(("LLM Direct\n(Qwen3-1.7B SFT)", heldout_diag["val_valid"].mean() if not heldout_diag.empty else 0))

    for ranker, bw in [("random", 1), ("random", 8), ("goal_count", 1), ("goal_count", 8), ("teacher_frequency", 8)]:
        subset = heldout_cs[(heldout_cs["ranker"] == ranker) & (heldout_cs["beam_width"] == bw)]
        rate = subset["solved_sas"].mean() if not subset.empty else 0
        label = f"{ranker.replace('_', ' ').title()}\nbw={bw}"
        methods.append((label, rate))

    if not heldout_repair.empty:
        gc_repair = heldout_repair[heldout_repair["ranker"] == "goal_count"]
        repair_rate = gc_repair["repaired"].mean() if not gc_repair.empty else 0
        methods.append((f"Prefix Repair\n(Goal-Count)", repair_rate))

    if "empty_prefix" in ablation_rates:
        methods.append(("Empty→Complete\n(Goal-Count bw=4)", ablation_rates["empty_prefix"]))
    if "llm_prefix" in ablation_rates:
        methods.append(("LLM Prefix→Complete\n(Goal-Count bw=4)", ablation_rates["llm_prefix"]))

    labels = [m[0] for m in methods]
    rates = [m[1] for m in methods]

    fig, ax = plt.subplots(figsize=(14, 8))

    colors = []
    for label in labels:
        if "LLM Direct" in label:
            colors.append("#e74c3c")
        elif "Random" in label:
            colors.append("#f39c12")
        elif "Goal" in label or "Empty" in label:
            colors.append("#27ae60")
        elif "Teacher" in label:
            colors.append("#3498db")
        elif "Repair" in label or "Prefix" in label:
            colors.append("#9b59b6")
        else:
            colors.append("#95a5a6")

    bars = ax.barh(range(len(labels)), rates, color=colors, edgecolor="black", linewidth=0.5, height=0.7)
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{rate:.1%}", va="center", fontsize=11, fontweight="bold")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Solved / Valid Rate (Heldout Domains)", fontsize=13)
    ax.set_title("Method Comparison: From SFT Failure to Structural Recovery",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, max(rates) * 1.15 + 0.05)
    ax.invert_yaxis()

    ax.axvline(x=rates[0], color="#e74c3c", linestyle=":", alpha=0.4, linewidth=1.5)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig29_money_figure.png", dpi=DPI)
    plt.close()
    print(f"  Methods: {len(methods)}")
    for label, rate in methods:
        print(f"    {label.replace(chr(10), ' ')}: {rate:.1%}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_repair_ablation()
    beam_width_curve()
    per_domain_breakdown()
    first_invalid_step_histogram()
    portfolio_leakage_check()
    updated_money_figure()
    print("\n=== All final experiments complete ===")
    print(f"Figures: {FIG_DIR}")
    print(f"Analysis: {ANALYSIS_DIR}")
