from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from planning_pivot.features import extract_instance_features
from planning_pivot.sas import parse_output_sas


def build_mode_label_table(mode_results: pd.DataFrame) -> pd.DataFrame:
    if mode_results.empty:
        return pd.DataFrame(columns=["instance_id", "best_mode"])

    rows: list[dict] = []
    for instance_id, group in mode_results.groupby("instance_id"):
        solved = group[group["solved_sas"] == True]
        if not solved.empty:
            best = solved.sort_values(["seconds", "plan_len"]).iloc[0]
        else:
            best = group.sort_values("seconds").iloc[0]
        rows.append({
            "instance_id": instance_id,
            "best_mode": best["mode"],
        })
    return pd.DataFrame(rows)


def fit_portfolio_selector(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
) -> dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score

    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    X = features[numeric_cols].fillna(0)

    models = {
        "dummy": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "logreg": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=574)),
        ]),
        "rf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=200, random_state=574, class_weight="balanced",
            )),
        ]),
    }

    unique_groups = groups.nunique()
    n_splits = min(5, unique_groups) if unique_groups > 1 else 2

    results: dict[str, Any] = {"cv_scores": {}}
    best_score = -1.0
    best_name = "dummy"

    for name, pipeline in models.items():
        fold_scores: list[float] = []

        if unique_groups > 1 and n_splits >= 2:
            gkf = GroupKFold(n_splits=n_splits)
            for train_idx, val_idx in gkf.split(X, labels, groups):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = labels.iloc[train_idx], labels.iloc[val_idx]
                pipeline.fit(X_train, y_train)
                preds = pipeline.predict(X_val)
                fold_scores.append(accuracy_score(y_val, preds))
        else:
            pipeline.fit(X, labels)
            preds = pipeline.predict(X)
            fold_scores.append(accuracy_score(labels, preds))

        mean_score = np.mean(fold_scores)
        results["cv_scores"][name] = {
            "mean_accuracy": float(mean_score),
            "fold_scores": fold_scores,
        }

        if mean_score > best_score:
            best_score = mean_score
            best_name = name

    final_model = models[best_name]
    final_model.fit(X, labels)
    results["best_model_name"] = best_name
    results["best_model"] = final_model
    results["feature_columns"] = numeric_cols

    return results


def evaluate_portfolio_selector(
    selector: Any,
    features: pd.DataFrame,
    labels: pd.Series,
    mode_results: pd.DataFrame,
) -> pd.DataFrame:
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    X = features[numeric_cols].fillna(0)

    predictions = selector.predict(X)
    rows: list[dict] = []

    for i, (idx, row) in enumerate(features.iterrows()):
        instance_id = row.get("instance_id", idx)
        predicted_mode = predictions[i]
        oracle_mode = labels.iloc[i] if i < len(labels) else "unknown"

        inst_results = mode_results[mode_results["instance_id"] == instance_id]
        pred_results = inst_results[inst_results["mode"] == predicted_mode]
        oracle_results = inst_results[inst_results["mode"] == oracle_mode]

        pred_solved = pred_results["solved_sas"].any() if not pred_results.empty else False
        oracle_solved = oracle_results["solved_sas"].any() if not oracle_results.empty else False

        rows.append({
            "instance_id": instance_id,
            "predicted_mode": predicted_mode,
            "oracle_mode": oracle_mode,
            "predicted_solved": pred_solved,
            "oracle_solved": oracle_solved,
            "regret": int(oracle_solved) - int(pred_solved),
        })

    return pd.DataFrame(rows)


def run_portfolio_experiment(config) -> pd.DataFrame:
    from planning_pivot.diagnostics import build_instance_index

    output_root = Path(config.data.output_root)
    port_dir = output_root / "portfolio"
    port_dir.mkdir(parents=True, exist_ok=True)

    cs_path = output_root / "constrained_search" / "mode_results.csv"
    if not cs_path.exists():
        empty = pd.DataFrame()
        for fname in ["features.csv", "labels.csv", "cv_results.csv", "heldout_results.csv", "summary.csv"]:
            empty.to_csv(port_dir / fname, index=False)
        return empty

    mode_results = pd.read_csv(cs_path)
    instances = build_instance_index(config)
    sas_cache_dir = output_root / "sas_cache"

    diag_path = output_root / "diagnostics" / "diagnostics.csv"
    diag_df = pd.read_csv(diag_path) if diag_path.exists() else None

    feature_rows: list[dict] = []
    seen = set()
    for _, irow in instances.iterrows():
        iid = str(irow["instance_id"])
        base = iid.rsplit("_", 1)[0] if "_" in iid else iid
        if base in seen:
            continue
        seen.add(base)

        sas_path = sas_cache_dir / base / "output.sas"
        task = None
        if sas_path.exists():
            try:
                task = parse_output_sas(sas_path)
            except Exception:
                pass

        feats = extract_instance_features(irow, task, diag_df)
        feats["instance_id"] = base
        feature_rows.append(feats)

    features_df = pd.DataFrame(feature_rows)
    features_df.to_csv(port_dir / "features.csv", index=False)

    labels_df = build_mode_label_table(mode_results)
    labels_df.to_csv(port_dir / "labels.csv", index=False)

    merged = features_df.merge(labels_df, on="instance_id", how="inner")
    if merged.empty:
        for fname in ["cv_results.csv", "heldout_results.csv", "summary.csv"]:
            pd.DataFrame().to_csv(port_dir / fname, index=False)
        return pd.DataFrame()

    feature_cols = [c for c in features_df.columns if c not in ("instance_id", "domain_name", "split")]
    X = merged[feature_cols] if all(c in merged.columns for c in feature_cols) else merged.select_dtypes(include=[np.number])
    y = merged["best_mode"]
    groups = merged["domain_name"] if "domain_name" in merged.columns else pd.Series(range(len(merged)))

    selector_info = fit_portfolio_selector(X, y, groups)

    cv_rows = []
    for name, scores in selector_info["cv_scores"].items():
        cv_rows.append({
            "model": name,
            "mean_accuracy": scores["mean_accuracy"],
            "n_folds": len(scores["fold_scores"]),
        })
    pd.DataFrame(cv_rows).to_csv(port_dir / "cv_results.csv", index=False)

    joblib.dump(selector_info["best_model"], port_dir / "selector.joblib")

    eval_df = evaluate_portfolio_selector(
        selector_info["best_model"], X, y, mode_results,
    )
    eval_df.to_csv(port_dir / "heldout_results.csv", index=False)

    summary_rows = [{
        "best_model": selector_info["best_model_name"],
        "selected_solved_rate": eval_df["predicted_solved"].mean() if not eval_df.empty else 0.0,
        "oracle_solved_rate": eval_df["oracle_solved"].mean() if not eval_df.empty else 0.0,
        "mean_regret": eval_df["regret"].mean() if not eval_df.empty else 0.0,
    }]
    pd.DataFrame(summary_rows).to_csv(port_dir / "summary.csv", index=False)

    return eval_df
