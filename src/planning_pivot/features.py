from __future__ import annotations

from typing import Any

import pandas as pd

from planning_pivot.sas import SasTask, applicable_operators


def extract_instance_features(
    instance_row: pd.Series,
    task: SasTask | None,
    diagnostics_df: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    features: dict[str, Any] = {}

    features["domain_name"] = str(instance_row.get("domain_name", "unknown"))
    features["split"] = str(instance_row.get("split", "unknown"))

    teacher_len_val = instance_row.get("teacher_plan_len")
    features["teacher_plan_len"] = int(teacher_len_val) if pd.notna(teacher_len_val) else 0

    if task is not None:
        features["n_sas_variables"] = len(task.variables)
        features["n_sas_operators"] = len(task.operators)
        features["n_goal_atoms"] = len(task.goals)

        init_applicable = applicable_operators(task, task.init)
        features["initial_applicable_count"] = len(init_applicable)

        costs = [op.cost for op in task.operators]
        features["mean_operator_cost"] = sum(costs) / len(costs) if costs else 0.0
        features["max_operator_cost"] = max(costs) if costs else 0
    else:
        features["n_sas_variables"] = 0
        features["n_sas_operators"] = 0
        features["n_goal_atoms"] = 0
        features["initial_applicable_count"] = 0
        features["mean_operator_cost"] = 0.0
        features["max_operator_cost"] = 0

    domain_name = features["domain_name"]
    if diagnostics_df is not None and not diagnostics_df.empty:
        dom_rows = diagnostics_df[diagnostics_df["domain_name"] == domain_name]
        if not dom_rows.empty:
            features["empty_plan_rate_by_domain"] = (dom_rows["surface_status"] == "empty_plan").mean()
            if "schema_valid_action_frac" in dom_rows.columns:
                features["schema_valid_frac_by_domain"] = dom_rows["schema_valid_action_frac"].mean()
            else:
                features["schema_valid_frac_by_domain"] = 0.0

            compact_rows = dom_rows[dom_rows["representation"] == "compact"]
            if not compact_rows.empty:
                features["compact_parsed_action_rate"] = (
                    compact_rows["surface_status"] == "parsed_actions"
                ).mean()
            else:
                features["compact_parsed_action_rate"] = 0.0
        else:
            features["empty_plan_rate_by_domain"] = 0.0
            features["schema_valid_frac_by_domain"] = 0.0
            features["compact_parsed_action_rate"] = 0.0
    else:
        features["empty_plan_rate_by_domain"] = 0.0
        features["schema_valid_frac_by_domain"] = 0.0
        features["compact_parsed_action_rate"] = 0.0

    return features
