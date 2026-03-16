from __future__ import annotations

from typing import Callable


class HumanPrompter:
    """Build human-in-the-loop action plans from a health report."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ):
        self.input_fn = input_fn
        self.output_fn = output_fn
        self._auto_tasks: dict = {"missing": {}, "cardinality": {}}
        self._human_tasks: dict = {"missing": {}, "cardinality": {}}
        self._human_decisions: dict = {"missing": {}, "cardinality": {}}

    def parse_report(self, health_report: dict) -> dict:
        """Separate auto-resolved tasks from human-required tasks."""
        self._auto_tasks = {"missing": {}, "cardinality": {}}
        self._human_tasks = {"missing": {}, "cardinality": {}}

        for col_name, details in health_report.get("missing_data", {}).items():
            label = str(details.get("traffic_light", "")).lower()
            missing_pct = float(details.get("missing_pct", 0.0))

            if label == "green":
                self._auto_tasks["missing"][col_name] = {
                    "action": "basic_impute",
                    "reason": "green_zone",
                    "missing_pct": missing_pct,
                }
            elif label == "red":
                self._auto_tasks["missing"][col_name] = {
                    "action": "drop_column",
                    "reason": "red_zone",
                    "missing_pct": missing_pct,
                }
            elif label == "yellow":
                self._human_tasks["missing"][col_name] = {
                    "missing_pct": missing_pct,
                    "traffic_light": "Yellow",
                }

        for col_name, details in health_report.get("cardinality", {}).items():
            unique_count = int(details.get("n_unique", 0))
            ask_human = bool(details.get("ask_human", False))

            if ask_human:
                self._human_tasks["cardinality"][col_name] = {
                    "unique_count": unique_count,
                }
            else:
                self._auto_tasks["cardinality"][col_name] = {
                    "action": "auto_encode",
                    "reason": "within_limit",
                    "unique_count": unique_count,
                }

        return {
            "auto_tasks": self._auto_tasks,
            "human_tasks": self._human_tasks,
        }

    def prompt_missing_yellow_zone(self, col_name: str, pct: float) -> dict:
        """Ask user what to do for yellow-zone missingness."""
        prompt = (
            f"Column [{col_name}] has [{pct:.2f}]% missing data. "
            "Choose: [1] Drop [2] Basic Impute [3] Smart KNN Impute: "
        )
        choice = self._prompt_choice(prompt, valid={"1", "2", "3"})

        mapping = {
            "1": {"action": "drop_column"},
            "2": {"action": "basic_impute"},
            "3": {"action": "smart_knn_impute"},
        }
        decision = {"column": col_name, "missing_pct": round(float(pct), 2), **mapping[choice]}
        self._human_decisions["missing"][col_name] = decision
        return decision

    def prompt_high_cardinality(self, col_name: str, unique_count: int) -> dict:
        """Ask user what to do for high-cardinality columns."""
        prompt = (
            f"Column [{col_name}] has[{unique_count}] unique categories. "
            "Choose: [1] Drop [2] Keep Top 10 [3] Auto-Encode anyway: "
        )
        choice = self._prompt_choice(prompt, valid={"1", "2", "3"})

        mapping = {
            "1": {"action": "drop_column"},
            "2": {"action": "keep_top_10"},
            "3": {"action": "auto_encode"},
        }
        decision = {"column": col_name, "unique_count": int(unique_count), **mapping[choice]}
        self._human_decisions["cardinality"][col_name] = decision
        return decision

    def build_action_plan(self) -> dict:
        """Compile auto-decisions and human decisions into cleaner instructions."""
        missing_actions = {}
        cardinality_actions = {}

        for col_name, task in self._auto_tasks.get("missing", {}).items():
            missing_actions[col_name] = task["action"]
        for col_name, task in self._human_decisions.get("missing", {}).items():
            missing_actions[col_name] = task["action"]

        for col_name, task in self._auto_tasks.get("cardinality", {}).items():
            cardinality_actions[col_name] = task["action"]
        for col_name, task in self._human_decisions.get("cardinality", {}).items():
            cardinality_actions[col_name] = task["action"]

        cleaner_instructions = {
            "drop_columns": sorted(
                [
                    col
                    for col, action in {**missing_actions, **cardinality_actions}.items()
                    if action == "drop_column"
                ]
            ),
            "missing_imputation": {
                col: action
                for col, action in missing_actions.items()
                if action in {"basic_impute", "smart_knn_impute"}
            },
            "cardinality_handling": {
                col: action
                for col, action in cardinality_actions.items()
                if action in {"keep_top_10", "auto_encode"}
            },
            "knn_k": 5,
        }

        return {
            "auto_tasks": self._auto_tasks,
            "human_tasks": self._human_tasks,
            "human_decisions": self._human_decisions,
            "actions": {
                "missing": missing_actions,
                "cardinality": cardinality_actions,
            },
            "cleaner_instructions": cleaner_instructions,
        }

    def _prompt_choice(self, prompt: str, valid: set[str]) -> str:
        while True:
            choice = self.input_fn(prompt).strip()
            if choice in valid:
                return choice
            self.output_fn(f"Invalid choice '{choice}'. Valid options: {sorted(valid)}")
