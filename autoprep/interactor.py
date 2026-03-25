from __future__ import annotations

from typing import Callable, Dict, List


class HumanPrompter:
    """Build human-in-the-loop action plans. All decisions are user-driven (no auto mode)."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ):
        self.input_fn = input_fn
        self.output_fn = output_fn
        self._human_tasks: dict = {"missing": {}, "cardinality": {}, "patterns": {}}
        self._human_decisions: dict = {"missing": {}, "cardinality": {}, "patterns": {}}

    def parse_report(self, health_report: dict) -> dict:
        """All issues → human_tasks (no auto mode). User decides everything."""
        self._human_tasks = {"missing": {}, "cardinality": {}, "patterns": {}}

        # All missing data → ask user (green, yellow, red)
        for col_name, details in health_report.get("missing_data", {}).items():
            missing_pct = float(details.get("missing_pct", 0.0))
            traffic_light = str(details.get("traffic_light", "")).lower()
            self._human_tasks["missing"][col_name] = {
                "missing_pct": missing_pct,
                "traffic_light": traffic_light,
            }

        # All cardinality issues → ask user
        for col_name, details in health_report.get("cardinality", {}).items():
            unique_count = int(details.get("n_unique", 0))
            self._human_tasks["cardinality"][col_name] = {
                "unique_count": unique_count,
            }

        return {
            "human_tasks": self._human_tasks,
        }

    def prompt_missing_yellow_zone(self, col_name: str, pct: float, traffic_light: str = "Yellow") -> dict:
        """Ask user what to do for ANY missing data (regardless of traffic light)."""
        light_color = traffic_light.lower()
        
        if light_color == "green":
            prompt = (
                f"Column [{col_name}] has [{pct:.2f}]% missing (LOW risk). "
                "Choose: [1] Drop [2] Basic Impute [3] Smart KNN Impute: "
            )
        elif light_color == "red":
            prompt = (
                f"Column [{col_name}] has [{pct:.2f}]% missing (HIGH risk). "
                "Choose: [1] Drop [2] Basic Impute [3] Smart KNN Impute: "
            )
        else:  # yellow
            prompt = (
                f"Column [{col_name}] has [{pct:.2f}]% missing (MODERATE risk). "
                "Choose: [1] Drop [2] Basic Impute [3] Smart KNN Impute: "
            )

        choice = self._prompt_choice(prompt, valid={"1", "2", "3"})

        mapping = {
            "1": {"action": "drop_column"},
            "2": {"action": "basic_impute"},
            "3": {"action": "smart_knn_impute"},
        }
        decision = {"column": col_name, "missing_pct": round(float(pct), 2), "traffic_light": traffic_light, **mapping[choice]}
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
        """Compile all human decisions into cleaner and encoder instructions."""
        missing_actions = {}
        cardinality_actions = {}
        pattern_mappings = {}

        # Missing data decisions
        for col_name, task in self._human_decisions.get("missing", {}).items():
            missing_actions[col_name] = task["action"]

        # Cardinality decisions
        for col_name, task in self._human_decisions.get("cardinality", {}).items():
            cardinality_actions[col_name] = task["action"]

        # Pattern decisions (optional mappings)
        for col_name, task in self._human_decisions.get("patterns", {}).items():
            pattern_mappings[col_name] = task

        cleaner_instructions = {
            "drop_columns": sorted(
                [
                    col
                    for col, action in {**missing_actions, **cardinality_actions}.items()
                    if action == "drop_column"
                ] + [col for col, task in pattern_mappings.items() if task.get("action") == "skip_column"]
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
            "pattern_mappings": pattern_mappings,
            "knn_k": 5,
        }

        return {
            "human_tasks": self._human_tasks,
            "human_decisions": self._human_decisions,
            "actions": {
                "missing": missing_actions,
                "cardinality": cardinality_actions,
                "patterns": pattern_mappings,
            },
            "cleaner_instructions": cleaner_instructions,
        }

    def prompt_numeric_text_pattern(self, col_name: str, samples: List[str], match_pct: float) -> dict:
        """Ask user how to normalize numeric text (5 million, 2 lakh, etc)."""
        self.output_fn(f"\n🔍 Column: [{col_name}] has NUMERIC TEXT patterns ({match_pct:.1f}% match)")
        self.output_fn(f"   Examples: {samples}")
        
        prompt = (
            f"How to handle [{col_name}]? "
            "[1] Skip this column [2] Auto-normalize (million→1M, lakh→100K) [3] Define custom mapping: "
        )
        choice = self._prompt_choice(prompt, valid={"1", "2", "3"})

        if choice == "1":
            decision = {"column": col_name, "action": "skip_column", "reason": "user_skip", "pattern": "numeric_text"}
        elif choice == "2":
            decision = {"column": col_name, "action": "normalize_numeric_text", "pattern": "numeric_text"}
        else:  # choice == "3"
            mappings = self._collect_numeric_mapping()
            decision = {
                "column": col_name, 
                "action": "normalize_numeric_text",
                "pattern": "numeric_text",
                "custom_mappings": mappings
            }

        self._human_decisions["patterns"][col_name] = decision
        return decision

    def prompt_categorical_variations(self, col_name: str, detected_category: str, 
                                     variations: List[str], samples: List[str], 
                                     match_pct: float) -> dict:
        """Ask user how to normalize categorical variations (m/male/man, f/female, etc)."""
        self.output_fn(f"\n🔍 Column: [{col_name}] looks like [{detected_category}] ({match_pct:.1f}% match)")
        self.output_fn(f"   Variations found: {variations}")
        self.output_fn(f"   Examples: {samples}")
        
        prompt = (
            f"How to handle [{col_name}]? "
            "[1] Skip [2] Auto-standardize as '{detected_category}' [3] Define custom mapping: "
        )
        choice = self._prompt_choice(prompt, valid={"1", "2", "3"})

        if choice == "1":
            decision = {
                "column": col_name,
                "action": "skip_column",
                "reason": "user_skip",
                "pattern": "categorical_mixed"
            }
        elif choice == "2":
            decision = {
                "column": col_name,
                "action": "standardize_categorical",
                "pattern": "categorical_mixed",
                "detected_category": detected_category
            }
        else:  # choice == "3"
            mappings = self._collect_categorical_mapping(col_name)
            decision = {
                "column": col_name,
                "action": "standardize_categorical",
                "pattern": "categorical_mixed",
                "custom_mappings": mappings
            }

        self._human_decisions["patterns"][col_name] = decision
        return decision

    def _collect_numeric_mapping(self) -> Dict[str, float]:
        """Interactively build numeric text → number mapping."""
        mappings = {}
        self.output_fn("Enter custom mappings (enter 'done' when finished):")
        
        while True:
            text_val = self.input_fn("  Text value (e.g., '5 million'): ").strip()
            if text_val.lower() == "done":
                break
            
            num_val_str = self.input_fn(f"  → Numeric value for '{text_val}': ").strip()
            try:
                mappings[text_val] = float(num_val_str)
            except ValueError:
                self.output_fn(f"  ❌ Invalid number '{num_val_str}'. Try again.")
        
        return mappings

    def _collect_categorical_mapping(self, col_name: str) -> Dict[str, str]:
        """Interactively build categorical variation → standard mapping."""
        mappings = {}
        self.output_fn(f"Define mappings for [{col_name}] (enter 'done' when finished):")
        
        while True:
            variant = self.input_fn("  Variant value (e.g., 'm'): ").strip()
            if variant.lower() == "done":
                break
            
            standard = self.input_fn(f"  → Standard value for '{variant}': ").strip()
            mappings[variant.lower()] = standard
        
        return mappings

    def _prompt_choice(self, prompt: str, valid: set[str]) -> str:
        while True:
            choice = self.input_fn(prompt).strip()
            if choice in valid:
                return choice
            self.output_fn(f"Invalid choice '{choice}'. Valid options: {sorted(valid)}")
