"""Streamlit-based interactive prompter for human-in-the-loop mode"""
import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
from autoprep.interactor import HumanPrompter as BaseHumanPrompter


class StreamlitHumanPrompter(BaseHumanPrompter):
    """
    Streamlit-compatible version of HumanPrompter.
    Uses Streamlit UI (radio buttons) instead of terminal input().
    Pre-collects decisions via UI, then pipeline can query them without blocking.
    """
    
    def __init__(self, streamlit_decisions: dict = None):
        super().__init__()
        # Pre-made decisions from Streamlit UI
        self.streamlit_decisions = streamlit_decisions or {
            "missing": {}, 
            "cardinality": {}, 
            "patterns": {}
        }
        # Initialize the parent's decision tracking
        self._human_decisions = {"missing": {}, "cardinality": {}, "patterns": {}}
        self._human_tasks = {"missing": {}, "cardinality": {}, "patterns": {}}
    
    def prompt_missing_yellow_zone(self, col_name: str, pct: float, traffic_light: str = "Yellow") -> dict:
        """Return pre-decided action from Streamlit UI."""
        if col_name in self.streamlit_decisions.get("missing", {}):
            decision_data = self.streamlit_decisions["missing"][col_name]
            action = decision_data.get("action", "basic_impute")
        else:
            action = "basic_impute"  # Default if not in decisions
        
        # Track decision
        decision = {
            "column": col_name,
            "action": action,
            "missing_pct": round(float(pct), 2),
            "traffic_light": traffic_light,
        }
        self._human_decisions["missing"][col_name] = decision
        return decision
    
    def prompt_high_cardinality(self, col_name: str, unique_count: int) -> dict:
        """Return pre-decided action from Streamlit UI."""
        if col_name in self.streamlit_decisions.get("cardinality", {}):
            decision_data = self.streamlit_decisions["cardinality"][col_name]
            action = decision_data.get("action", "auto_encode")
        else:
            action = "auto_encode"  # Default if not in decisions
        
        # Track decision
        decision = {
            "column": col_name,
            "action": action,
            "unique_count": int(unique_count),
        }
        self._human_decisions["cardinality"][col_name] = decision
        return decision

    def prompt_numeric_text_pattern(self, col_name: str, samples: list, match_pct: float) -> dict:
        """Apply pre-decided pattern action from Streamlit UI."""
        if col_name in self.streamlit_decisions.get("patterns", {}):
            decision_data = self.streamlit_decisions["patterns"][col_name]
            decision = {
                "column": col_name,
                "pattern": "numeric_text",
                "action": decision_data.get("action", "normalize_numeric_text"),
                "custom_mappings": decision_data.get("custom_mappings")
            }
        else:
            decision = {
                "column": col_name,
                "pattern": "numeric_text",
                "action": "normalize_numeric_text"
            }
        
        self._human_decisions["patterns"][col_name] = decision
        return decision

    def prompt_categorical_variations(self, col_name: str, detected_category: str, 
                                     variations: list, samples: list, match_pct: float) -> dict:
        """Apply pre-decided categorical action from Streamlit UI."""
        if col_name in self.streamlit_decisions.get("patterns", {}):
            decision_data = self.streamlit_decisions["patterns"][col_name]
            decision = {
                "column": col_name,
                "pattern": "categorical_mixed",
                "detected_category": detected_category,
                "action": decision_data.get("action", "standardize_categorical"),
                "custom_mappings": decision_data.get("custom_mappings")
            }
        else:
            decision = {
                "column": col_name,
                "pattern": "categorical_mixed",
                "detected_category": detected_category,
                "action": "standardize_categorical"
            }
        
        self._human_decisions["patterns"][col_name] = decision
        return decision


    def review_ai_mapping_categories(
        self, col_name: str, ai_mapping: Dict[str, str], unique_values: list
    ) -> Dict[str, str]:
        """
        Display AI-generated category mapping in Streamlit data_editor for human review.
        
        Args:
            col_name: Column name
            ai_mapping: Dictionary mapping original → standardized values
            unique_values: List of unique values from column
        
        Returns:
            Approved mapping dictionary (may be edited by user)
        """
        st.subheader(f"🤖 AI-Assisted Standardization: {col_name}")
        
        # Build DataFrame for editing
        edit_data = []
        for orig_val in unique_values:
            standardized = ai_mapping.get(orig_val, orig_val)
            edit_data.append({
                "Original Value": orig_val,
                "AI Proposed Value": standardized,
                "Approve": True
            })
        
        edit_df = pd.DataFrame(edit_data)
        
        st.info(
            f"👆 Review the AI-generated mappings below. "
            f"Edit any proposed values, mark False to skip mapping for that row."
        )
        
        # Let user edit the mapping
        edited_df = st.data_editor(
            edit_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Original Value": st.column_config.TextColumn(
                    "Original Value",
                    disabled=True,  # Don't allow editing original values
                    width="medium"
                ),
                "AI Proposed Value": st.column_config.TextColumn(
                    "AI Proposed Value",
                    width="medium",
                    help="Edit this to change the standardized value"
                ),
                "Approve": st.column_config.CheckboxColumn(
                    "Apply?",
                    width="small",
                    help="Uncheck to skip mapping for this value"
                )
            },
            key=f"ai_mapping_editor_{col_name}"
        )
        
        # Convert back to dictionary, only including approved rows
        approved_mapping = {}
        for idx, row in edited_df.iterrows():
            if row["Approve"]:
                approved_mapping[row["Original Value"]] = row["AI Proposed Value"]
        
        # Summary
        st.caption(f"✅ {len(approved_mapping)} mappings approved, {len(edit_df) - len(approved_mapping)} skipped")
        
        return approved_mapping

    def review_ai_mapping_numbers(
        self, col_name: str, ai_mapping: Dict[str, Any], unique_values: list
    ) -> Dict[str, Any]:
        """
        Display AI-generated numeric mapping in Streamlit data_editor for human review.
        
        Args:
            col_name: Column name
            ai_mapping: Dictionary mapping original → numeric values
            unique_values: List of unique values from column
        
        Returns:
            Approved mapping dictionary (may be edited by user)
        """
        st.subheader(f"🤖 AI-Assisted Numeric Standardization: {col_name}")
        
        # Build DataFrame for editing
        edit_data = []
        for orig_val in unique_values:
            standardized = ai_mapping.get(orig_val, None)
            edit_data.append({
                "Original Value": str(orig_val),
                "AI Proposed Number": standardized,
                "Approve": True
            })
        
        edit_df = pd.DataFrame(edit_data)
        
        st.info(
            f"👆 Review the AI-generated mappings below. "
            f"Edit any numeric values, mark False to skip mapping for that row."
        )
        
        # Let user edit the mapping
        edited_df = st.data_editor(
            edit_df,
            use_container_width=True,
            num_rows="fixed",
            column_config={
                "Original Value": st.column_config.TextColumn(
                    "Original Value",
                    disabled=True,
                    width="medium"
                ),
                "AI Proposed Number": st.column_config.NumberColumn(
                    "AI Proposed Number",
                    width="medium",
                    help="Edit this to change the numeric value",
                    format="%.2f"
                ),
                "Approve": st.column_config.CheckboxColumn(
                    "Apply?",
                    width="small",
                    help="Uncheck to skip mapping for this value"
                )
            },
            key=f"ai_mapping_numbers_{col_name}"
        )
        
        # Convert back to dictionary, only including approved rows
        approved_mapping = {}
        for idx, row in edited_df.iterrows():
            if row["Approve"]:
                orig = row["Original Value"]
                proposed = row["AI Proposed Number"]
                if pd.notna(proposed):
                    approved_mapping[orig] = proposed
        
        # Summary
        st.caption(f"✅ {len(approved_mapping)} mappings approved, {len(edit_df) - len(approved_mapping)} skipped")
        
        return approved_mapping

    def prompt_ai_fallback_warning(self):
        """Display warning when AI is unavailable and fallback is used."""
        st.warning(
            "⚠️ **AI Engine Unavailable** — Reverting to basic text cleaning. "
            "Applying: lowercase, trim whitespace, remove special characters. "
            "For custom mappings, use the manual tools below."
        )


def collect_user_decisions_streamlit(health_report: dict, ambiguous_columns: dict = None) -> dict:
    """
    Collect MINIMAL user decisions via Streamlit UI (auto-handle low-risk columns).
    Only ask about columns that actually need attention.
    
    Args:
        health_report: Data quality profiling results
        ambiguous_columns: Pattern detection results (numeric text, categorical variations)
    
    Returns:
        decisions dict with user choices (and auto-decisions for low-risk columns)
    """
    decisions = {"missing": {}, "cardinality": {}, "patterns": {}}
    ambiguous_columns = ambiguous_columns or {}
    
    # === MISSING DATA: Only ask about YELLOW & RED (skip GREEN) ===
    missing_data = health_report.get("missing_data", {})
    
    if missing_data:
        # Filter: only Yellow and Red zones need human input
        problem_cols = {
            col: details for col, details in missing_data.items()
            if details.get("traffic_light", "").lower() in ["yellow", "red"]
        }
        
        if problem_cols:
            st.subheader("📊 Missing Data — Needs Your Decision")
            
            for col, details in problem_cols.items():
                pct = details.get("missing_pct", 0)
                traffic_light = details.get("traffic_light", "Yellow").lower()
                
                if traffic_light == "red":
                    emoji = "🔴"
                    default_action = "drop_column"
                    suggestion = " (Recommend: Drop)"
                else:
                    emoji = "🟡"
                    default_action = "basic_impute"
                    suggestion = " (Recommend: Basic Impute)"
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"{emoji} **{col}** — {pct:.1f}% missing{suggestion}")
                with col2:
                    choice = st.selectbox(
                        "Action",
                        ["Drop column", "Basic Impute", "KNN Impute"],
                        key=f"missing_{col}",
                        label_visibility="collapsed"
                    )
                
                mapping = {
                    "Drop column": "drop_column",
                    "Basic Impute": "basic_impute",
                    "KNN Impute": "smart_knn_impute"
                }
                decisions["missing"][col] = {
                    "action": mapping.get(choice, default_action),
                    "missing_pct": pct,
                    "traffic_light": details.get("traffic_light")
                }
            
            st.divider()
        
        # Auto-decide GREEN zone columns (low risk)
        green_cols = {
            col: details for col, details in missing_data.items()
            if details.get("traffic_light", "").lower() == "green"
        }
        for col, details in green_cols.items():
            decisions["missing"][col] = {
                "action": "basic_impute",
                "missing_pct": details.get("missing_pct", 0),
                "traffic_light": "Green"
            }
    
    # === CARDINALITY: Only ask about HIGH-CARDINALITY columns ===
    cardinality = health_report.get("cardinality", {})
    
    if cardinality:
        # Filter: only columns exceeding the limit
        problem_cols = {
            col: details for col, details in cardinality.items()
            if details.get("ask_human")  # ask_human=True when n_unique > cardinality_limit
        }
        
        if problem_cols:
            st.subheader("🏷️ High-Cardinality Columns — Needs Your Decision")
            
            for col, details in problem_cols.items():
                unique_count = details.get("n_unique", 0)
                limit = details.get("cardinality_limit", 50)
                
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.write(f"**{col}** — {unique_count} unique values (limit: {limit})")
                with col2:
                    choice = st.selectbox(
                        "Action",
                        ["Drop", "Keep Top 10", "Encode All", "Keep as Text"],
                        key=f"cardinality_{col}",
                        label_visibility="collapsed"
                    )
                
                mapping = {
                    "Drop": "drop_column",
                    "Keep Top 10": "keep_top_10",
                    "Encode All": "auto_encode",
                    "Keep as Text": "skip_encoding"
                }
                decisions["cardinality"][col] = {
                    "action": mapping.get(choice, "auto_encode"),
                    "unique_count": unique_count
                }
            
            st.divider()
        
        # Auto-decide NORMAL cardinality columns (below limit)
        normal_cols = {
            col: details for col, details in cardinality.items()
            if not details.get("ask_human")
        }
        for col, details in normal_cols.items():
            decisions["cardinality"][col] = {
                "action": "auto_encode",
                "unique_count": details.get("n_unique", 0)
            }
    
    # === PATTERNS: Handle numeric text and categorical variations ===
    if ambiguous_columns:
        st.subheader("🔍 Ambiguous Data Patterns")
        
        for col, pattern_info in ambiguous_columns.items():
            pattern_type = pattern_info.get('type')
            samples = pattern_info.get('samples', [])
            match_pct = pattern_info.get('match_pct', 0)
            
            if pattern_type == 'numeric_text':
                st.write(f"**{col}** — Numeric text ({match_pct:.1f}%)")
                st.caption(f"Examples: {', '.join(str(s) for s in samples[:3])}")
                
                col1, col2 = st.columns([2, 1])
                with col2:
                    choice = st.selectbox(
                        "Action",
                        ["Skip", "Auto-normalize"],
                        key=f"pattern_numeric_{col}",
                        label_visibility="collapsed"
                    )
                
                action_map = {
                    "Skip": "skip_column",
                    "Auto-normalize": "normalize_numeric_text"
                }
                
                decisions["patterns"][col] = {
                    "action": action_map[choice],
                    "pattern": "numeric_text"
                }
                st.divider()
            
            elif pattern_type == 'categorical_mixed':
                detected_cat = pattern_info.get('detected_category', 'unknown')
                variations = pattern_info.get('variations', [])
                
                st.write(f"**{col}** — Categorical variations detected")
                st.caption(f"Detected as: {detected_cat}")
                
                col1, col2 = st.columns([2, 1])
                with col2:
                    choice = st.selectbox(
                        "Action",
                        ["Skip", "Standardize"],
                        key=f"pattern_cat_{col}",
                        label_visibility="collapsed"
                    )
                
                action_map = {
                    "Skip": "skip_column",
                    "Standardize": "standardize_categorical"
                }
                
                decisions["patterns"][col] = {
                    "action": action_map[choice],
                    "pattern": "categorical_mixed",
                    "detected_category": detected_cat
                }
                st.divider()
    
    # Summary of auto-decisions
    auto_count = (
        len([c for c in decisions["missing"].values() if c.get("traffic_light") == "Green"]) +
        len([c for c in decisions["cardinality"].values() if c.get("unique_count", 999) <= 50])
    )
    if auto_count > 0:
        st.info(f"✅ Auto-handled {auto_count} low-risk columns. Only asked about problem columns.")
    
    return decisions

