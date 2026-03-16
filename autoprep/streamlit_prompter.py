"""Streamlit-based interactive prompter for human-in-the-loop mode"""
import streamlit as st
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
        self.streamlit_decisions = streamlit_decisions or {"missing": {}, "cardinality": {}}
        # Initialize the parent's decision tracking
        self._human_decisions = {"missing": {}, "cardinality": {}}
        self._auto_tasks = {"missing": {}, "cardinality": {}}
        self._human_tasks = {"missing": {}, "cardinality": {}}
    
    def prompt_missing_yellow_zone(self, col_name: str, pct: float) -> str:
        """
        Return pre-decided action (from Streamlit UI) instead of prompting.
        """
        if col_name in self.streamlit_decisions.get("missing", {}):
            action = self.streamlit_decisions["missing"][col_name].get("action", "basic_impute")
        else:
            action = "basic_impute"  # Default if not in decisions
        
        # Track decision
        self._human_decisions["missing"][col_name] = {
            "column": col_name,
            "action": action,
            "missing_pct": round(float(pct), 2),
        }
        return action
    
    def prompt_high_cardinality(self, col_name: str, unique_count: int) -> str:
        """
        Return pre-decided action (from Streamlit UI) instead of prompting.
        """
        if col_name in self.streamlit_decisions.get("cardinality", {}):
            action = self.streamlit_decisions["cardinality"][col_name].get("action", "auto_encode")
        else:
            action = "auto_encode"  # Default if not in decisions
        
        # Track decision
        self._human_decisions["cardinality"][col_name] = {
            "column": col_name,
            "action": action,
            "unique_count": int(unique_count),
        }
        return action


def collect_user_decisions_streamlit(health_report: dict) -> dict:
    """
    Collect user decisions via Streamlit UI for Yellow zones and high cardinality columns.
    
    Returns:
        decisions dict with user choices (to pass to StreamlitHumanPrompter)
    """
    decisions = {"missing": {}, "cardinality": {}}
    
    # Collect decisions for missing data (Yellow zones)
    missing_data = health_report.get("missing_data", {})
    yellow_zones = {col: details for col, details in missing_data.items() 
                   if details.get("traffic_light") == "Yellow"}
    
    if yellow_zones:
        st.subheader("Missing Data - Attention Required")
        st.markdown("These columns have moderate amounts of missing data. Choose how to handle them.")
        for col, details in yellow_zones.items():
            pct = details.get("missing_pct", 0)
            st.write(f"**{col}** ({pct:.1f}% missing)")
            choice = st.radio(
                f"How should we handle `{col}`?",
                options=[
                    "Drop column",
                    "Basic Impute (median/mode)",
                    "Smart KNN Impute"
                ],
                key=f"missing_{col}",
                horizontal=True
            )
            
            # Map choice to action
            mapping = {
                "Drop column": "drop_column",
                "Basic Impute (median/mode)": "basic_impute",
                "Smart KNN Impute": "smart_knn_impute"
            }
            decisions["missing"][col] = {
                "action": mapping.get(choice, "basic_impute"),
                "missing_pct": pct
            }
            st.divider()
    
    # Collect decisions for high cardinality
    cardinality = health_report.get("cardinality", {})
    high_card = {col: details for col, details in cardinality.items() 
                 if details.get("ask_human")}
    
    if high_card:
        st.subheader("Categorical Columns - High Cardinality")
        st.markdown("These columns have many unique values and need special handling.")
        for col, details in high_card.items():
            unique_count = details.get("n_unique", 0)
            st.write(f"**{col}** ({unique_count} unique values)")
            choice = st.radio(
                f"How should we handle `{col}`?",
                options=[
                    "Drop column",
                    "Keep top 10 values",
                    "Auto-encode anyway"
                ],
                key=f"cardinality_{col}",
                horizontal=True
            )
            
            # Map choice to action
            mapping = {
                "Drop column": "drop_column",
                "Keep top 10 values": "keep_top_10",
                "Auto-encode anyway": "auto_encode"
            }
            decisions["cardinality"][col] = {
                "action": mapping.get(choice, "auto_encode"),
                "unique_count": unique_count
            }
            st.divider()
    
    if not yellow_zones and not high_card:
        st.success("No problematic columns detected. Using automatic handling.")
    
    return decisions

