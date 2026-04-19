"""
AutoPrep - 4-Stage Interactive Data Preprocessing Pipeline
Modern, professional UI with human-in-the-loop capabilities
"""

from dotenv import load_dotenv
load_dotenv()

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
import streamlit as st
from autoprep.config import parse_settings
from autoprep.profiler import DataProfiler
from autoprep.llm_agent import LLMAssistant
from autoprep.loader import DataLoader
from autoprep.pipeline import AutoPrepPipeline
from autoprep.ai_suggestions import (
    AIColumnAnalyzer, 
    AIOutlierExplainer, 
    AIInteractionSuggester,
    AIQualityReportGenerator
)

# ── Initialize session state ──────────────────────────────────────────────────
if "stage" not in st.session_state:
    st.session_state.stage = 1  # Start at Stage 1: Upload

session_keys = [
    "df_raw", "df_analyzed", "df_processed",
    "file_path", "column_profiles", "auto_mappings", "approved_mappings",
    "health_report", "quality_report", "interactions", "outlier_explanations"
]

for key in session_keys:
    if key not in st.session_state:
        st.session_state[key] = None

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPrep - Data Preprocessing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modern theme styling ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Stage progress indicators */
    .stage-indicator {
        display: flex;
        justify-content: space-between;
        margin: 2rem 0;
        position: relative;
    }
    
    .stage-item {
        flex: 1;
        text-align: center;
        font-weight: 500;
        padding: 1rem 0.5rem;
        border-radius: 8px;
        background-color: #f0f0f0;
        margin: 0 0.5rem;
        position: relative;
    }
    
    .stage-item.active {
        background-color: #2e88de;
        color: white;
        transform: scale(1.05);
    }
    
    .stage-item.completed {
        background-color: #2bcc71;
        color: white;
    }
    
    /* Professional typography */
    h1, h2, h3 {
        color: #1a1a1a;
        font-weight: 600;
        letter-spacing: -0.3px;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    h1 { font-size: 2.2rem; }
    h2 { font-size: 1.6rem; }
    h3 { font-size: 1.25rem; }
    
    hr { margin: 2rem 0; border: 1px solid #e0e0e0; }
    
    button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease;
    }
    
    .dataframe {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    .stSuccess { background-color: #ecfdf5 !important; color: #065f46 !important; }
    .stWarning { background-color: #fffbeb !important; color: #92400e !important; }
    .stInfo { background-color: #eff6ff !important; color: #0c4a6e !important; }
    .stError { background-color: #fef2f2 !important; color: #7f1d1d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── SIDEBAR: Global Configuration ─────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Pipeline Config")
    st.caption("Core preprocessing settings")
    st.divider()
    
    missing_strategy = st.selectbox(
        "Missing Values",
        ["auto", "drop", "median", "mode"],
        index=0,
        help="Strategy for handling missingness",
    )
    
    outlier_action = st.selectbox(
        "Outliers",
        ["keep", "clip", "remove"],
        index=1,
        help="How to handle statistical outliers",
    )
    
    detect_pii = st.checkbox("🔒 Detect sensitive data (slow)", value=False)
    
    st.divider()
    st.info("✅ Automated: AI mapping · Date extraction · Multi-stage analysis", icon="🤖")
    
    # Show current stage
    st.divider()
    st.markdown(f"**Current Stage:** {st.session_state.stage}")


# ── HELPER FUNCTIONS ──────────────────────────────────────────────────────────

def stage_progress_bar():
    """Render 4-stage progress indicator."""
    stages = ["📤 UPLOAD", "🔍 ANALYZE", "⚙️ PREPROCESS", "📊 EXPORT"]
    current_stage = st.session_state.stage or 1  # Default to 1 if None
    stage_colors = []
    stage_labels = []
    
    for i, stage_name in enumerate(stages, 1):
        if i < current_stage:
            stage_colors.append("completed")
            stage_labels.append(f"✅ {stage_name}")
        elif i == current_stage:
            stage_colors.append("active")
            stage_labels.append(f"→ {stage_name}")
        else:
            stage_colors.append("pending")
            stage_labels.append(f"⬜ {stage_name}")
    
    cols = st.columns(4)
    for col, label, color in zip(cols, stage_labels, stage_colors):
        with col:
            if color == "completed":
                st.success(label)
            elif color == "active":
                st.info(label)
            else:
                st.caption(label)


def render_column_profiles(profiles: Dict):
    """Render AI-generated column profiles."""
    if not profiles:
        st.info("No column profiles available")
        return
    
    st.markdown("#### 🔍 Column Analysis")
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Columns", len(profiles))
    with col2:
        issues_count = sum(len(p.get("quality_issues", [])) for p in profiles.values())
        st.metric("Quality Issues", issues_count)
    with col3:
        encoding_types = set(p.get("suggested_encoding", "") for p in profiles.values())
        st.metric("Encoding Types", len(encoding_types))
    
    # Detailed profiles
    with st.expander("📋 Detailed Column Profiles", expanded=False):
        for col_name, profile in profiles.items():
            with st.container(border=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.markdown(f"**{col_name}**")
                    st.caption(f"Type: {profile.get('semantic_type', 'unknown')} | {profile.get('dtype')}")
                
                with col2:
                    encoding = profile.get("suggested_encoding", "auto")
                    st.caption(f"Encoding: {encoding}")
                
                with col3:
                    card = profile.get("cardinality", 0)
                    st.caption(f"Cardinality: {card}")
                
                if profile.get("quality_issues"):
                    st.warning(f"⚠️ Issues: {', '.join(profile['quality_issues'])}")
                
                st.caption(f"Action: {profile.get('ai_action', 'inspect')}")


def render_outlier_explanations(explanations: Dict):
    """Render AI-generated outlier explanations."""
    if not explanations:
        st.info("No outliers detected")
        return
    
    st.markdown("#### 🎯 Outlier Analysis")
    
    for col, analysis in explanations.items():
        try:
            with st.expander(f"{col} ({analysis.get('outlier_count', 0)} outliers)", expanded=False):
                st.metric("Outlier Count", f"{analysis.get('outlier_count', 0)} ({analysis.get('outlier_pct', 0)}%)")
                
                recommended = analysis.get('recommended_action', 'keep')
                st.markdown(f"**Recommended Action:** {recommended}")
                
                exp_list = analysis.get('explanations', [])
                if exp_list:
                    with st.expander("Detailed Explanations", expanded=False):
                        for exp in exp_list:
                            st.write(f"• **Value:** {exp.get('value', 'N/A')}")
                            reasons = exp.get('possible_reasons', [])
                            st.write(f"  **Possible Reasons:** {', '.join(reasons) if reasons else 'N/A'}")
                            st.write(f"  **Context:** {exp.get('context', 'N/A')}")
        except Exception as e:
            st.warning(f"Could not render analysis for {col}: {str(e)}")


def render_interaction_suggestions(suggestions: Dict):
    """Render AI feature interaction suggestions."""
    if not suggestions or not any(suggestions.values()):
        st.info("No interaction suggestions (might not apply to your data)")
        return
    
    st.markdown("#### 🔗 Feature Interaction Suggestions")
    
    # Numeric interactions
    if suggestions.get("interaction_suggestions"):
        with st.expander(f"Numeric Interactions ({len(suggestions['interaction_suggestions'])})", expanded=True):
            for interaction in suggestions["interaction_suggestions"]:
                st.write(f"• **{interaction['description']}** ({interaction['type']})")
                st.caption(f"Reason: {interaction['reason']}")
    
    # Temporal suggestions
    if suggestions.get("temporal_suggestions"):
        with st.expander(f"Temporal Features ({len(suggestions['temporal_suggestions'])})", expanded=False):
            for suggestion in suggestions["temporal_suggestions"]:
                st.write(f"• {suggestion['description']}")
    
    # Domain suggestions
    if suggestions.get("domain_suggestions"):
        with st.expander(f"Domain-Specific Features ({len(suggestions['domain_suggestions'])})", expanded=False):
            for suggestion in suggestions["domain_suggestions"]:
                st.write(f"• {suggestion['description']}")


# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

st.title("🚀 AutoPrep - Data Preprocessing")
st.markdown("**Upload → Analyze → Preprocess → Export**")

stage_progress_bar()
st.divider()

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         STAGE 1: UPLOAD                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if st.session_state.stage == 1:
    st.header("📤 Stage 1: Upload Your Data")
    st.markdown("Start by uploading your dataset. Supported formats: CSV, Excel, JSON, Parquet")
    st.divider()
    
    uploaded_file = st.file_uploader(
        "📁 Choose a file",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
        label_visibility="collapsed",
    )
    
    if uploaded_file:
        # Save temp file
        suffix = Path(uploaded_file.name).suffix
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uploaded_file.read())
        tmp_file.flush()
        st.session_state.file_path = tmp_file.name
        
        st.success(f"✅ File loaded: **{uploaded_file.name}**")
        
        # Load and show preview
        try:
            loader = DataLoader()
            df_preview = loader.load_data(st.session_state.file_path)
            st.session_state.df_raw = df_preview.copy()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Rows", f"{len(df_preview):,}")
            with col2:
                st.metric("Columns", len(df_preview.columns))
            
            with st.expander("📋 Data Preview (first 20 rows)", expanded=True):
                st.dataframe(df_preview.head(20), use_container_width=True)
            
            st.divider()
            
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                if st.button("▶️ Next: Analyze", type="primary", use_container_width=True):
                    st.session_state.stage = 2
                    st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Upload a CSV, Excel, JSON, or Parquet file to proceed")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         STAGE 2: ANALYZE                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

elif st.session_state.stage == 2:
    st.header("🔍 Stage 2: Analyze Your Data")
    st.markdown("AI inspects data quality, detects issues, and suggests transformations")
    st.divider()
    
    if st.session_state.df_raw is None:
        st.error("❌ No data loaded. Go back to Stage 1")
        if st.button("← Back to Upload"):
            st.session_state.stage = 1
            st.rerun()
    else:
        df = st.session_state.df_raw
        
        # Show file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Input Rows", f"{len(df):,}")
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.divider()
        st.subheader("⏳ Processing...")
        
        with st.spinner("🤖 Running AI analysis..."):
            # Initialize AI tools
            analyzer = AIColumnAnalyzer(LLMAssistant())
            explainer = AIOutlierExplainer()
            suggester = AIInteractionSuggester()
            report_gen = AIQualityReportGenerator()
            llm_agent = LLMAssistant()
            profiler = DataProfiler()
            
            # 1. Column semantic analysis
            st.caption("🔍 Analyzing columns...")
            column_profiles = analyzer.profile_columns(df)
            st.session_state.column_profiles = column_profiles
            
            # 2. Create mappings for messy values using generic intelligent standardization
            st.caption("🤖 Generating auto-mappings...")
            llm_candidates = profiler.detect_llm_candidates(df)
            auto_mappings = {}
            
            # Combine column names from both detection profiles (they return dicts, not lists)
            category_candidates = llm_candidates.get("profile_a_messy_categories", {})
            number_candidates = llm_candidates.get("profile_b_messy_numbers", {})
            all_candidate_cols = list(category_candidates.keys()) + list(number_candidates.keys())
            
            for col_name in all_candidate_cols:
                unique_vals = df[col_name].dropna().unique().tolist()
                if len(unique_vals) <= 200:  # Only if reasonable
                    # Use generic intelligent standardization
                    mapping = llm_agent.standardize_column_values(unique_vals, col_name)
                    if mapping:
                        auto_mappings[col_name] = {"type": "auto", "mapping": mapping}
            
            st.session_state.auto_mappings = auto_mappings
            
            # 3. Apply mappings with proper type conversion
            df_mapped = df.copy()
            for col_name, config in auto_mappings.items():
                if config.get("mapping"):
                    mapping = config["mapping"]
                    
                    print(f"\n🔄 Applying mapping to column '{col_name}':")
                    print(f"   Mapping keys: {list(mapping.keys())[:10]}")
                    print(f"   Mapping values (first 10): {list(mapping.values())[:10]}")
                    
                    # Check original values in dataframe
                    df_col_values = df[col_name].dropna().unique()
                    print(f"   Original DF values (first 5): {list(df_col_values)[:5]}")
                    
                    # Apply the mapping
                    matched_count = 0
                    for orig_val in df_col_values:
                        if orig_val in mapping:
                            matched_count += 1
                    print(f"   Matched {matched_count}/{len(df_col_values)} unique values in mapping")
                    
                    df_mapped[col_name] = df_mapped[col_name].map(mapping).fillna(df_mapped[col_name])
                    
                    # Debug: Check values after mapping
                    df_mapped_values = df_mapped[col_name].dropna().unique()
                    print(f"   After mapping, unique values: {list(df_mapped_values)[:10]}")
                    print(f"   Data types after mapping: {df_mapped[col_name].dtype}")
                    
                    # Check if mapping has numeric values and convert column if needed
                    non_null_values = [v for v in mapping.values() if v is not None]
                    has_numeric = any(isinstance(v, (int, float)) for v in non_null_values)
                    print(f"   Has numeric values in mapping: {has_numeric}")
                    print(f"   Sample numeric values: {[v for v in non_null_values if isinstance(v, (int, float))][:5]}")
                    
                    if non_null_values and has_numeric:
                        # Column should be numeric - convert it
                        print(f"   Attempting pd.to_numeric conversion...")
                        try:
                            df_mapped[col_name] = pd.to_numeric(df_mapped[col_name], errors='coerce')
                            print(f"✅ Converted {col_name} to numeric after mapping")
                            print(f"   Final dtype: {df_mapped[col_name].dtype}")
                            print(f"   Final values (first 5): {df_mapped[col_name].dropna().unique()[:5]}")
                        except Exception as e:
                            print(f"⚠️ Could not convert {col_name} to numeric: {e}")
                            import traceback
                            traceback.print_exc()
            
            st.session_state.df_analyzed = df_mapped
            
            # 4. Generate health report
            st.caption("📊 Generating quality report...")
            health_report = profiler.generate_health_report(df_mapped)
            st.session_state.health_report = health_report
            
            # 5. Outlier analysis
            st.caption("🎯 Analyzing outliers...")
            outlier_explanations = {}
            try:
                from sklearn.preprocessing import StandardScaler
                numeric_cols = st.session_state.df_analyzed.select_dtypes(include=['float64', 'int64']).columns
                for col in numeric_cols:
                    try:
                        series = st.session_state.df_analyzed[col].fillna(st.session_state.df_analyzed[col].median())
                        scaler = StandardScaler()
                        z_scores = np.abs(scaler.fit_transform(series.values.reshape(-1, 1)))
                        outlier_idx = np.where(z_scores > 3)[0]
                        
                        if len(outlier_idx) > 0:
                            explanation = explainer.explain_outliers(st.session_state.df_analyzed, col, outlier_idx.tolist())
                            outlier_explanations[col] = explanation
                    except:
                        pass
            except:
                pass
            
            st.session_state.outlier_explanations = outlier_explanations
            
            # 6. Feature interaction suggestions (if has datetime/numeric mix)
            st.caption("🔗 Suggesting feature interactions...")
            interactions = suggester.suggest_interactions(df_mapped)
            st.session_state.interactions = interactions
        
        st.success("✅ Analysis complete!")
        st.divider()
        
        # Show analysis results
        tabs = st.tabs(["📋 Column Profiles", "🎯 Outliers", "🔗 Features", "📊 Quality"])
        
        with tabs[0]:
            render_column_profiles(column_profiles)
        
        with tabs[1]:
            render_outlier_explanations(outlier_explanations)
        
        with tabs[2]:
            render_interaction_suggestions(interactions)
        
        with tabs[3]:
            st.markdown("#### Data Quality Summary")
            st.json(health_report if health_report else {})
        
        st.divider()
        
        # Mapping review
        if auto_mappings:
            st.subheader("🔄 Review Auto-Generated Mappings")
            
            review_data = []
            for col_name, config in auto_mappings.items():
                mapping = config["mapping"]
                for orig, mapped in mapping.items():
                    review_data.append({
                        "Column": col_name,
                        "Original": orig,
                        "Proposed": str(mapped),
                        "✓": True
                    })
            
            edited = st.data_editor(
                pd.DataFrame(review_data),
                use_container_width=True,
                height=300,
                column_config={"✓": st.column_config.CheckboxColumn(required=False)}
            )
            
            st.caption(f"**{len(edited)} mappings** across **{len(auto_mappings)} columns**")
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.stage = 1
                st.rerun()
        with col2:
            if st.button("▶️ Preprocess", type="primary", use_container_width=True):
                st.session_state.stage = 3
                st.rerun()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                      STAGE 3: PREPROCESS                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

elif st.session_state.stage == 3:
    st.header("⚙️ Stage 3: Preprocess Your Data")
    st.markdown("Run the full preprocessing pipeline with your decisions")
    st.divider()
    
    if st.session_state.df_analyzed is None:
        st.error("❌ No analyzed data available. Go back")
        if st.button("← Back to Analyze"):
            st.session_state.stage = 2
            st.rerun()
    else:
        st.subheader("⏳ Running pipeline...")
        
        with st.spinner("Processing your data..."):
            try:
                pipeline = AutoPrepPipeline(
                    missing_strategy=missing_strategy,
                    outlier_action=outlier_action,
                    extract_date_features=True,
                    drop_low_variance=True,
                )
                
                # Save to temp
                temp_path = Path(tempfile.gettempdir()) / "analyzed_data.csv"
                st.session_state.df_analyzed.to_csv(temp_path, index=False)
                
                # Run pipeline
                df_processed, report = pipeline.run(str(temp_path))
                st.session_state.df_processed = df_processed
                
                st.success(f"✅ Pipeline complete: {df_processed.shape[0]:,} rows × {df_processed.shape[1]} columns")
                
            except Exception as e:
                st.error(f"❌ Pipeline error: {str(e)}")
        
        st.divider()
        
        if st.session_state.df_processed is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Output Rows", f"{len(st.session_state.df_processed):,}")
            with col2:
                st.metric("Output Columns", len(st.session_state.df_processed.columns))
            
            with st.expander("📋 Processed Data Preview", expanded=False):
                st.dataframe(st.session_state.df_processed.head(20), use_container_width=True)
        
        st.divider()
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.stage = 2
                st.rerun()
        with col2:
            if st.button("▶️ Export", type="primary", use_container_width=True):
                st.session_state.stage = 4
                st.rerun()


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                         STAGE 4: EXPORT                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

elif st.session_state.stage == 4:
    st.header("📊 Stage 4: Export & Results")
    st.markdown("Download your processed data and review transformation details")
    st.divider()
    
    if st.session_state.df_processed is None:
        st.error("❌ No processed data. Go back to preprocess")
        if st.button("← Back"):
            st.session_state.stage = 3
            st.rerun()
    else:
        df = st.session_state.df_processed
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Rows", f"{len(df):,}")
        with col2:
            st.metric("Final Columns", len(df.columns))
        with col3:
            st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
        
        st.divider()
        
        # Tabs for different views
        tabs = st.tabs(["📥 Download", "📋 Preview", "📊 Stats"])
        
        with tabs[0]:
            st.subheader("Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = df.to_csv(index=False).encode()
                st.download_button(
                    "⬇️ CSV",
                    csv,
                    file_name="processed_data.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                parquet = df.to_parquet(index=False)
                st.download_button(
                    "⬇️ Parquet",
                    parquet,
                    file_name="processed_data.parquet",
                    mime="application/octet-stream",
                    use_container_width=True
                )
        
        with tabs[1]:
            st.subheader("Data Preview")
            st.dataframe(df, use_container_width=True, height=400)
        
        with tabs[2]:
            st.subheader("Statistics")
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
            if len(numeric_cols) > 0:
                st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            else:
                st.info("No numeric columns to display")
        
        st.divider()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.stage = 3
                st.rerun()
        with col2:
            if st.button("🔄 Start Over", use_container_width=True):
                # Reset all state
                for key in session_keys:
                    st.session_state[key] = None
                st.session_state.stage = 1
                st.rerun()
