"""
AutoPrep - Interactive Data Preprocessing Pipeline
Modern, professional UI with human-in-the-loop capabilities
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from autoprep.config import parse_settings
from autoprep.profiler import DataProfiler

# ── Initialize session state ──────────────────────────────────────────────────
if "health_report_data" not in st.session_state:
    st.session_state.health_report_data = None
if "user_decisions" not in st.session_state:
    st.session_state.user_decisions = None

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPrep",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── modern theme styling ──────────────────────────────────────────────────────
st.markdown(
    """
    <style>
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
    
    /* Better spacing */
    hr { margin: 2rem 0; border: 1px solid #e0e0e0; }
    
    /* Enhanced button styling */
    button {
        border-radius: 8px !important;
        font-weight: 500 !important;
        letter-spacing: 0.3px;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease;
    }
    
    /* Data table improvements */
    .dataframe {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Status messages */
    .stSuccess { background-color: #ecfdf5 !important; color: #065f46 !important; }
    .stWarning { background-color: #fffbeb !important; color: #92400e !important; }
    .stInfo { background-color: #eff6ff !important; color: #0c4a6e !important; }
    .stError { background-color: #fef2f2 !important; color: #7f1d1d !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def render_profile(profile: dict):
    """Render a DataProfiler output dict as Streamlit widgets."""
    shape = profile.get("shape", {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", f"{shape.get('rows', 0):,}")
    with col2:
        st.metric("Columns", shape.get('cols', 0))

    # dtypes
    with st.expander("Column Data Types", expanded=False):
        st.dataframe(
            pd.DataFrame.from_dict(profile.get("dtypes", {}), orient="index", columns=["dtype"]),
            use_container_width=True,
        )

    # missing
    missing = profile.get("missing", {})
    with st.expander(f"Missing Values ({len(missing)} columns)", expanded=True):
        if missing:
            st.dataframe(
                pd.DataFrame(missing).T.rename(columns={"count": "Missing Count", "pct": "Missing %"}),
                use_container_width=True,
            )
        else:
            st.info("No missing values detected.")

    # numerical
    numerical = profile.get("numerical", {})
    with st.expander(f"Numerical Columns ({len(numerical)} found)", expanded=True):
        if numerical:
            st.dataframe(pd.DataFrame(numerical).T.round(3), use_container_width=True)
        else:
            st.info("No numerical columns in this dataset.")

    # categorical
    categorical = profile.get("categorical", {})
    with st.expander(f"Categorical Columns ({len(categorical)} found)", expanded=True):
        if categorical:
            for col, info in categorical.items():
                st.markdown(f"**{col}** — {info['n_unique']} unique values, {info['missing']} missing")
                top5 = info.get("top_5", {})
                if top5:
                    try:
                        st.dataframe(
                            pd.DataFrame.from_dict(top5, orient="index", columns=["Count"]),
                            use_container_width=True,
                        )
                    except Exception:
                        st.write(top5)
        else:
            st.info("No categorical columns in this dataset.")

    # temporal
    temporal = profile.get("temporal", {})
    with st.expander(f"Date/Time Columns ({len(temporal)} found)", expanded=True):
        if temporal:
            st.dataframe(pd.DataFrame(temporal).T, use_container_width=True)
        else:
            st.info("No date/time columns in this dataset.")


def render_report_section(title: str, data: dict):
    """Render report sections with JSON expansion."""
    with st.expander(title, expanded=False):
        st.json(data)


def render_health_report(health_report: dict):
    """Render data quality assessment with status indicators."""
    if not health_report:
        st.info("No assessment data available.")
        return

    # Missing data assessment
    missing = health_report.get("missing_data", {})
    if missing:
        st.subheader("Missing Data Assessment")
        missing_rows = []
        for col, details in missing.items():
            label = details.get("traffic_light", "Unknown")
            status_display = label.upper()
            if label == "Green":
                status_display = "OK"
            elif label == "Yellow":
                status_display = "ATTENTION"
            elif label == "Red":
                status_display = "CRITICAL"
            
            missing_rows.append({
                "Column": col,
                "Status": status_display,
                "Missing %": f"{details.get('missing_pct', 0):.1f}%",
                "Missing Count": int(details.get("missing_count", 0)),
            })
        if missing_rows:
            st.dataframe(pd.DataFrame(missing_rows), use_container_width=True)
    
    # Cardinality analysis
    cardinality = health_report.get("cardinality", {})
    if cardinality:
        st.subheader("Cardinality Analysis")
        card_rows = []
        for col, details in cardinality.items():
            status = "Review" if details.get("ask_human") else "OK"
            card_rows.append({
                "Column": col,
                "Status": status,
                "Unique Values": int(details.get("n_unique", 0)),
                "Limit": int(details.get("cardinality_limit", 50)),
            })
        if card_rows:
            st.dataframe(pd.DataFrame(card_rows), use_container_width=True)
    
    # Outlier detection
    outliers = health_report.get("outliers", {})
    if outliers:
        st.subheader("Outlier Detection Results")
        outlier_rows = []
        for col, details in outliers.items():
            if details.get("outlier_count", 0) > 0:
                outlier_rows.append({
                    "Column": col,
                    "Outlier Count": int(details.get("outlier_count", 0)),
                    "Outlier Ratio": f"{details.get('outlier_ratio', 0):.4f}",
                    "Method": details.get("method", "iqr").upper(),
                })
        if outlier_rows:
            st.dataframe(pd.DataFrame(outlier_rows), use_container_width=True)
        else:
            st.info("No outliers detected in the dataset.")


def render_action_plan(action_plan: dict):
    """Render preprocessing decisions and automatic actions."""
    if not action_plan:
        st.info("No action plan available in this mode.")
        return
    
    st.subheader("Preprocessing Actions")
    
    auto_tasks = action_plan.get("auto_tasks", {})
    decisions = action_plan.get("human_decisions", {})
    
    if auto_tasks.get("missing") or auto_tasks.get("cardinality"):
        st.write("**Automatic Handling:**")
        auto_summary = []
        for col, task in auto_tasks.get("missing", {}).items():
            action = task.get('action', 'unknown').replace('_', ' ').title()
            auto_summary.append(f"• {col}: {action} ({task.get('missing_pct', 0):.1f}% missing)")
        for col, task in auto_tasks.get("cardinality", {}).items():
            action = task.get('action', 'unknown').replace('_', ' ').title()
            auto_summary.append(f"• {col}: {action} ({task.get('unique_count', 0)} unique)")
        if auto_summary:
            st.write("\n".join(auto_summary))
    
    if decisions.get("missing") or decisions.get("cardinality"):
        st.write("**Your Decisions:**")
        decision_summary = []
        for col, dec in decisions.get("missing", {}).items():
            action = dec.get('action', 'unknown').replace('_', ' ').title()
            decision_summary.append(f"• {col}: {action} ({dec.get('missing_pct', 0):.1f}% missing)")
        for col, dec in decisions.get("cardinality", {}).items():
            action = dec.get('action', 'unknown').replace('_', ' ').title()
            decision_summary.append(f"• {col}: {action} ({dec.get('unique_count', 0)} unique)")
        if decision_summary:
            st.write("\n".join(decision_summary))


# ── SIDEBAR: Pipeline Configuration ───────────────────────────────────────────
with st.sidebar:
    st.title("Pipeline Configuration")
    st.caption("Customize preprocessing behavior below")

    st.divider()
    
    # Interactive mode
    st.subheader("Mode")
    interactive_mode = st.checkbox("Enable Interactive Mode", value=False, help="Make decisions for columns that need attention")
    
    # KNN settings
    try:
        settings = parse_settings()
        default_knn = settings.algorithms.knn_k
    except Exception:
        default_knn = 5
    
    knn_k = st.slider("KNN Neighbors", min_value=2, max_value=15, value=default_knn, help="For intelligent missing value imputation")

    st.divider()
    
    # Data cleaning options
    st.subheader("Data Cleaning")
    missing_strategy = st.selectbox(
        "Missing Value Strategy",
        ["auto", "mean", "median", "mode", "ffill", "bfill", "drop", "constant"],
        index=0,
        help="Method for filling missing values",
    )
    missing_threshold = st.slider(
        "Drop Column Threshold (%)",
        min_value=0, max_value=100, value=50, step=5,
        help="Drop columns with >X% missing",
    )
    outlier_method = st.selectbox(
        "Outlier Detection",
        ["iqr", "zscore", "none"],
        index=0,
        help="Statistical method for detecting outliers",
    )
    outlier_action = st.selectbox(
        "Outlier Handling",
        ["none", "clip", "remove"],
        index=0,
        help="Action to take on detected outliers",
    )

    st.divider()
    
    # Encoding options
    st.subheader("Categorical Encoding")
    encoding_strategy = st.selectbox(
        "Encoding Strategy",
        ["auto", "onehot", "label", "frequency"],
        index=0,
        help="Strategy for converting categories to numbers",
    )
    onehot_max_cardinality = st.slider(
        "One-Hot Max Cardinality",
        min_value=2, max_value=50, value=10,
        help="Use one-hot for categories with ≤X unique values",
    )

    st.divider()
    
    # Feature engineering
    st.subheader("Feature Engineering")
    extract_date_features = st.checkbox("Extract Date Features", value=True, help="Create year/month/day columns")
    drop_identifiers = st.checkbox("Drop ID Columns", value=True, help="Remove identifier-like columns")
    drop_low_variance = st.checkbox("Drop Low-Variance", value=True, help="Remove nearly-constant columns")
    drop_high_correlation = st.checkbox("Drop Redundant", value=True, help="Remove highly correlated columns")

    st.divider()
    
    # Output options
    st.subheader("Output")
    visualize = st.checkbox("Generate Visualizations", value=True, help="Create analysis plots")


# ── MAIN AREA: Data Input and Pipeline ────────────────────────────────────────
st.title("Data Preprocessing Pipeline")
st.markdown("Load your dataset, configure options in the sidebar, and run the pipeline.")

st.divider()

# Data source selection
col1, col2 = st.columns([3, 1])
with col1:
    data_source = st.radio(
        "Data Source",
        ["Use Sample Data", "Upload File"],
        horizontal=True,
        label_visibility="collapsed",
    )

file_path_to_use: str | None = None
tmp_file = None

if data_source == "Use Sample Data":
    sample_path = Path(__file__).parent / "data" / "sample.csv"
    if sample_path.exists():
        file_path_to_use = str(sample_path)
        sample_df = pd.read_csv(sample_path)
        st.success(f"Sample loaded: {sample_df.shape[0]:,} rows, {sample_df.shape[1]} columns")
    else:
        st.error("Sample file not found. Please upload a file.")
else:
    uploaded = st.file_uploader(
        "Choose a file",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
        help="CSV, Excel, JSON, or Parquet",
        label_visibility="collapsed",
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uploaded.read())
        tmp_file.flush()
        file_path_to_use = tmp_file.name
        st.success(f"Loaded: {uploaded.name}")

st.divider()

# Data analysis and health report
if file_path_to_use:
    try:
        # Show preview for CSV/TSV
        preview_df = None
        if file_path_to_use.endswith((".csv", ".tsv")):
            try:
                preview_df = pd.read_csv(file_path_to_use)
            except Exception:
                pass
        
        if preview_df is not None:
            with st.expander("Data Preview (first 20 rows)", expanded=True):
                st.dataframe(preview_df.head(20), use_container_width=True)
        
        # Analyze button
        col1, col2, col3 = st.columns([1, 3, 2])
        with col1:
            analyze_clicked = st.button("Analyze Data", type="primary", use_container_width=True)
        
        if analyze_clicked:
            from autoprep.loader import DataLoader
            loader = DataLoader()
            df_for_health = loader.load_data(file_path_to_use)
            profiler = DataProfiler()
            st.session_state.health_report_data = profiler.generate_health_report(df_for_health)
            st.rerun()
        
        # Show assessment if available
        if st.session_state.health_report_data is not None:
            with st.expander("Data Quality Assessment", expanded=True):
                render_health_report(st.session_state.health_report_data)
            
            if not interactive_mode:
                st.success("Assessment complete. Ready to run the pipeline.")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.divider()

# Interactive decision interface
if interactive_mode and st.session_state.health_report_data is not None:
    from autoprep.streamlit_prompter import collect_user_decisions_streamlit
    
    st.subheader("Your Preprocessing Choices")
    st.markdown("Select how to handle columns that need attention below.")
    
    user_decisions = collect_user_decisions_streamlit(st.session_state.health_report_data)
    st.session_state.user_decisions = user_decisions
    
    st.divider()

# Run pipeline button
run_disabled = file_path_to_use is None or (interactive_mode and st.session_state.health_report_data is None)
run_btn = st.button("Run Pipeline", type="primary", disabled=run_disabled, use_container_width=True)

if run_disabled and not run_btn:
    if file_path_to_use is None:
        st.info("Please select or upload a dataset to begin.")
    elif interactive_mode:
        st.info("Click 'Analyze Data' above to assess your dataset first.")

if run_btn:
    from autoprep.pipeline import AutoPrepPipeline
    from autoprep.streamlit_prompter import StreamlitHumanPrompter
    
    figures_dir = str(Path(__file__).parent / "reports" / "figures")
    
    # Create prompter with decisions if interactive mode
    prompter = None
    if interactive_mode and st.session_state.user_decisions is not None:
        prompter = StreamlitHumanPrompter(streamlit_decisions=st.session_state.user_decisions)

    pipeline = AutoPrepPipeline(
        missing_strategy=missing_strategy,
        missing_threshold=missing_threshold / 100.0,
        outlier_method=outlier_method,
        outlier_action=outlier_action,
        encoding_strategy=encoding_strategy,
        onehot_max_cardinality=onehot_max_cardinality,
        extract_date_features=extract_date_features,
        drop_identifiers=drop_identifiers,
        drop_low_variance=drop_low_variance,
        drop_high_correlation=drop_high_correlation,
        visualize=visualize,
        output_dir=figures_dir,
        interactive_mode=interactive_mode,
        human_prompter=prompter,
    )

    with st.spinner("Processing your data..."):
        try:
            df_processed, report = pipeline.run(file_path_to_use)
            rows_in = report['raw_profile']['shape']['rows']
            rows_out = report['processed_profile']['shape']['rows']
            cols = report['processed_profile']['shape']['cols']
            st.success(f"Pipeline complete: {rows_in:,} rows → {rows_out:,} rows, {cols} final columns")
        except Exception as exc:
            st.error(f"Pipeline error: {str(exc)}")
            st.stop()

    st.divider()
    
    # Results tabs
    if interactive_mode and report.get("action_plan"):
        tabs = st.tabs(["Results", "Raw Data", "Cleaned Data", "Assessment", "Report", "Visualizations"])
        tab_data, tab_raw_profile, tab_cleaned_profile, tab_health, tab_report, tab_figs = tabs
    else:
        tabs = st.tabs(["Results", "Raw Data", "Cleaned Data", "Report", "Visualizations"])
        tab_data, tab_raw_profile, tab_cleaned_profile, tab_report, tab_figs = tabs
        tab_health = None

    # Results tab
    with tab_data:
        st.subheader("Processed Data")
        st.write(f"{df_processed.shape[0]:,} rows × {df_processed.shape[1]} columns")
        st.dataframe(df_processed, use_container_width=True)

        csv_bytes = df_processed.to_csv(index=False).encode()
        st.download_button(
            "Download Results (CSV)",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
        )

    # Raw data profile
    with tab_raw_profile:
        st.subheader("Input Data Profile")
        render_profile(report["raw_profile"])

    # Cleaned data profile
    with tab_cleaned_profile:
        st.subheader("After Cleaning")
        st.caption("Following deduplication, type correction, imputation, and outlier handling")
        render_profile(report["cleaned_profile"])

    # Assessment tab (if interactive)
    if tab_health is not None:
        with tab_health:
            st.subheader("Assessment & Actions")
            render_health_report(report.get("health_report", {}))
            st.divider()
            render_action_plan(report.get("action_plan", {}))

    # Report tab
    with tab_report:
        st.subheader("Detailed Report")

        proc_numerical = report["processed_profile"].get("numerical", {})
        with st.expander("Summary Statistics", expanded=True):
            if proc_numerical:
                stats_df = pd.DataFrame(proc_numerical).T
                display_cols = [c for c in ["min", "mean", "50%", "max", "std", "skewness", "kurtosis"] if c in stats_df.columns]
                stats_df = stats_df[display_cols].rename(columns={"50%": "median"})
                st.dataframe(stats_df.round(3), use_container_width=True)
            else:
                st.info("No numerical columns in results.")

        render_report_section("Cleaning Changes", report["cleaning"])
        render_report_section("Encoding Applied", report["encoding"])
        render_report_section("Features Engineered", report["feature_engineering"])

        st.download_button(
            "Download Full Report (JSON)",
            data=json.dumps(report, indent=2),
            file_name="pipeline_report.json",
            mime="application/json",
        )

    # Visualizations tab
    with tab_figs:
        st.subheader("Generated Visualizations")
        figures_path = Path(__file__).parent / "reports" / "figures"
        if figures_path.exists():
            fig_files = sorted(figures_path.glob("*.png"))
            if fig_files:
                for fig_file in fig_files:
                    st.image(str(fig_file), use_column_width=True, caption=fig_file.stem)
            else:
                st.info("No visualizations generated (disabled in settings).")
        else:
            st.info("No visualizations generated (disabled in settings).")
