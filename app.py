"""
AutoPrep - Interactive Data Preprocessing Pipeline
Modern, professional UI with human-in-the-loop capabilities
"""

from dotenv import load_dotenv
load_dotenv()  # Load GROQ_API_KEY from .env FIRST

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from autoprep.config import parse_settings
from autoprep.profiler import DataProfiler
from autoprep.llm_agent import LLMAssistant
from autoprep.loader import DataLoader
from autoprep.pipeline import AutoPrepPipeline

# ── Initialize session state ──────────────────────────────────────────────────
if "health_report_data" not in st.session_state:
    st.session_state.health_report_data = None
if "user_decisions" not in st.session_state:
    st.session_state.user_decisions = None
if "ambiguous_columns" not in st.session_state:
    st.session_state.ambiguous_columns = None
if "df_for_analysis" not in st.session_state:
    st.session_state.df_for_analysis = None
if "approved_mappings" not in st.session_state:
    st.session_state.approved_mappings = {}
if "quality_report" not in st.session_state:
    st.session_state.quality_report = None
if "auto_mappings" not in st.session_state:
    st.session_state.auto_mappings = {}

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
    st.caption("Essential settings only - most preprocessing is automated")

    st.divider()
    
    st.subheader("⚙️ Core Settings")
    
    # ONLY ESSENTIAL SETTINGS
    missing_strategy = st.selectbox(
        "Missing Values",
        ["auto", "drop", "median", "mode"],
        index=0,
        help="How to handle missing values if necessary",
    )
    
    outlier_action = st.selectbox(
        "Outliers",
        ["keep", "remove", "clip"],
        index=0,
        help="Action on statistical outliers",
    )
    
    st.divider()
    
    st.subheader("🔒 Optional Scans")
    detect_pii = st.checkbox("Scan for sensitive data (PII)", value=False, help="Enable only if needed")
    
    st.divider()
    
    st.info("✅ Automated: AI mapping, date extraction, low-variance dropping, one-hot encoding (auto-detect cardinality)")

    # Set defaults for automated features
    drop_low_variance = True
    extract_date_features = True
    encoding_strategy = "auto"
    onehot_max_cardinality = 10


# ── MAIN AREA: Data Input and Pipeline ────────────────────────────────────────
st.title("AutoPrep - Data Preprocessing")
st.markdown("Upload data → Auto-analyze → Review AI mappings → Run pipeline")

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
            from autoprep.pattern_cleaner import PatternDetector
            
            loader = DataLoader()
            df_raw = loader.load_data(file_path_to_use)
            profiler = DataProfiler()
            
            # STEP 1: Auto-generate ALL MAPPINGS first (no user clicks needed)
            print("[Step 1] Detecting columns that need mapping...")
            llm_candidates = profiler.detect_llm_candidates(df_raw)
            
            all_mappings = {}  # Store all auto-generated mappings
            llm_agent = LLMAssistant()
            
            # Profile A: Messy categories
            for col_name in llm_candidates.get("profile_a_messy_categories", []):
                unique_vals = df_raw[col_name].unique().tolist()
                print(f"📡 Auto-generating mapping for {col_name}...")
                mapping = llm_agent.map_messy_categories(
                    [v for v in unique_vals if pd.notna(v)],
                    col_name
                )
                all_mappings[col_name] = {"type": "category", "mapping": mapping, "unique_values": unique_vals}
            
            # Profile B: Messy numbers
            for col_name in llm_candidates.get("profile_b_messy_numbers", []):
                unique_vals = df_raw[col_name].unique().tolist()
                print(f"📡 Auto-generating mapping for {col_name}...")
                mapping = llm_agent.map_messy_numbers(
                    [v for v in unique_vals if pd.notna(v)],
                    col_name
                )
                all_mappings[col_name] = {"type": "numeric", "mapping": mapping, "unique_values": unique_vals}
            
            st.session_state.auto_mappings = all_mappings
            print(f"[Step 1] Generated {len(all_mappings)} mappings")
            
            # STEP 2: Apply mappings to actual data
            print("[Step 2] Applying mappings to data...")
            df_mapped = df_raw.copy()
            for col_name, mapping_config in all_mappings.items():
                mapping = mapping_config["mapping"]
                if mapping:
                    df_mapped[col_name] = llm_agent.apply_mapping_to_dataframe(df_mapped, col_name, mapping)[col_name]
                    print(f"  ✅ Applied mapping to {col_name}")
            
            st.session_state.df_for_analysis = df_mapped  # Save MAPPED data
            st.session_state.df_raw = df_raw  # Save raw for comparison
            print("[Step 2] Mappings applied")
            
            # STEP 3: Analyze MAPPED data (not raw data)
            print("[Step 3] Analyzing cleaned/mapped data...")
            st.session_state.health_report_data = profiler.generate_health_report(df_mapped)
            
            # Only generate quality report if needed
            if detect_pii:
                st.session_state.quality_report = llm_agent.generate_data_quality_report(df_mapped)
            else:
                # Generate quality report without PII detection
                quality_report = llm_agent.generate_data_quality_report(df_mapped)
                quality_report["pii_data"] = {}  # Clear PII data if not requested
                st.session_state.quality_report = quality_report
            
            st.rerun()
        
        # Show assessment if available
        if st.session_state.health_report_data is not None:
            with st.expander("Data Quality Assessment", expanded=True):
                render_health_report(st.session_state.health_report_data)
            
            # Show comprehensive quality report if available
            if st.session_state.quality_report is not None:
                st.divider()
                st.subheader("📊 Data Quality Assessment")
                
                qr = st.session_state.quality_report
                
                # QUICK SUMMARY (always visible)
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Missing Data %", f"{qr['missing_values']['overall_missing_pct']:.1f}%")
                with col2:
                    st.metric("Duplicates %", f"{qr['duplicates']['exact_duplicates'].get('pct', 0):.1f}%")
                with col3:
                    st.metric("Data Quality Issues", len(qr["action_items"]))
                with col4:
                    st.metric("PII Columns", len(qr["pii_data"]))
                
                # ACTION ITEMS (critical)
                if qr["action_items"]:
                    with st.container(border=True):
                        st.write("**⚠️ ACTION ITEMS - Review these:**")
                        for item in qr["action_items"]:
                            st.write(f"• {item['message']}")
                
                # Detailed tabs (collapsed by default)
                with st.expander("📖 Detailed Analysis (click to expand)", expanded=False):
                    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
                        ["Missing", "Types", "Outliers", "Duplicates", "PII", "Stats", "Validation"]
                    )
                    
                    with tab1:
                        if qr["missing_values"]["columns_with_missing"]:
                            st.dataframe(
                                pd.DataFrame(qr["missing_values"]["columns_with_missing"]).T,
                                use_container_width=True
                            )
                        else:
                            st.success("No missing values")
                    
                    with tab2:
                        st.dataframe(pd.DataFrame(qr["data_types"]).T, use_container_width=True)
                    
                    with tab3:
                        if qr["outliers"]["numeric_columns"]:
                            st.dataframe(
                                pd.DataFrame(qr["outliers"]["numeric_columns"]).T,
                                use_container_width=True
                            )
                        else:
                            st.success("No outliers detected")
                    
                    with tab4:
                        st.write(f"Exact duplicates: {qr['duplicates']['exact_duplicates']['count']}")
                        if qr["duplicates"]["exact_duplicates"]["found"]:
                            st.warning(qr['duplicates']['suggestion'])
                    
                    with tab5:
                        if qr["pii_data"]:
                            st.error("🔴 PII DETECTED:")
                            for col, pii_list in qr["pii_data"].items():
                                st.write(f"**{col}:** {[p['type'] for p in pii_list]}")
                        else:
                            st.success("✅ No PII detected")
                    
                    with tab6:
                        stats_data = qr["statistics"]
                        if stats_data["numeric_summary"]:
                            st.dataframe(pd.DataFrame(stats_data["numeric_summary"]).T, use_container_width=True)
                            if stats_data["high_correlations"]:
                                st.write("**High Correlations:**")
                                st.dataframe(pd.DataFrame(stats_data["high_correlations"]), use_container_width=True)
                        else:
                            st.info("No numeric data")
                    
                    with tab7:
                        domain_data = qr["domain_validation"]
                        if domain_data["violations"]:
                            st.warning(f"{len(domain_data['violations'])} validation issues")
                            for col, violations in domain_data["validations"].items():
                                with st.expander(f"**{col}**"):
                                    for v in violations:
                                        st.write(f"- {v['rule']}: {v.get('invalid_pct', v.get('future_pct', 0)):.1f}% violations")
                        else:
                            st.success("✅ All domain rules valid")
            
            # ── LLM-ASSISTED MAPPING ──────────────────────────────────────
            # Only show if data has been analyzed
            if st.session_state.df_for_analysis is not None:
                st.divider()
                st.subheader("🤖 AI-Assisted Value Mapping")
                st.markdown("Use AI to standardize messy categories and numbers.")
            
            # ── AUTO-GENERATED MAPPINGS REVIEW ────────────────────────────
            if st.session_state.auto_mappings:
                st.divider()
                st.subheader("🤖 Review Auto-Generated Mappings")
                st.markdown("AI has auto-generated mappings for messy columns. Review and edit below, then apply.")
                
                # Build review table for all mappings
                review_data = []
                for col_name, col_data in st.session_state.auto_mappings.items():
                    mapping = col_data["mapping"]
                    for original, proposed in mapping.items():
                        review_data.append({
                            "Column": col_name,
                            "Original Value": original,
                            "AI Proposed": str(proposed),
                            "Approve": True
                        })
                
                if review_data:
                    st.write(f"**Total mappings to review: {len(review_data)} across {len(st.session_state.auto_mappings)} columns**")
                    
                    # Edit mappings
                    edited_df = st.data_editor(
                        pd.DataFrame(review_data),
                        use_container_width=True,
                        height=400,
                        column_config={
                            "AI Proposed": st.column_config.TextColumn(width=200),
                            "Approve": st.column_config.CheckboxColumn(width=100)
                        }
                    )
                    
                    if st.button("Apply Approved Mappings", type="primary", use_container_width=True):
                        # Only apply rows where Approve=True
                        approved = edited_df[edited_df["Approve"] == True]
                        
                        for _, row in approved.iterrows():
                            col = row["Column"]
                            orig = row["Original Value"]
                            new = row["AI Proposed"]
                            
                            if col not in st.session_state.approved_mappings:
                                st.session_state.approved_mappings[col] = {}
                            st.session_state.approved_mappings[col][orig] = new
                        st.success(f"✅ Applied {len(approved)} mappings!")
                        st.session_state.ready_for_pipeline = True
                else:
                    st.info("No mappings generated - data appears clean!")
                st.success("✅ Ready to run pipeline")
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

st.divider()

# Run pipeline button
run_disabled = file_path_to_use is None
run_btn = st.button("▶️ Run Pipeline", type="primary", disabled=run_disabled, use_container_width=True)

if run_disabled and not run_btn:
    st.info("📁 Upload a CSV file to get started")

if run_btn:
    figures_dir = str(Path(__file__).parent / "reports" / "figures")
    
    # Create prompter with decisions if interactive mode
    pipeline = AutoPrepPipeline(
        missing_strategy=missing_strategy,
        outlier_action=outlier_action,
        extract_date_features=extract_date_features,
        drop_low_variance=drop_low_variance,
    )

    with st.spinner("Processing your data..."):
        try:
            # Use the already-mapped data from analysis (auto-applied)
            if st.session_state.df_for_analysis is not None:
                st.info("✅ Using auto-applied mappings from analysis...")
                df_to_run = st.session_state.df_for_analysis.copy()
            else:
                # Fallback: load raw data if no analysis yet
                st.warning("⚠️ No mappings applied. Running on raw data.")
                loader = DataLoader()
                df_to_run = loader.load_data(file_path_to_use)
            
            # Save to temporary file for pipeline
            temp_path = Path(tempfile.gettempdir()) / f"mapped_{Path(file_path_to_use).name}"
            df_to_run.to_csv(temp_path, index=False)
            
            df_processed, report = pipeline.run(str(temp_path))
            st.success(f"✅ Pipeline complete: {report['raw_profile']['shape']['rows']:,} rows processed")
        except Exception as exc:
            st.error(f"Error: {str(exc)}")
            st.stop()

    st.divider()
    
    # Results tabs
    tabs = st.tabs(["Processed Data", "Raw Profile", "Cleaned Profile", "Visualizations"])
    tab_data, tab_raw_profile, tab_cleaned_profile, tab_figs = tabs

    # Processed data
    with tab_data:
        st.subheader("Output Data")
        st.write(f"**{df_processed.shape[0]:,} rows** × **{df_processed.shape[1]} columns**")
        
        # Explain transformations
        with st.expander("📖 Data Transformations Applied", expanded=False):
            st.markdown("""
            **Encoding:** Categorical columns (low-cardinality) are one-hot encoded to 0/1 for ML models
            - Example: Channel → Channel_AbbTakk=1, Channel_DawnNews=0, etc.
            
            **Normalization (StandardScaler):** Continuous numeric columns are standardized around 0
            - Formula: (value - mean) / std_dev
            - Result: Negative values are normal and expected
            - Example: Revenue -0.127 means "0.127 std devs below average revenue"
            - This enables proper correlation and ML model training
            
            **Text Columns:** High-cardinality text (names, headlines, etc.) stays readable
            
            **Date Features:** Extracted to year, month, day, dayofweek, quarter, is_weekend
            """)
        
        st.dataframe(df_processed, use_container_width=True)

        csv_bytes = df_processed.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download (CSV)",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
        )

    # Raw profile
    with tab_raw_profile:
        st.subheader("Input Profile")
        render_profile(report["raw_profile"])

    # Cleaned profile
    with tab_cleaned_profile:
        st.subheader("After Cleaning")
        st.caption("Following deduplication, type correction, imputation, and outlier handling")
        render_profile(report["cleaned_profile"])

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
