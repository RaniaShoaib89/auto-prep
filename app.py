"""
AutoPrep - Interactive Streamlit Interface
Run with:  streamlit run app.py
"""

import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AutoPrep",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── helpers ───────────────────────────────────────────────────────────────────

def render_profile(profile: dict):
    """Render a DataProfiler output dict as Streamlit widgets."""
    shape = profile.get("shape", {})
    st.markdown(f"**Shape:** {shape.get('rows', '?'):,} rows × {shape.get('cols', '?')} columns")

    # dtypes
    with st.expander("Column Data Types", expanded=False):
        st.dataframe(
            pd.DataFrame.from_dict(profile.get("dtypes", {}), orient="index", columns=["dtype"]),
            use_container_width=True,
        )

    # missing
    missing = profile.get("missing", {})
    with st.expander(f"Missing Values ({len(missing)} columns affected)", expanded=True):
        if missing:
            st.dataframe(
                pd.DataFrame(missing).T.rename(columns={"count": "Missing Count", "pct": "Missing %"}),
                use_container_width=True,
            )
        else:
            st.success("No missing values.")

    # numerical
    numerical = profile.get("numerical", {})
    with st.expander(f"Numerical Summary ({len(numerical)} columns)", expanded=True):
        if numerical:
            st.dataframe(pd.DataFrame(numerical).T.round(3), use_container_width=True)
        else:
            st.info("No numerical columns.")

    # categorical
    categorical = profile.get("categorical", {})
    with st.expander(f"Categorical Summary ({len(categorical)} columns)", expanded=True):
        if categorical:
            for col, info in categorical.items():
                st.markdown(f"**{col}** - {info['n_unique']} unique, {info['missing']} missing")
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
            st.info("No categorical columns.")

    # temporal
    temporal = profile.get("temporal", {})
    with st.expander(f"Temporal Summary ({len(temporal)} columns)", expanded=True):
        if temporal:
            st.dataframe(pd.DataFrame(temporal).T, use_container_width=True)
        else:
            st.info("No datetime columns.")


def render_report_section(title: str, data: dict):
    with st.expander(title, expanded=False):
        st.json(data)


# -- sidebar - pipeline configuration -------------------------------------------
with st.sidebar:
    st.title("⚙️ AutoPrep")
    st.caption("Configure pipeline steps below, then upload data and click **Run**.")

    st.divider()
    st.subheader("🧹 Cleaning")
    missing_strategy = st.selectbox(
        "Missing value strategy",
        ["auto", "mean", "median", "mode", "ffill", "bfill", "drop", "constant"],
        index=0,
        help="'auto' picks mean/median/mode per column type.",
    )
    missing_threshold = st.slider(
        "Drop column if missing > X%",
        min_value=0, max_value=100, value=50, step=5,
        help="Columns with more missing values than this threshold are dropped.",
    )
    outlier_method = st.selectbox(
        "Outlier detection method",
        ["iqr", "zscore", "none"],
        index=0,
    )
    outlier_action = st.selectbox(
        "Outlier action",
        ["clip", "remove", "none"],
        index=0,
    )

    st.divider()
    st.subheader("🔡 Encoding")
    encoding_strategy = st.selectbox(
        "Encoding strategy",
        ["auto", "onehot", "label", "frequency"],
        index=0,
        help="'auto' uses one-hot for low-cardinality, label for high-cardinality.",
    )
    onehot_max_cardinality = st.slider(
        "One-hot max cardinality",
        min_value=2, max_value=50, value=10,
        help="Columns with more unique values than this use label encoding instead.",
    )

    st.divider()
    st.subheader("🔧 Feature Engineering")
    extract_date_features = st.checkbox("Extract datetime features", value=True)
    drop_identifiers = st.checkbox("Drop identifier-like columns", value=True)
    drop_low_variance = st.checkbox("Drop low-variance columns", value=True)
    drop_high_correlation = st.checkbox("Drop highly correlated columns", value=True)

    st.divider()
    st.subheader("📊 Visualisation")
    visualize = st.checkbox("Generate figures", value=True)

# ── main area ─────────────────────────────────────────────────────────────────
st.title("AutoPrep - Automated Data Preprocessing")
st.markdown("Upload a dataset (or use the built-in sample), configure the pipeline in the sidebar, then hit **Run Pipeline**.")

# ── data source ───────────────────────────────────────────────────────────────
data_col, _ = st.columns([2, 1])
with data_col:
    data_source = st.radio(
        "Data source",
        ["Use built-in sample (data/sample.csv)", "Upload your own file"],
        horizontal=True,
    )

file_path_to_use: str | None = None
tmp_file = None  # keep a reference to prevent GC cleanup

if data_source == "Use built-in sample (data/sample.csv)":
    sample_path = Path(__file__).parent / "data" / "sample.csv"
    if sample_path.exists():
        file_path_to_use = str(sample_path)
        st.info(f"Using **{sample_path}** - {pd.read_csv(sample_path).shape[0]} rows")
    else:
        st.error("sample.csv not found. Please upload a file instead.")
else:
    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "tsv", "xlsx", "xls", "json", "parquet"],
        help="Supported: CSV, TSV, Excel (.xlsx/.xls), JSON, Parquet",
    )
    if uploaded is not None:
        suffix = Path(uploaded.name).suffix
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_file.write(uploaded.read())
        tmp_file.flush()
        file_path_to_use = tmp_file.name
        st.success(f"Uploaded: **{uploaded.name}**")

# ── preview raw data ──────────────────────────────────────────────────────────
if file_path_to_use:
    try:
        preview_df = pd.read_csv(file_path_to_use) if file_path_to_use.endswith((".csv", ".tsv")) else None
        if preview_df is not None:
            with st.expander("Raw data preview", expanded=True):
                st.dataframe(preview_df.head(20), use_container_width=True)
    except Exception:
        pass

st.divider()

# ── run button ────────────────────────────────────────────────────────────────
run_disabled = file_path_to_use is None
run_btn = st.button("▶  Run Pipeline", type="primary", disabled=run_disabled, use_container_width=False)

if run_disabled and not run_btn:
    st.caption("Select or upload a dataset to enable the pipeline.")

if run_btn:
    from autoprep.pipeline import AutoPrepPipeline

    figures_dir = str(Path(__file__).parent / "reports" / "figures")

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
    )

    with st.spinner("Running pipeline..."):
        try:
            df_processed, report = pipeline.run(file_path_to_use)
            st.success(
                f"Pipeline complete!  "
                f"{report['raw_profile']['shape']['rows']:,} rows in → "
                f"{report['processed_profile']['shape']['rows']:,} rows out, "
                f"{report['processed_profile']['shape']['cols']} columns."
            )
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    # ── results tabs ──────────────────────────────────────────────────────────
    tab_data, tab_raw_profile, tab_cleaned_profile, tab_report, tab_figs = st.tabs(
        ["📋 Processed Data", "📊 Raw Profile", "📊 Cleaned Profile", "📝 Full Report", "🖼️ Figures"]
    )

    with tab_data:
        st.subheader("Processed DataFrame")
        st.caption(f"{df_processed.shape[0]:,} rows × {df_processed.shape[1]} columns")
        st.dataframe(df_processed, use_container_width=True)

        csv_bytes = df_processed.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download as CSV",
            data=csv_bytes,
            file_name="processed_data.csv",
            mime="text/csv",
        )

    with tab_raw_profile:
        st.subheader("Raw Data Profile")
        render_profile(report["raw_profile"])

    with tab_cleaned_profile:
        st.subheader("Cleaned Data Profile")
        st.caption("After deduplication, type inference, missing-value imputation, and outlier handling — before encoding and feature engineering.")
        render_profile(report["cleaned_profile"])

    with tab_report:
        st.subheader("Pipeline Report")

        # Summary Statistics — numerical columns of the processed data
        proc_numerical = report["processed_profile"].get("numerical", {})
        with st.expander("Summary Statistics", expanded=True):
            if proc_numerical:
                stats_df = pd.DataFrame(proc_numerical).T
                display_cols = [c for c in ["min", "mean", "50%", "max", "std", "skewness", "kurtosis"] if c in stats_df.columns]
                stats_df = stats_df[display_cols].rename(columns={"50%": "median"})
                st.dataframe(stats_df.round(3), use_container_width=True)
            else:
                st.info("No numerical columns in processed data.")

        render_report_section("Cleaning Summary", report["cleaning"])
        render_report_section("Encoding Summary", report["encoding"])
        render_report_section("Feature Engineering Summary", report["feature_engineering"])

        st.download_button(
            "⬇ Download full report (JSON)",
            data=json.dumps(report, indent=2, default=str).encode(),
            file_name="report.json",
            mime="application/json",
        )

    with tab_figs:
        st.subheader("Generated Figures")
        figures: list[str] = report.get("figures", [])
        if not figures:
            st.info("No figures generated (visualisation was disabled or no plottable columns).")
        else:
            # Group by prefix (raw / cleaned)
            prefixes = sorted({Path(f).name.split("_")[0] for f in figures})
            fig_tabs = st.tabs([p.capitalize() for p in prefixes]) if len(prefixes) > 1 else [st.container()]
            prefix_map = {p: tab for p, tab in zip(prefixes, fig_tabs)}

            for fig_path in figures:
                prefix = Path(fig_path).name.split("_")[0]
                container = prefix_map.get(prefix, st.container())
                if os.path.exists(fig_path):
                    with container:
                        col, _ = st.columns([1, 1])
                        with col:
                            st.image(fig_path, caption=Path(fig_path).name)
                else:
                    st.warning(f"Figure not found: {fig_path}")
