import json
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

from autoprep.loader import DataLoader
from autoprep.cleaner import DataCleaner
from autoprep.encoder import CategoricalEncoder
from autoprep.features import FeatureEngineer
from autoprep.profiler import DataProfiler
from autoprep.visualizer import DataVisualizer
from autoprep.interactor import HumanPrompter


class AutoPrepPipeline:
    """
    End-to-end automated data preprocessing pipeline.

    Steps:
    1. Load      — read CSV / TSV / Excel / JSON / Parquet
    2. Profile   — capture raw data statistics
    3. Clean     — remove duplicates, impute missing values, handle outliers
    4. Encode    — convert categorical columns to numeric representations
    5. Engineer  — extract datetime features, drop redundant / id-like columns
    6. Visualize — save exploratory plots (raw and processed)
    7. Profile   — capture processed data statistics
    8. Return    — (processed_df, report_dict)

    Usage::

        pipeline = AutoPrepPipeline()
        df_clean, report = pipeline.run("data/my_data.csv")
    """

    def __init__(
        self,
        # ── cleaning ─────────────────────────────────────────────────────────
        missing_strategy: str = "auto",
        missing_threshold: float = 0.5,
        outlier_method: str = "iqr",
        outlier_action: str = "none",
        # ── encoding ─────────────────────────────────────────────────────────
        encoding_strategy: str = "auto",
        onehot_max_cardinality: int = 10,
        ordinal_categories: dict = None,
        # ── feature engineering ───────────────────────────────────────────────
        extract_date_features: bool = True,
        drop_identifiers: bool = True,
        drop_low_variance: bool = True,
        drop_high_correlation: bool = True,
        # ── visualisation ─────────────────────────────────────────────────────
        visualize: bool = True,
        output_dir: str = "reports/figures",
        # ── human in the loop ────────────────────────────────────────────────
        interactive_mode: bool = False,
        human_prompter: HumanPrompter | None = None,
    ):
        self.loader = DataLoader()

        self.cleaner = DataCleaner(
            missing_strategy=missing_strategy,
            missing_threshold=missing_threshold,
            outlier_method=outlier_method,
            outlier_action=outlier_action,
        )
        self.encoder = CategoricalEncoder(
            strategy=encoding_strategy,
            onehot_max_cardinality=onehot_max_cardinality,
            ordinal_categories=ordinal_categories,
        )
        self.engineer = FeatureEngineer(
            extract_date_features=extract_date_features,
            drop_datetime_cols=True,
            drop_identifiers=drop_identifiers,
            drop_low_variance=drop_low_variance,
            drop_high_correlation=drop_high_correlation,
        )
        self.profiler = DataProfiler()
        self.visualize_flag = visualize
        self.output_dir = output_dir
        self.interactive_mode = interactive_mode
        self.human_prompter = human_prompter

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, file_path: str, **loader_kwargs) -> tuple[pd.DataFrame, dict]:
        """
        Run the full pipeline on *file_path*.

        Returns
        -------
        df_processed : pd.DataFrame
            Fully cleaned, encoded, and engineered DataFrame ready for modelling.
        report : dict
            Summary of every transformation applied plus before/after profiles.
        """
        # 1. Load
        df_raw = self.loader.load_data(file_path, **loader_kwargs)
        print(f"[AutoPrep] Loaded     : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} cols")

        # 2. Profile raw
        raw_profile = self.profiler.profile(df_raw)

        # 2b. Generate diagnostics and optionally build human-in-the-loop plan
        health_report = self.profiler.generate_health_report(df_raw)
        action_plan = None

        if self.interactive_mode:
            prompter = self.human_prompter or HumanPrompter()
            task_split = prompter.parse_report(health_report)

            for col_name, details in task_split.get("human_tasks", {}).get("missing", {}).items():
                prompter.prompt_missing_yellow_zone(col_name, float(details.get("missing_pct", 0.0)))

            for col_name, details in task_split.get("human_tasks", {}).get("cardinality", {}).items():
                prompter.prompt_high_cardinality(col_name, int(details.get("unique_count", 0)))

            action_plan = prompter.build_action_plan()
            df_raw = self._apply_cleaner_instructions(
                df_raw,
                action_plan.get("cleaner_instructions", {}),
            )

        # 2c. Fix Int64 dtypes BEFORE cleaning (prevent casting errors during imputation/encoding)
        df_raw = self._fix_int64_dtypes(df_raw)

        # 3. Clean (type inference happens here — datetimes are detected)
        df = self.cleaner.fit_transform(df_raw)
        print(f"[AutoPrep] Cleaned    : {df.shape[0]:,} rows × {df.shape[1]} cols")

        # 3b. Profile cleaned data (before encoding/engineering strips temporal + categorical)
        cleaned_profile = self.profiler.profile(df)

        # 4. Visualize on raw + cleaned-pre-encoding data
        #    - raw        : original categories, distributions, missing data
        #    - pre_encode : temporal plots work because datetimes are now detected
        figures: list[str] = []
        if self.visualize_flag:
            viz = DataVisualizer(output_dir=self.output_dir)
            figures += viz.visualize_all(df_raw, prefix="raw")
            figures += viz.visualize_all(df, prefix="cleaned")
            print(f"[AutoPrep] Figures    : {len(figures)} saved to '{self.output_dir}'")

        # 5. Encode
        df = self.encoder.fit_transform(df)
        print(f"[AutoPrep] Encoded    : {df.shape[0]:,} rows × {df.shape[1]} cols")

        # 6. Feature engineering
        df = self.engineer.fit_transform(df)
        print(f"[AutoPrep] Engineered : {df.shape[0]:,} rows × {df.shape[1]} cols")

        # 7. Profile processed
        processed_profile = self.profiler.profile(df)

        report = {
            "raw_profile": raw_profile,
            "health_report": health_report,
            "action_plan": action_plan,
            "cleaned_profile": cleaned_profile,
            "cleaning": self.cleaner.report,
            "encoding": self.encoder.report,
            "feature_engineering": self.engineer.report,
            "processed_profile": processed_profile,
            "figures": figures,
        }

        return df, report

    def run_and_save(
        self,
        file_path: str,
        output_csv: str = "reports/processed_data.csv",
        report_json: str = "reports/report.json",
        **loader_kwargs,
    ) -> tuple[pd.DataFrame, dict]:
        """Run the pipeline and persist results to disk."""
        import os

        df, report = self.run(file_path, **loader_kwargs)

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[AutoPrep] Saved CSV  : {output_csv}")

        # JSON-serialise the report (convert non-serialisable types)
        with open(report_json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"[AutoPrep] Saved JSON : {report_json}")

        return df, report

    def _apply_cleaner_instructions(self, df: pd.DataFrame, instructions: dict) -> pd.DataFrame:
        """Apply human/auto decisions before default cleaner rules."""
        df = df.copy()
        
        # Fix nullable Int64 dtypes before processing
        df = self._fix_int64_dtypes(df)

        drop_columns = [col for col in instructions.get("drop_columns", []) if col in df.columns]
        if drop_columns:
            df = df.drop(columns=drop_columns)

        for col, action in instructions.get("cardinality_handling", {}).items():
            if col not in df.columns:
                continue
            if action == "keep_top_10":
                top_vals = set(df[col].dropna().value_counts().head(10).index)
                df[col] = df[col].where(df[col].isin(top_vals), "__OTHER__")

        missing_actions = instructions.get("missing_imputation", {})
        knn_k = int(instructions.get("knn_k", 5))

        knn_cols = [
            col
            for col, action in missing_actions.items()
            if action == "smart_knn_impute"
            and col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
        ]
        if knn_cols:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                imputer = KNNImputer(n_neighbors=knn_k)
                df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

        for col, action in missing_actions.items():
            if col not in df.columns:
                continue
            if action == "basic_impute":
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    mode_val = df[col].mode(dropna=True)
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])
            elif action == "smart_knn_impute":
                if not pd.api.types.is_numeric_dtype(df[col]):
                    mode_val = df[col].mode(dropna=True)
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val.iloc[0])

        return df

    def _fix_int64_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert nullable Int64/Int32/Int16/Int8 to float64 to prevent casting errors."""
        for col in df.columns:
            try:
                dtype_str = str(df[col].dtype)
                if "Int" in dtype_str and any(x in dtype_str for x in ["Int64", "Int32", "Int16", "Int8"]):
                    df[col] = df[col].astype("float64")
            except Exception:
                pass
        return df
