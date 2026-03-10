import json
import pandas as pd

from autoprep.loader import DataLoader
from autoprep.cleaner import DataCleaner
from autoprep.encoder import CategoricalEncoder
from autoprep.features import FeatureEngineer
from autoprep.profiler import DataProfiler
from autoprep.visualizer import DataVisualizer


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
        outlier_action: str = "clip",
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
