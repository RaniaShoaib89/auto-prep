import numpy as np
import pandas as pd
from scipy import stats


class DataCleaner:
    """
    Cleans a DataFrame by:
    - Removing duplicate rows
    - Inferring and coercing column data types
    - Dropping columns with too many missing values
    - Imputing remaining missing values
    - Detecting and handling outliers in numerical columns
    """

    def __init__(
        self,
        missing_strategy: str = "auto",
        # auto | drop | mean | median | mode | ffill | bfill | constant
        missing_fill_value=None,
        missing_threshold: float = 0.5,
        outlier_method: str = "iqr",      # iqr | zscore | none
        outlier_action: str = "clip",     # clip | remove | none
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
    ):
        self.missing_strategy = missing_strategy
        self.missing_fill_value = missing_fill_value
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_action = outlier_action
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier
        self.report: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = self._drop_duplicates(df)
        df = self._infer_types(df)
        df = self._drop_high_missing_columns(df)
        df = self._handle_missing(df)
        df = self._handle_outliers(df)
        return df

    # ── duplicates ──────────────────────────────────────────────────────────

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        self.report["duplicates_removed"] = before - len(df)
        return df

    # ── type inference ───────────────────────────────────────────────────────

    def _infer_types(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            # In pandas 3.0 string columns are StringDtype, not object dtype
            if not pd.api.types.is_string_dtype(df[col]):
                continue
            # Try numeric
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() / max(len(df), 1) > 0.8:
                df[col] = converted
                continue
            # Try datetime (use format='mixed' to avoid format-inference warnings)
            try:
                converted_dt = pd.to_datetime(df[col], format="mixed", errors="coerce")
                if converted_dt.notna().sum() / max(len(df), 1) > 0.8:
                    df[col] = converted_dt
            except Exception:
                pass
        return df

    # ── missing values ────────────────────────────────────────────────────────

    def _drop_high_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_frac = df.isnull().mean()
        drop_cols = missing_frac[missing_frac > self.missing_threshold].index.tolist()
        self.report["columns_dropped_high_missing"] = drop_cols
        return df.drop(columns=drop_cols)

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_info = {}
        num_cols = set(df.select_dtypes(include=[np.number]).columns)
        cat_cols = set(df.select_dtypes(include=["string", "category"]).columns)
        date_cols = set(df.select_dtypes(include=["datetime"]).columns)

        for col in df.columns:
            n_missing = int(df[col].isnull().sum())
            if n_missing == 0:
                continue
            missing_info[col] = n_missing
            strategy = self._resolve_strategy(col, num_cols, cat_cols, date_cols)
            df = self._fill_column(df, col, strategy)

        self.report["missing_values_filled"] = missing_info
        return df

    def _resolve_strategy(self, col, num_cols, cat_cols, date_cols):
        if self.missing_strategy != "auto":
            return self.missing_strategy
        if col in num_cols:
            return "median"
        if col in date_cols:
            return "ffill"
        return "mode"

    def _fill_column(self, df: pd.DataFrame, col: str, strategy: str) -> pd.DataFrame:
        if strategy == "drop":
            df = df.dropna(subset=[col])
        elif strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            if not mode_val.empty:
                df[col] = df[col].fillna(mode_val.iloc[0])
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
        elif strategy == "constant":
            df[col] = df[col].fillna(self.missing_fill_value)
        return df

    # ── outliers ──────────────────────────────────────────────────────────────

    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.outlier_method == "none" or self.outlier_action == "none":
            return df

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_info = {}

        for col in num_cols:
            series = df[col].dropna()
            if len(series) < 4:
                continue

            if self.outlier_method == "iqr":
                lower, upper = self._iqr_bounds(series)
            elif self.outlier_method == "zscore":
                lower, upper = self._zscore_bounds(series)
            else:
                continue

            mask = (df[col] < lower) | (df[col] > upper)
            n_out = int(mask.sum())
            if n_out == 0:
                continue

            outlier_info[col] = n_out
            if self.outlier_action == "clip":
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif self.outlier_action == "remove":
                df = df[~mask]

        self.report["outliers_handled"] = outlier_info
        return df

    def _iqr_bounds(self, series: pd.Series):
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        return q1 - self.iqr_multiplier * iqr, q3 + self.iqr_multiplier * iqr

    def _zscore_bounds(self, series: pd.Series):
        z = np.abs(stats.zscore(series))
        clean = series[z < self.zscore_threshold]
        return clean.min(), clean.max()
