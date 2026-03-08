
import numpy as np
import pandas as pd


class FeatureEngineer:
    """
    Minimal, targeted feature engineering:
    - Extracts year/month/day/dayofweek/quarter/is_weekend from datetime columns
    - Drops identifier-like columns (nearly all-unique strings/ints)
    - Drops near-zero-variance numerical features
    - Drops highly correlated (redundant) numerical features
    - Optionally drops raw datetime columns after extraction
    """

    def __init__(
        self,
        extract_date_features: bool = True,
        drop_datetime_cols: bool = True,
        drop_identifiers: bool = True,
        identifier_unique_threshold: float = 0.95,
        drop_low_variance: bool = True,
        variance_threshold: float = 0.01,
        drop_high_correlation: bool = True,
        correlation_threshold: float = 0.95,
    ):
        self.extract_date_features = extract_date_features
        self.drop_datetime_cols = drop_datetime_cols
        self.drop_identifiers = drop_identifiers
        self.identifier_unique_threshold = identifier_unique_threshold
        self.drop_low_variance = drop_low_variance
        self.variance_threshold = variance_threshold
        self.drop_high_correlation = drop_high_correlation
        self.correlation_threshold = correlation_threshold
        self.report: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if self.extract_date_features:
            df = self._extract_datetime_features(df)
        if self.drop_identifiers:
            df = self._drop_identifier_columns(df)
        if self.drop_low_variance:
            df = self._drop_low_variance_cols(df)
        if self.drop_high_correlation:
            df = self._drop_correlated_cols(df)
        return df

    # ── datetime feature extraction ───────────────────────────────────────────

    def _extract_datetime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
        new_features = []

        for col in dt_cols:
            p = col  # prefix
            df[f"{p}_year"] = df[col].dt.year
            df[f"{p}_month"] = df[col].dt.month
            df[f"{p}_day"] = df[col].dt.day
            df[f"{p}_dayofweek"] = df[col].dt.dayofweek   # Mon=0, Sun=6
            df[f"{p}_quarter"] = df[col].dt.quarter
            df[f"{p}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            # Only add hour if it carries information
            if df[col].dt.hour.nunique() > 1:
                df[f"{p}_hour"] = df[col].dt.hour
                new_features.append(f"{p}_hour")
            new_features += [
                f"{p}_year", f"{p}_month", f"{p}_day",
                f"{p}_dayofweek", f"{p}_quarter", f"{p}_is_weekend",
            ]

        self.report["datetime_features_extracted"] = new_features

        if self.drop_datetime_cols and dt_cols:
            df = df.drop(columns=dt_cols)

        return df

    # ── identifier columns ───────────────────────────────────────────────────

    def _drop_identifier_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop string or integer columns where nearly every value is unique (ID-like)."""
        if len(df) <= 10:
            self.report["identifier_columns_dropped"] = []
            return df

        drop_cols = []
        # Check both string and integer/float columns for ID-like uniqueness
        candidate_dtypes = ["string"] + [np.number]
        for col in df.select_dtypes(include=candidate_dtypes).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio >= self.identifier_unique_threshold:
                drop_cols.append(col)

        self.report["identifier_columns_dropped"] = drop_cols
        return df.drop(columns=drop_cols)

    # ── low variance ──────────────────────────────────────────────────────────

    def _drop_low_variance_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_cols = [
            col for col in num_cols
            if df[col].std() < self.variance_threshold
        ]
        self.report["low_variance_dropped"] = drop_cols
        return df.drop(columns=drop_cols)

    # ── high correlation ──────────────────────────────────────────────────────

    def _drop_correlated_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            self.report["high_correlation_dropped"] = []
            return df

        corr = df[num_cols].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_cols = [
            col for col in upper.columns
            if any(upper[col] > self.correlation_threshold)
        ]
        self.report["high_correlation_dropped"] = drop_cols
        return df.drop(columns=drop_cols)
