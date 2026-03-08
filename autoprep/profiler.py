import numpy as np
import pandas as pd


class DataProfiler:
    """Generates a statistical profile of a DataFrame before and after preprocessing."""

    def profile(self, df: pd.DataFrame) -> dict:
        return {
            "shape": {"rows": df.shape[0], "cols": df.shape[1]},
            "dtypes": df.dtypes.astype(str).to_dict(),
            "missing": self._missing_summary(df),
            "numerical": self._numerical_summary(df),
            "categorical": self._categorical_summary(df),
            "temporal": self._temporal_summary(df),
        }

    # ── missing ───────────────────────────────────────────────────────────────

    def _missing_summary(self, df: pd.DataFrame) -> dict:
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        n = len(df)
        return {
            col: {"count": int(cnt), "pct": round(cnt / n * 100, 2)}
            for col, cnt in missing.items()
        }

    # ── numerical ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_binary_indicator(series: pd.Series) -> bool:
        """True if column only contains 0/1 (encoded categorical indicator)."""
        unique_vals = set(series.dropna().unique())
        return unique_vals.issubset({0, 1, 0.0, 1.0})

    def _numerical_summary(self, df: pd.DataFrame) -> dict:
        num_df = df.select_dtypes(include=[np.number])
        # Exclude binary indicator columns — they are encoded categoricals
        real_num_cols = [
            col for col in num_df.columns
            if not self._is_binary_indicator(num_df[col])
        ]
        num_df = num_df[real_num_cols]
        if num_df.empty:
            return {}
        desc = num_df.describe().T
        desc["skewness"] = num_df.skew()
        desc["kurtosis"] = num_df.kurtosis()
        return desc.round(4).to_dict(orient="index")

    # ── categorical ───────────────────────────────────────────────────────────

    def _categorical_summary(self, df: pd.DataFrame) -> dict:
        summary = {}
        for col in df.select_dtypes(include=["string", "category"]).columns:
            vc = df[col].value_counts()
            summary[col] = {
                "n_unique": int(df[col].nunique()),
                "top_5": vc.head(5).to_dict(),
                "missing": int(df[col].isnull().sum()),
            }
        return summary

    # ── temporal ─────────────────────────────────────────────────────────────

    def _temporal_summary(self, df: pd.DataFrame) -> dict:
        summary = {}
        for col in df.select_dtypes(include=["datetime"]).columns:
            col_min, col_max = df[col].min(), df[col].max()
            summary[col] = {
                "min": str(col_min),
                "max": str(col_max),
                "range_days": int((col_max - col_min).days)
                if pd.notna(col_min) and pd.notna(col_max)
                else None,
                "missing": int(df[col].isnull().sum()),
            }
        return summary
