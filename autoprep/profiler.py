import numpy as np
import pandas as pd


class DataProfiler:
    """Generates statistical and diagnostics profiles of a DataFrame."""

    def __init__(
        self,
        missing_green_zone: tuple[float, float] = (0.0, 0.10),
        missing_yellow_zone: tuple[float, float] = (0.11, 0.79),
        missing_red_zone: tuple[float, float] = (0.80, 1.0),
        cardinality_limit: int = 50,
        zscore_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
    ):
        self.missing_green_zone = missing_green_zone
        self.missing_yellow_zone = missing_yellow_zone
        self.missing_red_zone = missing_red_zone
        self.cardinality_limit = cardinality_limit
        self.zscore_threshold = zscore_threshold
        self.iqr_multiplier = iqr_multiplier

    # ── diagnostics API ──────────────────────────────────────────────────────

    def assess_missing_data(self, df: pd.DataFrame) -> dict:
        """Return missingness metrics and traffic-light label for each column."""
        n_rows = max(len(df), 1)
        findings = {}

        for col in df.columns:
            missing_count = int(df[col].isna().sum())
            missing_ratio = missing_count / n_rows
            findings[col] = {
                "missing_count": missing_count,
                "missing_ratio": round(missing_ratio, 4),
                "missing_pct": round(missing_ratio * 100, 2),
                "traffic_light": self._missing_traffic_label(missing_ratio),
            }

        return findings

    def assess_outliers(self, df: pd.DataFrame, method: str = "iqr") -> dict:
        """Count outliers per numeric column using IQR or z-score bounds."""
        findings = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        for col in numeric_cols:
            series = df[col].dropna()
            if len(series) < 4:
                findings[col] = {
                    "method": method,
                    "outlier_count": 0,
                    "outlier_ratio": 0.0,
                }
                continue

            if method == "zscore":
                std = series.std(ddof=0)
                if std == 0 or pd.isna(std):
                    mask = pd.Series(False, index=df.index)
                else:
                    zscores = ((df[col] - series.mean()) / std).abs()
                    mask = zscores > self.zscore_threshold
            else:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.iqr_multiplier * iqr
                upper = q3 + self.iqr_multiplier * iqr
                mask = (df[col] < lower) | (df[col] > upper)

            outlier_count = int(mask.fillna(False).sum())
            findings[col] = {
                "method": method,
                "outlier_count": outlier_count,
                "outlier_ratio": round(outlier_count / max(len(df), 1), 4),
            }

        return findings

    def assess_cardinality(self, df: pd.DataFrame) -> dict:
        """Flag high-cardinality string columns as potential review candidates."""
        findings = {}
        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns

        for col in text_cols:
            n_unique = int(df[col].nunique(dropna=True))
            findings[col] = {
                "n_unique": n_unique,
                "cardinality_limit": self.cardinality_limit,
                "ask_human": n_unique > self.cardinality_limit,
            }

        return findings

    def detect_messy_categories(self, df: pd.DataFrame, min_unique: int = 3, max_unique: int = 50) -> dict:
        """
        Profile A: Detect columns with messy/inconsistent categorical values.
        
        Criteria:
        - Data type is object/string/category
        - Between min_unique and max_unique unique values (default 3-50)
        
        Returns:
            Dictionary mapping column names to detection info.
        """
        findings = {}
        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns

        for col in text_cols:
            n_unique = int(df[col].nunique(dropna=True))
            if min_unique <= n_unique <= max_unique:
                unique_values = df[col].dropna().unique().tolist()
                findings[col] = {
                    "n_unique": n_unique,
                    "profile": "A_messy_categories",
                    "unique_values": unique_values,
                    "eligible_for_ai": True,
                }

        return findings

    def detect_messy_numbers(self, df: pd.DataFrame, digit_threshold: float = 0.80) -> dict:
        """
        Profile B: Detect columns with messy number-like strings.
        
        Criteria:
        - Data type is object/string
        - 80%+ (configurable) of non-null values contain digits
        - Examples: "$100", "5 mil", "0", "1.5k"
        
        Args:
            df: DataFrame to profile
            digit_threshold: Proportion of values containing digits (default 0.80)
        
        Returns:
            Dictionary mapping column names to detection info.
        """
        import re
        
        findings = {}
        text_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in text_cols:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue
            
            # Count values containing at least one digit
            has_digit = non_null.astype(str).apply(
                lambda x: bool(re.search(r'\d', x))
            ).sum()
            
            digit_ratio = has_digit / len(non_null)
            
            if digit_ratio >= digit_threshold:
                unique_values = non_null.unique().tolist()
                findings[col] = {
                    "n_unique": len(unique_values),
                    "profile": "B_messy_numbers",
                    "digit_ratio": round(digit_ratio, 3),
                    "unique_values": unique_values,
                    "eligible_for_ai": True,
                }

        return findings

    def detect_llm_candidates(self, df: pd.DataFrame) -> dict:
        """
        Combined detection: returns both Profile A and Profile B candidates for LLM mapping.
        """
        return {
            "profile_a_messy_categories": self.detect_messy_categories(df),
            "profile_b_messy_numbers": self.detect_messy_numbers(df),
        }

    def generate_health_report(self, df: pd.DataFrame) -> dict:
        """Package diagnostics for downstream interactive decision-making."""
        return {
            "shape": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "missing_data": self.assess_missing_data(df),
            "outliers": self.assess_outliers(df, method="iqr"),
            "cardinality": self.assess_cardinality(df),
        }

    def _missing_traffic_label(self, ratio: float) -> str:
        g_lo, g_hi = self.missing_green_zone
        y_lo, y_hi = self.missing_yellow_zone
        r_lo, r_hi = self.missing_red_zone

        if g_lo <= ratio <= g_hi:
            return "Green"
        if y_lo <= ratio <= y_hi:
            return "Yellow"
        if r_lo <= ratio <= r_hi:
            return "Red"
        return "Unknown"

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

    @staticmethod
    def _is_id_like(series: pd.Series, threshold: float = 0.95) -> bool:
        """True for columns where nearly every value is unique (row identifiers)."""
        if len(series) <= 10:
            return False
        return series.nunique() / len(series) >= threshold

    _DATE_FEATURE_SUFFIXES = (
        "_year", "_month", "_day", "_dayofweek",
        "_quarter", "_is_weekend", "_hour",
    )

    def _is_date_feature(self, col: str) -> bool:
        """True for columns extracted from a datetime column by FeatureEngineer."""
        return col.endswith(self._DATE_FEATURE_SUFFIXES)

    def _numerical_summary(self, df: pd.DataFrame) -> dict:
        num_df = df.select_dtypes(include=[np.number])
        # Exclude binary indicator columns, ID-like columns, and date-extracted features
        real_num_cols = [
            col for col in num_df.columns
            if not self._is_binary_indicator(num_df[col])
            and not self._is_id_like(num_df[col])
            and not self._is_date_feature(col)
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
                # Convert keys to plain str to avoid pandas StringDtype key issues
                "top_5": {str(k): int(v) for k, v in vc.head(5).items()},
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
