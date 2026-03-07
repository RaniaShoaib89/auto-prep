import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


class CategoricalEncoder:
    """
    Encodes categorical (object / category) columns using an appropriate strategy.

    Strategies (auto selection):
    - label      → binary columns (2 unique values)
    - onehot     → low-cardinality columns (≤ onehot_max_cardinality unique values)
    - frequency  → medium / high cardinality (encode as relative frequency)
    - ordinal    → columns listed in ordinal_categories dict with a known ordering
    """

    def __init__(
        self,
        strategy: str = "auto",
        # auto | onehot | label | frequency | ordinal
        onehot_max_cardinality: int = 10,
        drop_first: bool = True,
        ordinal_categories: dict = None,
        # {col_name: [cat1, cat2, ...]} in ascending order
    ):
        self.strategy = strategy
        self.onehot_max_cardinality = onehot_max_cardinality
        self.drop_first = drop_first
        self.ordinal_categories = ordinal_categories or {}
        self._encoders: dict = {}
        self.report: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        cat_cols = df.select_dtypes(include=["string", "category"]).columns.tolist()
        encoding_log = {}

        for col in cat_cols:
            n_unique = int(df[col].nunique(dropna=True))
            strategy = self._resolve_strategy(col, n_unique)
            encoding_log[col] = {"strategy": strategy, "n_unique": n_unique}

            if strategy == "onehot":
                df = self._onehot(df, col)
            elif strategy == "label":
                df = self._label(df, col)
            elif strategy == "frequency":
                df = self._frequency(df, col)
            elif strategy == "ordinal":
                df = self._ordinal(df, col)

        self.report["categorical_encoding"] = encoding_log
        return df

    # ── strategy resolution ───────────────────────────────────────────────────

    def _resolve_strategy(self, col: str, n_unique: int) -> str:
        if self.strategy != "auto":
            return self.strategy
        if col in self.ordinal_categories:
            return "ordinal"
        if n_unique == 2:
            return "label"
        if n_unique <= self.onehot_max_cardinality:
            return "onehot"
        return "frequency"

    # ── encoding methods ──────────────────────────────────────────────────────

    def _onehot(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        dummies = pd.get_dummies(
            df[col], prefix=col, drop_first=self.drop_first
        ).astype(int)
        df = df.drop(columns=[col])
        return pd.concat([df, dummies], axis=1)

    def _label(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        self._encoders[col] = le
        return df

    def _frequency(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        freq_map = df[col].value_counts(normalize=True).to_dict()
        df[col] = df[col].map(freq_map)
        self._encoders[col] = freq_map
        return df

    def _ordinal(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        categories = self.ordinal_categories.get(
            col, sorted(df[col].dropna().unique().tolist())
        )
        oe = OrdinalEncoder(
            categories=[categories],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df[col] = oe.fit_transform(df[[col]])
        self._encoders[col] = oe
        return df
