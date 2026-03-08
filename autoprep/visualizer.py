import os
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import seaborn as sns


class DataVisualizer:
    """
    Produces exploratory visualisation plots and saves them as PNG files.

    Plot groups:
    - Missing data  : horizontal bar chart of missing %
    - Numerical     : distribution histograms + box-plots
    - Categorical   : value-count bar charts
    - Temporal      : raw time-series + monthly aggregation per numeric column
    - Correlation   : heatmap of Pearson correlations
    """

    def __init__(self, output_dir: str = "reports/figures", figsize: tuple = (10, 6)):
        self.output_dir = output_dir
        self.figsize = figsize
        os.makedirs(output_dir, exist_ok=True)

    def visualize_all(self, df: pd.DataFrame, prefix: str = "") -> list[str]:
        saved = []
        saved += self._plot_missing(df, prefix)
        saved += self._plot_numerical(df, prefix)
        saved += self._plot_categorical(df, prefix)
        saved += self._plot_temporal(df, prefix)
        saved += self._plot_correlation(df, prefix)
        return saved

    # ── missing data ──────────────────────────────────────────────────────────

    def _plot_missing(self, df: pd.DataFrame, prefix: str) -> list[str]:
        missing_pct = df.isnull().mean() * 100
        missing_pct = missing_pct[missing_pct > 0].sort_values(ascending=False)
        if missing_pct.empty:
            return []

        fig, ax = plt.subplots(figsize=self.figsize)
        missing_pct.plot(kind="barh", ax=ax, color="salmon", edgecolor="white")
        ax.set_xlabel("Missing (%)")
        ax.set_title("Missing Data by Column")
        ax.axvline(50, color="red", linestyle="--", linewidth=0.8, label="50% threshold")
        ax.legend()
        plt.tight_layout()
        return [self._save(fig, prefix, "missing_data")]

    # ── helpers: column classification ───────────────────────────────────────

    @staticmethod
    def _is_binary_indicator(series: pd.Series) -> bool:
        """True for 0/1 encoded categorical columns — not meaningful as continuous."""
        return set(series.dropna().unique()).issubset({0, 1, 0.0, 1.0})

    @staticmethod
    def _is_id_like(series: pd.Series, threshold: float = 0.95) -> bool:
        """True for columns that are almost entirely unique (row identifiers)."""
        if len(series) == 0:
            return False
        return series.nunique() / len(series) >= threshold

    def _real_num_cols(self, df: pd.DataFrame) -> list[str]:
        """Numeric columns that are neither binary indicators nor ID-like."""
        return [
            col for col in df.select_dtypes(include=[np.number]).columns
            if not self._is_binary_indicator(df[col]) and not self._is_id_like(df[col])
        ]

    # ── numerical ─────────────────────────────────────────────────────────────

    def _plot_numerical(self, df: pd.DataFrame, prefix: str) -> list[str]:
        num_cols = self._real_num_cols(df)
        if not num_cols:
            return []

        saved = []
        n = len(num_cols)
        ncols = min(3, n)
        nrows = math.ceil(n / ncols)

        # Histograms
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        for i, col in enumerate(num_cols):
            axes_flat[i].hist(df[col].dropna(), bins=30, color="steelblue", edgecolor="white")
            axes_flat[i].set_title(col, fontsize=9)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle("Numerical Feature Distributions", y=1.01, fontsize=12)
        plt.tight_layout()
        saved.append(self._save(fig, prefix, "numerical_distributions"))

        # Box-plots
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]
        for i, col in enumerate(num_cols):
            axes_flat[i].boxplot(
                df[col].dropna(), vert=True, patch_artist=True,
                boxprops=dict(facecolor="lightblue"),
                medianprops=dict(color="navy"),
            )
            axes_flat[i].set_title(col, fontsize=9)
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle("Numerical Feature Boxplots", y=1.01, fontsize=12)
        plt.tight_layout()
        saved.append(self._save(fig, prefix, "numerical_boxplots"))

        return saved

    # ── categorical ───────────────────────────────────────────────────────────

    def _plot_categorical(self, df: pd.DataFrame, prefix: str) -> list[str]:
        cat_cols = df.select_dtypes(include=["string", "category"]).columns.tolist()
        saved = []
        for col in cat_cols:
            top = df[col].value_counts().head(15)
            fig, ax = plt.subplots(figsize=self.figsize)
            top.plot(kind="bar", ax=ax, color="teal", edgecolor="white")
            ax.set_title(f"Value Counts: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            # Sanitize column name for filename
            safe_col = "".join(c if c.isalnum() or c in "-_" else "_" for c in col)
            saved.append(self._save(fig, prefix, f"categorical_{safe_col}"))
        return saved

    # ── temporal ─────────────────────────────────────────────────────────────

    def _plot_temporal(self, df: pd.DataFrame, prefix: str) -> list[str]:
        dt_cols = df.select_dtypes(include=["datetime"]).columns.tolist()
        if not dt_cols:
            return []

        num_cols = self._real_num_cols(df)  # excludes IDs and binary indicators
        saved = []

        for dt_col in dt_cols:
            df_sorted = df.sort_values(dt_col).copy()
            safe_dt = "".join(c if c.isalnum() or c in "-_" else "_" for c in dt_col)

            for num_col in num_cols[:5]:  # cap at 5 numeric columns
                safe_num = "".join(c if c.isalnum() or c in "-_" else "_" for c in num_col)

                # Raw time-series
                fig, ax = plt.subplots(figsize=self.figsize)
                ax.plot(df_sorted[dt_col], df_sorted[num_col], alpha=0.7, linewidth=1, color="steelblue")
                ax.set_title(f"{num_col} over time")
                ax.set_xlabel(dt_col)
                ax.set_ylabel(num_col)
                plt.xticks(rotation=30, ha="right")
                plt.tight_layout()
                saved.append(self._save(fig, prefix, f"temporal_{safe_dt}_{safe_num}"))

                # Monthly aggregation (only when enough data points)
                if len(df_sorted) > 10:
                    df_sorted["__period"] = df_sorted[dt_col].dt.to_period("M")
                    monthly = df_sorted.groupby("__period")[num_col].mean()
                    fig, ax = plt.subplots(figsize=self.figsize)
                    monthly.plot(ax=ax, marker="o", color="darkorange", linewidth=1.5)
                    ax.set_title(f"Monthly avg {num_col}")
                    ax.set_xlabel("Month")
                    ax.set_ylabel(f"Avg {num_col}")
                    plt.xticks(rotation=30, ha="right")
                    plt.tight_layout()
                    saved.append(self._save(fig, prefix, f"temporal_monthly_{safe_dt}_{safe_num}"))

            # Drop the helper column if it was added
            if "__period" in df_sorted.columns:
                df_sorted = df_sorted.drop(columns=["__period"])

        return saved

    # ── correlation ───────────────────────────────────────────────────────────

    def _plot_correlation(self, df: pd.DataFrame, prefix: str) -> list[str]:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            return []

        corr = num_df.corr()
        dim = max(8, len(corr.columns))
        fig, ax = plt.subplots(figsize=(dim, dim - 1))
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="coolwarm",
            ax=ax, square=True, linewidths=0.5,
            vmin=-1, vmax=1,
        )
        ax.set_title("Correlation Matrix")
        plt.tight_layout()
        return [self._save(fig, prefix, "correlation_matrix")]

    # ── helpers ───────────────────────────────────────────────────────────────

    def _save(self, fig, prefix: str, name: str) -> str:
        fname = f"{prefix}_{name}.png" if prefix else f"{name}.png"
        path = os.path.join(self.output_dir, fname)
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return path
