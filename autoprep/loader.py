import os
import pandas as pd

SUPPORTED_EXTENSIONS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".xls": "excel",
    ".xlsx": "excel",
    ".json": "json",
    ".parquet": "parquet",
}


class DataLoader:
    """Load tabular data from CSV, TSV, Excel, JSON, and Parquet files."""

    def load_data(
        self,
        file_path: str,
        sheet_name=0,
        encoding: str = "utf-8",
        **kwargs,
    ) -> pd.DataFrame:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        fmt = SUPPORTED_EXTENSIONS.get(ext)

        if fmt is None:
            raise ValueError(
                f"Unsupported file format '{ext}'. "
                f"Supported: {list(SUPPORTED_EXTENSIONS)}"
            )

        if fmt == "csv":
            return self._read_csv(file_path, encoding=encoding, **kwargs)
        elif fmt == "tsv":
            return self._load_delimited(file_path, sep="\t", encoding=encoding, **kwargs)
        elif fmt == "excel":
            return self._read_excel(file_path, sheet_name=sheet_name, **kwargs)
        elif fmt == "json":
            return self._cast_object_to_string(pd.read_json(file_path, encoding=encoding, **kwargs))
        elif fmt == "parquet":
            return self._cast_object_to_string(pd.read_parquet(file_path, **kwargs))

    def _read_csv(self, file_path: str, encoding: str = "utf-8", **kwargs) -> pd.DataFrame:
        """Read CSV with encoding fallback (utf-8 -> latin-1/cp1252)."""
        return self._load_delimited(file_path, sep=",", encoding=encoding, **kwargs)

    def _read_excel(self, file_path: str, sheet_name=0, **kwargs) -> pd.DataFrame:
        """Read Excel workbook (.xlsx/.xls)."""
        df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
        return self._cast_object_to_string(df)

    def _cast_object_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure string columns are StringDtype (not object) for consistent downstream handling."""
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("string")
        return self._convert_nullable_int_to_float(df)

    def _convert_nullable_int_to_float(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert nullable Int64/Int32 dtypes to float64 to avoid Int64 casting issues."""
        for col in df.columns:
            try:
                dtype_str = str(df[col].dtype)
                if "Int" in dtype_str and ("Int64" in dtype_str or "Int32" in dtype_str or "Int16" in dtype_str or "Int8" in dtype_str):
                    df[col] = df[col].astype("float64")
            except Exception:
                pass
        return df

    def _load_delimited(self, file_path: str, sep: str, encoding: str, **kwargs) -> pd.DataFrame:
        """Try primary encoding then fall back to common alternatives."""
        fallback_encodings = ["latin-1", "cp1252", "iso-8859-1"]
        for enc in [encoding] + fallback_encodings:
            try:
                df = pd.read_csv(file_path, sep=sep, encoding=enc, **kwargs)
                return self._cast_object_to_string(df)
            except UnicodeDecodeError:
                continue
        raise ValueError(
            f"Could not decode '{file_path}' with any known encoding "
            f"({[encoding] + fallback_encodings})."
        )

    def save_data(self, df: pd.DataFrame, file_path: str, **kwargs) -> None:
        """Persist DataFrame to CSV or Parquet based on target extension."""
        ext = os.path.splitext(file_path)[1].lower()
        output_dir = os.path.dirname(file_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if ext == ".csv":
            df.to_csv(file_path, index=False, **kwargs)
        elif ext == ".parquet":
            df.to_parquet(file_path, index=False, **kwargs)
        else:
            raise ValueError("save_data supports only '.csv' and '.parquet' outputs.")
