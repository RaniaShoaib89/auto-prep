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
            return self._load_delimited(file_path, sep=",", encoding=encoding, **kwargs)
        elif fmt == "tsv":
            return self._load_delimited(file_path, sep="\t", encoding=encoding, **kwargs)
        elif fmt == "excel":
            return self._cast_object_to_string(pd.read_excel(file_path, sheet_name=sheet_name, **kwargs))
        elif fmt == "json":
            return self._cast_object_to_string(pd.read_json(file_path, encoding=encoding, **kwargs))
        elif fmt == "parquet":
            return self._cast_object_to_string(pd.read_parquet(file_path, **kwargs))

    def _cast_object_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure string columns are StringDtype (not object) for consistent downstream handling."""
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype("string")
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
