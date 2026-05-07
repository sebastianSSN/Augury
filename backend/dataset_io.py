"""
Dataset ingestion layer: handles multiple file formats, encodings, and separators.
Returns a standardized DataFrame and a normalization report shown to the user.
"""
import io, re
import chardet
import pandas as pd
from fastapi import HTTPException

MAX_FILE_BYTES = 50 * 1024 * 1024

_SEPARATORS = [",", ";", "\t", "|"]

_ACCENT_TABLE = str.maketrans(
    "áéíóúüñÁÉÍÓÚÜÑ",
    "aeiouunAEIOUUN",
)


def clean_col_name(col: str) -> str:
    """Normalize a column name: strip whitespace, remove accents, replace non-word chars."""
    col = str(col).strip().translate(_ACCENT_TABLE)
    col = re.sub(r"[^\w]", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col or "col"


def _detect_encoding(content: bytes) -> str:
    result = chardet.detect(content[:50_000])
    return (result.get("encoding") or "utf-8").strip()


def _detect_separator(sample: str) -> str:
    first_lines = "\n".join(sample.splitlines()[:5])
    counts = {sep: first_lines.count(sep) for sep in _SEPARATORS}
    return max(counts, key=counts.get)


def load_dataset(content: bytes, filename: str = "dataset.csv") -> tuple[pd.DataFrame, dict]:
    """
    Load and standardize a dataset from raw bytes.
    Supports CSV (any encoding, any separator) and Excel (.xlsx / .xls).
    Returns (df, normalization_report).
    """
    report: dict = {
        "file_type": None,
        "encoding": None,
        "separator": None,
        "duplicates_removed": 0,
        "column_renames": [],
        "empty_cols_removed": [],
        "warnings": [],
    }

    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "csv"

    try:
        if ext in ("xlsx", "xls"):
            report["file_type"] = "excel"
            df = pd.read_excel(io.BytesIO(content))
        else:
            report["file_type"] = "csv"
            enc = _detect_encoding(content)
            report["encoding"] = enc
            try:
                text = content.decode(enc, errors="replace")
            except (LookupError, UnicodeDecodeError):
                text = content.decode("utf-8", errors="replace")
                report["encoding"] = "utf-8 (fallback)"
                report["warnings"].append("Encoding detectado no reconocido; se usó UTF-8.")

            sep = _detect_separator(text)
            report["separator"] = sep
            df = pd.read_csv(
                io.StringIO(text), sep=sep, on_bad_lines="skip", low_memory=False
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(df.columns) < 2:
        raise HTTPException(
            status_code=400, detail="El archivo necesita al menos 2 columnas."
        )

    # Normalize column names
    rename_map: dict[str, str] = {}
    seen: set[str] = set()
    for col in df.columns:
        new_col = clean_col_name(col)
        if new_col in seen:
            i = 2
            while f"{new_col}_{i}" in seen:
                i += 1
            new_col = f"{new_col}_{i}"
        if new_col != str(col):
            rename_map[str(col)] = new_col
        seen.add(new_col)

    if rename_map:
        df = df.rename(columns=rename_map)
        report["column_renames"] = [{"from": k, "to": v} for k, v in rename_map.items()]
        report["warnings"].append(
            f"Se normalizaron {len(rename_map)} nombre(s) de columna."
        )

    # Remove fully-empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)
        report["empty_cols_removed"] = empty_cols
        report["warnings"].append(
            f"Se eliminaron {len(empty_cols)} columna(s) completamente vacías."
        )

    # Remove duplicate rows
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    report["duplicates_removed"] = removed
    if removed:
        report["warnings"].append(f"Se eliminaron {removed} fila(s) duplicadas.")

    return df, report
