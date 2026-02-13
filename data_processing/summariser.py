import pandas as pd
import numpy as np
import config


def compute_summary(series, col_type):
    """Compute summary statistics for a column based on its assigned type.

    Args:
        series: pd.Series of raw data
        col_type: one of 'numeric', 'categorical', 'date', 'title_id'

    Returns:
        dict with type-specific summary statistics
    """
    count = int(series.notna().sum())
    missing = int(series.isna().sum())

    if col_type == "numeric":
        numeric = pd.to_numeric(series, errors="coerce")
        return {
            "type": "numeric",
            "count": count,
            "missing": missing,
            "mean": _safe_round(numeric.mean()),
            "median": _safe_round(numeric.median()),
            "std": _safe_round(numeric.std()),
            "min": _safe_round(numeric.min()),
            "max": _safe_round(numeric.max()),
        }

    if col_type == "date":
        dates = pd.to_datetime(series, errors="coerce", format="mixed")
        valid = dates.dropna()
        if len(valid) == 0:
            return {
                "type": "date",
                "count": count,
                "missing": missing,
                "min": None,
                "max": None,
                "range_days": None,
            }
        return {
            "type": "date",
            "count": count,
            "missing": missing,
            "min": str(valid.min().date()),
            "max": str(valid.max().date()),
            "range_days": (valid.max() - valid.min()).days,
        }

    if col_type == "title_id":
        non_null = series.dropna()
        sample = non_null.head(5).astype(str).tolist()
        return {
            "type": "title_id",
            "count": count,
            "missing": missing,
            "unique": int(non_null.nunique()),
            "sample_values": sample,
        }

    # categorical (default)
    non_null = series.dropna()
    value_counts = non_null.value_counts().head(config.TOP_N_CATEGORICAL)
    top_values = [
        {"value": str(val), "count": int(cnt)}
        for val, cnt in value_counts.items()
    ]
    return {
        "type": "categorical",
        "count": count,
        "missing": missing,
        "unique": int(non_null.nunique()),
        "top_values": top_values,
    }


def compute_all_summaries(df, column_meta):
    """Fill in the 'summary' column of column_meta for every column.

    Uses user_type to determine how to summarise.

    Returns:
        Updated column_meta DataFrame
    """
    for col_name in column_meta.index:
        col_type = column_meta.at[col_name, "user_type"]
        column_meta.at[col_name, "summary"] = compute_summary(df[col_name], col_type)
    return column_meta


def _safe_round(value, decimals=4):
    """Round a value, handling NaN/None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return round(float(value), decimals)
