import pandas as pd
import numpy as np
import config


TITLE_ID_KEYWORDS = {"id", "name", "title", "key", "code", "identifier", "uuid"}


def _detect_single_column(series):
    """Detect the type of a single column.

    Returns one of: 'numeric', 'categorical', 'date', 'title_id'
    """
    non_null = series.dropna()
    if len(non_null) == 0:
        return "categorical"

    # Check numeric
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # Try parsing as numeric
    coerced_numeric = pd.to_numeric(non_null, errors="coerce")
    if coerced_numeric.notna().sum() / len(non_null) > 0.8:
        return "numeric"

    # Check date
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # Try parsing as date
    try:
        coerced_dates = pd.to_datetime(non_null, errors="coerce", format="mixed")
        if coerced_dates.notna().sum() / len(non_null) > 0.8:
            return "date"
    except Exception:
        pass

    # Check title/ID
    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        n_unique = non_null.nunique()
        uniqueness_ratio = n_unique / len(non_null)

        # Check column name for ID-like keywords
        col_name_lower = str(series.name).lower()
        name_suggests_id = any(kw in col_name_lower for kw in TITLE_ID_KEYWORDS)

        if uniqueness_ratio > config.TITLE_ID_UNIQUENESS_THRESHOLD:
            return "title_id"
        if n_unique > config.TITLE_ID_MAX_CATEGORIES and uniqueness_ratio > 0.5:
            return "title_id"
        if name_suggests_id and uniqueness_ratio > 0.8:
            return "title_id"

    return "categorical"


def detect_column_types(df):
    """Analyse each column of df and return a metadata DataFrame.

    Returns:
        pd.DataFrame indexed by column name with columns:
        detected_type, user_type, include, summary
    """
    rows = []
    for col_name in df.columns:
        detected = _detect_single_column(df[col_name])
        rows.append({
            "column_name": col_name,
            "detected_type": detected,
            "user_type": detected,
            "include": detected != "title_id",
            "summary": {},
        })

    meta = pd.DataFrame(rows).set_index("column_name")
    return meta
