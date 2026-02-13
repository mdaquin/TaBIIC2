import pandas as pd
import numpy as np


def build_encoded_dataframe(df, column_meta):
    """Build an encoded/standardised DataFrame from raw data and column metadata.

    Processing by user_type (only for included columns):
    - numeric: z-score standardisation
    - categorical: one-hot encoding
    - date: convert to Unix timestamp, then z-score
    - title_id: excluded

    Returns:
        pd.DataFrame with all numeric columns
    """
    included = column_meta[column_meta["include"] == True]
    parts = []

    for col_name in included.index:
        col_type = included.at[col_name, "user_type"]

        if col_type == "title_id":
            continue

        if col_type == "numeric":
            numeric = pd.to_numeric(df[col_name], errors="coerce")
            parts.append(_standardise(numeric, col_name))

        elif col_type == "date":
            dates = pd.to_datetime(df[col_name], errors="coerce", format="mixed")
            timestamps = dates.astype(np.int64) // 10**9
            # Replace the int representation of NaT with NaN
            timestamps = timestamps.replace(dates.isna().map(
                lambda x: np.nan if x else None
            ))
            timestamps[dates.isna()] = np.nan
            parts.append(_standardise(timestamps, col_name))

        elif col_type == "categorical":
            dummies = pd.get_dummies(df[col_name], prefix=col_name)
            parts.append(dummies)

    if not parts:
        return pd.DataFrame(index=df.index)

    encoded = pd.concat(parts, axis=1)
    encoded = encoded.fillna(0)
    return encoded


def _standardise(series, name):
    """Z-score standardise a numeric series."""
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        result = pd.Series(0.0, index=series.index, name=name)
    else:
        result = (series - mean) / std
        result.name = name
    return result.to_frame()
