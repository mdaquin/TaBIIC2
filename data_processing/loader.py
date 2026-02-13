import os
import pandas as pd


def load_file(file_storage, filename):
    """Load an uploaded file into a pandas DataFrame.

    Args:
        file_storage: werkzeug FileStorage object from request.files
        filename: original filename (used for extension detection)

    Returns:
        pd.DataFrame

    Raises:
        ValueError: if format is unsupported or data is not valid tabular data
    """
    ext = os.path.splitext(filename)[1].lower().lstrip(".")

    try:
        if ext == "csv":
            df = pd.read_csv(file_storage)
        elif ext == "tsv":
            df = pd.read_csv(file_storage, sep="\t")
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(file_storage)
        elif ext == "json":
            df = pd.read_json(file_storage)
        else:
            raise ValueError(f"Unsupported file format: .{ext}")
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Failed to parse file: {e}")

    if df.empty or len(df.columns) == 0:
        raise ValueError("File contains no data (0 rows or 0 columns)")

    return df
