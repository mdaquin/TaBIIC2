import uuid
import pandas as pd
import numpy as np


def new_concept_id():
    return uuid.uuid4().hex[:8]


# -- Taxonomy creation --------------------------------------------------------

def create_taxonomy(df):
    """Initialize a new taxonomy with just the root concept."""
    root_id = new_concept_id()
    root = {
        "id": root_id,
        "name": "Root",
        "restrictions": [],
        "parent_ids": [],
        "child_ids": [],
        "row_indices": list(df.index),
        "origin": "root",
    }
    return {
        "concepts": {root_id: root},
        "root_id": root_id,
    }


# -- Row computation ----------------------------------------------------------

def compute_row_indices(concept, taxonomy, raw_df, column_meta):
    """Compute row indices for a concept based on parents and restrictions."""
    if concept["origin"] == "root":
        return list(raw_df.index)

    # Get base rows from parents
    if concept["origin"] == "union":
        parent_rows = set()
        for pid in concept["parent_ids"]:
            parent_rows |= set(taxonomy["concepts"][pid]["row_indices"])
        base_indices = sorted(parent_rows)
    elif concept["origin"] == "intersection":
        parent_rows = None
        for pid in concept["parent_ids"]:
            prows = set(taxonomy["concepts"][pid]["row_indices"])
            parent_rows = prows if parent_rows is None else parent_rows & prows
        base_indices = sorted(parent_rows) if parent_rows else []
    else:
        # restriction or complement: single parent
        if concept["parent_ids"]:
            parent = taxonomy["concepts"][concept["parent_ids"][0]]
            base_indices = parent["row_indices"]
        else:
            base_indices = list(raw_df.index)

    if not concept["restrictions"]:
        return base_indices

    return _apply_restrictions(base_indices, concept["restrictions"], raw_df, column_meta)


def _apply_restrictions(base_indices, restrictions, raw_df, column_meta):
    """Filter base_indices by applying all restrictions (AND logic)."""
    if not base_indices:
        return []

    subset = raw_df.loc[base_indices]
    mask = pd.Series(True, index=subset.index)

    for r in restrictions:
        col = r["column"]
        op = r["operator"]
        val = r["value"]
        col_type = column_meta.at[col, "user_type"]

        if col_type == "categorical":
            mask &= (subset[col].astype(str) == str(val))
        elif col_type == "numeric":
            numeric_col = pd.to_numeric(subset[col], errors="coerce")
            mask &= _apply_comparison(numeric_col, op, float(val))
        elif col_type == "date":
            date_col = pd.to_datetime(subset[col], errors="coerce", format="mixed")
            date_val = pd.to_datetime(val)
            mask &= _apply_comparison(date_col, op, date_val)

    return list(subset[mask].index)


def _apply_comparison(series, op, value):
    """Apply a comparison operator to a pandas Series."""
    if op == "=":
        return series == value
    elif op == ">":
        return series > value
    elif op == ">=":
        return series >= value
    elif op == "<":
        return series < value
    elif op == "<=":
        return series <= value
    return pd.Series(True, index=series.index)


# -- Coverage -----------------------------------------------------------------

def compute_coverage(concept, taxonomy):
    """Fraction of concept's rows covered by its children (0.0-1.0)."""
    if not concept["child_ids"] or not concept["row_indices"]:
        return 0.0

    parent_rows = set(concept["row_indices"])
    child_rows = set()
    for cid in concept["child_ids"]:
        child_rows |= set(taxonomy["concepts"][cid]["row_indices"])

    covered = parent_rows & child_rows
    return len(covered) / len(parent_rows)


# -- Operations ---------------------------------------------------------------

def add_subconcept(taxonomy, parent_id, restrictions, raw_df, column_meta, name=""):
    """Create a child concept with given restrictions under parent_id."""
    parent = taxonomy["concepts"][parent_id]

    new_id = new_concept_id()
    concept = {
        "id": new_id,
        "name": name,
        "restrictions": restrictions,
        "parent_ids": [parent_id],
        "child_ids": [],
        "row_indices": [],
        "origin": "restriction",
    }

    taxonomy["concepts"][new_id] = concept
    concept["row_indices"] = compute_row_indices(concept, taxonomy, raw_df, column_meta)
    parent["child_ids"].append(new_id)

    return new_id


def create_complement(taxonomy, concept_ids, raw_df, column_meta):
    """Create complement: parent rows NOT in any of the selected concepts.

    All selected concepts must share exactly one common parent.
    """
    concepts = [taxonomy["concepts"][cid] for cid in concept_ids]
    parent_sets = [set(c["parent_ids"]) for c in concepts]
    common_parents = parent_sets[0]
    for ps in parent_sets[1:]:
        common_parents &= ps

    if len(common_parents) != 1:
        raise ValueError("Selected concepts must share exactly one common parent")

    parent_id = list(common_parents)[0]
    parent = taxonomy["concepts"][parent_id]

    selected_rows = set()
    for cid in concept_ids:
        selected_rows |= set(taxonomy["concepts"][cid]["row_indices"])

    complement_rows = [r for r in parent["row_indices"] if r not in selected_rows]

    new_id = new_concept_id()
    concept = {
        "id": new_id,
        "name": "",
        "restrictions": [],
        "parent_ids": [parent_id],
        "child_ids": [],
        "row_indices": complement_rows,
        "origin": "complement",
    }

    taxonomy["concepts"][new_id] = concept
    parent["child_ids"].append(new_id)

    return new_id


def create_union(taxonomy, concept_ids, raw_df, column_meta):
    """Create union: new parent concept whose rows = union of selected concepts' rows."""
    union_rows = set()
    for cid in concept_ids:
        union_rows |= set(taxonomy["concepts"][cid]["row_indices"])

    new_id = new_concept_id()
    concept = {
        "id": new_id,
        "name": "",
        "restrictions": [],
        "parent_ids": [],
        "child_ids": list(concept_ids),
        "row_indices": sorted(union_rows),
        "origin": "union",
    }

    taxonomy["concepts"][new_id] = concept

    # Link selected concepts to this new parent
    for cid in concept_ids:
        c = taxonomy["concepts"][cid]
        if new_id not in c["parent_ids"]:
            c["parent_ids"].append(new_id)

    # Link orphan union to root so the graph stays connected
    root_id = taxonomy["root_id"]
    if not concept["parent_ids"] and new_id != root_id:
        concept["parent_ids"].append(root_id)
        taxonomy["concepts"][root_id]["child_ids"].append(new_id)

    return new_id


def create_intersection(taxonomy, concept_ids, raw_df, column_meta):
    """Create intersection: new child concept whose rows = intersection of selected."""
    intersection_rows = None
    for cid in concept_ids:
        rows = set(taxonomy["concepts"][cid]["row_indices"])
        intersection_rows = rows if intersection_rows is None else intersection_rows & rows

    new_id = new_concept_id()
    concept = {
        "id": new_id,
        "name": "",
        "restrictions": [],
        "parent_ids": list(concept_ids),
        "child_ids": [],
        "row_indices": sorted(intersection_rows) if intersection_rows else [],
        "origin": "intersection",
    }

    taxonomy["concepts"][new_id] = concept

    for cid in concept_ids:
        c = taxonomy["concepts"][cid]
        if new_id not in c["child_ids"]:
            c["child_ids"].append(new_id)

    return new_id


def delete_concept(taxonomy, concept_id):
    """Delete a concept and all its descendants. Unlink from parents."""
    concept = taxonomy["concepts"].get(concept_id)
    if not concept:
        return

    # Recursively delete children first
    for child_id in list(concept["child_ids"]):
        delete_concept(taxonomy, child_id)

    # Remove from parents' child_ids
    for parent_id in concept["parent_ids"]:
        parent = taxonomy["concepts"].get(parent_id)
        if parent and concept_id in parent["child_ids"]:
            parent["child_ids"].remove(concept_id)

    del taxonomy["concepts"][concept_id]


# -- Available restrictions ---------------------------------------------------

def get_available_restrictions(concept, taxonomy, raw_df, column_meta):
    """Return columns available for restrictions on this concept,
    with valid operators and value domains.

    Excludes title_id, non-included columns, and columns with <= 1 unique value.
    """
    if not concept["row_indices"]:
        return []

    subset = raw_df.loc[concept["row_indices"]]
    included = column_meta[
        (column_meta["include"] == True) & (column_meta["user_type"] != "title_id")
    ]

    result = []
    for col_name in included.index:
        col_type = included.at[col_name, "user_type"]
        col_data = subset[col_name].dropna()

        if col_type == "categorical":
            unique_vals = sorted(col_data.unique().astype(str).tolist())
            if len(unique_vals) <= 1:
                continue
            result.append({
                "column": col_name,
                "type": col_type,
                "operators": ["="],
                "values": unique_vals,
            })

        elif col_type == "numeric":
            numeric_data = pd.to_numeric(col_data, errors="coerce").dropna()
            if numeric_data.nunique() <= 1:
                continue
            result.append({
                "column": col_name,
                "type": col_type,
                "operators": ["=", ">", ">=", "<", "<="],
                "values": {
                    "min": round(float(numeric_data.min()), 4),
                    "max": round(float(numeric_data.max()), 4),
                    "mean": round(float(numeric_data.mean()), 4),
                    "median": round(float(numeric_data.median()), 4),
                },
            })

        elif col_type == "date":
            date_data = pd.to_datetime(col_data, errors="coerce", format="mixed").dropna()
            if date_data.nunique() <= 1:
                continue
            result.append({
                "column": col_name,
                "type": col_type,
                "operators": ["=", ">", ">=", "<", "<="],
                "values": {
                    "min": str(date_data.min().date()),
                    "max": str(date_data.max().date()),
                },
            })

    return result


# -- Serialization ------------------------------------------------------------

def serialize_taxonomy(taxonomy):
    """Serialize taxonomy for JSON API response (nodes + edges)."""
    nodes = []
    edges = []

    for cid, concept in taxonomy["concepts"].items():
        coverage = compute_coverage(concept, taxonomy)
        nodes.append({
            "id": concept["id"],
            "name": concept["name"],
            "restrictions": concept["restrictions"],
            "size": len(concept["row_indices"]),
            "coverage": round(coverage, 4),
            "origin": concept["origin"],
            "parent_ids": concept["parent_ids"],
            "child_ids": concept["child_ids"],
        })
        for child_id in concept["child_ids"]:
            edges.append({
                "source": concept["id"],
                "target": child_id,
            })

    return {
        "root_id": taxonomy["root_id"],
        "nodes": nodes,
        "edges": edges,
    }
