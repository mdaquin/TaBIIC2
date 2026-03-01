"""
HAC Taxonomy Builder
====================
Build an OWL/Turtle taxonomy from a CSV file using Hierarchical Agglomerative
Clustering (HAC).  Column preprocessing mirrors TaBIIC2:
  - numeric / date  → z-score standardisation
  - categorical     → one-hot encoding
  - title_id        → excluded from clustering (kept as individual properties)

The full HAC dendrogram is traversed depth-first up to `max_depth` levels.
Each node in the traversal becomes an OWL class; leaf nodes of the cut tree
(those at exactly max_depth, or natural singletons reached earlier) receive
rdf:type assertions for their data rows.  Internal nodes are connected via
rdfs:subClassOf; a reasoner can infer ancestor-class membership.

Usage
-----
    python hac_taxonomy.py <data.csv> <config.json> [-o output.ttl]

Config file schema
------------------
{
  "columns": {
    "<col_name>": {
      "type":    "numeric" | "categorical" | "date" | "title_id",
      "include": true | false          // default: true
    },
    ...
  },
  "hac": {
    "max_depth":        3,             // depth of dendrogram traversal (default 3)
    "linkage":          "ward",        // ward | complete | average | single
    "metric":           "euclidean",   // euclidean | cosine | … (see scipy docs)
    "concept_ns":       "http://example.org/hac/schema#",
    "entity_ns":        "http://example.org/hac/data/"
  }
}

Note: "ward" linkage requires metric="euclidean".
"""

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from rdflib import (
    Graph, Namespace, Literal, BNode, RDF, RDFS, OWL, XSD, URIRef
)
from rdflib.collection import Collection


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build an OWL taxonomy via HAC from a CSV file."
    )
    p.add_argument("csv", help="Input CSV file")
    p.add_argument("config", help="JSON config file")
    p.add_argument("-o", "--output", default=None,
                   help="Output Turtle file (default: <csv stem>.ttl)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    columns = cfg.get("columns", {})
    hac = cfg.get("hac", {})
    # Defaults
    hac.setdefault("max_depth", 3)
    hac.setdefault("linkage", "ward")
    hac.setdefault("metric", "euclidean")
    hac.setdefault("concept_ns", "http://example.org/hac/schema#")
    hac.setdefault("entity_ns", "http://example.org/hac/data/")
    return columns, hac


def build_column_meta(columns_cfg, df):
    """Return a DataFrame indexed by column name with 'user_type' and 'include'."""
    records = []
    for col in df.columns:
        if col in columns_cfg:
            cfg = columns_cfg[col]
            records.append({
                "column": col,
                "user_type": cfg.get("type", "numeric"),
                "include": cfg.get("include", True),
            })
        else:
            # Auto-detect absent columns as numeric, included
            records.append({
                "column": col,
                "user_type": "numeric",
                "include": True,
            })
    meta = pd.DataFrame(records).set_index("column")
    return meta


# ---------------------------------------------------------------------------
# Encoding (mirrors TaBIIC2 encoder.py)
# ---------------------------------------------------------------------------

def build_encoded_dataframe(df, column_meta):
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
    mean = series.mean()
    std = series.std()
    if std == 0 or pd.isna(std):
        result = pd.Series(0.0, index=series.index, name=name)
    else:
        result = (series - mean) / std
        result.name = name
    return result.to_frame()


# ---------------------------------------------------------------------------
# HAC + dendrogram → concept tree
# ---------------------------------------------------------------------------

def build_concept_tree_from_dendrogram(encoded_df, hac_cfg):
    """
    Run HAC and walk the resulting dendrogram depth-first up to max_depth
    levels below the root.

    Scipy linkage matrix Z has shape (n-1, 4): each row encodes one merge.
    Node indices 0..n-1 are original data points (leaves); indices n..2n-2
    are internal nodes in merge order.  The root of the dendrogram is node
    2n-2.

    Returns a list of concept dicts:
        id, name, parent_id, row_indices
    Only leaf concepts of the cut tree carry row_indices; internal concepts
    have row_indices=[] (rdfs:subClassOf lets a reasoner infer ancestry).
    """
    method = hac_cfg["linkage"]
    metric = hac_cfg["metric"]
    max_depth = int(hac_cfg["max_depth"])

    X = encoded_df.values.astype(float)
    row_index = encoded_df.index  # maps positional int → df index label
    n = len(row_index)

    concepts = []
    counter = [0]

    # --- helpers ---

    def get_leaf_positions(node):
        """Recursively collect all original-leaf positions under a node."""
        if node < n: return [int(node)]
        z_idx = node - n
        return (get_leaf_positions(int(Z[z_idx, 0])) +
                get_leaf_positions(int(Z[z_idx, 1])))

    def visit(node, parent_id, depth):
        counter[0] += 1
        cid = f"c{counter[0]:02d}"
        is_cut_leaf = (node < n) or (depth >= max_depth)

        if is_cut_leaf:
            positions = get_leaf_positions(node)
            row_indices = [row_index[p] for p in positions]
        else:
            row_indices = []

        concepts.append({
            "id": cid,
            "name": cid,
            "parent_id": parent_id,
            "row_indices": row_indices,
        })

        if not is_cut_leaf:
            z_idx = node - n
            visit(int(Z[z_idx, 0]), cid, depth + 1)
            visit(int(Z[z_idx, 1]), cid, depth + 1)

    # --- degenerate case: single row ---
    if n < 2:
        return [{
            "id": "root", "name": "Root", "parent_id": None,
            "row_indices": list(row_index),
        }]

    Z = linkage(X, method=method, metric=metric)

    # Root concept (no row_indices — individuals typed to leaf classes only)
    concepts.append({
        "id": "root",
        "name": "Root",
        "parent_id": None,
        "row_indices": [],
    })

    root_node = 2 * n - 2
    visit(root_node, "root", depth=1)

    return concepts


# ---------------------------------------------------------------------------
# OWL export
# ---------------------------------------------------------------------------

def _safe_local_name(name):
    safe = re.sub(r"[^\w-]", "_", str(name))
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe or not safe[0].isalpha():
        safe = "x_" + safe
    return safe


def _is_missing(val):
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def _make_literal(val, col_type):
    if col_type == "numeric":
        try:
            return Literal(float(val), datatype=XSD.decimal)
        except (TypeError, ValueError):
            return None
    elif col_type == "date":
        try:
            dt = pd.to_datetime(val)
            return Literal(str(dt.date()), datatype=XSD.date)
        except Exception:
            return None
    else:
        return Literal(str(val), datatype=XSD.string)


def export_owl(df, column_meta, concepts, concept_ns_uri, entity_ns_uri):
    if not concept_ns_uri.endswith(("#", "/")):
        concept_ns_uri += "#"
    if not entity_ns_uri.endswith(("#", "/")):
        entity_ns_uri += "/"

    g = Graph()
    CNS = Namespace(concept_ns_uri)
    ENS = Namespace(entity_ns_uri)
    g.bind("", CNS)
    g.bind("ent", ENS)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd", XSD)

    # --- Properties (one per column) ---
    prop_map = {}
    cat_value_map = {}  # (col_name, val_str) -> URIRef

    for col_name in column_meta.index:
        safe = _safe_local_name(col_name)
        prop_uri = CNS[safe]
        prop_map[col_name] = prop_uri
        col_type = column_meta.at[col_name, "user_type"]

        if col_type == "categorical":
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            # Create a named individual per unique value
            for val_str in df[col_name].dropna().astype(str).unique():
                safe_val = _safe_local_name(val_str)
                val_uri = CNS[f"{safe}_{safe_val}"]
                cat_value_map[(col_name, val_str)] = val_uri
                g.add((val_uri, RDF.type, OWL.NamedIndividual))
                g.add((val_uri, RDFS.label, Literal(val_str)))
        else:
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            if col_type == "numeric":
                g.add((prop_uri, RDFS.range, XSD.decimal))
            elif col_type == "date":
                g.add((prop_uri, RDFS.range, XSD.date))
            else:
                g.add((prop_uri, RDFS.range, XSD.string))

    # --- Classes (one per concept) ---
    class_map = {}
    for concept in concepts:
        local = _safe_local_name(concept["name"])
        class_uri = CNS[local]
        class_map[concept["id"]] = class_uri
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(concept["name"])))

    # --- rdfs:subClassOf ---
    for concept in concepts:
        if concept["parent_id"] is None:
            g.add((class_map[concept["id"]], RDFS.subClassOf, OWL.Thing))
        else:
            g.add((class_map[concept["id"]],
                   RDFS.subClassOf,
                   class_map[concept["parent_id"]]))

    # --- Individuals ---
    # Build row → concepts mapping
    row_concepts = {}
    for concept in concepts:
        for idx in concept["row_indices"]:
            row_concepts.setdefault(idx, []).append(concept["id"])

    for idx in df.index:
        ind_uri = ENS[f"entity_{idx}"]
        g.add((ind_uri, RDF.type, OWL.NamedIndividual))

        for cid in row_concepts.get(idx, []):
            g.add((ind_uri, RDF.type, class_map[cid]))

        for col_name in column_meta.index:
            val = df.at[idx, col_name]
            if _is_missing(val):
                continue
            col_type = column_meta.at[col_name, "user_type"]
            if col_type == "categorical":
                val_uri = cat_value_map.get((col_name, str(val)))
                if val_uri is not None:
                    g.add((ind_uri, prop_map[col_name], val_uri))
            else:
                literal = _make_literal(val, col_type)
                if literal is not None:
                    g.add((ind_uri, prop_map[col_name], literal))

    return g.serialize(format="turtle")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Output path
    if args.output:
        out_path = args.output
    else:
        stem = args.csv.rsplit(".", 1)[0]
        out_path = stem + ".ttl"

    # Load data
    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {args.csv}")

    # Load config
    columns_cfg, hac_cfg = load_config(args.config)
    column_meta = build_column_meta(columns_cfg, df)

    n_included = (
        column_meta[column_meta["include"] == True]
        .index.difference(
            column_meta[column_meta["user_type"] == "title_id"].index
        )
    ).shape[0]
    print(f"Columns included in clustering: {n_included}")

    # Encode
    encoded = build_encoded_dataframe(df, column_meta)
    if encoded.empty:
        print("ERROR: no columns available for clustering after encoding.", file=sys.stderr)
        sys.exit(1)
    print(f"Encoded matrix: {encoded.shape[0]} rows × {encoded.shape[1]} features")

    # HAC + dendrogram traversal
    print(f"Running HAC  linkage={hac_cfg['linkage']}  metric={hac_cfg['metric']}  "
          f"max_depth={hac_cfg['max_depth']}")
    concepts = build_concept_tree_from_dendrogram(encoded, hac_cfg)

    leaf_concepts = [c for c in concepts if c["row_indices"]]
    print(f"Concepts: {len(concepts)} total, {len(leaf_concepts)} leaves")
    for c in leaf_concepts:
        print(f"  {c['name']}: {len(c['row_indices'])} rows")

    # OWL export
    turtle = export_owl(
        df, column_meta, concepts,
        hac_cfg["concept_ns"], hac_cfg["entity_ns"]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(turtle)
    print(f"OWL ontology written to {out_path}")


if __name__ == "__main__":
    main()
