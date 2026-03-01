"""
FCA Taxonomy Builder
====================
Build an OWL/Turtle taxonomy from a CSV file using Formal Concept Analysis
(FCA).  Column preprocessing:
  - categorical  → one binary attribute per unique value  (col=val)
  - numeric/date → one binary attribute per discretisation bin  (col∈[lo,hi))
  - title_id     → excluded from the concept context

The formal concept lattice is explored BFS from the top concept (all objects,
common attributes) and limited to max_depth levels.  Each discovered concept
becomes an OWL class whose owl:equivalentClass axiom is built directly from
its intent (the set of shared binary attributes).  Hasse diagram edges are
computed post-hoc and emitted as rdfs:subClassOf.

Usage
-----
    python fca_taxonomy.py <data.csv> <config.json> [-o output.ttl]

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
  "fca": {
    "max_depth":   3,                  // depth of lattice BFS (default 3)
    "discretize": {
      "n_bins":   3,                   // bins per numeric/date column (default 3)
      "strategy": "quantile"           // "quantile" | "uniform" (default quantile)
    },
    "concept_ns":  "http://example.org/fca/schema#",
    "entity_ns":   "http://example.org/fca/data/"
  }
}
"""

import argparse
import json
import re
import sys
from collections import deque

import numpy as np
import pandas as pd
from rdflib import Graph, Namespace, Literal, BNode, RDF, RDFS, OWL, XSD
from rdflib.collection import Collection


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build an OWL taxonomy via FCA from a CSV file."
    )
    p.add_argument("csv", help="Input CSV file")
    p.add_argument("config", help="JSON config file")
    p.add_argument("-o", "--output", default=None,
                   help="Output Turtle file (default: <csv stem>.ttl)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config / column meta
# ---------------------------------------------------------------------------

def load_config(path):
    with open(path) as f:
        cfg = json.load(f)
    columns = cfg.get("columns", {})
    fca = cfg.get("fca", {})
    fca.setdefault("max_depth", 3)
    disc = fca.setdefault("discretize", {})
    disc.setdefault("n_bins", 3)
    disc.setdefault("strategy", "quantile")
    fca.setdefault("concept_ns", "http://example.org/fca/schema#")
    fca.setdefault("entity_ns", "http://example.org/fca/data/")
    return columns, fca


def build_column_meta(columns_cfg, df):
    records = []
    for col in df.columns:
        cfg = columns_cfg.get(col, {})
        records.append({
            "column": col,
            "user_type": cfg.get("type", "numeric"),
            "include": cfg.get("include", True),
        })
    return pd.DataFrame(records).set_index("column")


# ---------------------------------------------------------------------------
# Formal context construction
# ---------------------------------------------------------------------------

def build_formal_context(df, column_meta, discretize_cfg):
    """
    Build a binary formal context from the DataFrame.

    Returns a dict:
        objects      : frozenset of row indices
        attrs        : list of attribute-metadata dicts (one per binary attribute)
        attr_extents : {attr_name -> frozenset of row indices having that attribute}
        obj_intents  : {row_idx   -> frozenset of attr_names the row satisfies}
    """
    n_bins = int(discretize_cfg.get("n_bins", 3))
    strategy = discretize_cfg.get("strategy", "quantile")

    attr_list = []
    attr_extents = {}
    obj_intents = {idx: set() for idx in df.index}

    for col_name in column_meta.index:
        if not column_meta.at[col_name, "include"]:
            continue
        col_type = column_meta.at[col_name, "user_type"]
        if col_type == "title_id":
            continue

        if col_type == "categorical":
            _add_categorical_attrs(df, col_name, attr_list, attr_extents, obj_intents)
        elif col_type in ("numeric", "date"):
            _add_range_attrs(df, col_name, col_type, n_bins, strategy,
                             attr_list, attr_extents, obj_intents)

    obj_intents_frozen = {idx: frozenset(s) for idx, s in obj_intents.items()}
    return {
        "objects": frozenset(df.index),
        "attrs": attr_list,
        "attr_extents": attr_extents,
        "obj_intents": obj_intents_frozen,
    }


def _add_categorical_attrs(df, col_name, attr_list, attr_extents, obj_intents):
    for val in df[col_name].dropna().astype(str).unique():
        extent = frozenset(df.index[df[col_name].astype(str) == val])
        if not extent:
            continue
        attr_name = f"{col_name}={val}"
        attr_list.append({"name": attr_name, "col": col_name,
                           "kind": "categorical", "val": val})
        attr_extents[attr_name] = extent
        for idx in extent:
            obj_intents[idx].add(attr_name)


def _add_range_attrs(df, col_name, col_type, n_bins, strategy,
                     attr_list, attr_extents, obj_intents):
    if col_type == "numeric":
        series = pd.to_numeric(df[col_name], errors="coerce")
    else:
        series = pd.to_datetime(df[col_name], errors="coerce", format="mixed")
        series = series.astype(np.int64) // 10**9

    valid = series.dropna()
    if valid.empty or len(valid.unique()) < 2:
        return

    try:
        if strategy == "quantile":
            _, bin_edges = pd.qcut(valid, q=n_bins, duplicates="drop", retbins=True)
        else:
            _, bin_edges = pd.cut(valid, bins=n_bins, retbins=True)
    except ValueError:
        return

    n = len(bin_edges) - 1
    for i in range(n):
        lo = float(bin_edges[i])
        hi = float(bin_edges[i + 1])
        lo_incl = True                  # left-closed (pd.cut include_lowest)
        hi_incl = (i == n - 1)          # last bin is right-closed

        mask = (series >= lo) & (series <= hi if hi_incl else series < hi) & series.notna()
        extent = frozenset(df.index[mask])
        if not extent:
            continue

        lb = "["
        rb = "]" if hi_incl else ")"
        if col_type == "date":
            lo_str = pd.Timestamp(int(lo) * 10**9).strftime("%Y-%m-%d")
            hi_str = pd.Timestamp(int(hi) * 10**9).strftime("%Y-%m-%d")
            attr_name = f"{col_name}∈{lb}{lo_str},{hi_str}{rb}"
        else:
            attr_name = f"{col_name}∈{lb}{lo:.4g},{hi:.4g}{rb}"

        attr_list.append({
            "name": attr_name, "col": col_name, "kind": col_type,
            "lo": lo, "hi": hi, "lo_incl": lo_incl, "hi_incl": hi_incl,
        })
        attr_extents[attr_name] = extent
        for idx in extent:
            obj_intents[idx].add(attr_name)


# ---------------------------------------------------------------------------
# Concept discovery — BFS from top concept
# ---------------------------------------------------------------------------

def _intent_of_extent(extent, obj_intents):
    """Attributes shared by every object in the extent."""
    result = None
    for idx in extent:
        oi = obj_intents.get(idx, frozenset())
        result = oi if result is None else (result & oi)
    return result if result is not None else frozenset()


def _extent_of_intent(intent, attr_extents, all_objects):
    """Objects that satisfy every attribute in the intent."""
    result = all_objects
    for a in intent:
        result = result & attr_extents[a]
    return result


def discover_concepts(context, max_depth):
    """
    BFS from the top concept (extent = all objects).

    At each concept we try adding one attribute at a time to the intent.
    The resulting extent is closed (intent → extent) to obtain a proper
    formal concept.  Already-seen extents are skipped.

    Returns a list of concept dicts:
        id, name, extent (frozenset), intent (frozenset), depth
    """
    all_objects = context["objects"]
    attr_extents = context["attr_extents"]
    obj_intents = context["obj_intents"]
    all_attrs = frozenset(a["name"] for a in context["attrs"])

    def intent_of(extent):
        return _intent_of_extent(extent, obj_intents)

    def extent_of(intent):
        return _extent_of_intent(intent, attr_extents, all_objects)

    top_intent = intent_of(all_objects)
    root = {
        "id": "root", "name": "Root",
        "extent": all_objects, "intent": top_intent, "depth": 0,
    }

    seen = {all_objects: root}
    concepts = [root]
    queue = deque([root])
    counter = [0]

    while queue:
        current = queue.popleft()
        if current["depth"] >= max_depth:
            continue

        remaining = all_attrs - current["intent"]

        for attr in remaining:
            raw_ext = current["extent"] & attr_extents[attr]
            if not raw_ext:
                continue

            # Close: compute the formal concept containing this intersection
            new_intent = intent_of(raw_ext)
            new_extent = extent_of(new_intent)

            if new_extent == current["extent"] or new_extent in seen:
                continue

            counter[0] += 1
            cid = f"c{counter[0]:02d}"
            new_concept = {
                "id": cid, "name": cid,
                "extent": new_extent, "intent": new_intent,
                "depth": current["depth"] + 1,
            }
            seen[new_extent] = new_concept
            concepts.append(new_concept)
            queue.append(new_concept)

    return concepts


# ---------------------------------------------------------------------------
# Hasse diagram
# ---------------------------------------------------------------------------

def build_hasse_edges(concepts):
    """
    Compute direct parent-child edges in the Hasse diagram.

    P is a DIRECT parent of C iff:
      P.extent ⊋ C.extent  AND  no discovered concept M satisfies
      P.extent ⊋ M.extent ⊋ C.extent.

    Returns {concept_id: [direct_parent_id, ...]}
    """
    # Sort largest extent first (= highest in lattice)
    by_size = sorted(concepts, key=lambda c: len(c["extent"]), reverse=True)
    parent_map = {c["id"]: [] for c in concepts}

    for c in by_size:
        candidates = [p for p in by_size if p["extent"] > c["extent"]]
        for p in candidates:
            # p is a direct parent unless some other candidate sits between
            if not any(p["extent"] > m["extent"] > c["extent"]
                       for m in candidates):
                parent_map[c["id"]].append(p["id"])

    return parent_map


# ---------------------------------------------------------------------------
# OWL export helpers
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
            return Literal(str(pd.to_datetime(val).date()), datatype=XSD.date)
        except Exception:
            return None
    else:
        return Literal(str(val), datatype=XSD.string)


def _build_attr_restriction(g, attr_meta, prop_map, cat_value_map):
    """Return an OWL Restriction BNode for a single formal-context attribute."""
    col = attr_meta["col"]
    prop_uri = prop_map.get(col)
    if prop_uri is None:
        return None

    if attr_meta["kind"] == "categorical":
        val_uri = cat_value_map.get((col, attr_meta["val"]))
        if val_uri is None:
            return None
        restr = BNode()
        g.add((restr, RDF.type, OWL.Restriction))
        g.add((restr, OWL.onProperty, prop_uri))
        g.add((restr, OWL.hasValue, val_uri))
        return restr

    # numeric or date range
    base_dt = XSD.decimal if attr_meta["kind"] == "numeric" else XSD.date
    lo, hi = attr_meta["lo"], attr_meta["hi"]
    lo_incl, hi_incl = attr_meta["lo_incl"], attr_meta["hi_incl"]

    if base_dt == XSD.date:
        lo_lit = Literal(pd.Timestamp(int(lo) * 10**9).strftime("%Y-%m-%d"), datatype=XSD.date)
        hi_lit = Literal(pd.Timestamp(int(hi) * 10**9).strftime("%Y-%m-%d"), datatype=XSD.date)
    else:
        lo_lit = Literal(float(lo), datatype=XSD.decimal)
        hi_lit = Literal(float(hi), datatype=XSD.decimal)

    fb_lo = BNode()
    g.add((fb_lo, XSD.minInclusive if lo_incl else XSD.minExclusive, lo_lit))
    fb_hi = BNode()
    g.add((fb_hi, XSD.maxInclusive if hi_incl else XSD.maxExclusive, hi_lit))

    facet_list = BNode()
    Collection(g, facet_list, [fb_lo, fb_hi])

    dt_bnode = BNode()
    g.add((dt_bnode, RDF.type, RDFS.Datatype))
    g.add((dt_bnode, OWL.onDatatype, base_dt))
    g.add((dt_bnode, OWL.withRestrictions, facet_list))

    restr = BNode()
    g.add((restr, RDF.type, OWL.Restriction))
    g.add((restr, OWL.onProperty, prop_uri))
    g.add((restr, OWL.someValuesFrom, dt_bnode))
    return restr


# ---------------------------------------------------------------------------
# OWL export
# ---------------------------------------------------------------------------

def export_owl(df, column_meta, concepts, hasse_parents, context,
               concept_ns_uri, entity_ns_uri):
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
    cat_value_map = {}

    for col_name in column_meta.index:
        safe = _safe_local_name(col_name)
        prop_uri = CNS[safe]
        prop_map[col_name] = prop_uri
        col_type = column_meta.at[col_name, "user_type"]

        if col_type == "categorical":
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
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

    attr_meta_map = {a["name"]: a for a in context["attrs"]}

    # --- Classes ---
    class_map = {}
    for concept in concepts:
        local = _safe_local_name(concept["name"])
        class_uri = CNS[local]
        class_map[concept["id"]] = class_uri
        g.add((class_uri, RDF.type, OWL.Class))
        g.add((class_uri, RDFS.label, Literal(concept["name"])))

    # --- rdfs:subClassOf (Hasse edges) ---
    for concept in concepts:
        parents = hasse_parents.get(concept["id"], [])
        if not parents:
            g.add((class_map[concept["id"]], RDFS.subClassOf, OWL.Thing))
        else:
            for pid in parents:
                g.add((class_map[concept["id"]], RDFS.subClassOf, class_map[pid]))

    # --- owl:equivalentClass from intent ---
    for concept in concepts:
        if not concept["intent"]:
            continue  # Root with empty intent — no restriction to express

        restrictions = []
        for attr_name in concept["intent"]:
            a_meta = attr_meta_map.get(attr_name)
            if a_meta is None:
                continue
            restr = _build_attr_restriction(g, a_meta, prop_map, cat_value_map)
            if restr is not None:
                restrictions.append(restr)

        if not restrictions:
            continue

        if len(restrictions) == 1:
            eq_class = restrictions[0]
        else:
            eq_class = BNode()
            g.add((eq_class, RDF.type, OWL.Class))
            lst = BNode()
            Collection(g, lst, restrictions)
            g.add((eq_class, OWL.intersectionOf, lst))

        g.add((class_map[concept["id"]], OWL.equivalentClass, eq_class))

    # --- Individuals ---
    # Type each row to its most specific discovered concept(s).
    # "Most specific" = no other discovered concept has a strictly smaller
    # extent that also contains the row.
    row_to_concepts = {}
    for idx in df.index:
        containing = [c for c in concepts if idx in c["extent"]]
        minimal = [
            c for c in containing
            if not any(c2["id"] != c["id"]
                       and c2["extent"] < c["extent"]
                       and idx in c2["extent"]
                       for c2 in containing)
        ]
        row_to_concepts[idx] = [c["id"] for c in minimal]

    for idx in df.index:
        ind_uri = ENS[f"entity_{idx}"]
        g.add((ind_uri, RDF.type, OWL.NamedIndividual))

        for cid in row_to_concepts.get(idx, []):
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
    out_path = args.output or (args.csv.rsplit(".", 1)[0] + ".ttl")

    df = pd.read_csv(args.csv)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {args.csv}")

    columns_cfg, fca_cfg = load_config(args.config)
    column_meta = build_column_meta(columns_cfg, df)

    disc_cfg = fca_cfg["discretize"]
    max_depth = int(fca_cfg["max_depth"])

    print(f"Building formal context  "
          f"n_bins={disc_cfg['n_bins']}  strategy={disc_cfg['strategy']}")
    context = build_formal_context(df, column_meta, disc_cfg)
    print(f"  {len(context['objects'])} objects × {len(context['attrs'])} binary attributes")

    print(f"Discovering concepts  max_depth={max_depth}")
    concepts = discover_concepts(context, max_depth)
    print(f"  {len(concepts)} concepts found")

    print("Computing Hasse diagram...")
    hasse_parents = build_hasse_edges(concepts)

    leaf_concepts = [c for c in concepts
                     if not any(p == c["id"]
                                for parents in hasse_parents.values()
                                for p in parents)]
    print(f"  {len(leaf_concepts)} leaf concepts")

    turtle = export_owl(df, column_meta, concepts, hasse_parents, context,
                        fca_cfg["concept_ns"], fca_cfg["entity_ns"])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(turtle)
    print(f"OWL ontology written to {out_path}")


if __name__ == "__main__":
    main()
