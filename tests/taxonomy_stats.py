"""
Taxonomy Statistics
===================
Compute structural statistics for a collection of OWL/Turtle ontologies and
write them to a CSV file.

Usage
-----
    python taxonomy_stats.py onto1.ttl onto2.ttl ... [-o stats.csv]

Statistics reported
-------------------
  n_classes              Named OWL classes (URIRef subjects of rdf:type owl:Class
                         or of rdfs:subClassOf)
  n_individuals          Named individuals (owl:NamedIndividual or rdf:type a
                         named class)
  n_object_properties    owl:ObjectProperty declarations
  n_datatype_properties  owl:DatatypeProperty declarations
  n_restrictions         owl:Restriction BNodes (one per condition in a class
                         expression)
  n_equiv_class          Classes carrying an owl:equivalentClass axiom
  n_levels               Length of the longest root-to-leaf path + 1
  n_roots                Classes with no named superclass
  n_leaves               Classes with no named subclass
  n_multi_parent         Classes with more than one named superclass (DAG nodes)
  avg_depth / std_depth  Mean and std dev of class depth (longest path from root)
  avg_branching          Mean number of direct subclasses per non-leaf class
  std_branching          Std dev of branching factor
  avg_inst / med_inst / std_inst
                         Mean, median, std dev of instance count across ALL classes
  avg_leaf_inst / med_leaf_inst / std_leaf_inst
                         Same, restricted to leaf classes only
  n_empty_classes        Classes with zero direct instances
  coverage               Fraction of individuals typed to at least one non-root class
"""

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict, deque

from rdflib import Graph, RDF, RDFS, OWL
from rdflib.term import URIRef, BNode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute statistics for OWL/Turtle ontology files."
    )
    p.add_argument("ontologies", nargs="+", help="Turtle (.ttl) files to analyse")
    p.add_argument("-o", "--output", default="stats.csv",
                   help="Output CSV file (default: stats.csv)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def load_graph(path):
    g = Graph()
    g.parse(path, format="turtle")
    return g


def get_named_classes(g):
    """Named (URIRef) OWL classes, excluding owl:Thing itself."""
    classes = set()
    for s in g.subjects(RDF.type, OWL.Class):
        if isinstance(s, URIRef) and s != OWL.Thing:
            classes.add(s)
    # rdfs:subClassOf often declares classes implicitly
    for s, o in g.subject_objects(RDFS.subClassOf):
        if isinstance(s, URIRef) and s != OWL.Thing:
            classes.add(s)
        if isinstance(o, URIRef) and o != OWL.Thing:
            classes.add(o)
    return classes


def get_subclass_graph(g, classes):
    """
    Return two adjacency dicts (both keyed by URIRef):
        parent_to_children : parent  -> set of direct named subclasses
        child_to_parents   : child   -> set of direct named superclasses
    owl:Thing is treated as the implicit root and excluded from both dicts.
    """
    p2c = defaultdict(set)
    c2p = defaultdict(set)
    for child, parent in g.subject_objects(RDFS.subClassOf):
        if not isinstance(child, URIRef) or not isinstance(parent, URIRef):
            continue
        if parent == OWL.Thing or child == OWL.Thing:
            continue
        if child in classes and parent in classes:
            p2c[parent].add(child)
            c2p[child].add(parent)
    return p2c, c2p


def get_individuals(g, classes):
    """
    Named individuals: subjects of rdf:type owl:NamedIndividual, OR subjects
    of rdf:type <NamedClass> (for ontologies that omit the NamedIndividual
    declaration).
    """
    inds = set()
    for s in g.subjects(RDF.type, OWL.NamedIndividual):
        if isinstance(s, URIRef):
            inds.add(s)
    for s, o in g.subject_objects(RDF.type):
        if isinstance(s, URIRef) and isinstance(o, URIRef) and o in classes:
            inds.add(s)
    return inds


# ---------------------------------------------------------------------------
# Depth computation (longest path from any root in the DAG)
# ---------------------------------------------------------------------------

def compute_depths(classes, roots, p2c, c2p):
    """
    Topological DP (Kahn's algorithm variant) giving the longest path from
    any root to each class.  Handles DAGs with multiple parents correctly.
    """
    # In-degree = number of named parents
    in_degree = {c: len(c2p.get(c, set())) for c in classes}
    depth = {}

    queue = deque()
    for r in roots:
        depth[r] = 0
        queue.append(r)

    while queue:
        node = queue.popleft()
        for child in p2c.get(node, set()):
            # Update child's depth with the best seen so far
            depth[child] = max(depth.get(child, 0), depth[node] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    # Unreachable classes (disconnected from roots) get depth 0
    for c in classes:
        if c not in depth:
            depth[c] = 0

    return depth


# ---------------------------------------------------------------------------
# Per-ontology statistics
# ---------------------------------------------------------------------------

def _safe_std(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _safe_mean(values):
    return statistics.mean(values) if values else 0.0


def _safe_median(values):
    return statistics.median(values) if values else 0.0


def compute_stats(path):
    try:
        g = load_graph(path)
    except Exception as e:
        print(f"  ERROR loading {path}: {e}", file=sys.stderr)
        return None

    classes = get_named_classes(g)
    p2c, c2p = get_subclass_graph(g, classes)

    roots = {c for c in classes if not c2p.get(c)}
    leaves = {c for c in classes if not p2c.get(c)}
    multi_parent = {c for c in classes if len(c2p.get(c, set())) > 1}

    depths = compute_depths(classes, roots, p2c, c2p)
    depth_values = list(depths.values())
    n_levels = (max(depth_values) + 1) if depth_values else 0

    # Branching factor: over all non-leaf classes
    branching = [len(p2c[c]) for c in classes if p2c.get(c)]

    # Individuals and instance counts
    individuals = get_individuals(g, classes)

    inst_count = defaultdict(int)
    for ind in individuals:
        for cls in g.objects(ind, RDF.type):
            if isinstance(cls, URIRef) and cls in classes:
                inst_count[cls] += 1

    all_counts = [inst_count.get(c, 0) for c in classes]
    leaf_counts = [inst_count.get(c, 0) for c in leaves]

    # Individuals typed to at least one non-root class
    covered = set()
    for ind in individuals:
        for cls in g.objects(ind, RDF.type):
            if isinstance(cls, URIRef) and cls in classes and cls not in roots:
                covered.add(ind)
                break
    coverage = len(covered) / len(individuals) if individuals else 0.0

    # OWL axiom counts
    n_restrictions = sum(
        1 for _ in g.subjects(RDF.type, OWL.Restriction)
        if isinstance(_, BNode)
    )
    n_equiv = sum(
        1 for s, _ in g.subject_objects(OWL.equivalentClass)
        if isinstance(s, URIRef) and s in classes
    )
    n_obj_props = sum(1 for _ in g.subjects(RDF.type, OWL.ObjectProperty))
    n_dt_props = sum(1 for _ in g.subjects(RDF.type, OWL.DatatypeProperty))

    return {
        "file":                 path,
        "n_classes":            len(classes),
        "n_individuals":        len(individuals),
        "n_object_properties":  n_obj_props,
        "n_datatype_properties":n_dt_props,
        "n_restrictions":       n_restrictions,
        "n_equiv_class":        n_equiv,
        "n_levels":             n_levels,
        "n_roots":              len(roots),
        "n_leaves":             len(leaves),
        "n_multi_parent":       len(multi_parent),
        "avg_depth":            round(_safe_mean(depth_values), 3),
        "std_depth":            round(_safe_std(depth_values), 3),
        "avg_branching":        round(_safe_mean(branching), 3),
        "std_branching":        round(_safe_std(branching), 3),
        "avg_inst":             round(_safe_mean(all_counts), 3),
        "med_inst":             round(_safe_median(all_counts), 3),
        "std_inst":             round(_safe_std(all_counts), 3),
        "avg_leaf_inst":        round(_safe_mean(leaf_counts), 3),
        "med_leaf_inst":        round(_safe_median(leaf_counts), 3),
        "std_leaf_inst":        round(_safe_std(leaf_counts), 3),
        "n_empty_classes":      sum(1 for c in all_counts if c == 0),
        "coverage":             round(coverage, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

COLUMNS = [
    "file",
    "n_classes", "n_individuals",
    "n_object_properties", "n_datatype_properties",
    "n_restrictions", "n_equiv_class",
    "n_levels", "n_roots", "n_leaves", "n_multi_parent",
    "avg_depth", "std_depth",
    "avg_branching", "std_branching",
    "avg_inst", "med_inst", "std_inst",
    "avg_leaf_inst", "med_leaf_inst", "std_leaf_inst",
    "n_empty_classes", "coverage",
]


def main():
    args = parse_args()
    rows = []

    for path in args.ontologies:
        print(f"Analysing {path} ...", end=" ", flush=True)
        stats = compute_stats(path)
        if stats:
            rows.append(stats)
            print(f"{stats['n_classes']} classes, "
                  f"{stats['n_individuals']} individuals, "
                  f"{stats['n_levels']} levels")
        else:
            print("skipped")

    if not rows:
        print("No ontologies processed.", file=sys.stderr)
        sys.exit(1)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nStats written to {args.output}")


if __name__ == "__main__":
    main()
