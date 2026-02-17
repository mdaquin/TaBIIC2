"""RDF/OWL export for TaBIIC2 taxonomies."""

import re
import pandas as pd
from rdflib import Graph, Namespace, Literal, BNode, RDF, RDFS, OWL, XSD
from rdflib.collection import Collection


def export_taxonomy_as_turtle(taxonomy, raw_df, column_meta,
                               concept_ns_uri, entity_ns_uri):
    """Export the taxonomy as an OWL ontology in Turtle format."""
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

    concepts = taxonomy["concepts"]

    # 1. Name map for all concepts
    name_map = _build_name_map(concepts)

    # 2. Properties (one per included column)
    included_cols = column_meta[column_meta["include"] == True]
    prop_map = {}
    for col_name in included_cols.index:
        safe = _safe_local_name(col_name)
        prop_uri = CNS[safe]
        prop_map[col_name] = prop_uri
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        col_type = included_cols.at[col_name, "user_type"]
        if col_type == "numeric":
            g.add((prop_uri, RDFS.range, XSD.decimal))
        elif col_type == "date":
            g.add((prop_uri, RDFS.range, XSD.date))
        else:
            g.add((prop_uri, RDFS.range, XSD.string))

    # 3. Classes (one per concept)
    class_map = {}
    for cid, concept in concepts.items():
        local = _safe_local_name(name_map[cid])
        class_uri = CNS[local]
        class_map[cid] = class_uri
        g.add((class_uri, RDF.type, OWL.Class))
        if concept["name"]:
            g.add((class_uri, RDFS.label, Literal(concept["name"])))

    # 4. rdfs:subClassOf
    for cid, concept in concepts.items():
        for parent_id in concept["parent_ids"]:
            if parent_id in class_map:
                g.add((class_map[cid], RDFS.subClassOf, class_map[parent_id]))
        if concept["origin"] == "root":
            g.add((class_map[cid], RDFS.subClassOf, OWL.Thing))

    # 5. owl:equivalentClass axioms
    for cid, concept in concepts.items():
        _add_equivalent_class(g, concept, class_map, prop_map, column_meta)

    # 6. Individuals
    row_concepts = {}
    for cid, concept in concepts.items():
        for idx in concept["row_indices"]:
            row_concepts.setdefault(idx, []).append(cid)

    for idx in raw_df.index:
        ind_uri = ENS[f"row_{idx}"]
        g.add((ind_uri, RDF.type, OWL.NamedIndividual))

        for cid in row_concepts.get(idx, []):
            g.add((ind_uri, RDF.type, class_map[cid]))

        for col_name in included_cols.index:
            val = raw_df.at[idx, col_name]
            if _is_missing(val):
                continue
            col_type = included_cols.at[col_name, "user_type"]
            literal = _make_literal(val, col_type)
            if literal is not None:
                g.add((ind_uri, prop_map[col_name], literal))

    return g.serialize(format="turtle")


# -- Naming helpers -----------------------------------------------------------

def _build_name_map(concepts):
    """Map concept IDs to display names for URIs."""
    name_map = {}
    counter = 0
    for cid, concept in concepts.items():
        if concept["name"]:
            name_map[cid] = concept["name"]
        elif concept["origin"] == "root":
            name_map[cid] = "Root"
        else:
            counter += 1
            name_map[cid] = f"c{counter:02d}"
    return name_map


def _safe_local_name(name):
    """Convert a name to a safe URI local name."""
    safe = re.sub(r"[^\w-]", "_", name)
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe or not safe[0].isalpha():
        safe = "x_" + safe
    return safe


# -- Literal helpers ----------------------------------------------------------

def _is_missing(val):
    if val is None:
        return True
    try:
        return pd.isna(val)
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


# -- OWL equivalentClass axioms ----------------------------------------------

def _add_equivalent_class(g, concept, class_map, prop_map, column_meta):
    origin = concept["origin"]
    class_uri = class_map[concept["id"]]

    if origin == "root":
        return
    elif origin in ("restriction", "wsom", "merge"):
        _add_restriction_equivalent(g, concept, class_uri, class_map,
                                    prop_map, column_meta)
    elif origin == "complement":
        _add_complement_equivalent(g, concept, class_uri, class_map)
    elif origin == "intersection":
        _add_intersection_equivalent(g, concept, class_uri, class_map)
    elif origin == "union":
        _add_union_equivalent(g, concept, class_uri, class_map)


def _add_restriction_equivalent(g, concept, class_uri, class_map,
                                prop_map, column_meta):
    """equivalentClass = intersectionOf(parent, OWL restrictions)."""
    if not concept["restrictions"] or not concept["parent_ids"]:
        return

    parent_class = class_map[concept["parent_ids"][0]]
    owl_restrictions = _build_owl_restrictions(
        g, concept["restrictions"], prop_map, column_meta
    )
    if not owl_restrictions:
        return

    members = [parent_class] + owl_restrictions
    intersection_bnode = BNode()
    g.add((intersection_bnode, RDF.type, OWL.Class))
    member_list = BNode()
    Collection(g, member_list, members)
    g.add((intersection_bnode, OWL.intersectionOf, member_list))
    g.add((class_uri, OWL.equivalentClass, intersection_bnode))


def _add_complement_equivalent(g, concept, class_uri, class_map):
    """equivalentClass = intersectionOf(parent, complementOf(source(s)))."""
    if not concept["parent_ids"] or not concept.get("source_ids"):
        return

    parent_class = class_map[concept["parent_ids"][0]]
    source_classes = [class_map[sid] for sid in concept["source_ids"]
                      if sid in class_map]
    if not source_classes:
        return

    if len(source_classes) == 1:
        to_complement = source_classes[0]
    else:
        union_bnode = BNode()
        g.add((union_bnode, RDF.type, OWL.Class))
        union_list = BNode()
        Collection(g, union_list, source_classes)
        g.add((union_bnode, OWL.unionOf, union_list))
        to_complement = union_bnode

    complement_bnode = BNode()
    g.add((complement_bnode, RDF.type, OWL.Class))
    g.add((complement_bnode, OWL.complementOf, to_complement))

    intersection_bnode = BNode()
    g.add((intersection_bnode, RDF.type, OWL.Class))
    member_list = BNode()
    Collection(g, member_list, [parent_class, complement_bnode])
    g.add((intersection_bnode, OWL.intersectionOf, member_list))
    g.add((class_uri, OWL.equivalentClass, intersection_bnode))


def _add_intersection_equivalent(g, concept, class_uri, class_map):
    """equivalentClass = intersectionOf(source concepts)."""
    source_classes = [class_map[sid] for sid in concept.get("source_ids", [])
                      if sid in class_map]
    if len(source_classes) < 2:
        return

    intersection_bnode = BNode()
    g.add((intersection_bnode, RDF.type, OWL.Class))
    member_list = BNode()
    Collection(g, member_list, source_classes)
    g.add((intersection_bnode, OWL.intersectionOf, member_list))
    g.add((class_uri, OWL.equivalentClass, intersection_bnode))


def _add_union_equivalent(g, concept, class_uri, class_map):
    """equivalentClass = unionOf(source concepts)."""
    source_classes = [class_map[sid] for sid in concept.get("source_ids", [])
                      if sid in class_map]
    if len(source_classes) < 2:
        return

    union_bnode = BNode()
    g.add((union_bnode, RDF.type, OWL.Class))
    union_list = BNode()
    Collection(g, union_list, source_classes)
    g.add((union_bnode, OWL.unionOf, union_list))
    g.add((class_uri, OWL.equivalentClass, union_bnode))


# -- OWL restriction building ------------------------------------------------

def _build_owl_restrictions(g, restrictions, prop_map, column_meta):
    """Convert restriction dicts to OWL Restriction BNodes."""
    by_column = {}
    for r in restrictions:
        by_column.setdefault(r["column"], []).append(r)

    result = []
    for col, col_restrictions in by_column.items():
        if col not in prop_map:
            continue
        col_type = (column_meta.at[col, "user_type"]
                    if col in column_meta.index else "categorical")
        prop_uri = prop_map[col]

        if col_type == "categorical":
            for r in col_restrictions:
                result.append(
                    _build_has_value_restriction(g, prop_uri, str(r["value"]))
                )
        elif col_type in ("numeric", "date"):
            bnode = _build_datatype_restriction(
                g, prop_uri, col_restrictions, col_type
            )
            if bnode is not None:
                result.append(bnode)

    return result


def _build_has_value_restriction(g, prop_uri, value):
    restr = BNode()
    g.add((restr, RDF.type, OWL.Restriction))
    g.add((restr, OWL.onProperty, prop_uri))
    g.add((restr, OWL.hasValue, Literal(value, datatype=XSD.string)))
    return restr


def _build_datatype_restriction(g, prop_uri, col_restrictions, col_type):
    base_datatype = XSD.decimal if col_type == "numeric" else XSD.date

    FACET_MAP = {
        ">=": XSD.minInclusive,
        ">": XSD.minExclusive,
        "<=": XSD.maxInclusive,
        "<": XSD.maxExclusive,
    }

    facet_pairs = []
    for r in col_restrictions:
        op = r["operator"]
        val = r["value"]
        if op == "=":
            lit = _restriction_literal(val, col_type)
            facet_pairs.append((XSD.minInclusive, lit))
            facet_pairs.append((XSD.maxInclusive, lit))
        elif op in FACET_MAP:
            lit = _restriction_literal(val, col_type)
            facet_pairs.append((FACET_MAP[op], lit))

    if not facet_pairs:
        return None

    facet_bnodes = []
    for facet_uri, lit in facet_pairs:
        fb = BNode()
        g.add((fb, facet_uri, lit))
        facet_bnodes.append(fb)

    facet_list = BNode()
    Collection(g, facet_list, facet_bnodes)

    datatype_bnode = BNode()
    g.add((datatype_bnode, RDF.type, RDFS.Datatype))
    g.add((datatype_bnode, OWL.onDatatype, base_datatype))
    g.add((datatype_bnode, OWL.withRestrictions, facet_list))

    restr = BNode()
    g.add((restr, RDF.type, OWL.Restriction))
    g.add((restr, OWL.onProperty, prop_uri))
    g.add((restr, OWL.someValuesFrom, datatype_bnode))
    return restr


def _restriction_literal(val, col_type):
    if col_type == "numeric":
        return Literal(float(val), datatype=XSD.decimal)
    elif col_type == "date":
        return Literal(str(val), datatype=XSD.date)
    return Literal(str(val), datatype=XSD.string)
