# TaBIIC 2 - Taxonomy Building Through Iterative and Interactive Clustering

A web application for interactively building taxonomies from tabular data. Upload a CSV, Excel, or JSON file, inspect and configure your columns, then construct a hierarchy of concepts — each representing a subset of rows defined by restrictions on column values.

## Features

- **Data tab** — Upload tabular files (CSV, TSV, XLSX, JSON). Columns are auto-detected as numeric, categorical, date, or title/ID. Adjust types, toggle inclusion, and view per-column summaries (statistics, top values, date ranges).
- **Taxonomy tab** — Build a directed acyclic graph (DAG) of concepts over your data:
  - **Add subconcept** — Define restrictions (column/operator/value) to carve out a subset of a parent concept's rows.
  - **Complement** — Create a sibling concept containing the parent's rows not covered by selected siblings.
  - **Union / Intersection** — Combine selected concepts into a new parent (union) or child (intersection).
  - **Rename / Delete / Reset** — Manage concepts inline.
- **Graph visualisation** — Interactive DAG rendered with Cytoscape.js. Nodes show row counts and coverage colour-coding. Supports multi-selection for set operations.
- **Dark mode** — Light/dark theme toggle with system preference detection.

## Installation

Requires Python 3.10+.

```bash
git clone <repo-url>
cd TaBIIC2
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate
python app.py
```

Open http://127.0.0.1:5000 in your browser. Upload a data file on the Data tab, then switch to the Taxonomy tab to start building concepts.
