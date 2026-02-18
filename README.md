# Taxonomy Building Through Iterative and Interactive Clustering

A web application for interactively building taxonomies from tabular data. Upload a CSV, Excel, or JSON file, inspect and configure your columns, then construct a hierarchy of concepts — each representing a subset of rows defined by restrictions on column values. Concepts can be created manually or discovered automatically using a Weighted Self-Organizing Map (WSOM).

## Features

### Data tab

- Upload tabular files (CSV, TSV, XLSX, JSON).
- Columns are auto-detected as **numeric**, **categorical**, **date**, or **title/ID** using dtype analysis, content parsing, and column-name heuristics.
- Adjust types and toggle inclusion per column.
- View per-column summaries: statistics for numeric, top values for categorical, date ranges for date columns.

### Taxonomy tab

Build a directed acyclic graph (DAG) of concepts over your data:

- **Add subconcept** — Define restrictions (column/operator/value) to carve out a subset of a parent concept's rows.
- **Edit restrictions** — Modify a concept's restrictions inline in the detail panel; row indices are recomputed for the concept and all its descendants.
- **Complement** — Create a sibling concept containing the parent's rows not covered by selected siblings.
- **Union / Intersection** — Combine selected concepts into a new parent (union) or child (intersection).
- **Find Intersections** — Automatically create non-empty pairwise intersections between all selected concepts.
- **Merge** — Combine concepts that share restrictions on the same numerical column into a single concept with a hull interval covering all originals. The merged concepts are replaced by the new one.
- **Find Sub-concepts (WSOM)** — Automatic sub-concept discovery using a Weighted Self-Organizing Map (see below).
- **Rename / Delete / Reset** — Manage concepts inline.

### WSOM automatic discovery

Select a concept and click **Find Sub-concepts** to train a WSOM on its rows:

1. Configure hyperparameters: map size (2–20), column specificity (sparsity coefficient), number of training iterations, and optionally ignore specific columns.
2. Training runs in a background thread with a live progress bar.
3. Rows are grouped by Best Matching Unit (BMU) on the WSOM grid.
4. Each cluster is characterised with human-readable restrictions on the column with the highest WSOM weight, using precision-based selection over quantile thresholds.
5. Clusters with identical restrictions are merged; subset clusters are removed.
6. Proposed sub-concepts appear in the graph with a dashed border. Validate to keep them, cancel to discard, or retry with different parameters.

### Visualisation & UI

- Interactive DAG rendered with **Cytoscape.js** (dagre layout). Nodes show row counts and coverage colour-coding (green = fully covered by children, white = no coverage).
- Multi-selection (Ctrl/Cmd+click or box-select) for set operations.
- Detail panel shows concept name, restrictions, size, coverage, origin, and parent/child links.
- **Dark mode** — Light/dark theme toggle with system preference detection and localStorage persistence.

## Installation

Requires Python 3.10+.

```bash
git clone <repo-url>
cd TaBIIC2
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs Flask, pandas, openpyxl, and [ksom](https://pypi.org/project/ksom/) (which brings in PyTorch).

## Usage

```bash
source .venv/bin/activate
python app.py
```

Open http://127.0.0.1:5000 in your browser. Upload a data file on the Data tab, then switch to the Taxonomy tab to start building concepts.
