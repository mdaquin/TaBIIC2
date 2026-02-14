import logging
import threading

logging.basicConfig(level=logging.INFO)

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from markupsafe import Markup
import config
from state import app_state
from data_processing import loader, detector, summariser, encoder
from taxonomy import (
    create_taxonomy, add_subconcept, create_complement,
    create_union, create_intersection, create_pairwise_intersections,
    create_merge, delete_concept,
    update_restrictions, serialize_taxonomy, get_available_restrictions,
)
from wsom_discovery import run_wsom_training

app = Flask(__name__)
app.config.from_object(config)


# -- Template helper ----------------------------------------------------------

def render_summary(summary, col_type):
    """Render column summary as HTML. Used as a Jinja global function
    and also called from the API to return HTML fragments."""
    if not summary:
        return ""

    if col_type == "numeric":
        rows = []
        for label, key in [("Mean", "mean"), ("Median", "median"),
                           ("Std Dev", "std"), ("Min", "min"), ("Max", "max")]:
            val = summary.get(key)
            rows.append(f"<tr><td>{label}</td><td>{val if val is not None else 'N/A'}</td></tr>")
        missing = summary.get("missing", 0)
        rows.append(f"<tr><td>Missing</td><td>{missing}</td></tr>")
        return Markup(f'<table class="summary-table">{"".join(rows)}</table>')

    if col_type == "categorical":
        unique = summary.get("unique", 0)
        missing = summary.get("missing", 0)
        html = f"<p>{unique} unique values, {missing} missing</p>"
        top = summary.get("top_values", [])
        if top:
            items = "".join(
                f"<tr><td>{v['value']}</td><td>{v['count']}</td></tr>" for v in top
            )
            html += f'<table class="summary-table"><tr><th>Value</th><th>Count</th></tr>{items}</table>'
        return Markup(html)

    if col_type == "date":
        missing = summary.get("missing", 0)
        mn = summary.get("min", "N/A")
        mx = summary.get("max", "N/A")
        rng = summary.get("range_days")
        rng_str = f"{rng} days" if rng is not None else "N/A"
        return Markup(
            f'<table class="summary-table">'
            f"<tr><td>Min</td><td>{mn}</td></tr>"
            f"<tr><td>Max</td><td>{mx}</td></tr>"
            f"<tr><td>Range</td><td>{rng_str}</td></tr>"
            f"<tr><td>Missing</td><td>{missing}</td></tr>"
            f"</table>"
        )

    if col_type == "title_id":
        unique = summary.get("unique", 0)
        missing = summary.get("missing", 0)
        samples = summary.get("sample_values", [])
        sample_str = ", ".join(samples[:5])
        return Markup(
            f"<p>{unique} unique values, {missing} missing</p>"
            f'<p class="sample-values">Sample: {sample_str}</p>'
        )

    return ""


app.jinja_env.globals["render_summary"] = render_summary


# -- Helper functions ---------------------------------------------------------

def _build_columns_response():
    result = []
    for col_name, row in app_state.column_meta.iterrows():
        result.append({
            "column_name": col_name,
            "detected_type": row["detected_type"],
            "user_type": row["user_type"],
            "include": bool(row["include"]),
            "summary": row["summary"],
        })
    return result


def _build_single_column_response(column_name):
    row = app_state.column_meta.loc[column_name]
    return {
        "column_name": column_name,
        "detected_type": row["detected_type"],
        "user_type": row["user_type"],
        "include": bool(row["include"]),
        "summary": row["summary"],
    }


# -- Routes -------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("data_tab"))


@app.route("/data")
def data_tab():
    if not app_state.is_loaded():
        return render_template("data_tab.html", loaded=False)

    return render_template(
        "data_tab.html",
        loaded=True,
        filename=app_state.filename,
        row_count=len(app_state.raw_df),
        col_count=len(app_state.raw_df.columns),
        encoded_col_count=len(app_state.encoded_df.columns),
        columns=_build_columns_response(),
    )


@app.route("/data/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        flash("No file selected")
        return redirect(url_for("data_tab"))

    file = request.files["file"]
    if file.filename == "":
        flash("No file selected")
        return redirect(url_for("data_tab"))

    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in config.UPLOAD_ALLOWED_EXTENSIONS:
        flash(f"Unsupported format: .{ext}")
        return redirect(url_for("data_tab"))

    try:
        app_state.reset()
        app_state.filename = file.filename
        app_state.raw_df = loader.load_file(file, file.filename)
        app_state.column_meta = detector.detect_column_types(app_state.raw_df)
        app_state.column_meta = summariser.compute_all_summaries(
            app_state.raw_df, app_state.column_meta
        )
        app_state.encoded_df = encoder.build_encoded_dataframe(
            app_state.raw_df, app_state.column_meta
        )
    except ValueError as e:
        app_state.reset()
        flash(f"Error loading file: {e}")
        return redirect(url_for("data_tab"))

    return redirect(url_for("data_tab"))


@app.route("/data/columns")
def get_columns():
    if not app_state.is_loaded():
        return jsonify({"error": "No data loaded"}), 400
    return jsonify(_build_columns_response())


@app.route("/data/columns/<column_name>", methods=["PUT"])
def update_column(column_name):
    if not app_state.is_loaded():
        return jsonify({"error": "No data loaded"}), 400

    if column_name not in app_state.column_meta.index:
        return jsonify({"error": f"Column '{column_name}' not found"}), 404

    data = request.get_json()
    changed = False

    if "user_type" in data:
        new_type = data["user_type"]
        if new_type not in ("numeric", "categorical", "date", "title_id"):
            return jsonify({"error": f"Invalid type: {new_type}"}), 400
        app_state.column_meta.at[column_name, "user_type"] = new_type
        app_state.column_meta.at[column_name, "summary"] = summariser.compute_summary(
            app_state.raw_df[column_name], new_type
        )
        changed = True

    if "include" in data:
        app_state.column_meta.at[column_name, "include"] = bool(data["include"])
        changed = True

    if changed:
        app_state.encoded_df = encoder.build_encoded_dataframe(
            app_state.raw_df, app_state.column_meta
        )

    col_data = _build_single_column_response(column_name)
    col_data["summary_html"] = str(render_summary(col_data["summary"], col_data["user_type"]))

    return jsonify({
        "column": col_data,
        "encoded_col_count": len(app_state.encoded_df.columns),
    })


@app.route("/data/reset", methods=["POST"])
def reset_data():
    app_state.reset()
    return redirect(url_for("data_tab"))


@app.route("/taxonomy")
def taxonomy_tab():
    if not app_state.is_loaded():
        return render_template("taxonomy_tab.html", loaded=False)

    if app_state.taxonomy is None:
        app_state.taxonomy = create_taxonomy(app_state.raw_df)

    return render_template(
        "taxonomy_tab.html",
        loaded=True,
        filename=app_state.filename,
        row_count=len(app_state.raw_df),
    )


@app.route("/taxonomy/api/graph")
def taxonomy_graph():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400
    return jsonify(serialize_taxonomy(app_state.taxonomy))


@app.route("/taxonomy/api/concept", methods=["POST"])
def add_concept():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    data = request.get_json()
    parent_id = data.get("parent_id")
    restrictions = data.get("restrictions", [])
    name = data.get("name", "")

    if parent_id not in app_state.taxonomy["concepts"]:
        return jsonify({"error": "Parent concept not found"}), 404

    if not restrictions:
        return jsonify({"error": "At least one restriction is required"}), 400

    try:
        new_id = add_subconcept(
            app_state.taxonomy, parent_id, restrictions,
            app_state.raw_df, app_state.column_meta, name
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    result = serialize_taxonomy(app_state.taxonomy)
    result["new_concept_id"] = new_id
    return jsonify(result)


@app.route("/taxonomy/api/concept/<concept_id>", methods=["PUT"])
def rename_concept(concept_id):
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    if concept_id not in app_state.taxonomy["concepts"]:
        return jsonify({"error": "Concept not found"}), 404

    data = request.get_json()
    if "name" in data:
        app_state.taxonomy["concepts"][concept_id]["name"] = data["name"]

    if "restrictions" in data:
        update_restrictions(
            app_state.taxonomy, concept_id, data["restrictions"],
            app_state.raw_df, app_state.column_meta
        )

    return jsonify(serialize_taxonomy(app_state.taxonomy))


@app.route("/taxonomy/api/concept/<concept_id>", methods=["DELETE"])
def delete_concept_route(concept_id):
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    if concept_id == app_state.taxonomy["root_id"]:
        return jsonify({"error": "Cannot delete root concept"}), 400

    if concept_id not in app_state.taxonomy["concepts"]:
        return jsonify({"error": "Concept not found"}), 404

    delete_concept(app_state.taxonomy, concept_id)
    return jsonify(serialize_taxonomy(app_state.taxonomy))


@app.route("/taxonomy/api/complement", methods=["POST"])
def complement_concepts():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    data = request.get_json()
    concept_ids = data.get("concept_ids", [])

    if not concept_ids:
        return jsonify({"error": "Select at least one concept"}), 400

    try:
        new_id = create_complement(
            app_state.taxonomy, concept_ids,
            app_state.raw_df, app_state.column_meta
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = serialize_taxonomy(app_state.taxonomy)
    result["new_concept_id"] = new_id
    return jsonify(result)


@app.route("/taxonomy/api/union", methods=["POST"])
def union_concepts():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    data = request.get_json()
    concept_ids = data.get("concept_ids", [])

    if len(concept_ids) < 2:
        return jsonify({"error": "Select at least two concepts for union"}), 400

    try:
        new_id = create_union(
            app_state.taxonomy, concept_ids,
            app_state.raw_df, app_state.column_meta
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    result = serialize_taxonomy(app_state.taxonomy)
    result["new_concept_id"] = new_id
    return jsonify(result)


@app.route("/taxonomy/api/intersection", methods=["POST"])
def intersect_concepts():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    data = request.get_json()
    concept_ids = data.get("concept_ids", [])

    if len(concept_ids) < 2:
        return jsonify({"error": "Select at least two concepts for intersection"}), 400

    try:
        new_id = create_intersection(
            app_state.taxonomy, concept_ids,
            app_state.raw_df, app_state.column_meta
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    result = serialize_taxonomy(app_state.taxonomy)
    result["new_concept_id"] = new_id
    return jsonify(result)


@app.route("/taxonomy/api/find-intersections", methods=["POST"])
def find_intersections():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    data = request.get_json()
    concept_ids = data.get("concept_ids", [])

    if len(concept_ids) < 2:
        return jsonify({"error": "Select at least two concepts"}), 400

    new_ids = create_pairwise_intersections(app_state.taxonomy, concept_ids)

    if not new_ids:
        return jsonify({"error": "No non-empty intersections found"}), 400

    result = serialize_taxonomy(app_state.taxonomy)
    result["new_concept_ids"] = new_ids
    return jsonify(result)


@app.route("/taxonomy/api/merge", methods=["POST"])
def merge_concepts():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    data = request.get_json()
    concept_ids = data.get("concept_ids", [])

    if len(concept_ids) < 2:
        return jsonify({"error": "Select at least two concepts to merge"}), 400

    try:
        new_id = create_merge(
            app_state.taxonomy, concept_ids,
            app_state.raw_df, app_state.column_meta
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    result = serialize_taxonomy(app_state.taxonomy)
    result["new_concept_id"] = new_id
    return jsonify(result)


@app.route("/taxonomy/api/concept/<concept_id>/restrictions")
def concept_restrictions(concept_id):
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    if concept_id not in app_state.taxonomy["concepts"]:
        return jsonify({"error": "Concept not found"}), 404

    concept = app_state.taxonomy["concepts"][concept_id]
    available = get_available_restrictions(
        concept, app_state.taxonomy,
        app_state.raw_df, app_state.column_meta
    )

    return jsonify({"concept_id": concept_id, "available": available})


@app.route("/taxonomy/api/reset", methods=["POST"])
def reset_taxonomy():
    if not app_state.is_loaded():
        return jsonify({"error": "No data loaded"}), 400
    app_state.reset_wsom()
    app_state.taxonomy = create_taxonomy(app_state.raw_df)
    return jsonify(serialize_taxonomy(app_state.taxonomy))


@app.route("/taxonomy/api/columns")
def get_taxonomy_columns():
    if app_state.column_meta is None:
        return jsonify([])
    included = app_state.column_meta[
        (app_state.column_meta["include"] == True)
        & (app_state.column_meta["user_type"] != "title_id")
    ]
    return jsonify([
        {"name": col, "type": included.at[col, "user_type"]}
        for col in included.index
    ])


# -- WSOM discovery -----------------------------------------------------------

def _annotate_proposed(result):
    """Mark nodes as proposed if they are in the WSOM proposed set."""
    proposed_set = set(app_state.wsom_proposed_ids)
    for node in result.get("nodes", []):
        node["proposed"] = node["id"] in proposed_set
    return result


def _remove_proposed_concepts():
    """Remove all proposed WSOM concepts from the taxonomy."""
    for cid in list(app_state.wsom_proposed_ids):
        concept = app_state.taxonomy["concepts"].get(cid)
        if concept:
            for pid in concept["parent_ids"]:
                parent = app_state.taxonomy["concepts"].get(pid)
                if parent and cid in parent["child_ids"]:
                    parent["child_ids"].remove(cid)
            del app_state.taxonomy["concepts"][cid]


@app.route("/taxonomy/api/wsom/start", methods=["POST"])
def wsom_start():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    if app_state.wsom_training:
        return jsonify({"error": "Training already in progress"}), 409

    data = request.get_json()
    concept_id = data.get("concept_id")
    map_size = int(data.get("map_size", 5))
    sparcity_coeff = float(data.get("sparcity_coeff", 0.01))
    n_epochs = int(data.get("n_epochs", 100))
    ignore_columns = data.get("ignore_columns", [])

    if concept_id not in app_state.taxonomy["concepts"]:
        return jsonify({"error": "Concept not found"}), 404

    concept = app_state.taxonomy["concepts"][concept_id]
    if len(concept["row_indices"]) < 4:
        return jsonify({"error": "Concept has too few rows (need at least 4)"}), 400

    if map_size < 2 or map_size > 20:
        return jsonify({"error": "Map size must be between 2 and 20"}), 400

    if n_epochs < 1 or n_epochs > 5000:
        return jsonify({"error": "Epochs must be between 1 and 5000"}), 400

    app_state.reset_wsom()
    app_state.wsom_training = True
    app_state.wsom_parent_id = concept_id
    app_state.wsom_total_epochs = n_epochs

    thread = threading.Thread(
        target=run_wsom_training,
        args=(app_state, concept_id, map_size, sparcity_coeff, n_epochs, ignore_columns),
        daemon=True,
    )
    app_state.wsom_thread = thread
    thread.start()

    return jsonify({"status": "started", "total_epochs": n_epochs})


@app.route("/taxonomy/api/wsom/progress")
def wsom_progress():
    if app_state.wsom_error:
        error = app_state.wsom_error
        app_state.reset_wsom()
        return jsonify({"status": "error", "error": error})

    if app_state.wsom_training:
        return jsonify({
            "status": "training",
            "progress": app_state.wsom_progress,
            "total": app_state.wsom_total_epochs,
        })

    if app_state.wsom_proposed_ids:
        result = _annotate_proposed(serialize_taxonomy(app_state.taxonomy))
        return jsonify({
            "status": "complete",
            "proposed_ids": app_state.wsom_proposed_ids,
            "graph": result,
        })

    return jsonify({"status": "idle"})


@app.route("/taxonomy/api/wsom/resolve", methods=["POST"])
def wsom_resolve():
    if app_state.taxonomy is None:
        return jsonify({"error": "No taxonomy initialized"}), 400

    data = request.get_json()
    action = data.get("action")

    if action == "validate":
        app_state.reset_wsom()
        return jsonify(serialize_taxonomy(app_state.taxonomy))

    elif action == "cancel":
        _remove_proposed_concepts()
        app_state.reset_wsom()
        return jsonify(serialize_taxonomy(app_state.taxonomy))

    elif action == "retry":
        parent_id = app_state.wsom_parent_id
        _remove_proposed_concepts()
        app_state.reset_wsom()
        result = serialize_taxonomy(app_state.taxonomy)
        result["retry_parent_id"] = parent_id
        return jsonify(result)

    return jsonify({"error": "Invalid action"}), 400


if __name__ == "__main__":
    app.run(debug=True)
