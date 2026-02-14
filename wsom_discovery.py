import logging
import torch
import numpy as np
import pandas as pd
from ksom.ksom import WSOM, cosine_distance
from taxonomy import new_concept_id

logger = logging.getLogger(__name__)


# -- Constants ----------------------------------------------------------------

MIN_ROWS_FOR_WSOM = 4
MAX_BATCH_SIZE = 512
N_BATCHES = 10

# -- Entry point (runs in background thread) ---------------------------------

def run_wsom_training(app_state, parent_id, map_size, sparcity_coeff, n_epochs, ignore_columns=None):
    """Train a WSOM on a concept's rows and create proposed subconcepts."""
    try:
        concept = app_state.taxonomy["concepts"][parent_id]
        row_indices = concept["row_indices"]

        if len(row_indices) < MIN_ROWS_FOR_WSOM:
            app_state.wsom_error = "Concept has too few rows for WSOM discovery"
            app_state.wsom_training = False
            return

        # 1. Prepare data tensor from encoded dataframe (filtered by ignore_columns)
        filtered_encoded = _filter_encoded_columns(
            app_state.encoded_df, app_state.column_meta, ignore_columns
        )
        subset = filtered_encoded.loc[row_indices]
        if subset.shape[1] == 0:
            app_state.wsom_error = "No encoded columns available"
            app_state.wsom_training = False
            return

        data_tensor = torch.tensor(
            subset.values.astype(np.float32), dtype=torch.float32
        )
        dim = data_tensor.shape[1]

        # 2. Create WSOM (sample_init needs exactly map_size^2 samples)
        n_cells = map_size * map_size
        if data_tensor.shape[0] >= n_cells:
            indices = torch.randperm(data_tensor.shape[0])[:n_cells]
            init_samples = data_tensor[indices]
        else:
            repeats = (n_cells // data_tensor.shape[0]) + 1
            init_samples = data_tensor.repeat(repeats, 1)[:n_cells]

        wsom = WSOM(
            map_size, map_size, dim,
            sample_init=init_samples,
            dist=cosine_distance,
            sparcity_coeff=sparcity_coeff,
            alpha_init=1e-2, alpha_drate=5e-8
        )

        # 3. Train (batched)
        n_samples = data_tensor.shape[0]
        batch_size = max(MIN_ROWS_FOR_WSOM, min(MAX_BATCH_SIZE, n_samples // N_BATCHES))

        optimizer = torch.optim.Adam(wsom.parameters(), lr=1e-2)
        wsom.train()
        for epoch in range(n_epochs):
            perm = torch.randperm(n_samples)
            total_dist, total_count, total_loss = 0.0, 0, 0.0
            for start in range(0, n_samples, batch_size):
                batch = data_tensor[perm[start:start + batch_size]]
                dist, count, loss = wsom.add(batch, optimizer)
                total_dist += dist * len(batch)
                total_count += count
                total_loss += loss * len(batch)
                logger.info(
                    "    ... dist=%.4f  count=%d  loss=%.4f",
                    dist, count, loss,
                )
            total_dist /= n_samples
            total_loss /= n_samples
            app_state.wsom_progress = epoch + 1
            logger.info(
                "Epoch %d/%d  dist=%.4f  count=%d  loss=%.4f",
                epoch + 1, n_epochs, total_dist, total_count, total_loss,
            )

        # 4. Select best column by WSOM weights
        best_col, best_col_type = _select_best_column(
            wsom, filtered_encoded, app_state.column_meta, ignore_columns
        )
        logger.info("Selected column: %s (type: %s)", best_col, best_col_type)

        if best_col is None:
            app_state.wsom_error = "No suitable column found from WSOM weights"
            app_state.wsom_training = False
            return

        # 5. Compute BMUs
        wsom.eval()
        with torch.no_grad():
            bmu, _dists = wsom(data_tensor)

        # 6. Group rows by BMU coordinate
        clusters = {}
        for i in range(bmu.shape[0]):
            key = (int(bmu[i, 0]), int(bmu[i, 1]))
            clusters.setdefault(key, []).append(row_indices[i])

        # 7. Characterize each cluster using the selected column
        cluster_results = {}
        parent_subset = app_state.raw_df.loc[row_indices]
        for key, indices in clusters.items():
            cluster_set = set(indices)
            if best_col_type == "categorical":
                result = _best_categorical(best_col, parent_subset, cluster_set)
            elif best_col_type == "numeric":
                result = _best_numeric(best_col, parent_subset, cluster_set)
            elif best_col_type == "date":
                result = _best_date(best_col, parent_subset, cluster_set)
            else:
                result = None

            cluster_results[key] = result
            if result:
                restrictions, prec, matching = result
                desc = "  ".join(
                    "%s %s %s" % (r["column"], r["operator"], r["value"])
                    for r in restrictions
                )
                logger.info(
                    "Cluster %s  len=%d  best: %s  precision=%.4f  support=%d",
                    str(key), len(indices), desc, prec, len(matching),
                )
            else:
                logger.info("Cluster %s  len=%d  no restriction found", str(key), len(indices))

        # 8. Deduplicate clusters with identical restrictions
        merged = _deduplicate_restrictions(cluster_results)

        # 9. Remove clusters whose support is a strict subset of another's
        merged = _remove_subset_clusters(merged)

        # 10. Create proposed concepts
        proposed_ids = []
        for restrictions, matching_indices in merged:
            new_id = new_concept_id()
            proposed = {
                "id": new_id,
                "name": "",
                "restrictions": restrictions,
                "parent_ids": [parent_id],
                "child_ids": [],
                "row_indices": sorted(matching_indices),
                "origin": "wsom",
                "source_ids": [parent_id],
            }
            app_state.taxonomy["concepts"][new_id] = proposed
            concept["child_ids"].append(new_id)
            proposed_ids.append(new_id)

        app_state.wsom_proposed_ids = proposed_ids
        app_state.wsom_training = False

    except Exception as e:
        app_state.wsom_error = str(e)
        app_state.wsom_training = False


# -- Column filtering ---------------------------------------------------------

def _filter_encoded_columns(encoded_df, column_meta, ignore_columns):
    """Return encoded_df with columns corresponding to ignore_columns removed."""
    if not ignore_columns:
        return encoded_df

    ignore_set = set(ignore_columns)
    included = column_meta[
        (column_meta["include"] == True) & (column_meta["user_type"] != "title_id")
    ]

    encoded_cols = list(encoded_df.columns)
    cols_to_drop = []
    idx = 0

    for col_name in included.index:
        col_type = included.at[col_name, "user_type"]
        if col_type == "title_id":
            continue

        if col_type in ("numeric", "date"):
            if col_name in ignore_set:
                cols_to_drop.append(encoded_cols[idx])
            idx += 1
        elif col_type == "categorical":
            prefix = col_name + "_"
            while idx < len(encoded_cols) and encoded_cols[idx].startswith(prefix):
                if col_name in ignore_set:
                    cols_to_drop.append(encoded_cols[idx])
                idx += 1

    if cols_to_drop:
        logger.info("Ignoring columns: %s", ", ".join(ignore_columns))
        return encoded_df.drop(columns=cols_to_drop)
    return encoded_df


# -- Column selection by WSOM weights -----------------------------------------

def _select_best_column(wsom, encoded_df, column_meta, ignore_columns=None):
    """Select the original column with highest WSOM weight.

    For categorical columns (one-hot encoded), the max effective weight
    across all encoding columns is used as the column's weight.

    Returns (column_name, column_type) or (None, None).
    """
    effective_weights = torch.sigmoid(wsom.weights).detach().numpy()
    ignore_set = set(ignore_columns) if ignore_columns else set()

    included = column_meta[
        (column_meta["include"] == True) & (column_meta["user_type"] != "title_id")
    ]

    encoded_cols = list(encoded_df.columns)

    best_col = None
    best_type = None
    best_weight = -1.0
    idx = 0

    for col_name in included.index:
        col_type = included.at[col_name, "user_type"]
        if col_type == "title_id":
            continue

        if col_name in ignore_set:
            continue  # Not in encoded_df, don't advance idx

        if col_type in ("numeric", "date"):
            w = float(effective_weights[idx])
            idx += 1
        elif col_type == "categorical":
            prefix = col_name + "_"
            w = 0.0
            while idx < len(encoded_cols) and encoded_cols[idx].startswith(prefix):
                w = max(w, float(effective_weights[idx]))
                idx += 1
        else:
            continue

        logger.info("Column %s (%s): weight=%.4f", col_name, col_type, w)

        if w > best_weight:
            best_weight = w
            best_col = col_name
            best_type = col_type

    return best_col, best_type


# -- Cluster characterisation (precision-based, single column) ----------------

def _precision(matching_indices, cluster_set):
    """Compute precision of a restriction's matching rows vs cluster membership."""
    tp = len(matching_indices & cluster_set)
    if tp == 0:
        return 0.0
    fp = len(matching_indices - cluster_set)
    return tp / (tp + fp)


def _best_categorical(col_name, parent_df, cluster_set):
    """Find the categorical value whose = restriction has highest precision."""
    col = parent_df[col_name].dropna()
    if len(col) == 0:
        return None

    best_prec = 0.0
    best_restrictions = None
    best_matching = None

    for value in col.unique():
        matching = set(col[col == value].index)
        prec = _precision(matching, cluster_set)
        if prec > best_prec:
            best_prec = prec
            best_restrictions = [{"column": col_name, "operator": "=", "value": str(value)}]
            best_matching = matching

    if best_restrictions:
        return best_restrictions, best_prec, best_matching
    return None


def _best_numeric(col_name, parent_df, cluster_set):
    """Find the best numeric interval (or single bound) with highest precision.

    Explores all pairs of 5%-quantile thresholds as [lower, upper) intervals,
    plus open-ended single bounds (>= or <).
    """
    col = pd.to_numeric(parent_df[col_name], errors="coerce").dropna()
    if len(col) < 2:
        return None

    thresholds = sorted(set(
        round(float(col.quantile(pct / 100)), 4)
        for pct in range(5, 100, 5)
    ))

    best_prec = 0.0
    best_restrictions = None
    best_matching = None

    # Try single bounds
    for t in thresholds:
        matching = set(col[col >= t].index)
        prec = _precision(matching, cluster_set)
        if prec > best_prec:
            best_prec = prec
            best_restrictions = [{"column": col_name, "operator": ">=", "value": t}]
            best_matching = matching

        matching = set(col[col < t].index)
        prec = _precision(matching, cluster_set)
        if prec > best_prec:
            best_prec = prec
            best_restrictions = [{"column": col_name, "operator": "<", "value": t}]
            best_matching = matching

    # Try intervals [lower, upper)
    for i in range(len(thresholds)):
        for j in range(i + 1, len(thresholds)):
            lower, upper = thresholds[i], thresholds[j]
            matching = set(col[(col >= lower) & (col < upper)].index)
            prec = _precision(matching, cluster_set)
            if prec > best_prec:
                best_prec = prec
                best_restrictions = [
                    {"column": col_name, "operator": ">=", "value": lower},
                    {"column": col_name, "operator": "<", "value": upper},
                ]
                best_matching = matching

    if best_restrictions:
        return best_restrictions, best_prec, best_matching
    return None


def _best_date(col_name, parent_df, cluster_set):
    """Find the decile date split (>= or <) with highest precision."""
    col = pd.to_datetime(parent_df[col_name], errors="coerce", format="mixed").dropna()
    if len(col) < 2:
        return None

    best_prec = 0.0
    best_restrictions = None
    best_matching = None

    for pct in range(10, 100, 10):
        threshold = col.quantile(pct / 100)
        date_str = str(threshold.date())

        matching = set(col[col >= threshold].index)
        prec = _precision(matching, cluster_set)
        if prec > best_prec:
            best_prec = prec
            best_restrictions = [{"column": col_name, "operator": ">=", "value": date_str}]
            best_matching = matching

        matching = set(col[col < threshold].index)
        prec = _precision(matching, cluster_set)
        if prec > best_prec:
            best_prec = prec
            best_restrictions = [{"column": col_name, "operator": "<", "value": date_str}]
            best_matching = matching

    if best_restrictions:
        return best_restrictions, best_prec, best_matching
    return None


# -- Restriction deduplication and subset removal -----------------------------

def _deduplicate_restrictions(cluster_results):
    """Deduplicate clusters that ended up with the same restrictions.

    Returns list of (restrictions_list, matching_indices_set) tuples.
    """
    seen = {}
    for _key, result in cluster_results.items():
        if result is None:
            continue
        restrictions, _prec, matching_indices = result
        rkey = tuple(
            (r["column"], r["operator"], str(r["value"])) for r in restrictions
        )
        if rkey not in seen:
            seen[rkey] = (restrictions, matching_indices)

    return list(seen.values())


def _remove_subset_clusters(merged):
    """Remove clusters whose support is a strict subset of another cluster's support."""
    if len(merged) <= 1:
        return merged

    items = [(r, s if isinstance(s, set) else set(s)) for r, s in merged]

    to_remove = set()
    for i in range(len(items)):
        if i in to_remove:
            continue
        for j in range(len(items)):
            if i == j or j in to_remove:
                continue
            if items[i][1] < items[j][1]:
                desc = "  ".join(
                    "%s %s %s" % (r["column"], r["operator"], r["value"])
                    for r in items[i][0]
                )
                logger.info(
                    "Removing subset cluster %s (support %d ⊂ %d)",
                    desc, len(items[i][1]), len(items[j][1]),
                )
                to_remove.add(i)
                break

    return [(r, sorted(s)) for i, (r, s) in enumerate(items) if i not in to_remove]
