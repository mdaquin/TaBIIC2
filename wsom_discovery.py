import logging
import torch
import numpy as np
import pandas as pd
from ksom.ksom import WSOM, cosine_distance
from taxonomy import new_concept_id

logger = logging.getLogger(__name__)


# -- Thresholds ---------------------------------------------------------------

F1_THRESHOLD = 0.5            # minimum F1-score for a restriction to be kept
MIN_ROWS_FOR_WSOM = 4
MAX_BATCH_SIZE = 512
N_BATCHES = 10
CLUSTER_MERGE_LIMIT = 0.10    # merge clusters smaller than this fraction of total rows

# -- Entry point (runs in background thread) ---------------------------------

def run_wsom_training(app_state, parent_id, map_size, sparcity_coeff, n_epochs):
    """Train a WSOM on a concept's rows and create proposed subconcepts."""
    try:
        concept = app_state.taxonomy["concepts"][parent_id]
        row_indices = concept["row_indices"]

        if len(row_indices) < MIN_ROWS_FOR_WSOM:
            app_state.wsom_error = "Concept has too few rows for WSOM discovery"
            app_state.wsom_training = False
            return

        # 1. Prepare data tensor from encoded dataframe
        subset = app_state.encoded_df.loc[row_indices]
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
            # Repeat data to fill the map
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

        # 4. Compute BMUs
        wsom.eval()
        with torch.no_grad():
            bmu, _dists = wsom(data_tensor)

        # 5. Group rows by BMU coordinate
        clusters = {}
        for i in range(bmu.shape[0]):
            key = (int(bmu[i, 0]), int(bmu[i, 1]))
            clusters.setdefault(key, []).append(row_indices[i])

        # 5b. Merge small clusters into nearest neighbour
        clusters = _merge_small_clusters(clusters, wsom, n_samples)

        # 6. Characterise each cluster
        cluster_restrictions = {}
        for key, indices in clusters.items():
            restrictions = _characterize_cluster(
                indices, row_indices,
                app_state.raw_df, app_state.column_meta,
            )
            cluster_restrictions[key] = restrictions
            logger.info(
                "Cluster %s  len=%d restrictions=%s",
                str(key), len(indices), str(restrictions)
            )

        # 7. Merge clusters with identical restriction sets
        merged = _merge_clusters(clusters, cluster_restrictions)

        # 8. Create proposed concepts
        proposed_ids = []
        for restrictions, indices in merged:
            if not restrictions:
                continue

            new_id = new_concept_id()
            proposed = {
                "id": new_id,
                "name": "",
                "restrictions": restrictions,
                "parent_ids": [parent_id],
                "child_ids": [],
                "row_indices": sorted(indices),
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


# -- Small cluster merging ----------------------------------------------------

def _merge_small_clusters(clusters, wsom, n_samples):
    """Merge clusters smaller than CLUSTER_MERGE_LIMIT of total rows into their
    nearest neighbour, using cosine distance between WSOM unit vectors."""
    min_size = max(1, int(CLUSTER_MERGE_LIMIT * n_samples))

    while len(clusters) > 1:
        small_keys = [k for k, v in clusters.items() if len(v) < min_size]
        if not small_keys:
            break

        smallest_key = min(small_keys, key=lambda k: len(clusters[k]))
        row, col = smallest_key
        small_vec = wsom.somap[row * wsom.ys + col]

        best_key = None
        best_dist = float("inf")
        for key in clusters:
            if key == smallest_key:
                continue
            r, c = key
            vec = wsom.somap[r * wsom.ys + c]
            dot = torch.dot(small_vec, vec)
            norms = torch.norm(small_vec) * torch.norm(vec)
            dist = 1.0 - (dot / norms).item() if norms > 0 else 2.0
            if dist < best_dist:
                best_dist = dist
                best_key = key

        logger.info(
            "Merging small cluster %s (len=%d) into %s (dist=%.4f)",
            smallest_key, len(clusters[smallest_key]), best_key, best_dist,
        )
        clusters[best_key].extend(clusters[smallest_key])
        del clusters[smallest_key]

    return clusters


# -- Cluster characterisation (F1-based) --------------------------------------

def _f1_score(matching_indices, cluster_set):
    """Compute F1-score of a restriction's matching rows vs cluster membership."""
    tp = len(matching_indices & cluster_set)
    if tp == 0:
        return 0.0
    fp = len(matching_indices - cluster_set)
    fn = len(cluster_set - matching_indices)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def _characterize_cluster(cluster_indices, parent_indices, raw_df, column_meta):
    """Find restrictions that best discriminate the cluster from the parent, by F1-score."""
    restrictions = []
    cluster_set = set(cluster_indices)
    included = column_meta[
        (column_meta["include"] == True) & (column_meta["user_type"] != "title_id")
    ]

    parent_subset = raw_df.loc[parent_indices]

    for col_name in included.index:
        col_type = included.at[col_name, "user_type"]

        if col_type == "categorical":
            result = _best_categorical(col_name, parent_subset, cluster_set)
        elif col_type == "numeric":
            result = _best_numeric(col_name, parent_subset, cluster_set)
        elif col_type == "date":
            result = _best_date(col_name, parent_subset, cluster_set)
        else:
            continue

        if result:
            r, f1 = result
            logger.info(
                "  restriction %s %s %s  f1=%.4f",
                r["column"], r["operator"], r["value"], f1,
            )
            restrictions.append(r)

    return restrictions


def _best_categorical(col_name, parent_df, cluster_set):
    """Find the categorical value whose = restriction has highest F1 vs the cluster."""
    col = parent_df[col_name].dropna()
    if len(col) == 0:
        return None

    best_f1 = 0.0
    best_restriction = None

    for value in col.unique():
        matching = set(col[col == value].index)
        f1 = _f1_score(matching, cluster_set)
        if f1 > best_f1:
            best_f1 = f1
            best_restriction = {"column": col_name, "operator": "=", "value": str(value)}

    if best_f1 >= F1_THRESHOLD:
        return best_restriction, best_f1
    return None


def _best_numeric(col_name, parent_df, cluster_set):
    """Find the decile split (>= or <) with highest F1 vs the cluster."""
    col = pd.to_numeric(parent_df[col_name], errors="coerce").dropna()
    if len(col) < 2:
        return None

    best_f1 = 0.0
    best_restriction = None

    for pct in range(10, 100, 10):
        threshold = col.quantile(pct / 100)

        matching = set(col[col >= threshold].index)
        f1 = _f1_score(matching, cluster_set)
        if f1 > best_f1:
            best_f1 = f1
            best_restriction = {"column": col_name, "operator": ">=", "value": round(float(threshold), 4)}

        matching = set(col[col < threshold].index)
        f1 = _f1_score(matching, cluster_set)
        if f1 > best_f1:
            best_f1 = f1
            best_restriction = {"column": col_name, "operator": "<", "value": round(float(threshold), 4)}

    if best_f1 >= F1_THRESHOLD:
        return best_restriction, best_f1
    return None


def _best_date(col_name, parent_df, cluster_set):
    """Find the decile date split (>= or <) with highest F1 vs the cluster."""
    col = pd.to_datetime(parent_df[col_name], errors="coerce", format="mixed").dropna()
    if len(col) < 2:
        return None

    best_f1 = 0.0
    best_restriction = None

    for pct in range(10, 100, 10):
        threshold = col.quantile(pct / 100)
        date_str = str(threshold.date())

        matching = set(col[col >= threshold].index)
        f1 = _f1_score(matching, cluster_set)
        if f1 > best_f1:
            best_f1 = f1
            best_restriction = {"column": col_name, "operator": ">=", "value": date_str}

        matching = set(col[col < threshold].index)
        f1 = _f1_score(matching, cluster_set)
        if f1 > best_f1:
            best_f1 = f1
            best_restriction = {"column": col_name, "operator": "<", "value": date_str}

    if best_f1 >= F1_THRESHOLD:
        return best_restriction, best_f1
    return None


# -- Cluster merging ----------------------------------------------------------

def _merge_clusters(clusters, cluster_restrictions):
    """Merge BMU clusters that have identical restriction sets.

    Returns list of (restrictions, merged_row_indices) tuples.
    """
    def _restriction_key(restrictions):
        return tuple(sorted(
            (r["column"], r["operator"], str(r["value"])) for r in restrictions
        ))

    groups = {}
    for bmu_key, indices in clusters.items():
        restrictions = cluster_restrictions[bmu_key]
        rkey = _restriction_key(restrictions)
        if rkey in groups:
            groups[rkey][1].extend(indices)
        else:
            groups[rkey] = (restrictions, list(indices))

    return list(groups.values())
