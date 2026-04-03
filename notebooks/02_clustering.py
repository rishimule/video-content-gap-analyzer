"""
02_clustering.py — Cluster Marengo embeddings and reduce to 2D with UMAP.

Loads the dataset with pre-computed 512-d embeddings from Phase 1,
runs KMeans clustering, computes cosine centroid distances, flags outliers,
and projects to 2D via UMAP for visualization.
"""

import os

import numpy as np
import fiftyone as fo
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import normalize
import umap


def main():
    # --- Load dataset ---
    dataset_name = "Voxel51/Safe_and_Unsafe_Behaviours"
    print(f"Loading dataset: {dataset_name}")
    try:
        dataset = fo.load_dataset(dataset_name)
    except ValueError:
        raise RuntimeError(
            f"Dataset '{dataset_name}' not found in FiftyOne. "
            "Run 01_embeddings.py first to create and embed it."
        )
    print(f"  {len(dataset)} samples\n")

    # --- Filter to embedded samples only ---
    print("Extracting embedded samples...")
    samples_list = []
    embeddings_list = []
    gt_labels = []

    for sample in dataset:
        try:
            emb = sample["embedding"]
        except (KeyError, AttributeError):
            continue
        if emb is not None:
            samples_list.append(sample)
            embeddings_list.append(emb)
            label = sample.ground_truth.label if sample.ground_truth else "unknown"
            gt_labels.append(label)

    if len(embeddings_list) == 0:
        raise RuntimeError("No samples have embeddings. Run 01_embeddings.py first.")

    embeddings = np.array(embeddings_list)
    n_samples = len(embeddings)
    print(f"  Embedding matrix: {embeddings.shape}")
    print(f"  Mean norm: {np.linalg.norm(embeddings, axis=1).mean():.4f}\n")

    # --- L2 normalize ---
    print("Normalizing embeddings to unit sphere...")
    embeddings_norm = normalize(embeddings, norm="l2")
    print(f"  Mean norm after: {np.linalg.norm(embeddings_norm, axis=1).mean():.4f}\n")

    # --- Choose n_clusters ---
    n_gt_categories = len(set(gt_labels))
    if n_samples < 4:
        n_clusters = 1
        print(f"Only {n_samples} samples — assigning all to cluster 0\n")
    else:
        # Use ground truth category count as hint, cap at reasonable range
        max_k = max(2, n_samples // 3)
        n_clusters = min(n_gt_categories, max_k)
        n_clusters = max(2, n_clusters)  # at least 2
        print(f"Choosing n_clusters = {n_clusters} for {n_samples} samples "
              f"({n_gt_categories} ground truth categories, "
              f"~{n_samples / n_clusters:.1f} samples/cluster)\n")

    # --- KMeans clustering ---
    print(f"Running KMeans (k={n_clusters})...")
    if n_clusters == 1:
        labels = np.zeros(n_samples, dtype=int)
        centroids_norm = embeddings_norm.mean(axis=0, keepdims=True)
        centroids_norm = normalize(centroids_norm, norm="l2")
        sil_score = None
    else:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_norm)
        centroids_norm = normalize(kmeans.cluster_centers_, norm="l2")

        sil_score = silhouette_score(embeddings_norm, labels, metric="cosine")
        print(f"  Silhouette score (cosine): {sil_score:.4f}")
        if sil_score < 0.1:
            print("  WARNING: Low silhouette score — clusters may not be well-separated")
            print("  (Expected for highly similar embeddings)")

    # Check cluster distribution
    unique, counts = np.unique(labels, return_counts=True)
    for cid, cnt in zip(unique, counts):
        print(f"  Cluster {cid}: {cnt} samples")

    if len(unique) == 1 and n_clusters > 1:
        print("  WARNING: All samples ended up in one cluster")
    print()

    # --- Centroid distances ---
    print("Computing cosine distances to centroids...")
    distances = np.array([
        cosine_distances(
            embeddings_norm[i : i + 1],
            centroids_norm[labels[i] : labels[i] + 1],
        )[0, 0]
        for i in range(n_samples)
    ])

    mean_dist = distances.mean()
    std_dist = distances.std()
    print(f"  Mean distance: {mean_dist:.6f}")
    print(f"  Std distance:  {std_dist:.6f}")

    # --- Outlier detection ---
    if std_dist > 0:
        threshold = mean_dist + 2 * std_dist
        is_outlier = distances > threshold
    else:
        threshold = mean_dist
        is_outlier = np.zeros(n_samples, dtype=bool)
        print("  NOTE: Zero std — no outliers flagged")

    outlier_count = is_outlier.sum()
    print(f"  Outlier threshold (mean + 2*std): {threshold:.6f}")
    print(f"  Outliers flagged: {outlier_count}\n")

    # --- UMAP reduction to 2D ---
    print("Running UMAP (512-d -> 2D)...")
    n_neighbors = min(5, n_samples - 1)
    reducer = umap.UMAP(
        n_components=2,
        metric="cosine",
        n_neighbors=n_neighbors,
        min_dist=0.1,
        random_state=42,
    )
    coords_2d = reducer.fit_transform(embeddings_norm)
    print(f"  x range: [{coords_2d[:, 0].min():.4f}, {coords_2d[:, 0].max():.4f}]")
    print(f"  y range: [{coords_2d[:, 1].min():.4f}, {coords_2d[:, 1].max():.4f}]\n")

    # --- Store fields on samples ---
    print("Writing fields to FiftyOne samples...")
    for i, sample in enumerate(samples_list):
        sample["cluster_id"] = int(labels[i])
        sample["centroid_distance"] = float(distances[i])
        sample["is_outlier"] = bool(is_outlier[i])
        sample["umap_x"] = float(coords_2d[i, 0])
        sample["umap_y"] = float(coords_2d[i, 1])
        sample.save()

    dataset.save()
    print(f"  All {n_samples} samples updated\n")

    # --- Summary ---
    print("=" * 50)
    print("CLUSTERING SUMMARY")
    print("=" * 50)
    print(f"Samples clustered:  {n_samples}")
    print(f"Number of clusters: {n_clusters}")
    if sil_score is not None:
        print(f"Silhouette score:   {sil_score:.4f} (cosine)")
    print()
    for cid, cnt in zip(unique, counts):
        print(f"  Cluster {cid}: {cnt} samples")
    print()
    print(f"Centroid distances — mean: {mean_dist:.6f}, std: {std_dist:.6f}")
    print(f"Outlier threshold:  {threshold:.6f}")
    print(f"Outliers flagged:   {outlier_count}")
    print()
    print(f"UMAP 2D — x: [{coords_2d[:, 0].min():.2f}, {coords_2d[:, 0].max():.2f}], "
          f"y: [{coords_2d[:, 1].min():.2f}, {coords_2d[:, 1].max():.2f}]")
    print()

    # --- Validate against ground truth ---
    if n_gt_categories > 1:
        ari = adjusted_rand_score(gt_labels, labels)
        print(f"GROUND TRUTH VALIDATION")
        print(f"  Categories: {set(gt_labels)}")
        print(f"  Adjusted Rand Index: {ari:.4f}")
        if ari > 0.5:
            print("  -> Good: clusters align well with ground truth categories")
        elif ari > 0.1:
            print("  -> Moderate: partial alignment with ground truth")
        else:
            print("  -> Low: clusters do not match ground truth categories")

        # Cross-tabulation
        from collections import Counter
        print("\n  Cluster vs Ground Truth:")
        for cid in range(n_clusters):
            cluster_labels = [gt_labels[i] for i in range(n_samples) if labels[i] == cid]
            dist = Counter(cluster_labels)
            parts = ", ".join(f"{lbl}: {cnt}" for lbl, cnt in dist.most_common())
            print(f"    Cluster {cid}: {parts}")
    print()

    # --- Sanity checks ---
    all_ok = True
    for sample in samples_list:
        for field in ["cluster_id", "centroid_distance", "is_outlier", "umap_x", "umap_y"]:
            if sample[field] is None:
                print(f"  FAIL: {os.path.basename(sample.filepath)} missing {field}")
                all_ok = False

    if all_ok:
        print("Sanity check: all fields written OK")
    print(f"Dataset: {dataset.name}")


if __name__ == "__main__":
    main()
