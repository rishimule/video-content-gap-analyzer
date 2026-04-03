"""
03_cluster_descriptions.py — Generate Pegasus descriptions for cluster representatives.

Loads the clustered dataset from Phase 2, selects the 2 samples nearest
each cluster centroid, generates one-sentence descriptions via Twelve Labs
Pegasus 1.2, and stores cluster_label on every sample.
"""

import os
import time
from datetime import datetime
from collections import defaultdict

import fiftyone as fo
from twelvelabs import TwelveLabs
from twelvelabs.types import VideoContext_AssetId
from twelvelabs.indexes import IndexesCreateRequestModelsItem

# --- Configuration ---
DATASET_NAME = "Voxel51/Safe_and_Unsafe_Behaviours"
REPS_PER_CLUSTER = 2
INDEX_NAME_PREFIX = "gap-analyzer"
PEGASUS_PROMPT = (
    "Describe the main activity, setting, and key objects visible "
    "in this video in one concise sentence."
)
POLL_INTERVAL = 10.0  # seconds between indexing status checks
RATE_LIMIT_WAIT = 30  # seconds to wait on rate limit


def find_cluster_representatives(dataset):
    """For each cluster_id, find the REPS_PER_CLUSTER samples closest to centroid."""
    clusters = defaultdict(list)

    for sample in dataset:
        try:
            cid = sample["cluster_id"]
            dist = sample["centroid_distance"]
        except (KeyError, AttributeError):
            continue
        if cid is not None and dist is not None:
            clusters[cid].append((dist, sample))

    representatives = {}
    for cid in sorted(clusters.keys()):
        sorted_samples = sorted(clusters[cid], key=lambda x: x[0])
        representatives[cid] = [s for _, s in sorted_samples[:REPS_PER_CLUSTER]]

    return representatives


def upload_asset(client, filepath):
    """Upload a video file as a Twelve Labs asset. Returns asset or None."""
    try:
        with open(filepath, "rb") as f:
            asset = client.assets.create(method="direct", file=f)
        return asset
    except Exception as e:
        print(f"UPLOAD FAILED: {e}")
        return None


def analyze_via_asset(client, asset_id, prompt):
    """Approach A: analyze directly using asset_id (no indexing)."""
    response = client.analyze(
        video=VideoContext_AssetId(asset_id=asset_id),
        prompt=prompt,
        temperature=0.2,
    )
    return response.data


def create_pegasus_index(client):
    """Create a Twelve Labs index with pegasus1.2 model support."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    index_name = f"{INDEX_NAME_PREFIX}-{timestamp}"

    print(f"  Creating Twelve Labs index: {index_name} ...")
    index = client.indexes.create(
        index_name=index_name,
        models=[
            IndexesCreateRequestModelsItem(
                model_name="pegasus1.2",
                model_options=["visual", "audio"],
            ),
        ],
    )
    print(f"  Index created: {index.id}")
    return index.id


def index_and_analyze(client, index_id, asset_id, prompt):
    """Approach B: index the asset in an index, wait for ready, then analyze."""
    # Index the asset
    resp = client.indexes.indexed_assets.create(
        index_id=index_id,
        asset_id=asset_id,
    )
    indexed_asset_id = resp.id
    print(f"indexing (id={indexed_asset_id}) ... ", end="", flush=True)

    # Poll until ready
    while True:
        detail = client.indexes.indexed_assets.retrieve(index_id, indexed_asset_id)
        if detail.status == "ready":
            break
        if detail.status == "failed":
            raise RuntimeError(f"Indexing failed for asset {asset_id}")
        time.sleep(POLL_INTERVAL)

    # Analyze
    response = client.analyze(
        video_id=indexed_asset_id,
        prompt=prompt,
        temperature=0.2,
    )
    return response.data


def generate_description(client, filepath, use_indexing, index_id):
    """
    Generate a one-sentence Pegasus description for a video.

    Returns (description_or_None, use_indexing_updated).
    If Approach A fails with a non-rate-limit error, switches to Approach B.
    """
    filename = os.path.basename(filepath)
    print(f"  {filename}: ", end="", flush=True)
    start = time.time()

    # Upload asset
    asset = upload_asset(client, filepath)
    if asset is None:
        return None, use_indexing

    # Try Approach A (direct asset analysis) or B (index-based)
    for attempt in range(2):  # at most 1 retry on rate limit
        try:
            if use_indexing:
                print("analyzing (indexed) ... ", end="", flush=True)
                text = index_and_analyze(client, index_id, asset.id, PEGASUS_PROMPT)
            else:
                print("analyzing (direct) ... ", end="", flush=True)
                text = analyze_via_asset(client, asset.id, PEGASUS_PROMPT)

            elapsed = time.time() - start
            if text:
                text = text.strip()
                print(f"OK ({elapsed:.1f}s)")
                return text, use_indexing
            else:
                print(f"empty response ({elapsed:.1f}s)")
                return None, use_indexing

        except Exception as e:
            elapsed = time.time() - start
            error_str = str(e).lower()

            # Rate limit: wait and retry once
            if "429" in str(e) or "rate" in error_str or "too many" in error_str:
                if attempt == 0:
                    print(f"rate limited, waiting {RATE_LIMIT_WAIT}s ... ", end="", flush=True)
                    time.sleep(RATE_LIMIT_WAIT)
                    continue
                else:
                    print(f"rate limited again, skipping ({elapsed:.1f}s)")
                    return None, use_indexing

            # Non-rate-limit error on Approach A: switch to Approach B
            if not use_indexing:
                print(f"direct analysis failed ({elapsed:.1f}s): {e}")
                print("  Switching to index-based approach (Approach B)...")
                return None, True  # signal caller to switch

            # Approach B failure
            print(f"FAILED ({elapsed:.1f}s): {e}")
            return None, use_indexing

    return None, use_indexing


def main():
    # --- Validate API key ---
    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TWELVELABS_API_KEY environment variable is not set. "
            "Set it with: export TWELVELABS_API_KEY=<your-key>"
        )

    client = TwelveLabs(api_key=api_key)

    # --- Load dataset ---
    print(f"Loading dataset: {DATASET_NAME}")
    try:
        dataset = fo.load_dataset(DATASET_NAME)
    except ValueError:
        raise RuntimeError(
            f"Dataset '{DATASET_NAME}' not found in FiftyOne. "
            "Run 01_embeddings.py and 02_clustering.py first."
        )
    print(f"  {len(dataset)} samples\n")

    # --- Validate clustering fields exist ---
    sample = dataset.first()
    try:
        _ = sample["cluster_id"]
        _ = sample["centroid_distance"]
    except (KeyError, AttributeError):
        raise RuntimeError(
            "No cluster_id/centroid_distance fields found. "
            "Run 02_clustering.py first."
        )

    # --- Find cluster representatives ---
    print("Finding cluster representatives...")
    representatives = find_cluster_representatives(dataset)

    for cid, reps in representatives.items():
        dists = [f"{r['centroid_distance']:.6f}" for r in reps]
        print(f"  Cluster {cid}: {len(reps)} reps (distances: {', '.join(dists)})")
    print()

    # --- Generate descriptions ---
    print("Generating Pegasus descriptions...")
    use_indexing = False
    index_id = None
    cluster_labels = {}

    for cid in sorted(representatives.keys()):
        samples = representatives[cid]
        descriptions = []

        print(f"Cluster {cid}:")

        # If we switched to Approach B and haven't created an index yet, create one
        if use_indexing and index_id is None:
            index_id = create_pegasus_index(client)

        for sample in samples:
            desc, use_indexing_new = generate_description(
                client, sample.filepath, use_indexing, index_id
            )

            # If approach switched, create index and retry this sample
            if use_indexing_new and not use_indexing:
                use_indexing = True
                if index_id is None:
                    index_id = create_pegasus_index(client)
                # Retry with indexing
                desc, _ = generate_description(
                    client, sample.filepath, use_indexing, index_id
                )

            if desc:
                descriptions.append(desc)

        # Combine descriptions into label
        if len(descriptions) >= 2:
            cluster_labels[cid] = f"{descriptions[0]}; {descriptions[1]}"
        elif len(descriptions) == 1:
            cluster_labels[cid] = descriptions[0]
        else:
            cluster_labels[cid] = f"Cluster {cid}"
            print("  WARNING: No descriptions generated, using fallback label")

        print(f"  -> Label: {cluster_labels[cid]}\n")

    # --- Apply labels to all samples ---
    print("Writing cluster_label to all samples...")
    updated = 0

    for sample in dataset:
        try:
            cid = sample["cluster_id"]
        except (KeyError, AttributeError):
            continue

        if cid is not None and cid in cluster_labels:
            sample["cluster_label"] = cluster_labels[cid]
            sample.save()
            updated += 1

    dataset.save()
    print(f"  {updated} samples updated\n")

    # --- Summary ---
    print("=" * 60)
    print("CLUSTER DESCRIPTION SUMMARY")
    print("=" * 60)

    for cid in sorted(cluster_labels.keys()):
        count = sum(
            1 for s in dataset
            if s.get_field("cluster_id") is not None and s["cluster_id"] == cid
        )
        print(f"  Cluster {cid} ({count} samples): \"{cluster_labels[cid]}\"")

    print(f"\nDataset: {dataset.name}")

    # --- Sanity check ---
    missing = 0
    for sample in dataset:
        try:
            cid = sample["cluster_id"]
        except (KeyError, AttributeError):
            continue
        if cid is not None:
            try:
                label = sample["cluster_label"]
                if label is None:
                    missing += 1
            except (KeyError, AttributeError):
                missing += 1

    if missing == 0:
        print("Sanity check: all clustered samples have cluster_label OK")
    else:
        print(f"WARNING: {missing} clustered samples missing cluster_label")


if __name__ == "__main__":
    main()
