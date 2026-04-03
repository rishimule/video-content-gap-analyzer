"""
01_embeddings.py — Generate Marengo embeddings for video dataset.

Loads the Safe & Unsafe Behaviours dataset from FiftyOne (or HuggingFace),
selects a balanced subset across categories, generates 512-d Marengo
embeddings via Twelve Labs API, and stores them in FiftyOne sample fields.
"""

import os
import time

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
from twelvelabs import TwelveLabs, VideoInputRequest, MediaSource

# --- Configuration ---
MAX_SAMPLES = 30  # Total samples to embed
DATASET_NAME = "Voxel51/Safe_and_Unsafe_Behaviours"


def main():
    # --- Validate API key ---
    api_key = os.environ.get("TWELVELABS_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TWELVELABS_API_KEY environment variable is not set. "
            "Set it with: export TWELVELABS_API_KEY=<your-key>"
        )

    client = TwelveLabs(api_key=api_key)

    # --- Load or reuse dataset ---
    if fo.dataset_exists(DATASET_NAME):
        print(f"Using existing dataset: {DATASET_NAME}")
        dataset = fo.load_dataset(DATASET_NAME)
    else:
        print(f"Loading {DATASET_NAME} from HuggingFace...")
        dataset = load_from_hub(DATASET_NAME)
    print(f"  Total samples: {len(dataset)}\n")

    # --- Select balanced subset across categories ---
    from collections import Counter

    all_labels = [s.ground_truth.label for s in dataset if s.ground_truth]
    categories = list(Counter(all_labels).keys())
    n_categories = len(categories)
    per_category = MAX_SAMPLES // n_categories

    print(f"Selecting {MAX_SAMPLES} samples ({per_category} per category)...")
    print(f"  Categories: {categories}\n")

    selected_ids = []
    for cat in categories:
        view = dataset.match({"ground_truth.label": cat}).limit(per_category)
        selected_ids.extend([s.id for s in view])
        print(f"  {cat}: {len(view)} samples selected")

    subset = dataset.select(selected_ids)
    print(f"\n  Subset size: {len(subset)}")

    # --- Generate embeddings (skip already-embedded samples) ---
    success_count = 0
    fail_count = 0
    skip_count = 0

    for i, sample in enumerate(subset, start=1):
        filepath = sample.filepath
        filename = os.path.basename(filepath)

        # Skip if already embedded
        try:
            if sample["embedding"] is not None:
                skip_count += 1
                print(f"[{i}/{len(subset)}] {filename} — already embedded, skipping")
                continue
        except (KeyError, AttributeError):
            pass

        print(f"[{i}/{len(subset)}] Embedding: {filename} ... ", end="", flush=True)

        start = time.time()
        try:
            with open(filepath, "rb") as f:
                asset = client.assets.create(method="direct", file=f)

            response = client.embed.v_2.create(
                input_type="video",
                model_name="marengo3.0",
                video=VideoInputRequest(
                    media_source=MediaSource(asset_id=asset.id),
                    embedding_option=["visual", "audio"],
                    embedding_scope=["asset"],
                    embedding_type=["fused_embedding"],
                ),
            )

            embedding = response.data[0].embedding
            sample["embedding"] = embedding
            sample.save()

            elapsed = time.time() - start
            success_count += 1
            print(f"{len(embedding)}-d ({elapsed:.1f}s)")

        except Exception as e:
            elapsed = time.time() - start
            fail_count += 1
            print(f"FAILED ({elapsed:.1f}s): {e}")
            continue

    # --- Save and summarize ---
    dataset.save()
    print(f"\nDone! {success_count} embedded, {fail_count} failed, {skip_count} skipped.")

    # Verify dimensionality on first successful sample
    for sample in subset:
        try:
            if sample["embedding"] is not None:
                print(f"Embedding dimensionality: {len(sample['embedding'])}")
                break
        except (KeyError, AttributeError):
            continue

    print(f"Dataset: {dataset.name} ({len(dataset)} total, {len(subset)} in subset)")


if __name__ == "__main__":
    main()
