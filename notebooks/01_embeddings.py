"""
01_embeddings.py — Generate Marengo embeddings for video dataset.

Loads the Safe & Unsafe Behaviours dataset from HuggingFace (10 samples),
generates 512-d Marengo embeddings via Twelve Labs API, and stores them
in FiftyOne sample fields.
"""

import os
import time

import fiftyone as fo
from fiftyone.utils.huggingface import load_from_hub
from twelvelabs import TwelveLabs, VideoInputRequest, MediaSource


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
    print("Loading Safe & Unsafe Behaviours dataset (max_samples=10)...")
    dataset = load_from_hub("Voxel51/Safe_and_Unsafe_Behaviours", max_samples=10)
    print(f"Loaded {len(dataset)} samples\n")

    # --- Generate embeddings ---
    success_count = 0
    fail_count = 0

    for i, sample in enumerate(dataset, start=1):
        filepath = sample.filepath
        filename = os.path.basename(filepath)
        print(f"[{i}/{len(dataset)}] Embedding: {filename}")

        start = time.time()
        try:
            # Upload video file to Twelve Labs
            with open(filepath, "rb") as f:
                asset = client.assets.create(method="direct", file=f)

            # Generate fused asset-level embedding
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

            # Extract embedding vector
            embedding = response.data[0].embedding
            sample["embedding"] = embedding
            sample.save()

            elapsed = time.time() - start
            success_count += 1
            print(f"  -> {len(embedding)}-d vector in {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start
            fail_count += 1
            print(f"  -> FAILED after {elapsed:.1f}s: {e}")
            continue

    # --- Save and summarize ---
    dataset.save()
    print(f"\nDone! {success_count} embedded, {fail_count} failed.")

    # Verify dimensionality on first successful sample
    for sample in dataset:
        if sample["embedding"] is not None:
            print(f"Embedding dimensionality: {len(sample['embedding'])}")
            break

    print(f"Dataset: {dataset.name} ({len(dataset)} samples)")


if __name__ == "__main__":
    main()
