import os
import json
from typing import List

from flask import Flask, request, jsonify
from deepface.commons.distance import findEuclideanDistance, findCosineDistance

from utils import (
    download_image,
    get_face_embedding,
    get_face_embedding_augmented,
    l2_normalize,
    MODEL_NAME,
    DETECTOR_BACKEND,
)


# Configuration (defaults). Can be overridden per-request via JSON fields.
DISTANCE_METRIC = "euclidean_l2"  # for Facenet; use "cosine" for ArcFace
CUSTOM_MATCH_THRESHOLD = 0.60      # typical for Facenet euclidean_l2; ArcFace cosine ~0.30


app = Flask(__name__)


@app.route("/compare", methods=["POST"])
def compare_images():
    # Parse and validate request
    try:
        data = request.get_json(force=True, silent=False)
    except Exception:
        return jsonify({"error": "Invalid JSON payload."}), 400

    selfie_url = data.get("selfie") if isinstance(data, dict) else None
    images = data.get("images") if isinstance(data, dict) else None
    # Optional overrides
    model_override = data.get("model")
    metric_override = data.get("metric")
    threshold_override = data.get("threshold")
    augment_selfie = bool(data.get("augment_selfie", True))
    augment_targets = bool(data.get("augment_targets", False))
    top_k = data.get("top_k", 8)
    return_distances = bool(data.get("return_distances", True))

    if not isinstance(selfie_url, str) or not selfie_url.strip():
        return jsonify({"error": "Field 'selfie' (string URL) is required."}), 400

    if not isinstance(images, list) or not images:
        return jsonify({"error": "Field 'images' (non-empty list of URLs) is required."}), 400

    # Prepare temp file for selfie
    selfie_path = "temp_selfie.jpg"

    # Download selfie
    if not download_image(selfie_url, selfie_path):
        # Cleanup just in case
        if os.path.exists(selfie_path):
            try:
                os.remove(selfie_path)
            except Exception:
                pass
        return jsonify({"error": "Could not download or validate the selfie image."}), 400

    # Resolve effective model/metric/threshold
    # Defaults: ArcFace + cosine + threshold 0.60 when not provided
    effective_model = model_override if isinstance(model_override, str) and model_override else "ArcFace"
    # If metric not provided, choose sensible default per model
    if isinstance(metric_override, str) and metric_override:
        effective_metric = metric_override
    else:
        effective_metric = "cosine" if effective_model.lower() in ("arcface",) else DISTANCE_METRIC
    if isinstance(threshold_override, (int, float)):
        effective_threshold = float(threshold_override)
    else:
        effective_threshold = 0.60 if effective_metric == "cosine" else CUSTOM_MATCH_THRESHOLD

    # Step 1: Compute selfie embedding once (optionally augmented)
    selfie_embedding = get_face_embedding_augmented(
        selfie_path,
        model_name=effective_model,
        detector_backend=DETECTOR_BACKEND,
        augment=augment_selfie,
    )
    if selfie_embedding is None:
        try:
            os.remove(selfie_path)
        except Exception:
            pass
        return jsonify({"error": "Could not detect a face in the provided selfie."}), 400

    matches: List[str] = []
    failed_images: List[str] = []
    scored_results = []

    # Step 2: Loop through targets
    for idx, url in enumerate(images):
        if not isinstance(url, str) or not url.strip():
            failed_images.append(url)
            continue

        target_path = f"temp_target_{idx}.jpg"

        try:
            if not download_image(url, target_path):
                failed_images.append(url)
                continue

            if augment_targets:
                target_embedding = get_face_embedding_augmented(
                    target_path,
                    model_name=effective_model,
                    detector_backend=DETECTOR_BACKEND,
                    augment=True,
                )
            else:
                target_embedding = get_face_embedding(
                    target_path,
                    model_name=effective_model,
                    detector_backend=DETECTOR_BACKEND,
                )
            if target_embedding is None:
                failed_images.append(url)
                continue

            # Step 3: Compute distance manually
            if effective_metric == "cosine":
                # Cosine works best on L2-normalized embeddings
                distance = findCosineDistance(l2_normalize(selfie_embedding), l2_normalize(target_embedding))
            else:
                # Euclidean_L2 expects normalized vectors
                distance = findEuclideanDistance(l2_normalize(selfie_embedding), l2_normalize(target_embedding))
            print(f"Image {idx} ({url}) - Distance: {distance:.4f} ({effective_metric})")

            if distance <= effective_threshold:
                matches.append(url)

            if return_distances:
                scored_results.append({"url": url, "distance": float(distance)})
        finally:
            # Clean up target file
            if os.path.exists(target_path):
                try:
                    os.remove(target_path)
                except Exception:
                    pass

    # Clean up selfie
    if os.path.exists(selfie_path):
        try:
            os.remove(selfie_path)
        except Exception:
            pass

    response = {
        "message": "Comparison complete",
        "model": effective_model,
        "detector": DETECTOR_BACKEND,
        "distance_metric": effective_metric,
        "threshold": effective_threshold,
        "matches": matches,
        "total_processed": len(images),
        "matches_found": len(matches),
        "failed_to_process": failed_images,
    }

    if return_distances and scored_results:
        # sort ascending by distance
        scored_results.sort(key=lambda x: x["distance"]) 
        response["results"] = scored_results
        if isinstance(top_k, int) and top_k > 0:
            response["top_k"] = scored_results[:top_k]

    return jsonify(response)


if __name__ == "__main__":
    # Suppress TF INFO and WARNING logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    app.run(host="0.0.0.0", port=5001, debug=True)