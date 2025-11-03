import os
import shutil
from typing import Optional, Tuple

import cv2
import numpy as np
import requests
from deepface import DeepFace


# Configuration constants
MODEL_NAME = "Facenet"
DETECTOR_BACKEND = "mtcnn"


def valid_image(path: str) -> bool:
    """Validate that the file at path is a readable image.

    Returns False if cv2 fails to load, size is zero, or image has < 2 dims.
    """
    try:
        img = cv2.imread(path)
        if img is None:
            return False
        if img.size == 0:
            return False
        if len(img.shape) < 2:
            return False
        return True
    except Exception:
        return False


def download_image(url: str, path: str) -> bool:
    """Download an image from a URL or copy from a local path to destination.

    - Includes a User-Agent header for HTTP(S) requests.
    - Raises for bad HTTP status codes.
    - Validates the saved image with valid_image().
    - Returns True if valid, otherwise False.
    """
    # Handle local file paths (e.g., file already on disk)
    if os.path.exists(url):
        shutil.copyfile(url, path)
        return valid_image(path)

    # Remote URL download
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/118.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Ensure directory exists
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(resp.content)

    return valid_image(path)


def _extract_embedding(result: object, context: str) -> Optional[np.ndarray]:
    """Internal: normalize DeepFace.represent outputs into a 1D numpy vector."""
    try:
        candidate = None

        if isinstance(result, list):
            if len(result) == 0:
                return None
            if all(isinstance(x, (float, int, np.floating, np.integer)) for x in result):
                candidate = result
            else:
                known_keys = ("embedding", "represent", "representation", "feature", "features", "vector")
                for item in result:
                    if isinstance(item, dict):
                        for k in known_keys:
                            if k in item:
                                candidate = item[k]
                                break
                        if candidate is not None:
                            break
                    elif isinstance(item, (list, tuple, np.ndarray)):
                        if len(item) > 0 and all(isinstance(x, (float, int, np.floating, np.integer)) for x in item):
                            candidate = item
                            break
                if candidate is None:
                    first = result[0]
                    if isinstance(first, dict):
                        for k in known_keys:
                            if k in first:
                                candidate = first[k]
                                break
                    elif isinstance(first, (list, tuple, np.ndarray)):
                        candidate = first

        elif isinstance(result, dict):
            for k in ("embedding", "represent", "representation", "feature", "features", "vector"):
                if k in result:
                    candidate = result[k]
                    break

        elif isinstance(result, (list, tuple, np.ndarray)):
            candidate = result

        if candidate is None or isinstance(candidate, (float, int)):
            print(f"Unexpected represent() output type for {context}: {type(result)}")
            return None

        vec = np.asarray(candidate, dtype=np.float32).reshape(-1)
        return vec
    except Exception as e:
        print(f"_extract_embedding error ({context}): {e}")
        return None


def get_face_embedding(image_path: str, model_name: Optional[str] = None, detector_backend: Optional[str] = None) -> Optional[np.ndarray]:
    """Compute the face embedding for an image using DeepFace.represent.

    Robust to different return formats across DeepFace versions.
    Returns the embedding vector (first detected face) or None.
    """
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=model_name or MODEL_NAME,
            detector_backend=detector_backend or DETECTOR_BACKEND,
            enforce_detection=True,
        )
        return _extract_embedding(result, image_path)
    except Exception as e:
        print(f"get_face_embedding error: {e}")
        return None


def get_face_embedding_from_array(image_bgr: np.ndarray, model_name: Optional[str] = None, detector_backend: Optional[str] = None) -> Optional[np.ndarray]:
    """Compute embedding when an image array is already loaded (BGR from cv2)."""
    try:
        if image_bgr is None:
            return None
        result = DeepFace.represent(
            img_path=image_bgr,
            model_name=model_name or MODEL_NAME,
            detector_backend=detector_backend or DETECTOR_BACKEND,
            enforce_detection=True,
        )
        return _extract_embedding(result, "np_image")
    except Exception as e:
        print(f"get_face_embedding_from_array error: {e}")
        return None


def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def get_face_embedding_augmented(image_path: str, model_name: Optional[str] = None, detector_backend: Optional[str] = None, augment: bool = True) -> Optional[np.ndarray]:
    """Compute an embedding with optional simple augmentation (horizontal flip) and average.

    This improves robustness to pose and mirroring.
    """
    if not augment:
        return get_face_embedding(image_path, model_name=model_name, detector_backend=detector_backend)

    img = cv2.imread(image_path)
    if img is None:
        return None

    emb1 = get_face_embedding_from_array(img, model_name=model_name, detector_backend=detector_backend)
    # Horizontal flip
    img_flipped = cv2.flip(img, 1)
    emb2 = get_face_embedding_from_array(img_flipped, model_name=model_name, detector_backend=detector_backend)

    if emb1 is None and emb2 is None:
        return None
    if emb1 is None:
        return emb2
    if emb2 is None:
        return emb1

    # Average normalized embeddings
    emb1_n = l2_normalize(emb1)
    emb2_n = l2_normalize(emb2)
    avg = l2_normalize((emb1_n + emb2_n) / 2.0)
    return avg

# Removed duplicate legacy block to avoid redefinitions