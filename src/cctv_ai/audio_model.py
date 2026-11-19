from __future__ import annotations

import hashlib
import io
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


YAMNET_MODEL_URL = (
    "https://www.kaggle.com/api/v1/models/google/yamnet/tfLite/"
    "classification-tflite/1/download"
)
YAMNET_MODEL_SHA256 = (
    "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de"
)
YAMNET_CLASS_MAP_URL = (
    "https://raw.githubusercontent.com/tensorflow/models/"
    "4d7bdd8c170ee90850f2f9ccef0f6d19b817de35/"
    "research/audioset/yamnet/yamnet_class_map.csv"
)
YAMNET_CLASS_MAP_SHA256 = (
    "cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2"
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "cctv-family-ai/0.1"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _valid_existing(path: Path, expected_sha256: str) -> bool:
    if not path.is_file():
        return False
    return _sha256(path.read_bytes()) == expected_sha256


def install_yamnet(model_dir: str | Path) -> dict[str, Any]:
    destination = Path(model_dir)
    model_path = destination / "yamnet.tflite"
    class_map_path = destination / "yamnet_class_map.csv"
    downloaded: list[str] = []

    if not _valid_existing(model_path, YAMNET_MODEL_SHA256):
        archive = _download(YAMNET_MODEL_URL)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            member = bundle.getmember("1.tflite")
            extracted = bundle.extractfile(member)
            if extracted is None:
                raise RuntimeError("YAMNet model archive did not contain 1.tflite")
            model_data = extracted.read()
        model_hash = _sha256(model_data)
        if model_hash != YAMNET_MODEL_SHA256:
            raise RuntimeError(
                "YAMNet model checksum mismatch: "
                f"expected {YAMNET_MODEL_SHA256}, got {model_hash}"
            )
        _atomic_write(model_path, model_data)
        downloaded.append("model")

    if not _valid_existing(class_map_path, YAMNET_CLASS_MAP_SHA256):
        class_map_data = _download(YAMNET_CLASS_MAP_URL)
        class_map_hash = _sha256(class_map_data)
        if class_map_hash != YAMNET_CLASS_MAP_SHA256:
            raise RuntimeError(
                "YAMNet class map checksum mismatch: "
                f"expected {YAMNET_CLASS_MAP_SHA256}, got {class_map_hash}"
            )
        _atomic_write(class_map_path, class_map_data)
        downloaded.append("class_map")

    return {
        "model_path": str(model_path.resolve()),
        "model_sha256": YAMNET_MODEL_SHA256,
        "class_map_path": str(class_map_path.resolve()),
        "class_map_sha256": YAMNET_CLASS_MAP_SHA256,
        "downloaded": downloaded,
        "model_source": YAMNET_MODEL_URL,
        "class_map_source": YAMNET_CLASS_MAP_URL,
    }
