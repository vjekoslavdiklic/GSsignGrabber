import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO


@dataclass
class Detection:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str


class SignClassifier(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = torch.nn.Linear(backbone.fc.in_features, num_classes)
        self.model = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_street_view_url(
    lat: float,
    lon: float,
    api_key: str,
    heading: int = 0,
    fov: int = 90,
    pitch: int = 0,
    size: str = "640x640",
) -> str:
    return (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={size}&location={lat},{lon}&heading={heading}&fov={fov}"
        f"&pitch={pitch}&key={api_key}"
    )


def fetch_image_from_url(url: str, timeout_s: int = 30) -> Image.Image:
    response = requests.get(url, timeout=timeout_s, stream=True)
    response.raise_for_status()
    return Image.open(response.raw).convert("RGB")


def load_image(image_path: Optional[str], image_url: Optional[str]) -> Image.Image:
    if image_path:
        return Image.open(image_path).convert("RGB")
    if image_url:
        return fetch_image_from_url(image_url)
    raise ValueError("Provide either image_path or image_url.")


def detect_signs(
    yolo_weights: str,
    image: Image.Image,
    class_filter: Optional[Sequence[str]] = None,
    conf_threshold: float = 0.25,
) -> List[Detection]:
    model = YOLO(yolo_weights)
    results = model.predict(np.array(image), conf=conf_threshold, verbose=False)
    detections: List[Detection] = []
    for result in results:
        if not result.boxes:
            continue
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = result.names.get(class_id, str(class_id))
            if class_filter and class_name not in class_filter:
                continue
            xyxy = tuple(int(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    bbox_xyxy=xyxy,
                    confidence=float(box.conf.item()),
                    class_id=class_id,
                    class_name=class_name,
                )
            )
    return detections


def load_classifier(checkpoint_path: str, labels: Sequence[str]) -> SignClassifier:
    model = SignClassifier(num_classes=len(labels))
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def classify_crops(
    classifier: SignClassifier,
    crops: Iterable[Image.Image],
    labels: Sequence[str],
) -> List[Tuple[str, float]]:
    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch = torch.stack([preprocess(crop) for crop in crops])
    with torch.no_grad():
        logits = classifier(batch)
        probs = torch.softmax(logits, dim=1)
        confs, indices = torch.max(probs, dim=1)
    return [(labels[idx], float(conf)) for idx, conf in zip(indices.tolist(), confs.tolist())]


def crop_image(image: Image.Image, bbox_xyxy: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = bbox_xyxy
    return image.crop((x1, y1, x2, y2))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_log(
    log_path: Path,
    rows: Iterable[dict],
    fieldnames: Sequence[str],
) -> None:
    file_exists = log_path.exists()
    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grab and classify traffic signs from Street View.")
    parser.add_argument("--image-path", type=str, help="Path to a local image.")
    parser.add_argument("--image-url", type=str, help="URL to an image.")
    parser.add_argument("--lat", type=float, help="Latitude for Street View.")
    parser.add_argument("--lon", type=float, help="Longitude for Street View.")
    parser.add_argument("--api-key", type=str, help="Google Street View API key.")
    parser.add_argument("--heading", type=int, default=0)
    parser.add_argument("--fov", type=int, default=90)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--yolo-weights", type=str, required=True)
    parser.add_argument("--yolo-classes", type=str, default="", help="Comma-separated class names to keep.")
    parser.add_argument("--classifier-weights", type=str, required=True)
    parser.add_argument("--classifier-labels", type=str, required=True, help="Path to text file with labels.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--conf-threshold", type=float, default=0.25)
    return parser.parse_args()


def load_labels(labels_path: str) -> List[str]:
    with open(labels_path, "r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip()]


def main() -> None:
    args = parse_args()
    image_url = args.image_url
    if not image_url and args.lat is not None and args.lon is not None and args.api_key:
        image_url = build_street_view_url(
            lat=args.lat,
            lon=args.lon,
            api_key=args.api_key,
            heading=args.heading,
            fov=args.fov,
            pitch=args.pitch,
        )

    image = load_image(args.image_path, image_url)

    class_filter = [c.strip() for c in args.yolo_classes.split(",") if c.strip()] or None
    detections = detect_signs(
        args.yolo_weights,
        image,
        class_filter=class_filter,
        conf_threshold=args.conf_threshold,
    )

    labels = load_labels(args.classifier_labels)
    classifier = load_classifier(args.classifier_weights, labels)

    output_dir = Path(args.output_dir)
    crops_dir = output_dir / "crops"
    ensure_dir(crops_dir)

    crops = [crop_image(image, det.bbox_xyxy) for det in detections]
    classifications = classify_crops(classifier, crops, labels) if crops else []

    log_rows = []
    for idx, (det, (label, conf)) in enumerate(zip(detections, classifications)):
        crop_path = crops_dir / f"sign_{idx:03d}.jpg"
        crops[idx].save(crop_path)
        log_rows.append(
            {
                "crop_path": str(crop_path),
                "bbox_x1": det.bbox_xyxy[0],
                "bbox_y1": det.bbox_xyxy[1],
                "bbox_x2": det.bbox_xyxy[2],
                "bbox_y2": det.bbox_xyxy[3],
                "det_confidence": det.confidence,
                "det_class_id": det.class_id,
                "det_class_name": det.class_name,
                "cls_label": label,
                "cls_confidence": conf,
            }
        )

    log_path = output_dir / "sign_log.csv"
    fieldnames = [
        "crop_path",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "det_confidence",
        "det_class_id",
        "det_class_name",
        "cls_label",
        "cls_confidence",
    ]
    write_log(log_path, log_rows, fieldnames)

    print(f"Saved {len(log_rows)} crops and log to {output_dir}")


if __name__ == "__main__":
    main()
