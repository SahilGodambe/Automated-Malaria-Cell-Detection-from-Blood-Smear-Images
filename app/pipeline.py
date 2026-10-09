from __future__ import annotations

import io
import base64
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

MODEL_FILES = {
    "thin": "malaria_thinsmear_44_retrainSudan_20P_4000C_separate.pb",
    "thick": "ThickSmearModel.h5.pb",
}
INPUT_SIZE = 44
MAX_CANDIDATES = 400
MODEL_LOCK = Lock()
_models = {}


class AnalysisConfigurationError(RuntimeError):
    pass


def available_models() -> dict[str, bool]:
    return {smear_type: (MODEL_DIR / filename).is_file() for smear_type, filename in MODEL_FILES.items()}


def _decode_image(image_bytes: bytes) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return np.asarray(image.convert("RGB"))
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image") from exc


def _load_model(smear_type: str):
    if smear_type in _models:
        return _models[smear_type]

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise AnalysisConfigurationError(
            "TensorFlow is not installed. Run .\\.venv\\Scripts\\python.exe -m pip install tensorflow-cpu."
        ) from exc

    with MODEL_LOCK:
        if smear_type in _models:
            return _models[smear_type]
        graph = tf.Graph()
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString((MODEL_DIR / MODEL_FILES[smear_type]).read_bytes())
        with graph.as_default():
            tf.import_graph_def(graph_def, name="")
        input_name = "conv2d_1_input:0"
        output_name = "dense_1/Softmax:0" if smear_type == "thin" else "output_node0:0"
        try:
            input_tensor = graph.get_tensor_by_name(input_name)
            output_tensor = graph.get_tensor_by_name(output_name)
        except KeyError as exc:
            raise AnalysisConfigurationError(
                f"The {smear_type} graph does not expose the expected TensorFlow nodes "
                f"{input_name} and {output_name}."
            ) from exc
        _models[smear_type] = (tf.compat.v1.Session(graph=graph), input_tensor, output_tensor)
        return _models[smear_type]


def _thick_candidate_centers(image: np.ndarray) -> tuple[list[tuple[int, int]], int]:
    padded = cv2.copyMakeBorder(
        image, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, INPUT_SIZE, cv2.BORDER_CONSTANT
    )
    reduced = cv2.resize(padded, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(reduced, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    border_mask = mask.copy()
    if contours:
        largest = max(range(len(contours)), key=lambda index: cv2.contourArea(contours[index]))
        for index, contour in enumerate(contours):
            if index != largest:
                cv2.drawContours(border_mask, [contour], -1, 1, -1)

    wbc_mask = np.clip(border_mask - mask, 0, 1).astype(np.uint8)
    wbc_mask = cv2.dilate(wbc_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    wbc_contours, _ = cv2.findContours(wbc_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    wbc_count = 0
    areas = [cv2.contourArea(contour) for contour in wbc_contours]
    average_area = sum(areas) / len(areas) if areas else 0
    if average_area:
        wbc_count = sum(round(area / average_area) for area in areas if 1000 < area < 10000)

    candidate_mask = np.ones(gray.shape, dtype=np.uint8)
    candidate_image = gray.astype(np.float32) * (1 - wbc_mask)
    candidate_image *= (1 - cv2.dilate(1 - border_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))))
    centers = []
    radius = INPUT_SIZE // 2
    for _ in range(MAX_CANDIDATES):
        _, maximum, _, location = cv2.minMaxLoc(candidate_image, mask=candidate_mask)
        if maximum <= 0:
            break
        x, y = location
        original_x = x * 2 - INPUT_SIZE
        original_y = y * 2 - INPUT_SIZE
        if (
            original_x - radius < 0
            or original_y - radius < 0
            or original_x + radius >= image.shape[1]
            or original_y + radius >= image.shape[0]
        ):
            break
        centers.append((original_x, original_y))
        cv2.circle(candidate_mask, (x, y), radius, 0, -1)

    return centers, wbc_count


def _candidate_centers(image: np.ndarray, smear_type: str) -> tuple[list[tuple[int, int]], int | None]:
    if smear_type == "thick":
        return _thick_candidate_centers(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    green = image[:, :, 1]
    _, mask = cv2.threshold(green, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    min_area = max(40, int(image.shape[0] * image.shape[1] / 250000))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centers.append((round(moments["m10"] / moments["m00"]), round(moments["m01"] / moments["m00"])))
        if len(centers) == MAX_CANDIDATES:
            break

    return centers, None


def _patches(image: np.ndarray, centers: list[tuple[int, int]]) -> np.ndarray:
    half = INPUT_SIZE // 2
    patches = []
    for x, y in centers:
        x0, y0 = max(0, x - half), max(0, y - half)
        x1, y1 = min(image.shape[1], x + half), min(image.shape[0], y + half)
        patch = image[y0:y1, x0:x1]
        patch = cv2.resize(patch, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        patches.append(patch.astype(np.float32) / 255.0)
    return np.asarray(patches, dtype=np.float32)


def _annotated_image(image: np.ndarray, detections: list[dict]) -> str:
    output = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    for detection in detections:
        color = (66, 121, 233) if detection["infected"] else (160, 180, 120)
        cv2.circle(output, (detection["x"], detection["y"]), 12, color, 2)
    success, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not success:
        raise ValueError("Could not encode the annotated result image")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii")


def analyze_image(image_bytes: bytes, smear_type: str) -> dict:
    image = _decode_image(image_bytes)
    model_path = MODEL_DIR / MODEL_FILES[smear_type]
    if not model_path.is_file():
        raise AnalysisConfigurationError(
            f"The {smear_type} model is not installed. Add {model_path.name} to models/ before analyzing images."
        )

    session, input_tensor, output_tensor = _load_model(smear_type)
    centers, wbc_count = _candidate_centers(image, smear_type)
    if not centers:
        raise ValueError("The thick-smear pipeline found no usable candidate regions")
    scores = []
    with MODEL_LOCK:
        for start in range(0, len(centers), 32):
            batch = _patches(image, centers[start:start + 32])
            scores.extend(session.run(output_tensor, {input_tensor: batch}).tolist())

    detections = []
    infected_count = 0
    for (x, y), output in zip(centers, scores):
        probabilities = np.asarray(output, dtype=np.float32).reshape(-1)
        if probabilities.size < 2:
            raise AnalysisConfigurationError("The model returned fewer than two class scores.")
        infected_index = 0 if smear_type == "thin" else 1
        infected = bool(probabilities[infected_index] >= 0.5)
        confidence = float(probabilities[infected_index] if infected else probabilities[1 - infected_index])
        infected_count += int(infected)
        detections.append({"x": x, "y": y, "infected": infected, "confidence": round(confidence, 4)})

    count_label = "infected cells" if smear_type == "thin" else "parasite candidates"
    return {
        "smear_type": smear_type,
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "candidate_count": len(detections),
        "infected_count": infected_count,
        "wbc_count": wbc_count,
        "summary": f"{infected_count} {count_label} across {len(detections)} candidates",
        "detections": detections,
        "annotated_image": _annotated_image(image, detections),
    }
