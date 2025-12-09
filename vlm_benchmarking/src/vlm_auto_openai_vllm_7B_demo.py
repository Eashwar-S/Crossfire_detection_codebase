import os
import io
import json
import re
import time
import base64
# import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict

from json_repair import repair_json
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from openai import OpenAI

# -----------------------------
# 0) PATHS / CONFIG
# -----------------------------
IMAGES_DIR = "../dataset/images"
LABELS_DIR = "../dataset/labels"     # ground-truth labels
OUT_IMAGE_DIR = "../vlm_7B_results/images"
METRICS_PATH = "../vlm_7B_results/metrics.txt"

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

IOU_THRESHOLD = 0.5  # For TP/FP and mAP

# -----------------------------
# ENV + OPENAI / vLLM CLIENT
# -----------------------------
load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")   # vLLM doesn't validate
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "leon-se/ForestFireVLM-7B-FP8-Dynamic")  # or your VLM model

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

# -----------------------------
# 1) PROMPT
# -----------------------------
def build_fire_prompt(img_w: int, img_h: int) -> str:
    coord_rules = (
        f"Image size: width={img_w}, height={img_h} pixels.\n"
        "Use 0-based pixel coordinates.\n"
        "Axes: x increases to the right; y increases downward.\n"
        "Bounding box format: [x_min, y_min, x_max, y_max] with integers.\n"
        "Constraints: 0 <= x_min < x_max <= width, 0 <= y_min < y_max <= height.\n"
    )

    # JSON schema with all fields from the paper table
    schema = (
        "{\n"
        '  "Smoke": "Yes" or "No",\n'
        '  "Flames": "Yes" or "No",\n'
        '  "Uncontrolled": "Yes" or "Closer investigation required" or "No forest fire visible",\n'
        '  "FireState": "Ignition Phase" or "Growth Phase" or "Fully Developed Phase" or '
        '"Decay Phase" or "Cannot be determined" or "No forest fire visible",\n'
        '  "FireType": "Ground Fire" or "Surface Fire" or "Crown Fire" or '
        '"Cannot be determined" or "No forest fire visible",\n'
        '  "FireIntensity": "Low" or "Moderate" or "High" or '
        '"Cannot be determined" or "No forest fire visible",\n'
        '  "FireSize": "Small" or "Medium" or "Large" or '
        '"Cannot be determined" or "No forest fire visible",\n'
        '  "FireHotspots": "Multiple hotspots" or "One hotspot" or '
        '"Cannot be determined" or "No forest fire visible",\n'
        '  "InfrastructureNearby": "Yes" or "No" or "Cannot be determined" or '
        '"No forest fire visible",\n'
        '  "PeopleNearby": "Yes" or "No" or "Cannot be determined" or '
        '"No forest fire visible",\n'
        '  "TreeVitality": "Vital" or "Moderate Vitality" or "Declining" or "Dead" or '
        '"Cannot be determined" or "No forest fire visible",\n'
        '  "fires": [ {"bbox": [x_min, y_min, x_max, y_max]} ],\n'
        '  "smoke": [ {"bbox": [x_min, y_min, x_max, y_max]} ]\n'
        "}\n"
    )

    prompt = f"""
You are an image analyst for aerial wildfire imagery.

Look only at the image and answer the following questions, using the allowed
options for each field. Do not make up new option strings.

Questions / fields:
1) Smoke: Is smoke from a forest fire visible in the image?
   - Options: "Yes", "No"

2) Flames: Are flames from a forest fire visible in the image?
   - Options: "Yes", "No"

3) Uncontrolled: Can you confirm that this is an uncontrolled forest fire?
   - Options: "Yes", "Closer investigation required", "No forest fire visible"

4) FireState: What is the current state of the forest fire?
   - Options: "Ignition Phase", "Growth Phase", "Fully Developed Phase",
              "Decay Phase", "Cannot be determined", "No forest fire visible"

5) FireType: What type of fire is it?
   - Options: "Ground Fire", "Surface Fire", "Crown Fire",
              "Cannot be determined", "No forest fire visible"

6) FireIntensity: What is the intensity of the fire?
   - Options: "Low", "Moderate", "High",
              "Cannot be determined", "No forest fire visible"

7) FireSize: What is the size of the fire?
   - Options: "Small", "Medium", "Large",
              "Cannot be determined", "No forest fire visible"

8) FireHotspots: Does the forest fire have multiple hotspots?
   - Options: "Multiple hotspots", "One hotspot",
              "Cannot be determined", "No forest fire visible"

9) InfrastructureNearby: Is there infrastructure visible near the forest fire?
   - Options: "Yes", "No", "Cannot be determined", "No forest fire visible"

10) PeopleNearby: Are there people visible near the forest fire?
    - Options: "Yes", "No", "Cannot be determined", "No forest fire visible"

11) TreeVitality: Describe the vitality of the trees around the fire.
    - Options: "Vital", "Moderate Vitality", "Declining", "Dead",
               "Cannot be determined", "No forest fire visible"

Additionally:
12) If there is visible FIRE, provide bounding boxes for each distinct region of fire.
13) If there is visible SMOKE, provide bounding boxes for each distinct region of smoke.

Bounding boxes:
- Fire regions: "fires": [{{"bbox": [x_min, y_min, x_max, y_max]}}, ...]
- Smoke regions: "smoke": [{{"bbox": [x_min, y_min, x_max, y_max]}}, ...]

Rules:
{coord_rules}
- If no fire is visible, set fire-related fields that have "No forest fire visible"
  as an option to exactly "No forest fire visible" and set "fires": [].
- If no smoke is visible, set "Smoke": "No" and "smoke": [].
- If the correct option is unclear from the image, choose the corresponding
  "Cannot be determined" option instead of guessing.
- Use the options EXACTLY as written (case and spelling).

Return a SINGLE JSON object only.
Do NOT include any text outside the JSON.
Use EXACT field names and value strings.

Output JSON schema:
{schema}
"""
    # Strip leading spaces so the prompt is clean
    return "\n".join(line.strip() for line in prompt.splitlines() if line.strip())


# -----------------------------
# 2) HELPER: JSON PARSE
# -----------------------------
def extract_json_obj(text: str) -> dict:
    """
    Extract first JSON object from model text output and repair if needed.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    json_str = match.group(0)
    repaired_json = repair_json(json_str)
    return json.loads(repaired_json)


def pil_to_base64_jpeg(image: Image.Image) -> str:
    """
    Convert a PIL.Image to a base64-encoded JPEG (for OpenAI-style image_url).
    """
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return b64


def run_vlm_on_image(image: Image.Image) -> Dict:
    """
    Run the VLM via an OpenAI-compatible API (e.g., vLLM Docker) and return
    parsed JSON with fires/smoke and questionnaire fields.
    """
    img_w, img_h = image.size
    prompt = build_fire_prompt(img_w, img_h)

    image_b64 = pil_to_base64_jpeg(image)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.0,
    )

    text = response.choices[0].message.content
    print("=== Raw model output ===")
    print(text)
    print("========================")

    result = extract_json_obj(text)

    # Normalize keys: we expect each item to have "bbox" OR "bbox_2d"
    for key in ("fires", "smoke"):
        if key in result and isinstance(result[key], list):
            for item in result[key]:
                if "bbox" not in item and "bbox_2d" in item:
                    item["bbox"] = item["bbox_2d"]

    # ---- Print questionnaire answers in terminal ----
    question_fields = [
        "Smoke",
        "Flames",
        "Uncontrolled",
        "FireState",
        "FireType",
        "FireIntensity",
        "FireSize",
        "FireHotspots",
        "InfrastructureNearby",
        "PeopleNearby",
        "TreeVitality",
    ]

    print("\n=== Parsed wildfire questionnaire ===")
    for field in question_fields:
        print(f"{field}: {result.get(field, 'N/A')}")
    print(f"Number of fire bboxes: {len(result.get('fires', []))}")
    print(f"Number of smoke bboxes: {len(result.get('smoke', []))}")
    print("=====================================\n")

    return result



# -----------------------------
# 3) YOLO LABELS → PIXEL BOXES
# -----------------------------
def load_yolo_labels(label_path: str, img_w: int, img_h: int) -> List[Tuple[int, List[float]]]:
    """
    Load YOLO TXT labels: class cx cy w h (normalized).
    Return list of (class_id, [x_min, y_min, x_max, y_max]) in pixel coords.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            x_min = (cx - w / 2.0) * img_w
            x_max = (cx + w / 2.0) * img_w
            y_min = (cy - h / 2.0) * img_h
            y_max = (cy + h / 2.0) * img_h
            boxes.append((cls_id, [x_min, y_min, x_max, y_max]))
    return boxes


# -----------------------------
# 4) IoU + MATCHING
# -----------------------------
def iou(box1: List[float], box2: List[float]) -> float:
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h

    area1 = max(0.0, x1_max - x1_min) * max(0.0, y1_max - y1_min)
    area2 = max(0.0, x2_max - x2_min) * max(0.0, y2_max - y2_min)

    union = area1 + area2 - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


@dataclass
class MatchStats:
    # For AP / mAP
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # For average IoU over matched GT boxes
    iou_sum: float = 0.0
    iou_count: int = 0


def match_predictions(
    pred_boxes: List[List[float]],
    gt_boxes: List[List[float]],
    iou_threshold: float = IOU_THRESHOLD,
) -> MatchStats:
    """
    Greedy matching:
      - For each GT box, find best IoU prediction that is not used yet.
      - If IoU >= threshold -> TP, else FN.
      - Unmatched predictions -> FP.
    """
    stats = MatchStats()
    used_pred = set()

    # Match GT -> best pred
    for gt in gt_boxes:
        best_iou = 0.0
        best_idx = None
        for i, pb in enumerate(pred_boxes):
            if i in used_pred:
                continue
            val = iou(pb, gt)
            if val > best_iou:
                best_iou = val
                best_idx = i

        if best_idx is not None and best_iou >= iou_threshold:
            stats.tp += 1
            stats.iou_sum += best_iou
            stats.iou_count += 1
            used_pred.add(best_idx)
        else:
            stats.fn += 1

    # Remaining predictions are FP
    stats.fp += (len(pred_boxes) - len(used_pred))
    return stats


# -----------------------------
# 5) MAIN LOOP
# -----------------------------
def main():
    total_time = 0.0
    num_inferred = 0

    # Global stats for mAP-like metric
    fire_stats = MatchStats()
    smoke_stats = MatchStats()

    lines_out = []

    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")]
    image_files.sort()

    for img_name in image_files:
        img_path = os.path.join(IMAGES_DIR, img_name)
        stem, _ = os.path.splitext(img_name)
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        # ---- Run VLM ----
        start = time.perf_counter()
        try:
            result = run_vlm_on_image(image)
        except Exception as e:
            print(f"[ERROR] VLM failed on {img_name}: {e}")
            continue
        end = time.perf_counter()

        infer_time = end - start
        total_time += infer_time
        num_inferred += 1

        # ---- Get predicted boxes ----
        pred_fire_boxes = []
        pred_smoke_boxes = []

        for item in result.get("fires", []):
            box = item.get("bbox")
            if box and len(box) == 4:
                pred_fire_boxes.append(list(map(float, box)))

        for item in result.get("smoke", []):
            box = item.get("bbox")
            if box and len(box) == 4:
                pred_smoke_boxes.append(list(map(float, box)))

        # ---- Load GT boxes (YOLO) ----
        gt_all = load_yolo_labels(label_path, img_w, img_h)

        gt_fire_boxes = [b for cls, b in gt_all if cls == 1]  # class 1 = fire
        gt_smoke_boxes = [b for cls, b in gt_all if cls == 0]  # class 0 = smoke

        # ---- Match + IoU stats per class ----
        fire_match = match_predictions(pred_fire_boxes, gt_fire_boxes, IOU_THRESHOLD)
        smoke_match = match_predictions(pred_smoke_boxes, gt_smoke_boxes, IOU_THRESHOLD)

        # accumulate global stats
        fire_stats.tp += fire_match.tp
        fire_stats.fp += fire_match.fp
        fire_stats.fn += fire_match.fn
        fire_stats.iou_sum += fire_match.iou_sum
        fire_stats.iou_count += fire_match.iou_count

        smoke_stats.tp += smoke_match.tp
        smoke_stats.fp += smoke_match.fp
        smoke_stats.fn += smoke_match.fn
        smoke_stats.iou_sum += smoke_match.iou_sum
        smoke_stats.iou_count += smoke_match.iou_count

        # Per-image IoU (mean over matched GT boxes for that image)
        fire_iou_img = fire_match.iou_sum / fire_match.iou_count if fire_match.iou_count > 0 else 0.0
        smoke_iou_img = smoke_match.iou_sum / smoke_match.iou_count if smoke_match.iou_count > 0 else 0.0

        # ---- Draw and save image with boxes ----
        out_img = image.copy()
        draw = ImageDraw.Draw(out_img)

        try:
            font = ImageFont.truetype("arial.ttf", 24)  # increase 24 → 32, etc. for larger text
        except OSError:
            font = ImageFont.load_default()

        for box in pred_fire_boxes:
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0], box[1]), "Fire", fill="red", font=font)

        for box in pred_smoke_boxes:
            draw.rectangle(box, outline="blue", width=3)
            draw.text((box[0], box[1]), "Smoke", fill="blue", font=font)

        out_name = f"{stem}_vlm_output.jpg"
        out_path = os.path.join(OUT_IMAGE_DIR, out_name)
        out_img.save(out_path)

        # ---- Log per-image ----
        fire_boxes_json = json.dumps(pred_fire_boxes)   # list of [x_min, y_min, x_max, y_max]
        smoke_boxes_json = json.dumps(pred_smoke_boxes)

        lines_out.append(
            f"{img_name}"
            f"\tSmokeIoU={smoke_iou_img:.4f}"
            f"\tFireIoU={fire_iou_img:.4f}"
            f"\tInferenceTimeSec={infer_time:.4f}"
            f"\tFireBoxes={fire_boxes_json}"
            f"\tSmokeBoxes={smoke_boxes_json}"
        )
        print(lines_out[-1])

    # -----------------------------
    # 6) GLOBAL METRICS (mAP-like)
    # -----------------------------
    def compute_precision_recall(stats: MatchStats) -> Tuple[float, float]:
        precision = stats.tp / (stats.tp + stats.fp) if (stats.tp + stats.fp) > 0 else 0.0
        recall = stats.tp / (stats.tp + stats.fn) if (stats.tp + stats.fn) > 0 else 0.0
        return precision, recall

    fire_prec, fire_rec = compute_precision_recall(fire_stats)
    smoke_prec, smoke_rec = compute_precision_recall(smoke_stats)

    # Simple AP approximation: AP ≈ precision at IoU>=0.5
    fire_ap = fire_prec
    smoke_ap = smoke_prec
    mAP = (fire_ap + smoke_ap) / 2.0

    fire_iou_mean = fire_stats.iou_sum / fire_stats.iou_count if fire_stats.iou_count > 0 else 0.0
    smoke_iou_mean = smoke_stats.iou_sum / smoke_stats.iou_count if smoke_stats.iou_count > 0 else 0.0
    avg_time = total_time / num_inferred if num_inferred > 0 else 0.0

    summary_lines = [
        "",
        "===== SUMMARY =====",
        f"Images processed: {num_inferred}",
        f"Avg inference time per image: {avg_time:.4f} sec",
        f"Fire:   TP={fire_stats.tp} FP={fire_stats.fp} FN={fire_stats.fn} "
        f"Precision={fire_prec:.4f} Recall={fire_rec:.4f} MeanIoU={fire_iou_mean:.4f} AP≈{fire_ap:.4f}",
        f"Smoke:  TP={smoke_stats.tp} FP={smoke_stats.fp} FN={smoke_stats.fn} "
        f"Precision={smoke_prec:.4f} Recall={smoke_rec:.4f} MeanIoU={smoke_iou_mean:.4f} AP≈{smoke_ap:.4f}",
        f"mAP@0.5 (Fire, Smoke avg): {mAP:.4f}",
    ]

    # Write metrics file
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        for line in lines_out + summary_lines:
            f.write(line + "\n")

    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()