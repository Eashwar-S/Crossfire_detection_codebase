import os
import json
import re
import time
from vllm import LLM, SamplingParams
from dataclasses import dataclass
from typing import List, Tuple, Dict
import json
from json_repair import repair_json
# from huggingface_hub import login
# from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
import torch
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# -----------------------------
# 0) PATHS / CONFIG
# -----------------------------
IMAGES_DIR = "../dataset/images"
LABELS_DIR = "../dataset/labels"     # ground-truth labels
OUT_IMAGE_DIR = "../vlm_results/images"
METRICS_PATH = "../vlm_results/metrics.txt"

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

IOU_THRESHOLD = 0.5  # For TP/FP and mAP


# -----------------------------
# 1) PROMPT + CONVERSATION
# -----------------------------
def build_fire_prompt(img_w: int, img_h: int) -> str:
    coord_rules = (
        f"Image size: width={img_w}, height={img_h} pixels.\n"
        "Use 0-based pixel coordinates.\n"
        "Axes: x increases to the right; y increases downward.\n"
        "Bounding box format: [x_min, y_min, x_max, y_max] with integers.\n"
        "Constraints: 0 <= x_min < x_max <= width, 0 <= y_min < y_max <= height.\n"
    )

    schema = (
        "{\n"
        '  "FirePresent": "Yes" or "No",\n'
        '  "SmokePresent": "Yes" or "No",\n'
        '  "fires": [ {"bbox": [x_min, y_min, x_max, y_max]} ],\n'
        '  "smoke": [ {"bbox": [x_min, y_min, x_max, y_max]} ]\n'
        "}\n"
    )

    prompt = f"""
You are an image analyst.

Look only at the image and answer:
1) Is there any visible FIRE?
2) Is there any visible SMOKE?
3) If present, return bounding boxes for each distinct region of fire and smoke.

Bounding boxes:
- Fire regions: "fires": [{{"bbox": [x_min, y_min, x_max, y_max]}}, ...]
- Smoke regions: "smoke": [{{"bbox": [x_min, y_min, x_max, y_max]}}, ...]

Rules:
{coord_rules}
If there is no fire, set "FirePresent": "No" and "fires": [].
If there is no smoke, set "SmokePresent": "No" and "smoke": [].

Return a SINGLE JSON object only.
Do NOT include any text outside the JSON.
Use EXACT field names and value strings.

Output JSON schema:
{schema}
"""
    return "\n".join(line.strip() for line in prompt.splitlines() if line.strip())


def build_conversation_for_fire(image: Image.Image):
    img_w, img_h = image.size
    prompt = build_fire_prompt(img_w, img_h)
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]
    }]
    return conversation

model_id = "leon-se/ForestFireVLM-3B"
llm = LLM(model=model_id,
            trust_remote_code=True,
            dtype="auto")
sampling_params = SamplingParams(max_tokens=128)

# processor = AutoProcessor.from_pretrained(model_id)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()


# -----------------------------
# 3) HELPER: PARSE VLM JSON
# -----------------------------
# def extract_json_obj(text: str) -> Dict:
#     """
#     Extract first JSON object from model text output.
#     Handles cases where there is extra text before/after.
#     """
#     match = re.search(r"\{.*\}", text, re.DOTALL)
#     if not match:
#         raise ValueError("No JSON object found in model output.")
#     json_str = match.group(0)
#     return json.loads(json_str)

def extract_json_obj(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output.")
    json_str = match.group(0)
    repaired_json = repair_json(json_str)
    return json.loads(repaired_json)


def run_vlm_on_image(image: Image.Image) -> Dict:
    """Run the VLM and return parsed JSON with fires/smoke."""
    conversation = build_conversation_for_fire(image)

    prompt = (
        "USER: <image>\n"
        f"{conversation}\n"
        "ASSISTANT:"
    )

    outputs = llm.generate(
        {
            "prompt":prompt,
            "multi_modal_data": {
                "image": image,
            },
        },
        sampling_params=sampling_params,
    )

    text = outputs[0].outputs[0].text

    result = extract_json_obj(text)

    # Normalize keys: we expect each item to have "bbox" OR "bbox_2d"
    for key in ("fires", "smoke"):
        if key in result and isinstance(result[key], list):
            for item in result[key]:
                if "bbox" not in item and "bbox_2d" in item:
                    item["bbox"] = item["bbox_2d"]
    return result


# -----------------------------
# 4) YOLO LABELS → PIXEL BOXES
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
# 5) IoU + MATCHING
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
# 6) MAIN LOOP
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
        # lines_out.append(
        #     f"{img_name}\tSmokeIoU={smoke_iou_img:.4f}\tFireIoU={fire_iou_img:.4f}\tInferenceTimeSec={infer_time:.4f}"
        # )
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
        # break

    # -----------------------------
    # 7) GLOBAL METRICS (mAP-like)
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
        f"Precision={fire_prec:.44f} Recall={fire_rec:.4f} MeanIoU={fire_iou_mean:.4f} AP≈{fire_ap:.4f}",
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
