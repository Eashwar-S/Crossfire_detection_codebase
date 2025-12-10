import os
import json
import re
import base64
import time
from dataclasses import dataclass
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams

# -----------------------------
# 1) CONFIGURATION
# -----------------------------

IMAGES_DIR = "/dataset/images"
LABELS_DIR = "/dataset/labels"  # YOLO TXT labels (class cx cy w h)
OUTPUT_FILE = "/workspace/results.jsonl"
MODEL_PATH = "leon-se/ForestFireVLM-3B"
OUT_IMAGE_DIR = "/dataset/vlm_3B_results/images"  # will store VLM bbox images here
METRICS_PATH = "dataset/vlm_3B_results/metrics.txt"

IOU_THRESHOLD = 0.5  # For TP/FP and IoU-based metrics


# -----------------------------
# 2) PROMPT + HELPERS
# -----------------------------

def build_fire_prompt(img_w: int, img_h: int) -> str:
    coord_rules = (
        f"Image size: width={img_w}, height={img_h} (pixels).\n"
        "0-based coords, x→right, y→down.\n"
        "bbox: [x_min, y_min, x_max, y_max] (ints).\n"
        "0 <= x_min < x_max <= width; 0 <= y_min < y_max <= height.\n"
    )

    # Schema with allowed values, written once, very compact
    schema = """
Fields and allowed values:

Smoke: "Yes" | "No"
Flames: "Yes" | "No"
Uncontrolled: "Yes" | "Probe" | "No fire"
FireState: "Ignition Phase" | "Growth Phase" | "Developed Phase" |
           "Decay Phase" | "Indeterminable" | "No fire"
FireIntensity: "Low" | "Moderate" | "High" |
               "Indeterminable" | "No fire"
PeopleNearby: "Yes" | "No" | "Indeterminable" | "No fire"
fires: list of fire bboxes:
  "fires": [ {"bbox": [x_min, y_min, x_max, y_max]}, ... ]
smoke: list of smoke bboxes:
  "smoke": [ {"bbox": [x_min, y_min, x_max, y_max]}, ... ]
""".strip()

    prompt = f"""
You are an image analyst for aerial wildfire imagery.

Return EXACTLY ONE JSON object and nothing else.
The JSON MUST be a single minified line with NO spaces and NO newlines, e.g.:
{{"Smoke":"Yes","Flames":"No",...}}

Use these field names and allowed values (do NOT invent others):

{schema}

Bounding boxes:
- Fire regions → "fires":[{{"bbox":[x_min,y_min,x_max,y_max]}},...]
- Smoke regions → "smoke":[{{"bbox":[x_min,y_min,x_max,y_max]}},...]
- Coordinate rules:
{coord_rules}

Logic:
- If no fire is visible:
  - For any field that allows "No forest fire visible", use exactly that string.
  - Set "fires":[].
- If no smoke is visible:
  - Set "Smoke":"No".
  - Set "smoke":[].
- If the image does not clearly support a more specific choice,
  use the corresponding "Indeterminable" option.
- Do NOT output any explanation text, only the JSON object.

Output format (example shape, not actual values):
{{"Smoke":"Yes","Flames":"No","Uncontrolled":"Probe",
"FireState":"Growth Phase","FireIntensity":"Moderate",
"PeopleNearby":"No",
"fires":[{{"bbox":[x_min,y_min,x_max,y_max]}}],"smoke":[...]}}
"""
    # Keep prompt compact by stripping indentation/blank lines
    return "\n".join(line.strip() for line in prompt.splitlines() if line.strip())


def extract_json_obj(text: str) -> dict:
    """Extract first JSON object from model output, or return error wrapper."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {"error": "No JSON found", "raw_output": text}
    except Exception:
        return {"error": "JSON parse failed", "raw_output": text}


def encode_image_to_base64(path: str) -> str:
    """Read an image file and return base64-encoded bytes as a UTF-8 string."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -----------------------------
# 3) YOLO LABELS + IoU + MATCHING
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
    # Counts
    tp: int = 0
    fp: int = 0
    fn: int = 0
    # For mean IoU over matched GT boxes
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


def compute_precision_recall(stats: MatchStats):
    precision = stats.tp / (stats.tp + stats.fp) if (stats.tp + stats.fp) > 0 else 0.0
    recall = stats.tp / (stats.tp + stats.fn) if (stats.tp + stats.fn) > 0 else 0.0
    return precision, recall


def show_images_sequence(
    original: Image.Image,
    pred_img: Image.Image,
    gt_img: Image.Image,
    delay_sec: float = 3.0,
):
    """
    (Unused in headless Docker, kept for completeness.)
    """
    original.show()
    time.sleep(delay_sec)
    pred_img.show()
    time.sleep(delay_sec)
    gt_img.show()
    time.sleep(delay_sec)


def extract_bboxes(field_value):
    """Accept either [{'bbox': [...]}, {'bbox_2d': [...]}] or [[...], [...]]."""
    boxes = []
    if not isinstance(field_value, list):
        return boxes

    for item in field_value:
        bbox = None

        # Case 1: dict with "bbox" / "bbox_2d"
        if isinstance(item, dict):
            bbox = item.get("bbox") or item.get("bbox_2d")

        # Case 2: plain list like [x_min, y_min, x_max, y_max]
        elif isinstance(item, list) and len(item) == 4:
            bbox = item

        # Normalize and validate
        if bbox and len(bbox) == 4:
            try:
                boxes.append([float(v) for v in bbox])
            except (TypeError, ValueError):
                continue

    return boxes


# -----------------------------
# 4) MAIN
# -----------------------------

def main():
    print(f"Loading model: {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        max_model_len=2048,
        gpu_memory_utilization=0.9,
        dtype="float16",
        limit_mm_per_prompt={"image": 1},
    )

    print(f"Scanning images in {IMAGES_DIR}...")
    if not os.path.exists(IMAGES_DIR):
        print(f"ERROR: {IMAGES_DIR} does not exist. Check your Docker mounts.")
        return

    # Ensure output directory for VLM-bbox images exists
    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)

    image_files = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()

    sampling_params = SamplingParams(
        temperature=0.0,      # greedy
        top_p=1.0,
        max_tokens=128,       # can try going lower once JSON is stable
        # stop=["}\n", "}\r", "}"],  # optional: stop at end of JSON
    )

    question_fields = [
        "Smoke", "Flames", "Uncontrolled", "FireState", "FireType",
        "FireIntensity", "FireSize", "FireHotspots",
        "InfrastructureNearby", "PeopleNearby",
    ]

    # Global metrics
    fire_stats = MatchStats()
    smoke_stats = MatchStats()
    total_time = 0.0
    num_done = 0

    print(f"Processing {len(image_files)} images (one by one)...")

    with open(OUTPUT_FILE, "w") as f_out:
        for img_file in image_files:
            img_path = os.path.join(IMAGES_DIR, img_file)
            stem, _ = os.path.splitext(img_file)
            label_path = os.path.join(LABELS_DIR, stem + ".txt")

            print(f"\n--- {img_file} ---")

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Skipping {img_file}: {e}")
                continue

            img_w, img_h = image.size
            prompt_text = build_fire_prompt(img_w, img_h)

            # Encode image as base64
            s_t = time.time()
            b64_img = encode_image_to_base64(img_path)
            e_t = time.time()
            encode_time = e_t - s_t
            print(f"Image encoding time: {encode_time:.3f} seconds")

            data_url = f"data:image/jpeg;base64,{b64_img}"

            msg = {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": b64_img,
                        }
                    },
                ],
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }

            # ---- Run inference for this image ----
            start_t = time.perf_counter()
            try:
                outputs = llm.chat([msg], sampling_params)
            except Exception as e:
                print(f"Model error on image {img_file}: {e}")
                continue
            end_t = time.perf_counter()
            per_image_time = end_t - start_t

            total_time += per_image_time
            num_done += 1

            generated_text = outputs[0].outputs[0].text
            result = extract_json_obj(generated_text)

            print(f"=== Result for: {img_file} ===")
            print(f"InferenceTimeSec: {per_image_time:.3f}")

            # Parse questionnaire + bboxes
            if "error" in result:
                print(f"Error parsing JSON: {result.get('raw_output')}")
                fires = []
                smoke = []
            else:
                for field in question_fields:
                    val = result.get(field, "N/A")
                    print(f"{field}: {val}")

                fires = extract_bboxes(result.get("fires", []))
                smoke = extract_bboxes(result.get("smoke", []))

                print(f"Number of fire bboxes: {len(fires)}")
                print(f"Number of smoke bboxes: {len(smoke)}")

            print("=====================================")

            # Log raw result to JSONL
            final_record = {
                "file": img_file,
                "inference_time_sec": per_image_time,
                "result": result,
            }
            f_out.write(json.dumps(final_record) + "\n")

            # ---- Metrics vs YOLO ground truth ----
            gt_all = load_yolo_labels(label_path, img_w, img_h)
            gt_fire_boxes = [b for cls, b in gt_all if cls == 1]  # fire = class 1
            gt_smoke_boxes = [b for cls, b in gt_all if cls == 0]  # smoke = class 0

            fire_match = match_predictions(fires, gt_fire_boxes, IOU_THRESHOLD)
            smoke_match = match_predictions(smoke, gt_smoke_boxes, IOU_THRESHOLD)

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

            # -----------------------------
            # Draw prediction image and SAVE it
            # -----------------------------
            pred_img = image.copy()
            draw_pred = ImageDraw.Draw(pred_img)

            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except OSError:
                font = ImageFont.load_default()

            # VLM predictions: fire (red), smoke (blue)
            for box in fires:
                draw_pred.rectangle(box, outline="red", width=3)
                draw_pred.text((box[0], box[1]), "Fire", fill="red", font=font)

            for box in smoke:
                draw_pred.rectangle(box, outline="blue", width=3)
                draw_pred.text((box[0], box[1]), "Smoke", fill="blue", font=font)

            # Save VLM predictions image
            out_pred_path = os.path.join(OUT_IMAGE_DIR, f"{stem}_vlm.png")
            pred_img.save(out_pred_path)
            print(f"Saved VLM bbox image: {out_pred_path}")

    # -----------------------------
    # Summary metrics
    # -----------------------------
    if num_done > 0:
        avg_time = total_time / num_done
    else:
        avg_time = 0.0

    fire_prec, fire_rec = compute_precision_recall(fire_stats)
    smoke_prec, smoke_rec = compute_precision_recall(smoke_stats)

    fire_iou_mean = (
        fire_stats.iou_sum / fire_stats.iou_count if fire_stats.iou_count > 0 else 0.0
    )
    smoke_iou_mean = (
        smoke_stats.iou_sum / smoke_stats.iou_count if smoke_stats.iou_count > 0 else 0.0
    )

    # Simple AP approximation: AP ≈ precision@IoU>=0.5
    fire_ap = fire_prec
    smoke_ap = smoke_prec
    mAP = (fire_ap + smoke_ap) / 2.0

    print("\n===== SUMMARY METRICS =====")
    print(f"Images processed: {num_done}")
    print(f"Avg inference time per image: {avg_time:.4f} sec")
    print(
        f"Fire:   TP={fire_stats.tp} FP={fire_stats.fp} FN={fire_stats.fn} "
        f"Precision={fire_prec:.4f} Recall={fire_rec:.4f} "
        f"MeanIoU={fire_iou_mean:.4f} AP≈{fire_ap:.4f}"
    )
    print(
        f"Smoke:  TP={smoke_stats.tp} FP={smoke_stats.fp} FN={smoke_stats.fn} "
        f"Precision={smoke_prec:.4f} Recall={smoke_rec:.4f} "
        f"MeanIoU={smoke_iou_mean:.4f} AP≈{smoke_ap:.4f}"
    )
    print(f"mAP@0.5 (Fire, Smoke avg): {mAP:.4f}")
    print("Done! Terminating.")


if __name__ == "__main__":
    main()
