import os
import io
import json
import re
import time
import base64
from dataclasses import dataclass
from typing import Dict, List

from json_repair import repair_json
from dotenv import load_dotenv
from PIL import Image

from datasets import load_dataset
from openai import OpenAI

# -----------------------------
# 0) CONFIG
# -----------------------------
DATASET_NAME = "leon-se/ForestFireInsights-Eval"
DATASET_SPLIT = "train"          # same default as the authors' repo
RESULTS_DIR = "./ffvlm_7bfp8_eval"
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.txt")

os.makedirs(RESULTS_DIR, exist_ok=True)

load_dotenv()

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")  # vLLM ignores
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "leon-se/ForestFireVLM-7B-FP8-Dynamic")

client = OpenAI(
    base_url=OPENAI_BASE_URL,
    api_key=OPENAI_API_KEY,
)

# -----------------------------
# 1) PROMPT + CLIENT
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


def pil_to_base64_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_json_obj(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in model output.")
    json_str = m.group(0)
    repaired = repair_json(json_str)
    return json.loads(repaired)


def run_vlm_on_image(image: Image.Image) -> Dict:
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
        max_tokens=256,
        temperature=0.0,
    )

    text = response.choices[0].message.content
    print("RAW MODEL OUTPUT:\n", text)
    result = extract_json_obj(text)

    # Normalize optional key name "bbox_2d"
    for key in ("fires", "smoke"):
        if key in result and isinstance(result[key], list):
            for item in result[key]:
                if "bbox" not in item and "bbox_2d" in item:
                    item["bbox"] = item["bbox_2d"]

    return result


# -----------------------------
# 2) DATASET GROUND-TRUTH MAPPING
# -----------------------------
def get_image_from_sample(sample) -> Image.Image:
    """
    ForestFireInsights-Eval is an HF dataset; the authors' repo uses the 'image'
    column. If your local column name differs, change this function.
    """
    img = sample["image"]
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    # HF Image feature returns dict {'bytes': ..., 'path': ...} when decoded=False.
    # If that happens, convert manually:
    if isinstance(img, dict) and "bytes" in img:
        return Image.open(io.BytesIO(img["bytes"])).convert("RGB")
    raise TypeError("Unable to interpret image field; adjust get_image_from_sample.")



def get_smoke_gt(sample) -> int:
    """
    Map dataset ground-truth to a binary label for smoke:
       1 = smoke visible
       0 = no smoke visible
      -1 = unknown / cannot be determined (we skip these samples)

    For leon-se/ForestFireInsights-Eval, the label is stored in:
        sample["gt_dict"]["forest_fire_smoke_visible"]  # "Yes" or "No"
    """
    if "gt_dict" not in sample:
        raise KeyError("sample['gt_dict'] missing; dataset schema changed?")

    gt_dict = sample["gt_dict"]
    if "forest_fire_smoke_visible" not in gt_dict:
        raise KeyError(
            "Key 'forest_fire_smoke_visible' not found in gt_dict; "
            "dataset schema changed?"
        )

    value = gt_dict["forest_fire_smoke_visible"]
    v = str(value).strip().lower()

    if v == "yes":
        return 1
    if v == "no":
        return 0

    # If somehow it's something else, skip this sample
    return -1


# -----------------------------
# 3) METRIC CONTAINER
# -----------------------------
@dataclass
class SmokeStats:
    tp: int = 0
    tn: int = 0
    fp: int = 0
    fn: int = 0


def update_stats(stats: SmokeStats, y_true: int, y_pred: int):
    if y_true == 1 and y_pred == 1:
        stats.tp += 1
    elif y_true == 0 and y_pred == 0:
        stats.tn += 1
    elif y_true == 0 and y_pred == 1:
        stats.fp += 1
    elif y_true == 1 and y_pred == 0:
        stats.fn += 1


# -----------------------------
# 4) MAIN LOOP
# -----------------------------
def main():
    ds = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    print(f"Loaded dataset {DATASET_NAME} split={DATASET_SPLIT} with {len(ds)} samples")

    stats = SmokeStats()
    total_time = 0.0
    num_inferred = 0

    lines = []

    for idx, sample in enumerate(ds):
        image = get_image_from_sample(sample)
        smoke_gt = get_smoke_gt(sample)

        # Skip "cannot be determined" samples if get_smoke_gt returns -1
        if smoke_gt == -1:
            continue

        start = time.perf_counter()
        try:
            result = run_vlm_on_image(image)
        except Exception as e:
            print(f"[ERROR] VLM failed on index {idx}: {e}")
            continue
        end = time.perf_counter()

        infer_time = end - start
        total_time += infer_time
        num_inferred += 1

        smoke_pred_str = str(result.get("SmokePresent", "No")).strip().lower()
        smoke_pred = 1 if smoke_pred_str.startswith("y") else 0

        update_stats(stats, smoke_gt, smoke_pred)

        lines.append(
            f"idx={idx}\tSmokeGT={smoke_gt}\tSmokePred={smoke_pred}"
            f"\tInferenceTimeSec={infer_time:.4f}"
        )
        print(lines[-1])

    # -----------------------------
    # 5) GLOBAL METRICS
    # -----------------------------
    tp, tn, fp, fn = stats.tp, stats.tn, stats.fp, stats.fn
    denom = tp + tn + fp + fn

    accuracy = (tp + tn) / denom if denom > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    avg_time = total_time / num_inferred if num_inferred > 0 else 0.0

    summary_lines = [
        "",
        "===== SUMMARY (Smoke Detection) =====",
        f"Samples evaluated (after skipping 'cannot be determined'): {denom}",
        f"Images actually inferred: {num_inferred}",
        f"TP={tp} TN={tn} FP={fp} FN={fn}",
        f"Accuracy = (TP+TN)/(TP+TN+FP+FN) = {accuracy:.4f}",
        f"Precision = TP/(TP+FP) = {precision:.4f}",
        f"Recall = TP/(TP+FN) = {recall:.4f}",
        f"F1 = 2*Precision*Recall/(Precision+Recall) = {f1:.4f}",
        f"Mean inference time per image = {avg_time:.4f} seconds",
    ]

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        for line in lines + summary_lines:
            f.write(line + "\n")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
