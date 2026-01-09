import os
import json
import re
import base64
import time

# [ADDED] Imports for drawing
from PIL import Image, ImageDraw, ImageFont
from vllm import LLM, SamplingParams

# -----------------------------
# 1) CONFIGURATION
# -----------------------------

IMAGES_DIR = "/dataset/images"
OUTPUT_FILE = "/workspace/results.jsonl"
MODEL_PATH = "leon-se/ForestFireVLM-7B-FP8-Dynamic"

# [ADDED] Output config for images
OUT_IMAGE_DIR = "/workspace/vlm_7B_results/images"
# Ensure the directory exists
os.makedirs(OUT_IMAGE_DIR, exist_ok=True)

# CONFIG FOR VOTING
NUM_SAMPLES = 5  # Number of times to query the model per image
TEMPERATURE = 0.1 # Higher temp needed for diversity

# -----------------------------
# 2) PROMPT + HELPERS
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
# 3) MAIN
# -----------------------------

def main():
    print(f"Loading model: {MODEL_PATH}...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="float16",
        limit_mm_per_prompt={"image": 8},
    )

    print(f"Scanning images in {IMAGES_DIR}...")
    if not os.path.exists(IMAGES_DIR):
        print(f"ERROR: {IMAGES_DIR} does not exist. Check your Docker mounts.")
        return

    image_files = [
        f for f in os.listdir(IMAGES_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    image_files.sort()

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        n=NUM_SAMPLES,
        max_tokens=256,
    )

    question_fields = [
        "Smoke", "Flames", "Uncontrolled", "FireState", "FireType",
        "FireIntensity", "FireSize", "FireHotspots",
        "InfrastructureNearby", "PeopleNearby", "TreeVitality",
    ]

    total_time = 0.0
    num_done = 0
    BATCH_SIZE = 1

    print(f"Processing {len(image_files)} images in batches of {BATCH_SIZE}...")
    with open(OUTPUT_FILE, "w") as f_out:
        for start_idx in range(0, len(image_files), BATCH_SIZE):
            batch_files = image_files[start_idx:start_idx + BATCH_SIZE]
            messages_batch = []
            meta_batch = []

            for img_file in batch_files:
                img_path = os.path.join(IMAGES_DIR, img_file)
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception as e:
                    print(f"Skipping {img_file}: {e}")
                    continue

                img_w, img_h = image.size
                prompt_text = build_fire_prompt(img_w, img_h)
                b64_img = encode_image_to_base64(img_path)
                data_url = f"data:image/jpeg;base64,{b64_img}"

                msg = {
                    "role": "user",
                    "parts": [
                        {"text": prompt_text},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64_img}},
                    ],
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }

                messages_batch.append(msg)
                # [MODIFIED] Store original image object in metadata so we can draw on it later
                meta_batch.append({"file": img_file, "orig_image": image})

            if not messages_batch:
                continue

            start_t = time.perf_counter()
            try:
                outputs = llm.chat(messages_batch, sampling_params)
            except Exception as e:
                print(f"Model error on batch starting at {batch_files[0]}: {e}")
                continue
            end_t = time.perf_counter()

            batch_time = end_t - start_t
            per_image_time = batch_time / len(outputs)
            total_time += batch_time
            num_done += len(outputs)

            for meta, out in zip(meta_batch, outputs):
                img_file = meta["file"]
                # We use the first sample for drawing/saving
                generated_text = out.outputs[0].text

                result = extract_json_obj(generated_text)

                print(f"=== Result for: {img_file} ===")
                print(f"InferenceTimeSec (approx): {per_image_time:.3f}")
                if "error" in result:
                    print(f"Error parsing JSON: {result.get('raw_output')}")
                else:
                    for field in question_fields:
                        val = result.get(field, "N/A")
                        print(f"{field}: {val}")

                    fires = result.get("fires", [])
                    smoke = result.get("smoke", [])
                    print(f"Fire bboxes: {len(fires) if isinstance(fires, list) else 0}")
                    print(f"Smoke bboxes: {len(smoke) if isinstance(smoke, list) else 0}")

                    # ----------------------------------------------------
                    # [ADDED] Draw and Save Image Logic
                    # ----------------------------------------------------
                    try:
                        orig_image = meta["orig_image"]
                        out_img = orig_image.copy()
                        draw = ImageDraw.Draw(out_img)

                        try:
                            # Try to load a nice font, fallback to default
                            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
                        except OSError:
                            font = ImageFont.load_default()

                        # Draw Fire (Red)
                        if isinstance(fires, list):
                            for item in fires:
                                # Handle both straight list [x,y,x,y] or dict wrapper {"bbox": ...}
                                bbox = item.get("bbox") if isinstance(item, dict) else item
                                if isinstance(bbox, list) and len(bbox) == 4:
                                    draw.rectangle(bbox, outline="red", width=3)
                                    draw.text((bbox[0], bbox[1]), "Fire", fill="red", font=font)

                        # Draw Smoke (Blue)
                        if isinstance(smoke, list):
                            for item in smoke:
                                bbox = item.get("bbox") if isinstance(item, dict) else item
                                if isinstance(bbox, list) and len(bbox) == 4:
                                    draw.rectangle(bbox, outline="blue", width=3)
                                    draw.text((bbox[0], bbox[1]), "Smoke", fill="blue", font=font)

                        # Save to OUT_IMAGE_DIR
                        # Note: img_file usually includes extension (e.g. "test.jpg")
                        out_name = f"{os.path.splitext(img_file)[0]}_vlm_output.jpg"
                        out_path = os.path.join(OUT_IMAGE_DIR, out_name)
                        out_img.save(out_path)
                        print(f"Saved annotated image: {out_path}")

                    except Exception as e:
                        print(f"Error drawing/saving image for {img_file}: {e}")
                    # ----------------------------------------------------

                print("=====================================")

                final_record = {
                    "file": img_file,
                    "inference_time_sec": per_image_time,
                    "result": result,
                }
                f_out.write(json.dumps(final_record) + "\n")

    if num_done > 0:
        avg_time = total_time / num_done
        print(f"\nAverage (approx) inference time: {avg_time:.3f} sec over {num_done} images")
    
    print(f"Done! Results saved to {OUTPUT_FILE}")
    print(f"Annotated images saved to {OUT_IMAGE_DIR}")

if __name__ == "__main__":
    main()