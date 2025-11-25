'''
Terminal 1 run:
    cd Crossfire_detection_codebase/vlm_benchmarking/src/ \
    export MODEL_HANDLE="nvidia/Qwen2.5-VL-7B-Instruct-FP4"  \
    docker run --name trtllm_vlm_server --rm -it --gpus all --ipc host --network host   -e HF_TOKEN="$HF_TOKEN" -e MODEL_HANDLE="$MODEL_HANDLE"   -v $HOME/.cache/huggingface/:/root/.cache/huggingface/   nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev   bash -c 'hf download "$MODEL_HANDLE" && trtllm-serve "$MODEL_HANDLE" --max_batch_size 4 --trust_remote_code --port 8355'

Terminal 2 run:
cd Crossfire_detection_codebase/vlm_benchmarking/src/
source .venv/bin/activate

    or you may have to do:
        python3 -m venv .venv
        source .venv/bin/activate
        python -m pip install --upgrade pip

python3 vlm_auto_openai_vllm_modified.py
'''

import os
import json
import re
import time
import base64
import asyncio
import io
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Robust JSON parsing
try:
    from json_repair import repair_json
except ImportError:
    repair_json = None

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont, ImageOps
from openai import AsyncOpenAI

# -----------------------------
# CONFIGURATION
# -----------------------------
IMAGES_DIR = "../dataset/images"
LABELS_DIR = "../dataset/labels"
OUT_IMAGE_DIR = "../vlm_results/images_modified"
METRICS_PATH = "../vlm_results/metrics_modified.txt"

# --- CRITICAL TUNING ---
TARGET_SIZE = 1024           # Fixed square size for stable batching
BATCH_SIZE = 4               # Async concurrency (pushes 4 images at once)
MAX_OUTPUT_TOKENS = 512      # Increased to allow for the "Description" text
IOU_THRESHOLD = 0.5

os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

load_dotenv()

# CHECK YOUR PORT (Usually 8355 for TRT-LLM)
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:8355/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "dummy-key")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "nvidia/Qwen2.5-VL-7B-Instruct-FP4")

client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=OPENAI_API_KEY)

# -----------------------------
# 1) PRE-PROCESSING (Reasoning & Formatting)
# -----------------------------
def build_fire_prompt() -> str:
    # WE FORCE THE MODEL TO DESCRIBE FIRST ("Reasoning Step")
    # This prevents the model from skipping straight to empty brackets [].
    return """You are a Fire Safety AI.
1. ANALYZE: Look for fire or smoke.
2. DESCRIBE: Write 1 sentence describing the fire/smoke location.
3. DETECT: Output bounding boxes in pixel coordinates.

Output strictly JSON:
{
  "description": "string", 
  "fires": [[x1,y1,x2,y2],...], 
  "smoke": [[x1,y1,x2,y2],...]
}
"""

def process_image_for_model(image_path: str) -> Tuple[str, Image.Image]:
    """
    Forces image to be RGB and padded to a perfect square.
    This fixes the 'Tensor mismatch' error in batching.
    """
    img = Image.open(image_path).convert("RGB") # Drop Alpha channel
    
    # Resize maintaining aspect ratio
    img.thumbnail((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
    
    # Pad to exact square
    delta_w = TARGET_SIZE - img.size[0]
    delta_h = TARGET_SIZE - img.size[1]
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
    new_img = ImageOps.expand(img, padding, fill=(0, 0, 0))
    
    # Encode
    buf = io.BytesIO()
    new_img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    
    return b64, new_img

def extract_json_obj(text: str) -> dict:
    """Robust extraction that finds the JSON block inside the reasoning text."""
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match: match = re.search(r"\{.*", text, re.DOTALL)
    
    json_str = match.group(0) if match else text
    
    try:
        return json.loads(json_str)
    except:
        if repair_json:
            try: return json.loads(repair_json(json_str))
            except: pass
        return {"fires": [], "smoke": []}

# -----------------------------
# 2) ASYNC INFERENCE
# -----------------------------
async def process_single_image(file_path: str, semaphore: asyncio.Semaphore) -> Dict:
    async with semaphore:
        try:
            b64, processed_img = process_image_for_model(file_path)
            prompt = build_fire_prompt()

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": prompt}
                ]
            }]

            start = time.perf_counter()
            response = await client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                max_tokens=MAX_OUTPUT_TOKENS,
                temperature=0.4,  # Raised temp to encourage detection
                stop=["```"]
            )
            duration = time.perf_counter() - start
            
            text = response.choices[0].message.content
            res = extract_json_obj(text)
            
            # --- DEBUG: CHECK IF REASONING IS WORKING ---
            # If this prints a description, the logic is fixed.
            if "WEB09236" in file_path: 
                print(f"\n[DEBUG WEB09236] Model Thought: {res.get('description', 'No desc')}")
                print(f"[DEBUG WEB09236] Raw: {text[:100]}...\n")
            
            return {
                "file_path": file_path,
                "duration": duration,
                "result": res,
                "img_obj": processed_img
            }
        except Exception as e:
            print(f"❌ Error on {os.path.basename(file_path)}: {e}")
            return None

async def main_async():
    files = sorted([os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) if f.lower().endswith(".jpg")])
    if not files: print("No images."); return

    semaphore = asyncio.Semaphore(BATCH_SIZE)
    tasks = [process_single_image(f, semaphore) for f in files]
    
    print(f"🚀 Starting ASYNC inference on {len(files)} images (Batch: {BATCH_SIZE})...")
    print(f"ℹ️  Strategy: Force 1024px Square + Chain-of-Thought Description")
    
    total_start = time.perf_counter()
    results = await asyncio.gather(*tasks)
    total_duration = time.perf_counter() - total_start
    
    valid_results = [r for r in results if r is not None]
    
    # --- RESULTS & SAVING ---
    print("💾 Saving results...")
    lines_out = []
    
    for item in valid_results:
        path = item['file_path']
        res = item['result']
        img = item['img_obj']
        
        # Clean boxes
        def clean(lst):
            out = []
            for b in lst:
                if isinstance(b, list) and len(b)==4: out.append(b)
            return out
        
        fires = clean(res.get("fires", []))
        smoke = clean(res.get("smoke", []))
        
        draw = ImageDraw.Draw(img)
        for b in fires: draw.rectangle(b, outline="red", width=4)
        for b in smoke: draw.rectangle(b, outline="blue", width=4)
        
        stem = os.path.splitext(os.path.basename(path))[0]
        img.save(os.path.join(OUT_IMAGE_DIR, f"{stem}_out.jpg"))
        
        desc = res.get("description", "N/A")[:30] # Log first 30 chars of desc
        log = f"{os.path.basename(path)} | {item['duration']:.2f}s | F:{len(fires)} S:{len(smoke)} | Desc: {desc}..."
        lines_out.append(log)
        print(log)

    avg_time = total_duration / len(valid_results) if valid_results else 0
    print(f"\n===== STATS =====")
    print(f"Total Wall Time: {total_duration:.2f}s")
    print(f"Effective Avg Time: {avg_time:.4f}s")
    print(f"Target: 1.0s - 1.5s (Reasoning adds ~0.5s latency)")
    
    with open(METRICS_PATH, "w") as f:
        f.write("\n".join(lines_out))

if __name__ == "__main__":
    asyncio.run(main_async())