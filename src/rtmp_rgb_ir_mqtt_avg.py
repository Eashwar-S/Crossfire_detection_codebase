# October 4th 11:56 PM
# GUI link: http://192.168.200.55:8000/
#!/usr/bin/env python3
import os
import time
import cv2
import argparse
import numpy as np
from ultralytics import YOLO

# --------- Flask imports ----------- live view link: http://192.168.200.55:8000/
from flask import Flask, Response, send_from_directory, jsonify
import threading

# --------- Flask creation -----------
app = Flask(__name__, static_folder='.')
frame_lock = threading.Lock()
latest_ir_frame = None 
latest_rgb_frame = None
global status_message
status_message = "System initialized."

# ---------- Stream & FFmpeg backend ----------
URL = "rtmp://192.168.200.55/live/ir"

# --------- Flask http routes -----------
@app.route('/status_text')
def status_text():
    pos = {
        "air_lat": _latest.get("air_lat"),
        "air_lon": _latest.get("air_lon"),
        "air_alt": _latest.get("air_h"),
        "heading": _latest.get("head"),
        "pitch": _latest.get("pitch"),
        "roll": _latest.get("roll"),
        "lrf_lat": _latest.get("lrf_lat"),
        "lrf_lon": _latest.get("lrf_lon"),
        "lrf_alt": _latest.get("lrf_alt"),
        "lrf_dist": _latest.get("lrf_dist"),
        "ts": _latest.get("ts")
    }
    return jsonify({"text": status_message, "position": pos})

@app.route('/ir_feed')
def ir_feed():
    return Response(_mjpeg_generator(lambda: latest_ir_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rgb_feed')
def rgb_feed():
    return Response(_mjpeg_generator(lambda: latest_rgb_frame),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# >>>>>>>>>>>>>>>>>>>>>>>>>> ADDED: MQTT (integrated) <<<<<<<<<<<<<<<<<<<<<<<<<<
import json, threading
from datetime import datetime
from importlib.metadata import version as _paho_version
import paho
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
load_dotenv()

MQTT_HOST = os.getenv("MQTT_HOST") or os.getenv("HOST_ADDR", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")
# We DO NOT use SN filtering per your request.

# cache latest LRF target + aircraft pose from MQTT
_latest = {
    "ts": None,                # milliseconds (if provided) or None
    "air_lat": None, "air_lon": None, "air_h": None,
    "head": None, "pitch": None, "roll": None,
    "lrf_lat": None, "lrf_lon": None, "lrf_alt": None, "lrf_dist": None
}

_DETECTIONS_LOG = "detections.txt"
_AVG_LOG        = "detections_avg.txt"
_WAVG_LOG       = "detections_wavg.txt"

# Running aggregates (for LRF target)
_avg_acc  = {"n": 0, "sum_lat": 0.0, "sum_lon": 0.0}
_wavg_acc = {"sum_w": 0.0, "sum_w_lat": 0.0, "sum_w_lon": 0.0}

# Flask helper function
def _mjpeg_generator(get_frame_fn, jpeg_quality=80, target_sleep=0.03):
    """Yield multipart JPEG frames from the current numpy BGR frame returned by get_frame_fn()."""
    while True:
        with frame_lock:
            frame = get_frame_fn()
        if frame is None:
            time.sleep(0.01)
            continue
        ok, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            time.sleep(0.01)
            continue
        jpg_bytes = jpg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpg_bytes + b'\r\n')
        time.sleep(target_sleep)

def _iso_ts(ts_ms):
    if isinstance(ts_ms, (int, float)):
        return datetime.fromtimestamp(ts_ms / 1000.0).isoformat()
    return datetime.utcnow().isoformat()

def _maybe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def _extract_from_osd_message(m: dict):
    """
    Supports both shapes:
      A) {"data": {"host": {...}}, "timestamp": ...}
      B) {"host": {...}, "timestamp": ...}
    Within host, we look for:
      - Aircraft fields: latitude, longitude, height, attitude_head/pitch/roll
      - LRF fields under any child dict with keys measure_target_*
    """
    if not isinstance(m, dict):
        return

    # timestamp (ms)
    ts = m.get("timestamp")
    ts = _maybe_float(ts)

    # resolve 'host' node
    root = m.get("data", m)
    host = root.get("host") if isinstance(root, dict) else None
    if not isinstance(host, dict):
        # Some payloads may not have host; nothing to do
        if ts is not None:
            _latest["ts"] = ts
        return

    # aircraft pose (if provided at host level)
    air_lat = _maybe_float(host.get("latitude"))
    air_lon = _maybe_float(host.get("longitude"))
    air_h   = _maybe_float(host.get("height"))
    head    = _maybe_float(host.get("attitude_head"))
    pitch   = _maybe_float(host.get("attitude_pitch"))
    roll    = _maybe_float(host.get("attitude_roll"))

    # LRF block: under host there can be payload-indexed dicts such as "99-0-0"
    lrf_lat = lrf_lon = lrf_alt = lrf_dist = None
    for k, v in host.items():
        if not isinstance(v, dict):
            continue
        # Look for measure_target_* keys
        if "measure_target_latitude" in v or "measure_target_distance" in v:
            lrf_lat = _maybe_float(v.get("measure_target_latitude"))
            lrf_lon = _maybe_float(v.get("measure_target_longitude"))
            lrf_alt = _maybe_float(v.get("measure_target_altitude"))
            lrf_dist = _maybe_float(v.get("measure_target_distance"))
            # Prefer first block that contains these; break
            break

    # Update cache
    _latest["ts"] = ts if ts is not None else _latest["ts"]
    _latest["air_lat"] = air_lat if air_lat is not None else _latest["air_lat"]
    _latest["air_lon"] = air_lon if air_lon is not None else _latest["air_lon"]
    _latest["air_h"]   = air_h   if air_h   is not None else _latest["air_h"]
    _latest["head"]    = head    if head    is not None else _latest["head"]
    _latest["pitch"]   = pitch   if pitch   is not None else _latest["pitch"]
    _latest["roll"]    = roll    if roll    is not None else _latest["roll"]

    _latest["lrf_lat"]  = lrf_lat  if lrf_lat  is not None else _latest["lrf_lat"]
    _latest["lrf_lon"]  = lrf_lon  if lrf_lon  is not None else _latest["lrf_lon"]
    _latest["lrf_alt"]  = lrf_alt  if lrf_alt  is not None else _latest["lrf_alt"]
    _latest["lrf_dist"] = lrf_dist if lrf_dist is not None else _latest["lrf_dist"]

# def calcluatecoords(m: dict):
#     ts = m.get("timestamp")
#     ts = _maybe_float(ts)

#     # resolve 'host' node
#     root = m.get("data", m)
#     host = root.get("host") if isinstance(root, dict) else None
#     if not isinstance(host, dict):
#         # Some payloads may not have host; nothing to do
#         if ts is not None:
#             _latest["ts"] = ts
#         return

#     # aircraft pose (if provided at host level)
#     air_lat = _maybe_float(host.get("latitude"))
#     air_lon = _maybe_float(host.get("longitude"))
#     air_h   = _maybe_float(host.get("height"))
#     head    = _maybe_float(host.get("attitude_head"))
#     pitch   = _maybe_float(host.get("attitude_pitch"))
#     roll    = _maybe_float(host.get("attitude_roll"))
#     px = int(host.get["cameras"][0]["ir_metering_point"]["x"] * 640)
#     py = int(host.get["cameras"][0]["ir_metering_point"]["y"] * 512)
#     flr_lat = _maybe_float(host.get("measure_target_latitude")) 
#     flr_lon = _maybe_float(host.get("measure_target_longitude")) 
#     status_message = f"pixels: {px} {py}"
#     R_earth = 6378137.0
#     px_center_x = 320
#     px_center_y = 256
#     theta = 45

#     dx = px - px_center_x
#     dy = py - px_center_y

#     # keep same small-angle pinhole approximation used in your working code
#     fov_x, fov_y = 0.6, 0.45
#     alpha_x = dx / px_center_x * (fov_x / 2)
#     alpha_y = dy / px_center_y * (fov_y / 2)

#     # replicate the math style from your example code (theta in degrees + alpha in radians was used there)
#     # to preserve the exact behavior you said works in your system
#     theta_total = theta + alpha_y
#     d_forward = air_h * np.tan(-theta_total)
#     d_side = air_h * np.tan(alpha_x)

#     heading_rad = np.deg2rad(head)
#     north_disp = d_forward * np.cos(heading_rad) - d_side * np.sin(heading_rad)
#     east_disp  = d_forward * np.sin(heading_rad) + d_side * np.cos(heading_rad)

#     dlat = north_disp / R_earth * (180/np.pi)
#     dlon = east_disp / (R_earth * np.cos(np.deg2rad(Lat0))) * (180/np.pi)

#     lat = flr_lat + dlat
#     lon = flr_lon + dlon

#     status_message = f"Estimated location: {lat}, {lon}"
#     return lat, lon



def on_connect(client, userdata, flags, rc_or_reason, properties=None):
    try:
        rc_val = rc_or_reason.value
        rc_text = rc_or_reason.getName()
    except Exception:
        rc_val = rc_or_reason
        rc_text = str(rc_or_reason)
    print(f"[MQTT] Connected to {MQTT_HOST}:{MQTT_PORT} rc={rc_text}")
    if rc_val == 0:
        client.subscribe("thing/product/+/osd")
        client.subscribe("thing/product/+/status")
        print("[MQTT] Subscribed to: thing/product/+/osd, thing/product/+/status")

def on_message(client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
    try:
        payload = msg.payload.decode("utf-8")
        m = json.loads(payload)
    except Exception as e:
        print(f"[MQTT] {msg.topic} decode error: {e}")
        return

    if msg.topic.endswith("/osd"):
        _extract_from_osd_message(m)
        # Debug print so you can see when LRF updates arrive
        if _latest["lrf_lat"] is not None and _latest["lrf_lon"] is not None:
            print(f"[MQTT LRF] lat={_latest['lrf_lat']}, lon={_latest['lrf_lon']}, dist={_latest['lrf_dist']}")
    elif msg.topic.endswith("/status") and m.get("method") == "update_topo":
        reply = {
            "tid": m.get("tid"),
            "bid": m.get("bid"),
            "timestamp": (m.get("timestamp") or 0) + 2,
            "data": {"result": 0},
        }
        client.publish(msg.topic + "_reply", json.dumps(reply))

def client_factory():
    main_ver = int(_paho_version("paho-mqtt").split(".")[0])
    if main_ver == 1:
        c = mqtt.Client(transport="tcp")
    else:
        c = mqtt.Client(paho.mqtt.enums.CallbackAPIVersion.VERSION2, transport="tcp")
    c.on_connect = on_connect
    c.on_message = on_message
    if MQTT_USERNAME or MQTT_PASSWORD:
        c.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    return c

def start_mqtt_nonblocking():
    c = client_factory()
    print(f"[MQTT] Connecting to {MQTT_HOST}:{MQTT_PORT} ...")
    c.connect(MQTT_HOST, MQTT_PORT, 60)
    t = threading.Thread(target=c.loop_forever, daemon=True)
    t.start()
    return c

# ---------- helpers for IR/RGB detection & logging ----------
def _boxes_from_mask(mask: np.ndarray, min_area: int):
    boxes = []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        boxes.append((x, y, x + w, y + h))
    return boxes

def _intersects(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0, min(ay2, by2) - max(ay1, by1))
    return (iw > 0) and (ih > 0)

def _get_yolo_boxes(model: YOLO, frame_bgr: np.ndarray, imgsz: int, conf: float, classes, rect: bool, device):
    kwargs = dict(source=frame_bgr, imgsz=imgsz, conf=conf,
                  classes=classes, rect=rect, verbose=False, agnostic_nms=False)
    if device is not None:
        try:
            kwargs["device"] = device
        except Exception:
            pass
    r = model.predict(**kwargs)[0]
    boxes = []
    if getattr(r, "boxes", None) is not None and getattr(r.boxes, "xyxy", None) is not None:
        try:
            for x1, y1, x2, y2, *_ in r.boxes.xyxy.cpu().numpy():
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        except Exception:
            pass
    return boxes

def _weight_from_center(yolo_boxes, rgb_w, rgb_h):
    if not yolo_boxes:
        return 0.0
    cx0, cy0 = rgb_w * 0.5, rgb_h * 0.5
    diag = (rgb_w**2 + rgb_h**2) ** 0.5
    best_w = 0.0
    for (x1, y1, x2, y2) in yolo_boxes:
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        d = ((cx - cx0)**2 + (cy - cy0)**2) ** 0.5
        d_norm = d / max(1e-6, diag)
        w = 1.0 / (1e-6 + d_norm)
        if w > best_w:
            best_w = w
    return min(best_w, 1e6)

def _update_running_average(lat: float, lon: float):
    if lat is None or lon is None:
        return None
    _avg_acc["n"] += 1
    _avg_acc["sum_lat"] += lat
    _avg_acc["sum_lon"] += lon
    return (_avg_acc["sum_lat"] / _avg_acc["n"], _avg_acc["sum_lon"] / _avg_acc["n"])

def _update_weighted_average(lat: float, lon: float, yolo_boxes, rgb_w: int, rgb_h: int):
    if lat is None or lon is None:
        return None
    w = _weight_from_center(yolo_boxes, rgb_w, rgb_h)
    if w <= 0.0:
        return None
    _wavg_acc["sum_w"] += w
    _wavg_acc["sum_w_lat"] += w * lat
    _wavg_acc["sum_w_lon"] += w * lon
    return (_wavg_acc["sum_w_lat"] / _wavg_acc["sum_w"],
            _wavg_acc["sum_w_lon"] / _wavg_acc["sum_w"])

def _log_detection(frame_id: int):
    # Write LRF target values (required)
    ts_str = _iso_ts(_latest["ts"])
    line = f"{ts_str},{frame_id},{_latest['lrf_lat']},{_latest['lrf_lon']},{_latest['lrf_dist']}\n"
    try:
        with open(_DETECTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG] Failed to write to '{_DETECTIONS_LOG}': {e}")

def _log_avg(frame_id: int, avg_lat: float, avg_lon: float):
    ts_str = _iso_ts(_latest["ts"])
    line = f"{ts_str},{frame_id},{avg_lat},{avg_lon}\n"
    try:
        with open(_AVG_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG] Failed to write to '{_AVG_LOG}': {e}")

def _log_wavg(frame_id: int, wavg_lat: float, wavg_lon: float):
    ts_str = _iso_ts(_latest["ts"])
    line = f"{ts_str},{frame_id},{wavg_lat},{wavg_lon}\n"
    try:
        with open(_WAVG_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG] Failed to write to '{_WAVG_LOG}': {e}")

# <<<<<<<<<<<<<<<<<<<<<<< END ADDED: MQTT <<<<<<<<<<<<<<<<<<<<<<<<<<

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "probesize;500000|analyzeduration;500000|fflags;discardcorrupt|rtmp_live;live|"
    "rw_timeout;2000000|stimeout;2000000"
)

def open_cap(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    return cap

def wait_for_first_frame(cap, timeout_s=4.0):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            return frame
        time.sleep(0.01)
    return None

def split_halves(frame: np.ndarray):
    h, w = frame.shape[:2]
    mid = w // 2
    left  = frame[:, :mid]
    right = frame[:, mid:]
    return left, right

# ---------- IR helpers ----------
H_DEFAULT = np.array([
    [7.84539671e-01, 3.23869026e-02, 3.66465224e+02],
    [2.86785918e-02, 7.60677032e-01, 2.69386770e+02],
    [4.78052584e-05, 3.51503005e-05, 1.00000000e+00]
], dtype=np.float64)

def ir_hsv_mask(bgr: np.ndarray, lower: tuple, upper: tuple) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    mask = cv2.inRange(hsv, lo, hi)
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.dilate(mask, k, iterations=1)
    return mask

def draw_boxes_from_mask(bgr: np.ndarray, mask: np.ndarray, min_area: int) -> np.ndarray:
    out = bgr
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(out, "fire", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
    return out

def maybe_warp_ir(ir_bgr: np.ndarray, use_warp: bool, H: np.ndarray, dst_w: int, dst_h: int,
                  src_w: int, src_h: int) -> np.ndarray:
    if not use_warp:
        return ir_bgr
    if ir_bgr.shape[1] != src_w or ir_bgr.shape[0] != src_h:
        ir_bgr = cv2.resize(ir_bgr, (src_w, src_h), interpolation=cv2.INTER_LINEAR)
    warped = cv2.warpPerspective(
        ir_bgr, H, (dst_w, dst_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return warped

# ---------- RGB YOLO ----------
def load_yolo(weights: str, imgsz: int, is_pt_hint: bool):
    ext = os.path.splitext(weights)[1].lower()
    task = "detect" if ext == ".onnx" else None
    model = YOLO(weights, task=task)
    if is_pt_hint and ext == ".pt":
        try:
            model.fuse()
        except Exception:
            pass
    return model

def run_yolo(model: YOLO, frame_bgr: np.ndarray, imgsz: int, conf: float,
             classes, rect: bool, device):
    predict_kwargs = dict(source=frame_bgr, imgsz=imgsz, conf=conf,
                          classes=classes, rect=rect, verbose=False, agnostic_nms=False)
    if device is not None:
        try:
            predict_kwargs["device"] = device
        except Exception:
            pass
    results = model.predict(**predict_kwargs)
    annotated = results[0].plot()
    return annotated

def _get_yolo_boxes_dup(model: YOLO, frame_bgr: np.ndarray, imgsz: int, conf: float,
                        classes, rect: bool, device):
    # (kept to avoid changing your previous structure elsewhere if referenced)
    return _get_yolo_boxes(model, frame_bgr, imgsz, conf, classes, rect, device)

# ---------- Main ----------
def main():
    global status_message
    ## making Flask variables global and starting Flask thread
    global latest_ir_frame, latest_rgb_frame
    def _start_flask_server():
        app.run(host="0.0.0.0", port=8000, threaded=True, debug=False)

    # start flask in background
    flask_thread = threading.Thread(target=_start_flask_server, daemon=True)
    flask_thread.start()

    ap = argparse.ArgumentParser(description="RTMP split (IR|RGB) with YOLO (RGB) + IR HSV thresholding + MQTT LRF logging")
    # stream + skipping
    ap.add_argument("--url", default=URL, help="RTMP/HTTP(S) URL of composite stream (left=IR, right=RGB)")
    ap.add_argument("--skip", type=int, default=0, help="Frames to discard between processed frames")
    # YOLO (RGB)
    # ap.add_argument("--weights", default="yolo11m.onnx", help="YOLO weights (.onnx/.pt/.engine)")
    ap.add_argument("--weights", default="10-8K-100e-n.onnx", help="YOLO weights (.onnx/.pt/.engine)")
    ap.add_argument("--conf", type=float, default=0.50, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (multiple of 32)")
    ap.add_argument("--rect", action="store_true", help="Minimal padding to match stride (faster on .onnx)")
    ap.add_argument("--classes", type=int, nargs="*", default=None, help="Optional class filter, e.g. 0 1")
    ap.add_argument("--device", default=None, help="Torch device for .pt (e.g. 0 or 'cpu'); ignored for .onnx")
    # IR thresholds
    ap.add_argument("--ir-hsv-lower", type=str, default="12,42,39", help="HSV lower (H[0-179],S,V) for IR")
    ap.add_argument("--ir-hsv-upper", type=str, default="166,255,255", help="HSV upper (H[0-179],S,V) for IR")
    ap.add_argument("--ir-min-area", type=int, default=200, help="Min contour area for IR boxes")
    # optional homography alignment for IR
    ap.add_argument("--ir-warp", action="store_true", help="Apply homography to IR before thresholding")
    ap.add_argument("--H", type=float, nargs=9, default=H_DEFAULT.flatten().tolist(),
                    help="3x3 homography row-major for IR warp")
    ap.add_argument("--src-wh", type=int, nargs=2, default=[1280, 720], help="IR src size expected before warp")
    ap.add_argument("--dst-wh", type=int, nargs=2, default=[1280, 720], help="IR warp destination size")
    # windows & saving
    ap.add_argument("--save-dir", default="captures", help="Directory to save snapshots")
    args = ap.parse_args()

    # Start MQTT listener (non-blocking)
    _ = start_mqtt_nonblocking()

    def _triple(s):
        vals = [int(x) for x in s.split(",")]
        if len(vals) != 3:
            raise ValueError("HSV triple must be H,S,V")
        return tuple(vals)
    hsv_lo = _triple(args.ir_hsv_lower)
    hsv_hi = _triple(args.ir_hsv_upper)

    model = load_yolo(args.weights, args.imgsz, is_pt_hint=True)

    cap = open_cap(args.url)
    if not cap.isOpened():
        print("ERROR: Could not open stream. Check network/RTMP publisher.")
        return

    first = wait_for_first_frame(cap, timeout_s=4.0)
    if first is None:
        print("ERROR: No decodable frames arrived; try increasing probesize/analyzeduration.")
        cap.release()
        return

    # cv2.namedWindow("IR (annotated)",  cv2.WINDOW_NORMAL)
    # cv2.namedWindow("RGB (YOLO)",      cv2.WINDOW_NORMAL)
    os.makedirs(args.save_dir, exist_ok=True)
    idx = 0
    print(f"Press 's' to save both halves, 'q' to quit. Skipping {args.skip} frames.")

    H_mat   = np.array(args.H, dtype=np.float64).reshape(3, 3)
    SRC_W, SRC_H = args.src_wh
    DST_W, DST_H = args.dst_wh
    last_info = 0.0
    frame_id = 0

    while True:
        # Read frame from RTMP
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        # Split composite frame into IR and RGB
        ir_left, rgb_right = split_halves(frame)

        # ---------- IR detection ----------
        ir_mask = ir_hsv_mask(ir_left, hsv_lo, hsv_hi)
        ir_anno = draw_boxes_from_mask(ir_left.copy(), ir_mask, args.ir_min_area)

        # Map IR mask onto RGB space for overlap heuristic
        rgb_h, rgb_w = rgb_right.shape[:2]
        ir_mask_on_rgb = cv2.resize(ir_mask, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        ir_boxes_on_rgb = _boxes_from_mask(ir_mask_on_rgb, args.ir_min_area)

        # ---------- Conditional YOLO ----------
        if len(ir_boxes_on_rgb) > 0:  # IR threshold met
            rgb_anno = run_yolo(model, rgb_right, imgsz=args.imgsz, conf=args.conf,
                                classes=args.classes, rect=args.rect, device=args.device)
            yolo_boxes = _get_yolo_boxes(model, rgb_right, imgsz=args.imgsz, conf=args.conf,
                                            classes=args.classes, rect=args.rect, device=args.device)

            # Check intersection for logging
            has_intersection = False
            for ib in ir_boxes_on_rgb:
                for yb in yolo_boxes:
                    if _intersects(ib, yb):
                        has_intersection = True
                        break
                if has_intersection:
                    break

            if has_intersection and _latest["lrf_lat"] is not None and _latest["lrf_lon"] is not None:
                print(f"----> INTERSECTION + LRF target at frame {frame_id} -> logging to detections.txt")
                _log_detection(frame_id)
                avg = _update_running_average(_latest["lrf_lat"], _latest["lrf_lon"])
                if avg is not None:
                    _log_avg(frame_id, avg[0], avg[1])
                wavg = _update_weighted_average(_latest["lrf_lat"], _latest["lrf_lon"], yolo_boxes, rgb_w, rgb_h)
                if wavg is not None:
                    _log_wavg(frame_id, wavg[0], wavg[1])
        else:
            # IR threshold not met → skip YOLO, send raw RGB
            rgb_anno = rgb_right.copy()
            yolo_boxes = []

        # ---------- Update Flask streams ----------
        with frame_lock:
            latest_ir_frame = ir_anno.copy()
            latest_rgb_frame = rgb_anno.copy()

        # Debug print
        print(f"Frame {frame_id}: IR boxes={len(ir_boxes_on_rgb)}, YOLO boxes={len(yolo_boxes)} | LRF ready={_latest['lrf_lat'] is not None}")

        # ---------- Save frames on keypress ----------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            ir_path  = os.path.join(args.save_dir, f"ir_{idx:06d}.png")
            rgb_path = os.path.join(args.save_dir, f"rgb_{idx:06d}.png")
            cv2.imwrite(ir_path, ir_anno)
            cv2.imwrite(rgb_path, rgb_anno)
            print(f"Saved {ir_path} and {rgb_path}")
            idx += 1

        frame_id += 1
        lrf_lat = _latest.get("lrf_lat")
        lrf_lon = _latest.get("lrf_lon")
        status_message = f"Coordinates: {lrf_lat} {lrf_lon}"


    cap.release()

if __name__ == "__main__":
    main()
