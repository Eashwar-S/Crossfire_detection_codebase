#!/usr/bin/env python3
import os
import time
import cv2
import argparse
import numpy as np
import math
from ultralytics import YOLO


# >>>>>>>>>>>>>>>>>>>>>>>>>> MQTT (integrated) <<<<<<<<<<<<<<<<<<<<<<<<<<
import json, threading
from datetime import datetime
from importlib.metadata import version as _paho_version
import paho
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from typing import Tuple
from flask import Flask, Response, send_from_directory, jsonify
load_dotenv()

# ---------- Stream & FFmpeg backend ----------
URL = "rtmp://192.168.200.55/live/ir"

# --------- Flask creation -----------
app = Flask(__name__, static_folder='.')
frame_lock = threading.Lock()
latest_ir_frame = None 
latest_rgb_frame = None
global status_message
global firedetected
status_message = "System initialized."


MQTT_HOST = os.getenv("MQTT_HOST") or os.getenv("HOST_ADDR", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
MQTT_USERNAME = os.getenv("MQTT_USERNAME")
MQTT_PASSWORD = os.getenv("MQTT_PASSWORD")

# cache latest LRF target + aircraft pose from MQTT
_latest = {
    "ts": None,                # milliseconds (if provided) or None
    "air_lat": None, "air_lon": None, "air_h": None,
    "head": None, "pitch": None, "roll": None,
    "lrf_lat": None, "lrf_lon": None, "lrf_alt": None, "lrf_dist": None,
    "gimbal_yaw": None, "gimbal_pitch": None, "fire": False, "fire_lat": None, "fire_lon": None
}

_DETECTIONS_LOG = "detections.txt"
_AVG_LOG        = "detections_avg.txt"
_WAVG_LOG       = "detections_wavg.txt"

# BBOX GEO logs & accumulators
_DETECTIONS_BBOX_LOG = "detections_bbox.txt"
_bbox_avg = {"n": 0, "sum_lat": 0.0, "sum_lon": 0.0}

# clustered outputs
_CLUSTERED_AVG_LOG  = "detections_avg_clustered.txt"
_CLUSTERED_WAVG_LOG = "detections_wavg_clustered.txt"
_CENTROID_LAT_LONG_DETECTIONS_LOG = "dectections_centroid_lat_long.txt"

# Running aggregates (for LRF target)
_avg_acc  = {"n": 0, "sum_lat": 0.0, "sum_lon": 0.0}
_wavg_acc = {"sum_w": 0.0, "sum_w_lat": 0.0, "sum_w_lon": 0.0}


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

    coords = {
        # "air_lat": _latest.get("lrf_lat"),
        # "air_lon": _latest.get("lrf_lon")
        "air_lat": _latest["air_lat"],
        "air_lon": _latest["air_lon"]
    }

    firecoords = {
        "fire_lat": _latest["fire_lat"],
        "fire_lon":_latest["fire_lon"]

    }
    head = _latest["head"]
    fire = _latest["fire"]

    return jsonify({"ts": ts, "text": status_message, "Heading": head, "coordinates": coords, "fire": fire, "firecoords": firecoords})

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
    
def bearing_deg(lat1, lon1, lat2, lon2):
    global status_message
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1)*math.sin(φ2) - math.sin(φ1)*math.cos(φ2)*math.cos(Δλ)
    θ = math.atan2(y, x)
    headf = (math.degrees(θ) + 360) % 360
    # lat_t = _latest["air_lat"]
    # lon_l = _latest["air_lon"]
    # status_message = f"Heading: {headf}, Coordinates: {lat_t}, {lon_l}"
    return headf  # 0=North, clockwise


def calculatecoords(xPixVal, yPixVal, height, pitch):
    global status_message
    # Constants
    dFOV = 54.143 * math.pi / 180 # degrees, 54.143 for 1.7x zoom, 82 for 1x zoom
    xPixMax = 720 
    yPixMax = 640
    # height = 35 # Put height of the drone during the flight in meters
    pitch = pitch * math.pi / 180 - 0.000001 # INPUT pitch of the camera, facing down: 90, facing 45: 45, facing forward: 0

    # Variable
    _latest["head"] = (_latest["head"] + 360) % 360
    heading = _latest["head"] * math.pi / 180 - 0.000001 # north: 0, east: 90, south: 180, west: 270
    lat = _latest["air_lat"] # latitude
    lon = _latest["air_lon"] # longitude

    # Calculating x and y FOV
    omega = math.atan(yPixMax/xPixMax)
    xFOV = math.cos(omega) * dFOV
    yFOV = math.sin(omega) * dFOV

    # Pixel -> Angle from the center
    xAngle = (xPixMax/2 - xPixVal) * xFOV/xPixMax
    yAngle = -(yPixMax/2 - yPixVal) * yFOV/yPixMax

    # Vector functions
    vn = (math.cos(pitch) - math.tan(yAngle) * math.sin(pitch)) * math.cos(heading) + math.tan(xAngle) * math.sin(heading)
    ve = (math.cos(pitch) - math.tan(yAngle) * math.sin(pitch)) * math.sin(heading) - math.tan(xAngle) * math.cos(heading)
    vd = math.sin(pitch) + math.tan(yAngle) * math.cos(pitch)

    # Displacement Calculations
    mn = height * vn/vd
    me = height * ve/vd

    # final equations
    latF = lat + (mn / 111132.0)
    lonF = lon + (me / (111132.0 * math.cos(lat * math.pi / 180)))

    hed = heading * 180 / math.pi

    status_message = f"Heading: {hed}, Pitch: {pitch}, meters north: {mn}, meters east: {me} Estimated coord of detection: {latF}, {lonF}"
    _latest["fire_lat"] = latF
    _latest["fire_lon"] = lonF
    return latF, lonF


global latin
global lonin
global t
latin = 0
lonin = 0
t = 0


def _extract_from_osd_message(m: dict):
    global latin
    global lonin
    global t

    if not isinstance(m, dict):
        return
    ts = _maybe_float(m.get("timestamp"))

    root = m.get("data", m)
    host = root.get("host") if isinstance(root, dict) else None
    if not isinstance(host, dict):
        if ts is not None:
            _latest["ts"] = ts
        return

    air_lat = _maybe_float(host.get("latitude"))
    air_lon = _maybe_float(host.get("longitude"))
    air_h   = _maybe_float(host.get("height"))
    head    = _maybe_float(host.get("attitude_head"))
    pitch   = _maybe_float(host.get("attitude_pitch"))
    roll    = _maybe_float(host.get("attitude_roll"))

    lrf_lat = lrf_lon = lrf_alt = lrf_dist = None
    gimbal_yaw = gimbal_pitch = None
    for _, v in host.items():
        if not isinstance(v, dict):
            continue
        if "measure_target_latitude" in v or "measure_target_distance" in v:
            lrf_lat  = _maybe_float(v.get("measure_target_latitude"))
            lrf_lon  = _maybe_float(v.get("measure_target_longitude"))
            lrf_alt  = _maybe_float(v.get("measure_target_altitude"))
            lrf_dist = _maybe_float(v.get("measure_target_distance"))
            gimbal_yaw   = _maybe_float(v.get("gimbal_yaw"))
            gimbal_pitch = _maybe_float(v.get("gimbal_pitch"))
            break

    _latest["ts"] = ts if ts is not None else _latest["ts"]

    lat_is_none = _latest.get("air_lat") is None

    if t == 0 or lat_is_none or (math.sqrt((air_lat if air_lat is not None else _latest["air_lat"] - 38.54936109489752)**2 + (air_lon if air_lon is not None else _latest["air_lon"] + 76.9539505811457)**2) + 0.00005) < math.sqrt((latin - 38.54936109489752)**2 + (lonin + 76.9539505811457)**2):
        _latest["air_lat"] = air_lat if air_lat is not None else _latest["air_lat"]
        latin = _latest["air_lat"]
        _latest["air_lon"] = air_lon if air_lon is not None else _latest["air_lon"]
        lonin = _latest["air_lon"]
    else: # 54936109489752              9539505811457
        # print( math.sqrt(        (log_entry.get("air_lat") - latin)**2 + ( log_entry.get("air_lon") - lonin)**2 )    )
        if math.sqrt((air_lat if air_lat is not None else _latest["air_lat"] - latin)**2 + (air_lon if air_lon is not None else _latest["air_lon"] - lonin)**2) > 0.0002:
            _latest["air_lat"] = air_lat if air_lat is not None else _latest["air_lat"]
            _latest["air_lon"] = air_lon if air_lon is not None else _latest["air_lon"]
    t = 1

    # _latest["air_lat"] = air_lat if air_lat is not None else _latest["air_lat"]
    # _latest["air_lon"] = air_lon if air_lon is not None else _latest["air_lon"]
   
   
   
    _latest["air_h"]   = air_h   if air_h   is not None else _latest["air_h"]
    _latest["head"]    = head    if head    is not None else _latest["head"]
    _latest["pitch"]   = pitch   if pitch   is not None else _latest["pitch"]
    _latest["roll"]    = roll    if roll    is not None else _latest["roll"]

    _latest["lrf_lat"]  = lrf_lat  if lrf_lat  is not None else _latest["lrf_lat"]
    _latest["lrf_lon"]  = lrf_lon  if lrf_lon  is not None else _latest["lrf_lon"]
    _latest["lrf_alt"]  = lrf_alt  if lrf_alt  is not None else _latest["lrf_alt"]
    _latest["lrf_dist"] = lrf_dist if lrf_dist is not None else _latest["lrf_dist"]

    _latest["gimbal_yaw"]   = gimbal_yaw   if gimbal_yaw   is not None else _latest["gimbal_yaw"]
    _latest["gimbal_pitch"] = gimbal_pitch if gimbal_pitch is not None else _latest["gimbal_pitch"]

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
        # if _latest["lrf_lat"] is not None and _latest["lrf_lon"] is not None:
        #     print(f"[MQTT LRF] lat={_latest['lrf_lat']}, lon={_latest['lrf_lon']}, dist={_latest['lrf_dist']}")
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
                  classes=classes, rect=rect, verbose=True, agnostic_nms=False)
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
            (_wavg_acc["sum_w_lon"] / _wavg_acc["sum_w"]))

def _log_detection(frame_id: int):
    ts_str = _iso_ts(_latest["ts"])
    line = f"{ts_str},{frame_id},{_latest['lrf_lat']},{_latest['lrf_lon']},{_latest['lrf_dist']}\n"
    try:
        with open(_DETECTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG] Failed to write to '{_DETECTIONS_LOG}': {e}")

def _log_detection_centroid(frame_id: int, centroid_lat: float, centroid_long: float):
    ts_str = _iso_ts(_latest["ts"])
    line = f"{ts_str},{frame_id},{centroid_lat},{centroid_long}\n"
    try:
        with open(_CENTROID_LAT_LONG_DETECTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG] Failed to write to '{_CENTROID_LAT_LONG_DETECTIONS_LOG}': {e}")

def _log_detection_with_bbox(frame_id: int, yb):
    if yb is None or _latest.get("lrf_lat") is None or _latest.get("lrf_lon") is None:
        return _log_detection(frame_id)
    x1, y1, x2, y2 = map(int, yb)
    cx = int(0.5 * (x1 + x2))
    cy = int(0.5 * (y1 + y2))
    ts_str = _iso_ts(_latest["ts"])
    line = f"{ts_str},{frame_id},{_latest['lrf_lat']},{_latest['lrf_lon']},{_latest['lrf_dist']},{x1},{y1},{x2},{y2},{cx},{cy}\n"
    try:
        with open(_DETECTIONS_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG] Failed to write bbox row to '{_DETECTIONS_LOG}': {e}")

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

# ---------------- BBOX GEO helpers ----------------
def _meters_per_deg(lat_rad: float):
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(lat_rad)
    return m_per_deg_lat, m_per_deg_lon

def _add_meters_to_ll(lat_deg: float, lon_deg: float, dE_m: float, dN_m: float):
    lat_rad = np.deg2rad(lat_deg if lat_deg is not None else 0.0)
    m_per_deg_lat, m_per_deg_lon = _meters_per_deg(lat_rad)
    dlat = dN_m / m_per_deg_lat
    dlon = dE_m / max(1e-9, m_per_deg_lon)
    return lat_deg + dlat, lon_deg + dlon

def _estimate_bbox_centroid_geo(yb, rgb_w, rgb_h, hfov_deg, vfov_deg):
    if _latest["lrf_lat"] is None or _latest["lrf_lon"] is None:
        return None
    if _latest["air_h"] is None:
        return None

    x1, y1, x2, y2 = yb
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    cx0 = 0.5 * rgb_w
    cy0 = 0.5 * rgb_h
    dx_px = cx - cx0           # + right
    dy_px = cy - cy0           # + down

    h = float(_latest["air_h"])  # meters
    hfov = np.deg2rad(hfov_deg)
    vfov = np.deg2rad(vfov_deg)
    gsd_x = 2.0 * h * np.tan(hfov * 0.5) / max(1.0, rgb_w)
    gsd_y = 2.0 * h * np.tan(vfov * 0.5) / max(1.0, rgb_h)

    mx_cam = dx_px * gsd_x           # + right
    my_cam = dy_px * gsd_y           # + down
    dE_local = mx_cam
    dN_local = -my_cam

    yaw_deg = (_latest["head"] or 0.0) + (_latest["gimbal_yaw"] or 0.0)
    yaw = np.deg2rad(yaw_deg)
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    dE_world =  cos_y * dE_local - sin_y * dN_local
    dN_world =  sin_y * dE_local + cos_y * dN_local

    lat0 = float(_latest["lrf_lat"])
    lon0 = float(_latest["lrf_lon"])
    est_lat, est_lon = _add_meters_to_ll(lat0, lon0, dE_world, dN_world)
    return est_lat, est_lon, float(cx), float(cy)

def _log_bbox(frame_id: int, est_lat: float, est_lon: float, cx: float, cy: float):
    ts_str = _iso_ts(_latest["ts"])
    line = f"{ts_str},{frame_id},{est_lat},{est_lon},{int(cx)},{int(cy)}\n"
    try:
        with open(_DETECTIONS_BBOX_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception as e:
        print(f"[LOG] Failed to write to '{_DETECTIONS_BBOX_LOG}': {e}")

def _accum_bbox_avg(lat: float, lon: float):
    if lat is None or lon is None:
        return
    _bbox_avg["n"] += 1
    _bbox_avg["sum_lat"] += lat
    _bbox_avg["sum_lon"] += lon

def _write_bbox_final_average_as_last_entry():
    if _bbox_avg["n"] <= 0:
        return
    avg_lat = _bbox_avg["sum_lat"] / _bbox_avg["n"]
    avg_lon = _bbox_avg["sum_lon"] / _bbox_avg["n"]
    ts_str = _iso_ts(_latest["ts"])
    line = f"{ts_str},FINAL_AVG,{avg_lat},{avg_lon},-1,-1\n"
    try:
        with open(_DETECTIONS_BBOX_LOG, "a", encoding="utf-8") as f:
            f.write(line)
        print(f"[BBOX] Final average appended to { _DETECTIONS_BBOX_LOG }: {avg_lat:.7f}, {avg_lon:.7f}")
    except Exception as e:
        print(f"[LOG] Failed to append final average: {e}")

# ---------- clustering utilities (no external deps) ----------
def _load_lat_lon_from_detections(path):
    pts = []
    if not os.path.exists(path):
        return np.asarray(pts, dtype=float)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 4:
                continue
            try:
                lat = float(parts[2])
                lon = float(parts[3])
            except Exception:
                continue
            if np.isfinite(lat) and np.isfinite(lon):
                pts.append([lat, lon])
    return np.asarray(pts, dtype=float)

def yolo_once(model, frame_bgr, imgsz, conf, classes, rect, device):
    kwargs = dict(source=frame_bgr, imgsz=imgsz, conf=conf,
                  classes=classes, rect=rect, verbose=False, agnostic_nms=False)
    if device is not None:
        try:
            kwargs["device"] = device
        except Exception:
            pass
    res = model.predict(**kwargs)[0]    # single inference
    # boxes
    boxes = []
    if getattr(res, "boxes", None) is not None and getattr(res.boxes, "xyxy", None) is not None:
        try:
            for x1, y1, x2, y2, *_ in res.boxes.xyxy.cpu().numpy():
                boxes.append((int(x1), int(y1), int(x2), int(y2)))
        except Exception:
            pass
    # annotated image
    annotated = res.plot()
    return annotated, boxes

def _pairwise_within_eps_idx(pts, eps_deg):
    n = len(pts)
    nbrs = [[] for _ in range(n)]
    if n == 0:
        return nbrs
    for i in range(n):
        for j in range(i+1, n):
            d = np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1])
            if d <= eps_deg:
                nbrs[i].append(j)
                nbrs[j].append(i)
    return nbrs

def _dbscan_like_largest_cluster(pts, eps_deg=0.00008, min_samples=3):
    n = len(pts)
    if n == 0:
        return []
    nbrs = _pairwise_within_eps_idx(pts, eps_deg)
    is_core = np.array([len(nbrs[i]) >= (min_samples-1) for i in range(n)], dtype=bool)
    visited = np.zeros(n, dtype=bool)
    best_cluster = []
    for i in range(n):
        if visited[i] or not is_core[i]:
            continue
        stack = [i]
        cluster = set()
        visited[i] = True
        while stack:
            u = stack.pop()
            cluster.add(u)
            for v in nbrs[u]:
                if not visited[v]:
                    visited[v] = True
                    if is_core[v]:
                        stack.append(v)
                    cluster.add(v)
        if len(cluster) > len(best_cluster):
            best_cluster = list(cluster)
    return sorted(best_cluster)

def _weighted_average_inverse_distance(pts, center):
    if len(pts) == 0:
        return None
    d = np.hypot(pts[:,0] - center[0], pts[:,1] - center[1])
    w = 1.0 / (1e-9 + d)
    w_sum = np.sum(w)
    lat = np.sum(w * pts[:,0]) / w_sum
    lon = np.sum(w * pts[:,1]) / w_sum
    return float(lat), float(lon)

def _cluster_and_write_averages(detections_path, eps_deg, min_samples, out_avg_path, out_wavg_path):
    pts = _load_lat_lon_from_detections(detections_path)
    if pts.size == 0:
        print(f"[CLUSTER] No detections found in {detections_path}; skipping clustered averages.")
        return
    idxs = _dbscan_like_largest_cluster(pts, eps_deg=eps_deg, min_samples=min_samples)
    if len(idxs) == 0:
        print("[CLUSTER] No dense cluster found; skipping clustered averages.")
        return
    clustered = pts[idxs]
    avg_lat = float(np.mean(clustered[:,0]))
    avg_lon = float(np.mean(clustered[:,1]))
    wavg_lat, wavg_lon = _weighted_average_inverse_distance(clustered, [avg_lat, avg_lon])

    ts_str = _iso_ts(_latest["ts"])
    try:
        with open(out_avg_path, "w", encoding="utf-8") as f:
            f.write(f"{ts_str},CLUSTER,{avg_lat},{avg_lon}\n")
        with open(out_wavg_path, "w", encoding="utf-8") as f:
            f.write(f"{ts_str},CLUSTER,{wavg_lat},{wavg_lon}\n")
        print(f"[CLUSTER] Cluster size={len(clustered)} | avg=({avg_lat:.7f},{avg_lon:.7f}) "
              f"| wavg=({wavg_lat:.7f},{wavg_lon:.7f})")
        print(f"[CLUSTER] Wrote: {out_avg_path}, {out_wavg_path}")
    except Exception as e:
        print(f"[CLUSTER] Failed to write clustered averages: {e}")

# ---------- OpenCV capture helpers ----------
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
        # time.sleep(0.01)
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



def draw_boxes_from_mask(bgr: np.ndarray, mask: np.ndarray, min_area: int) -> Tuple[np.ndarray, bool]:
    """
    Draws bounding boxes on regions of the mask with area >= min_area.
    Returns:
        out: annotated BGR image
        detected: True if at least one valid contour (fire region) was found
    """
    out = bgr.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected = False
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(out, "fire", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
        detected = True

    return out, detected

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
    ap.add_argument("--weights", default="10-12K-100e-n.onnx", help="YOLO weights (.onnx/.pt/.engine)")
    ap.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (multiple of 32)")
    ap.add_argument("--rect", action="store_true", help="Minimal padding to match stride (faster on .onnx)")
    ap.add_argument("--classes", type=int, nargs="*", default=1, help="Optional class filter, e.g. 0 1")
    ap.add_argument("--device", default=0, help="Torch device for .pt (e.g. 0 or 'cpu'); ignored for .onnx")
    
    # Height Pitch
    ap.add_argument("--height", type=int, default="35", help="height of the drone in meters above the ground")
    ap.add_argument("--pitch", type=int, default="90", help="Pitch for the camera (90 is nadir) 45 is 45")

    
    # IR thresholds
    ap.add_argument("--ir-hsv-lower", type=str, default="12,25,39", help="HSV lower (H[0-179],S,V) for IR")
    ap.add_argument("--ir-hsv-upper", type=str, default="255,255,255", help="HSV upper (H[0-179],S,V) for IR")

    # ap.add_argument("--ir-hsv-lower", type=str, default="12,42,39", help="HSV lower (H[0-179],S,V) for IR")
    # ap.add_argument("--ir-hsv-upper", type=str, default="166,255,255", help="HSV upper (H[0-179],S,V) for IR")
    ap.add_argument("--ir-min-area", type=int, default=200, help="Min contour area for IR boxes")
    # optional homography alignment for IR
    ap.add_argument("--ir-warp", action="store_true", help="Apply homography to IR before thresholding")
    ap.add_argument("--H", type=float, nargs=9, default=H_DEFAULT.flatten().tolist(),
                    help="3x3 homography row-major for IR warp")
    ap.add_argument("--src-wh", type=int, nargs=2, default=[1280, 720], help="IR src size expected before warp")
    ap.add_argument("--dst-wh", type=int, nargs=2, default=[1280, 720], help="IR warp destination size")
    # windows & saving
    ap.add_argument("--save-dir", default="captures", help="Directory to save snapshots")
    # FOV for bbox->geo
    ap.add_argument("--hfov-deg", type=float, default=95.0, help="RGB horizontal FOV (deg)")
    ap.add_argument("--vfov-deg", type=float, default=90.0, help="RGB vertical FOV (deg)")
    # clustering CLI (default ~4 m latitude)
    # ap.add_argument("--cluster-eps-deg", type=float, default=0.0000359, help="DBSCAN-like epsilon in degrees (~4m lat)")
    # ap.add_argument("--cluster-eps-deg", type=float, default=0.00001796, help="DBSCAN-like epsilon in degrees (~2m lat)")
    # ap.add_argument("--cluster-eps-deg", type=float, default=0.000008983, help="DBSCAN-like epsilon in degrees (~1m lat)") #did not work
    
    ap.add_argument("--cluster-eps-deg", type=float, default=0.00001348, help="DBSCAN-like epsilon in degrees (~1.5m lat)")
    
    ap.add_argument("--cluster-min-samples", type=int, default=3, help="Min samples for dense core")
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

    cv2.namedWindow("IR (annotated)",  cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB (YOLO)",      cv2.WINDOW_NORMAL)
    os.makedirs(args.save_dir, exist_ok=True)
    idx = 0
    print(f"Press 's' to save both halves, 'q' to quit. Skipping {args.skip} frames.")

    H_mat   = np.array(args.H, dtype=np.float64).reshape(3, 3)
    SRC_W, SRC_H = args.src_wh
    DST_W, DST_H = args.dst_wh
    last_info = 0.0
    frame_id = 0

    # counterlat = 0
    # lat_cur = _latest["air_lat"]
    # lon_cur = _latest["air_lon"]
    # lat_prev = lat_cur
    # lon_prev = lon_cur
    # _latest["head"] = 0

    while True:
        # system_message = ""
        # if math.sqrt((lat_cur - lat_prev)**2 + (lon_cur - lon_prev)**2) > 0.0004818: # 00003818 #counterlat % 5 == 0:
            # _latest["head"] = bearing_deg(lat_prev, lon_prev, lat_cur, lon_cur)
            # lat_prev = lat_cur
            # lon_prev = lon_cur

        # counterlat += 1

        # frame skipping

        # lat_cur = _latest["air_lat"]
        # lon_cur = _latest["air_lon"]
        # heading = _latest["head"]
        # status_message = f"Coordinates: {lat_cur}, {lon_cur}. Heading: {heading}"




        for _ in range(max(0, args.skip)):
            cap.grab()
            # continue

        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            continue

        ir_left, rgb_right = split_halves(frame)
        print(f'RGB shape - {rgb_right.shape}')
        print(f'IR shape - {ir_left.shape}')
        rgb_anno = rgb_right.copy()  # safe default (we always show RGB every frame)

        # -------- IR detection (gates YOLO) --------
        ir_mask = ir_hsv_mask(ir_left, hsv_lo, hsv_hi)

        # If mask invalid or empty -> show both raw views, continue
        if ir_mask is None or np.count_nonzero(ir_mask) == 0:
            cv2.imshow("IR (annotated)", ir_left)
            cv2.imshow("RGB (YOLO)", rgb_anno)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                ir_path  = os.path.join(args.save_dir, f"ir_{idx:06d}.png")
                rgb_path = os.path.join(args.save_dir, f"rgb_{idx:06d}.png")
                cv2.imwrite(ir_path, ir_left)
                cv2.imwrite(rgb_path, rgb_anno)
                print(f"Saved {ir_path} and {rgb_path}")
                idx += 1
            frame_id += 1
            continue

        ir_anno, ir_detected = draw_boxes_from_mask(ir_left, ir_mask, args.ir_min_area)

        # If no valid IR blobs -> show both, continue
        if not ir_detected:
            cv2.imshow("IR (annotated)", ir_anno)
            cv2.imshow("RGB (YOLO)", rgb_anno)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                ir_path  = os.path.join(args.save_dir, f"ir_{idx:06d}.png")
                rgb_path = os.path.join(args.save_dir, f"rgb_{idx:06d}.png")
                cv2.imwrite(ir_path, ir_anno)
                cv2.imwrite(rgb_path, rgb_anno)
                print(f"Saved {ir_path} and {rgb_path}")
                idx += 1
            frame_id += 1
            with frame_lock:
                latest_ir_frame = ir_anno.copy()
                latest_rgb_frame = rgb_anno.copy()
            _latest["fire"] = False
            continue

        # -------- YOLO only when IR detected --------
        rgb_h, rgb_w = rgb_right.shape[:2]
        ir_mask_on_rgb = cv2.resize(ir_mask, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        ir_boxes_on_rgb = _boxes_from_mask(ir_mask_on_rgb, args.ir_min_area)

        try:
            rgb_anno, yolo_boxes = yolo_once(model, rgb_right, imgsz=args.imgsz, conf=args.conf,
                                 classes=args.classes, rect=args.rect, device=args.device)
            print(f'YOLO boxes - {yolo_boxes}')
        except Exception as e:
            rgb_anno = rgb_right.copy()
            print(f"[YOLO] inference error: {e}")
            yolo_boxes = []

        print(f"Frame {frame_id}: IR boxes={len(ir_boxes_on_rgb)}, YOLO boxes={len(yolo_boxes)} | "
              f"LRF ready={_latest['lrf_lat'] is not None}")

        with frame_lock:
            latest_ir_frame = ir_anno.copy()
            latest_rgb_frame = rgb_anno.copy()


        has_intersection = False
        first_matching_yolo_box = None
        for ib in ir_boxes_on_rgb:
            for yb in yolo_boxes:
                if _intersects(ib, yb):
                    has_intersection = True
                    first_matching_yolo_box = yb
                    break
            if has_intersection:
                break

        if has_intersection and _latest["lrf_lat"] is not None and _latest["lrf_lon"] is not None:
            print(f"----> INTERSECTION + LRF target at frame {frame_id} -> logging to detections.txt")
            _latest["fire"] = True

            _log_detection_with_bbox(frame_id, first_matching_yolo_box)

            avg = _update_running_average(_latest["lrf_lat"], _latest["lrf_lon"])
            if avg is not None:
                _log_avg(frame_id, avg[0], avg[1])

            wavg = _update_weighted_average(_latest["lrf_lat"], _latest["lrf_lon"], yolo_boxes, rgb_w, rgb_h)
            if wavg is not None:
                _log_wavg(frame_id, wavg[0], wavg[1])

            est = _estimate_bbox_centroid_geo(first_matching_yolo_box, rgb_w, rgb_h,
                                              args.hfov_deg, args.vfov_deg)
            if est is not None:
                est_lat, est_lon, cx, cy = est

                _log_bbox(frame_id, est_lat, est_lon, cx, cy)
                _accum_bbox_avg(est_lat, est_lon)

                # --- draw on RGB preview safely ---
                h, w = rgb_anno.shape[:2]
                xi, yi = int(round(cx)), int(round(cy))
                if 0 <= xi < w and 0 <= yi < h:
                    cv2.circle(rgb_anno, (xi, yi), 6, (0, 255, 0), 2)
                    cv2.putText(
                        rgb_anno,
                        f"{est_lat:.6f},{est_lon:.6f}",
                        (xi + 8, yi - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA
                    )
            
            if first_matching_yolo_box:
                x1, y1, x2, y2 = map(int, first_matching_yolo_box)
                x_pixel_centroid = (x1 + x2) // 2
                y_pixel_centroid = (y1 + y2) // 2

                centroid_lat, centroid_long = calculatecoords(x_pixel_centroid, y_pixel_centroid)
                _log_detection_centroid(frame_id, centroid_lat, centroid_long)
        else:
           _latest["fire"] = False



        # -------- Always show both windows (every frame) --------
        cv2.imshow("IR (annotated)", ir_anno)
        cv2.imshow("RGB (YOLO)", rgb_anno)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            ir_path  = os.path.join(args.save_dir, f"ir_{idx:06d}.png")
            rgb_path = os.path.join(args.save_dir, f"rgb_{idx:06d}.png")
            cv2.imwrite(ir_path, ir_anno)
            cv2.imwrite(rgb_path, rgb_anno)
            print(f"Saved {ir_path} and {rgb_path}")
            idx += 1

        frame_id += 1

    # write final bbox average line
    _write_bbox_final_average_as_last_entry()

    # post-run clustering from detections.txt
    _cluster_and_write_averages(
        detections_path=_DETECTIONS_LOG,
        eps_deg=args.cluster_eps_deg,
        min_samples=args.cluster_min_samples,
        out_avg_path=_CLUSTERED_AVG_LOG,
        out_wavg_path=_CLUSTERED_WAVG_LOG,
    )
    _latest["fire"] = False


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()