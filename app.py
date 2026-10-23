from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from ultralytics import YOLO
import easyocr
import cv2
import numpy as np
import re
from datetime import datetime
import sqlite3
import os
import csv
import io

# Import the improved OCR engine
from anpr_ocr_engine import (
    perform_enhanced_anpr_ocr,
    extract_state_and_rto,
    is_valid_indian_format,
    format_plate_for_display,
    clean_alphanumeric,
    INDIAN_STATES,
    RTO_MAPPINGS,
)

# ==========================================
# FLASK SETUP
# ==========================================
app = Flask(__name__)
CORS(app)

DATABASE = "anpr_history.db"

# Evidence folders
EVIDENCE_DIR = "evidence"
ORIGINAL_DIR = os.path.join(EVIDENCE_DIR, "original")
PLATES_DIR = os.path.join(EVIDENCE_DIR, "plates")

os.makedirs(ORIGINAL_DIR, exist_ok=True)
os.makedirs(PLATES_DIR, exist_ok=True)


# ==========================================
# DATABASE HELPERS
# ==========================================
def get_connection():
    connection = sqlite3.connect(DATABASE)
    connection.row_factory = sqlite3.Row
    return connection


def init_database():
    connection = sqlite3.connect(DATABASE)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            plate_number TEXT,
            ocr_confidence REAL,
            best_method TEXT,
            status TEXT,
            location TEXT,
            detection_time TEXT,
            plate_quality TEXT,
            alert TEXT,
            watchlist_match INTEGER DEFAULT 0,
            watchlist_reason TEXT,
            previous_detections INTEGER DEFAULT 0,
            latitude REAL,
            longitude REAL,
            original_image TEXT,
            plate_image TEXT
        )
    """)

    # Add columns if older anpr_history.db already exists.
    existing = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(detections)").fetchall()
    }

    new_columns = {
        "plate_quality": "TEXT",
        "alert": "TEXT",
        "watchlist_match": "INTEGER DEFAULT 0",
        "watchlist_reason": "TEXT",
        "previous_detections": "INTEGER DEFAULT 0",
        "latitude": "REAL",
        "longitude": "REAL",
        "original_image": "TEXT",
        "plate_image": "TEXT",
        "state_code": "TEXT",
        "state_name": "TEXT",
        "rto_code": "TEXT",
        "rto_name": "TEXT",
        "toll_gate_id": "INTEGER",
        "toll_gate_name": "TEXT",
        "validation_status": "TEXT",
        "is_demo": "INTEGER DEFAULT 0",
        "display_plate": "TEXT",
        # Camera / lane / duplicate tracking
        "camera_id": "TEXT",
        "lane_id": "TEXT",
        "source_type": "TEXT DEFAULT 'upload'",
        "is_duplicate": "INTEGER DEFAULT 0",
    }

    for column, definition in new_columns.items():
        if column not in existing:
            cursor.execute(
                f"ALTER TABLE detections ADD COLUMN {column} {definition}"
            )

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS watchlist (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE,
            reason TEXT,
            added_time TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_number TEXT UNIQUE,
            owner_name TEXT,
            registration_status TEXT,
            vehicle_details TEXT,
            notes TEXT
        )
    """)

    # Extend vehicles table
    existing_veh = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(vehicles)").fetchall()
    }
    new_veh_cols = {
        "state_code": "TEXT",
        "state_name": "TEXT",
        "rto_code": "TEXT",
        "rto_name": "TEXT",
        "manufacturer_model": "TEXT",
        "fuel_type": "TEXT",
        "vehicle_color": "TEXT",
        "registration_year": "INTEGER",
        "registration_date": "TEXT",
        "purchase_date": "TEXT",
        "purchase_condition": "TEXT",
        "previous_owners_count": "INTEGER DEFAULT 0",
        "previous_ownership_history": "TEXT",
        "insurance_expiry": "TEXT",
        "fitness_expiry": "TEXT",
        "pollution_expiry": "TEXT",
        "is_demo": "INTEGER DEFAULT 0",
    }
    for col, col_type in new_veh_cols.items():
        if col not in existing_veh:
            cursor.execute(f"ALTER TABLE vehicles ADD COLUMN {col} {col_type}")

    # Toll gates table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS toll_gates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            highway TEXT,
            city TEXT,
            state TEXT,
            latitude REAL,
            longitude REAL,
            address TEXT,
            is_demo INTEGER DEFAULT 1
        )
    """)

    # Migrate toll_gates columns added after initial creation
    existing_tg = {
        row[1]
        for row in cursor.execute("PRAGMA table_info(toll_gates)").fetchall()
    }
    for col, coltype in {
        "lane_count": "INTEGER DEFAULT 1",
        "camera_ids": "TEXT",
    }.items():
        if col not in existing_tg:
            cursor.execute(f"ALTER TABLE toll_gates ADD COLUMN {col} {coltype}")

    connection.commit()
    connection.close()
    print("Database initialized successfully!")


# ==========================================
# FILE / EVIDENCE HELPERS
# ==========================================
def safe_filename(filename):
    filename = os.path.basename(filename or "vehicle.jpg")
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    return filename


def save_original_evidence(image_bytes, filename, detection_time):
    stamp = detection_time.replace(":", "-").replace(" ", "_")
    base = os.path.splitext(safe_filename(filename))[0]
    path = os.path.join(ORIGINAL_DIR, f"{stamp}_{base}.jpg")

    with open(path, "wb") as f:
        f.write(image_bytes)

    return path


def save_plate_evidence(plate_image, filename, detection_time):
    stamp = detection_time.replace(":", "-").replace(" ", "_")
    base = os.path.splitext(safe_filename(filename))[0]
    path = os.path.join(PLATES_DIR, f"{stamp}_{base}_plate.jpg")

    cv2.imwrite(path, plate_image)
    return path


def relative_evidence_path(path):
    if not path:
        return None
    return os.path.relpath(path, ".").replace("\\", "/")


# ==========================================
# WATCHLIST / VEHICLE HELPERS
# ==========================================
def clean_ocr_text(text):
    text = str(text).upper()
    return re.sub(r"[^A-Z0-9]", "", text)


def check_watchlist(plate_number):
    if not plate_number or plate_number in ("Not Recognized", "Not Detected"):
        return False, None

    connection = get_connection()
    row = connection.execute(
        """
        SELECT reason
        FROM watchlist
        WHERE UPPER(plate_number) = ?
        """,
        (plate_number.upper(),)
    ).fetchone()
    connection.close()

    if row:
        return True, row["reason"]

    return False, None


def get_vehicle_details(plate_number):
    if not plate_number or plate_number in ("Not Recognized", "Not Detected"):
        return None

    connection = get_connection()
    row = connection.execute(
        """
        SELECT *
        FROM vehicles
        WHERE UPPER(plate_number) = ?
        """,
        (plate_number.upper(),)
    ).fetchone()
    connection.close()

    return dict(row) if row else None


# ==========================================
# SAVE DETECTION
# ==========================================
def save_detection(
    image_name,
    plate_number,
    ocr_confidence,
    best_method,
    status,
    location,
    detection_time,
    plate_quality="Very Poor",
    alert="None",
    watchlist_match=False,
    watchlist_reason=None,
    previous_detections=0,
    latitude=None,
    longitude=None,
    original_image=None,
    plate_image=None,
    state_code=None,
    state_name=None,
    rto_code=None,
    rto_name=None,
    toll_gate_id=None,
    toll_gate_name=None,
    validation_status=None,
    display_plate=None,
    is_demo=0,
    camera_id=None,
    lane_id=None,
    source_type="upload",
    is_duplicate=0,
):
    connection = get_connection()

    connection.execute(
        """
        INSERT INTO detections (
            image_name, plate_number, ocr_confidence, best_method, status,
            location, detection_time, plate_quality, alert,
            watchlist_match, watchlist_reason, previous_detections,
            latitude, longitude, original_image, plate_image,
            state_code, state_name, rto_code, rto_name,
            toll_gate_id, toll_gate_name, validation_status, display_plate, is_demo,
            camera_id, lane_id, source_type, is_duplicate
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            image_name, plate_number, ocr_confidence, best_method, status,
            location, detection_time, plate_quality, alert,
            1 if watchlist_match else 0, watchlist_reason, previous_detections,
            latitude, longitude, original_image, plate_image,
            state_code, state_name, rto_code, rto_name,
            toll_gate_id, toll_gate_name, validation_status, display_plate, is_demo,
            camera_id, lane_id, source_type, is_duplicate,
        )
    )

    connection.commit()
    connection.close()


def get_previous_detection_count(plate_number):
    if not plate_number:
        return 0

    connection = get_connection()
    row = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM detections
        WHERE plate_number = ?
        """,
        (plate_number,)
    ).fetchone()
    connection.close()

    return row["count"]


# ==========================================
# LOAD YOLO (number plate detection model - DO NOT REPLACE)
# ==========================================
print("Loading YOLO model...")
model = YOLO("runs/detect/train/weights/best.pt")
print("YOLO loaded successfully!")


# ==========================================
# LOAD EASY OCR
# ==========================================
print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)
print("EasyOCR loaded successfully!")


# ==========================================
# HOME
# ==========================================
@app.route("/")
def home():
    return "ANPR Backend is Running Successfully!"


# ==========================================
# PREDICT — uses improved anpr_ocr_engine
# ==========================================
@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    if not file.filename:
        return jsonify({"error": "No image filename"}), 400

    # Read uploaded image ONCE.
    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"error": "Empty image file"}), 400

    # Optional location/GPS/toll-gate data from frontend.
    location = request.form.get("location", "Rural Toll Plaza")
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")
    toll_gate_id = request.form.get("toll_gate_id")
    toll_gate_name = request.form.get("toll_gate_name")
    camera_id = request.form.get("camera_id", "CAM-01")
    lane_id = request.form.get("lane_id", "LANE-1")
    source_type = request.form.get("source_type", "upload")
    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        latitude = float(latitude) if latitude else None
    except (ValueError, TypeError):
        latitude = None

    try:
        longitude = float(longitude) if longitude else None
    except (ValueError, TypeError):
        longitude = None

    try:
        toll_gate_id = int(toll_gate_id) if toll_gate_id else None
    except (ValueError, TypeError):
        toll_gate_id = None

    # Save original evidence using the same bytes.
    original_image = save_original_evidence(
        image_bytes,
        file.filename,
        detection_time
    )

    # Decode image: try disk-read first (most faithful), fall back to imdecode
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # If we already saved the evidence file, re-read from disk to get the
    # identical decode path as direct cv2.imread (avoids HTTP JPEG artefacts)
    if original_image and os.path.isfile(original_image):
        disk_img = cv2.imread(original_image)
        if disk_img is not None:
            image = disk_img

    # ======================================
    # YOLO NUMBER PLATE DETECTION
    # ======================================
    import math as _math

    # Only resize if image is extremely large (> 1920px on any side).
    # Running YOLO on the original image gives better crops (no coord scaling).
    h_img, w_img = image.shape[:2]
    max_dim = 1920
    if max(h_img, w_img) > max_dim:
        scale_factor = max_dim / max(h_img, w_img)
        image_resized = cv2.resize(
            image,
            (int(w_img * scale_factor), int(h_img * scale_factor)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        image_resized = image
        scale_factor = 1.0

    # Run YOLO with standard defaults (conf=0.25, iou=0.7).
    # These match evaluate_anpr_ocr.py exactly for consistent results.
    results = model(image_resized, conf=0.25, iou=0.7, verbose=False)
    # Check if anything detected; if not, retry with lower threshold
    _any_detected = any(len(r.boxes) > 0 for r in results)
    if not _any_detected:
        results = model(image_resized, conf=0.10, iou=0.45, verbose=False)

    plate_detected = False
    saved_plate_image = None
    best_crop = None
    best_conf_score = -1.0

    for result in results:
        # Sort boxes by AREA (largest first) to match evaluate_anpr_ocr.py behavior.
        # The largest detected region is most likely the full plate (not a partial sub-region).
        boxes = sorted(
            result.boxes,
            key=lambda b: (b.xyxy[0][2] - b.xyxy[0][0]) * (b.xyxy[0][3] - b.xyxy[0][1]),
            reverse=True
        )

        for box in boxes:
            conf_score = float(box.conf[0])

            x1r, y1r, x2r, y2r = box.xyxy[0].tolist()

            # Scale back to original resolution.
            # KEY FIX: use floor for top-left and ceil for bottom-right
            # so integer truncation never clips the rightmost character.
            if scale_factor != 1.0:
                x1 = max(0, int(_math.floor(x1r / scale_factor)))
                y1 = max(0, int(_math.floor(y1r / scale_factor)))
                x2 = min(image.shape[1], int(_math.ceil(x2r / scale_factor)))
                y2 = min(image.shape[0], int(_math.ceil(y2r / scale_factor)))
            else:
                x1 = max(0, int(x1r))
                y1 = max(0, int(y1r))
                x2 = min(image.shape[1], int(_math.ceil(x2r)))
                y2 = min(image.shape[0], int(_math.ceil(y2r)))

            plate_w = x2 - x1
            plate_h = y2 - y1

            # 6% horizontal, 5% vertical padding — close to evaluate script (4%)
            # slightly more to handle HTTP-uploaded images
            pad_x = max(4, int(plate_w * 0.06))
            pad_y = max(3, int(plate_h * 0.05))
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(image.shape[1], x2 + pad_x)
            y2 = min(image.shape[0], y2 + pad_y)

            plate = image[y1:y2, x1:x2]

            # Skip crops too small for OCR
            if plate.size == 0 or (x2 - x1) < 40 or (y2 - y1) < 12:
                continue

            # Take largest-area box (first valid one after area-sort)
            if conf_score > best_conf_score:
                best_conf_score = conf_score
                best_crop = plate
                plate_detected = True

    # Save the best plate crop
    if plate_detected and best_crop is not None:
        saved_plate_image = save_plate_evidence(
            best_crop,
            file.filename,
            detection_time
        )

    # ==========================================
    # NO PLATE DETECTED
    # ==========================================
    if not plate_detected or best_crop is None:
        save_detection(
            file.filename, "Not Detected", 0, "None", "Detection Failed",
            location, detection_time, "Very Poor", "No number plate detected",
            False, None, 0, latitude, longitude,
            relative_evidence_path(original_image), None,
            "Unverified", "Unverified", "Unverified", "Unverified",
            toll_gate_id, toll_gate_name, "Plate Not Detected", "Not Detected",
            0, camera_id, lane_id, source_type, 0,
        )

        return jsonify({
            "plate_number": "Not Detected",
            "display_plate": "Not Detected",
            "previous_detections": 0,
            "ocr_confidence": 0,
            "best_method": "None",
            "plate_quality": "Very Poor",
            "alert": "No number plate detected",
            "status": "Detection Failed",
            "validation_status": "Plate Not Detected",
            "state_code": "Unverified",
            "state_name": "Unverified",
            "rto_code": "Unverified",
            "rto_name": "Unverified",
            "location": location,
            "detection_time": detection_time,
            "latitude": latitude,
            "longitude": longitude,
            "toll_gate_id": toll_gate_id,
            "toll_gate_name": toll_gate_name,
            "original_image": relative_evidence_path(original_image),
            "plate_image": None,
            "watchlist_match": False,
            "watchlist_reason": None,
            "vehicle_details": None
        })

    # ==========================================
    # IMPROVED OCR — via anpr_ocr_engine
    # ==========================================
    ocr_result = perform_enhanced_anpr_ocr(best_crop, reader)

    plate_number = ocr_result["plate_number"]
    display_plate = ocr_result.get("display_plate", plate_number)
    ocr_confidence = ocr_result["ocr_confidence"]
    best_method = ocr_result["best_method"]
    plate_quality = ocr_result["plate_quality"]
    status = ocr_result["status"]
    validation_status = ocr_result.get("validation_status", "Unverified")
    state_code = ocr_result.get("state_code", "Unverified")
    state_name = ocr_result.get("state_name", "Unverified")
    rto_code = ocr_result.get("rto_code", "Unverified")
    rto_name = ocr_result.get("rto_name", "Unverified")
    alert = ocr_result.get("alert", "None")

    # ==========================================
    # PREVIOUS DETECTIONS
    # ==========================================
    previous_detections = get_previous_detection_count(plate_number)

    # ==========================================
    # WATCHLIST
    # ==========================================
    watchlist_match, watchlist_reason = check_watchlist(plate_number)

    if watchlist_match:
        alert = "WATCHLIST MATCH"

    # ==========================================
    # OWNER / VEHICLE DETAILS
    # ==========================================
    vehicle_details = get_vehicle_details(plate_number)

    # ==========================================
    # SAVE DETECTION
    # ==========================================
    save_detection(
        file.filename, plate_number, ocr_confidence, best_method, status,
        location, detection_time, plate_quality, alert,
        watchlist_match, watchlist_reason, previous_detections,
        latitude, longitude,
        relative_evidence_path(original_image),
        relative_evidence_path(saved_plate_image),
        state_code, state_name, rto_code, rto_name,
        toll_gate_id, toll_gate_name, validation_status, display_plate
    )

    # ==========================================
    # RESPONSE
    # ==========================================
    return jsonify({
        "plate_number": plate_number,
        "display_plate": display_plate,
        "previous_detections": previous_detections,
        "ocr_confidence": ocr_confidence,
        "best_method": best_method,
        "plate_quality": plate_quality,
        "alert": alert,
        "status": status,
        "validation_status": validation_status,
        "state_code": state_code,
        "state_name": state_name,
        "rto_code": rto_code,
        "rto_name": rto_name,
        "location": location,
        "detection_time": detection_time,
        "latitude": latitude,
        "longitude": longitude,
        "toll_gate_id": toll_gate_id,
        "toll_gate_name": toll_gate_name,
        "original_image": relative_evidence_path(original_image),
        "plate_image": relative_evidence_path(saved_plate_image),
        "watchlist_match": watchlist_match,
        "watchlist_reason": watchlist_reason,
        "vehicle_details": vehicle_details,
        "methods_agree_count": ocr_result.get("methods_agree_count", 1),
        "is_low_resolution": ocr_result.get("is_low_resolution", False),
    })


# ==========================================
# HISTORY HELPERS
# ==========================================
def row_to_dict(row):
    d = dict(row)
    d["watchlist_match"] = bool(d.get("watchlist_match", 0))
    d["is_demo"] = bool(d.get("is_demo", 0))
    return d


# ==========================================
# HISTORY
# ==========================================
@app.route("/history", methods=["GET"])
def history():
    state = request.args.get("state", "").strip().upper()
    rto = request.args.get("rto", "").strip().upper()
    status_filter = request.args.get("status", "").strip()
    date_from = request.args.get("date_from", "").strip()
    date_to = request.args.get("date_to", "").strip()
    confidence_min = request.args.get("confidence_min", "").strip()
    watchlist_only = request.args.get("watchlist_only", "").strip()
    limit = request.args.get("limit", "200").strip()
    # Location filters (new)
    toll_gate_id_filter = request.args.get("toll_gate_id", "").strip()
    toll_gate_name_filter = request.args.get("toll_gate_name", "").strip()
    lane_id_filter = request.args.get("lane_id", "").strip()
    camera_id_filter = request.args.get("camera_id", "").strip()

    query = "SELECT * FROM detections WHERE 1=1"
    params = []

    if state:
        query += " AND UPPER(state_code) = ?"
        params.append(state)

    if rto:
        rto_clean = re.sub(r"\s+", " ", rto).strip()
        query += " AND UPPER(rto_code) = ?"
        params.append(rto_clean)

    if status_filter:
        query += " AND status = ?"
        params.append(status_filter)

    if date_from:
        query += " AND detection_time >= ?"
        params.append(date_from)

    if date_to:
        query += " AND detection_time <= ?"
        params.append(date_to + " 23:59:59")

    if confidence_min:
        try:
            query += " AND ocr_confidence >= ?"
            params.append(float(confidence_min))
        except ValueError:
            pass

    if watchlist_only == "1":
        query += " AND watchlist_match = 1"

    # Location filters
    if toll_gate_id_filter:
        try:
            query += " AND toll_gate_id = ?"
            params.append(int(toll_gate_id_filter))
        except ValueError:
            pass

    if toll_gate_name_filter:
        query += " AND LOWER(toll_gate_name) LIKE ?"
        params.append(f"%{toll_gate_name_filter.lower()}%")

    if lane_id_filter:
        query += " AND lane_id = ?"
        params.append(lane_id_filter)

    if camera_id_filter:
        query += " AND camera_id = ?"
        params.append(camera_id_filter)

    query += " ORDER BY id DESC"

    try:
        lim = int(limit)
        query += f" LIMIT {lim}"
    except ValueError:
        query += " LIMIT 200"

    connection = get_connection()
    rows = connection.execute(query, params).fetchall()
    connection.close()

    return jsonify([row_to_dict(row) for row in rows])


@app.route("/history", methods=["DELETE"])
def clear_history():
    connection = get_connection()
    connection.execute("DELETE FROM detections WHERE is_demo = 0 OR is_demo IS NULL")
    connection.commit()
    connection.close()

    return jsonify({
        "message": "Detection history cleared successfully (demo records preserved)"
    })


# ==========================================
# SEARCH HISTORY
# ==========================================
@app.route("/history/search", methods=["GET"])
def search_history():
    plate = request.args.get("plate", "").strip().upper()

    if not plate:
        return jsonify({
            "error": "Please provide a plate number"
        }), 400

    connection = get_connection()
    rows = connection.execute(
        """
        SELECT * FROM detections
        WHERE UPPER(plate_number) LIKE ?
        ORDER BY id DESC
        """,
        (f"%{plate}%",)
    ).fetchall()
    connection.close()

    return jsonify([row_to_dict(row) for row in rows])


# ==========================================
# STATISTICS
# ==========================================
@app.route("/stats", methods=["GET"])
def stats():
    connection = get_connection()

    total = connection.execute(
        "SELECT COUNT(*) AS c FROM detections"
    ).fetchone()["c"]

    recognized = connection.execute(
        "SELECT COUNT(*) AS c FROM detections WHERE status='Recognized'"
    ).fetchone()["c"]

    low_confidence = connection.execute(
        "SELECT COUNT(*) AS c FROM detections WHERE status='Low Confidence'"
    ).fetchone()["c"]

    failed = connection.execute(
        """
        SELECT COUNT(*) AS c FROM detections
        WHERE status IN ('Detection Failed', 'OCR Failed')
        """
    ).fetchone()["c"]

    watchlist_matches = connection.execute(
        "SELECT COUNT(*) AS c FROM detections WHERE watchlist_match=1"
    ).fetchone()["c"]

    repeated_detections = connection.execute(
        """
        SELECT COUNT(*) AS c FROM detections
        WHERE previous_detections > 0
        """
    ).fetchone()["c"]

    connection.close()

    return jsonify({
        "total_detections": total,
        "recognized": recognized,
        "low_confidence": low_confidence,
        "failed": failed,
        "watchlist_matches": watchlist_matches,
        "repeated_detections": repeated_detections
    })


# ==========================================
# STATE-WISE SUMMARY
# ==========================================
@app.route("/stats/state-summary", methods=["GET"])
def state_summary():
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT state_code, state_name,
               COUNT(*) AS total,
               SUM(CASE WHEN status='Recognized' THEN 1 ELSE 0 END) AS recognized,
               SUM(CASE WHEN status='Low Confidence' THEN 1 ELSE 0 END) AS low_confidence
        FROM detections
        WHERE state_code IS NOT NULL AND state_code != 'Unverified' AND state_code != ''
        GROUP BY state_code, state_name
        ORDER BY total DESC
        """
    ).fetchall()
    connection.close()
    return jsonify([dict(r) for r in rows])


# ==========================================
# RTO-WISE SUMMARY
# ==========================================
@app.route("/stats/rto-summary", methods=["GET"])
def rto_summary():
    state = request.args.get("state", "").strip().upper()
    connection = get_connection()

    if state:
        rows = connection.execute(
            """
            SELECT rto_code, rto_name, state_code,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='Recognized' THEN 1 ELSE 0 END) AS recognized
            FROM detections
            WHERE rto_code IS NOT NULL AND rto_code != 'Unverified' AND rto_code != ''
              AND UPPER(state_code) = ?
            GROUP BY rto_code, rto_name, state_code
            ORDER BY total DESC
            """,
            (state,)
        ).fetchall()
    else:
        rows = connection.execute(
            """
            SELECT rto_code, rto_name, state_code,
                   COUNT(*) AS total,
                   SUM(CASE WHEN status='Recognized' THEN 1 ELSE 0 END) AS recognized
            FROM detections
            WHERE rto_code IS NOT NULL AND rto_code != 'Unverified' AND rto_code != ''
            GROUP BY rto_code, rto_name, state_code
            ORDER BY total DESC
            """
        ).fetchall()
    connection.close()
    return jsonify([dict(r) for r in rows])


# ==========================================
# VEHICLE REGISTRY SEARCH WITH FILTERS
# ==========================================
@app.route("/vehicles/search", methods=["GET"])
def search_vehicles():
    plate = request.args.get("plate", "").strip().upper()
    state = request.args.get("state", "").strip().upper()
    rto = request.args.get("rto", "").strip().upper()

    query = "SELECT * FROM vehicles WHERE 1=1"
    params = []

    if plate:
        query += " AND UPPER(plate_number) LIKE ?"
        params.append(f"%{plate}%")

    if state:
        query += " AND UPPER(state_code) = ?"
        params.append(state)

    if rto:
        rto_clean = re.sub(r"\s+", " ", rto).strip()
        query += " AND UPPER(rto_code) = ?"
        params.append(rto_clean)

    query += " ORDER BY id DESC LIMIT 200"

    connection = get_connection()
    rows = connection.execute(query, params).fetchall()
    connection.close()

    return jsonify([dict(r) for r in rows])


# ==========================================
# FULL VEHICLE PROFILE (with detection history)
# ==========================================
@app.route("/vehicles/<plate_number>/profile", methods=["GET"])
def vehicle_full_profile(plate_number):
    plate_clean = clean_alphanumeric(plate_number)

    connection = get_connection()
    vehicle = connection.execute(
        "SELECT * FROM vehicles WHERE UPPER(plate_number) = ?",
        (plate_clean,)
    ).fetchone()

    history_rows = connection.execute(
        """
        SELECT id, image_name, ocr_confidence, best_method, status,
               location, detection_time, plate_quality, alert,
               watchlist_match, latitude, longitude, original_image, plate_image,
               toll_gate_name, validation_status, is_demo
        FROM detections
        WHERE UPPER(plate_number) = ?
        ORDER BY detection_time DESC
        LIMIT 50
        """,
        (plate_clean,)
    ).fetchall()
    connection.close()

    vehicle_data = dict(vehicle) if vehicle else None

    # If vehicle not in registry but has detections, build a basic profile
    if not vehicle_data:
        vehicle_data = {
            "plate_number": plate_clean,
            "owner_name": "Not available",
            "registration_status": "Not available",
            "manufacturer_model": "Not available",
            "fuel_type": "Not available",
            "vehicle_color": "Not available",
            "registration_year": None,
            "registration_date": "Not available",
            "purchase_date": "Not available",
            "purchase_condition": "Not available",
            "previous_owners_count": None,
            "previous_ownership_history": "Not available",
            "insurance_expiry": "Not available",
            "fitness_expiry": "Not available",
            "pollution_expiry": "Not available",
            "state_code": "Unverified",
            "state_name": "Unverified",
            "rto_code": "Unverified",
            "rto_name": "Unverified",
            "is_demo": 0,
            "notes": None,
        }
        # Try to extract state/RTO from plate itself
        try:
            sc, sn, rc, rn = extract_state_and_rto(plate_clean)
            vehicle_data["state_code"] = sc
            vehicle_data["state_name"] = sn
            vehicle_data["rto_code"] = rc
            vehicle_data["rto_name"] = rn
        except Exception:
            pass

    history_list = [row_to_dict(r) for r in history_rows]

    return jsonify({
        "vehicle": vehicle_data,
        "detection_history": history_list,
        "total_detections": len(history_list),
    })


# ==========================================
# WATCHLIST
# ==========================================
@app.route("/watchlist", methods=["GET"])
def get_watchlist():
    connection = get_connection()
    rows = connection.execute(
        "SELECT * FROM watchlist ORDER BY id DESC"
    ).fetchall()
    connection.close()

    return jsonify([dict(row) for row in rows])


@app.route("/watchlist", methods=["POST"])
def add_watchlist():
    data = request.get_json(silent=True) or {}

    plate_number = clean_ocr_text(data.get("plate_number", ""))
    reason = str(data.get("reason", "Watchlist vehicle"))

    if not plate_number:
        return jsonify({
            "error": "plate_number is required"
        }), 400

    added_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO watchlist
            (plate_number, reason, added_time)
            VALUES (?, ?, ?)
            """,
            (plate_number, reason, added_time)
        )
        connection.commit()
    except sqlite3.IntegrityError:
        connection.close()
        return jsonify({
            "error": "Plate already exists in watchlist"
        }), 409

    connection.close()

    return jsonify({
        "message": "Plate added to watchlist",
        "plate_number": plate_number,
        "reason": reason,
        "added_time": added_time
    }), 201


@app.route("/watchlist/<plate_number>", methods=["DELETE"])
def delete_watchlist(plate_number):
    plate_number = clean_ocr_text(plate_number)

    connection = get_connection()
    cursor = connection.execute(
        "DELETE FROM watchlist WHERE UPPER(plate_number)=?",
        (plate_number,)
    )
    connection.commit()
    deleted = cursor.rowcount
    connection.close()

    if deleted == 0:
        return jsonify({"error": "Plate not found"}), 404

    return jsonify({
        "message": "Plate removed from watchlist",
        "plate_number": plate_number
    })


@app.route("/watchlist/check/<plate_number>", methods=["GET"])
def watchlist_check(plate_number):
    plate_number = clean_ocr_text(plate_number)
    match, reason = check_watchlist(plate_number)

    return jsonify({
        "plate_number": plate_number,
        "watchlist_match": match,
        "watchlist_reason": reason
    })


# ==========================================
# VEHICLE / OWNER DETAILS
# ==========================================
@app.route("/vehicles", methods=["GET"])
def get_vehicles():
    connection = get_connection()
    rows = connection.execute(
        "SELECT * FROM vehicles ORDER BY id DESC"
    ).fetchall()
    connection.close()

    return jsonify([dict(row) for row in rows])


@app.route("/vehicles/<plate_number>", methods=["GET"])
def get_vehicle(plate_number):
    details = get_vehicle_details(clean_ocr_text(plate_number))

    if not details:
        return jsonify({"error": "Vehicle not found"}), 404

    return jsonify(details)


@app.route("/vehicles", methods=["POST"])
def add_vehicle():
    data = request.get_json(silent=True) or {}

    plate_number = clean_ocr_text(data.get("plate_number", ""))

    if not plate_number:
        return jsonify({
            "error": "plate_number is required"
        }), 400

    owner_name = data.get("owner_name")
    registration_status = data.get("registration_status")
    vehicle_details = data.get("vehicle_details")
    notes = data.get("notes")

    connection = get_connection()

    try:
        connection.execute(
            """
            INSERT INTO vehicles (
                plate_number, owner_name, registration_status,
                vehicle_details, notes
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (plate_number, owner_name, registration_status, vehicle_details, notes)
        )
        connection.commit()
    except sqlite3.IntegrityError:
        connection.close()
        return jsonify({
            "error": "Vehicle already exists"
        }), 409

    connection.close()

    return jsonify({
        "message": "Vehicle details added",
        "plate_number": plate_number
    }), 201


@app.route("/vehicles/<plate_number>", methods=["DELETE"])
def delete_vehicle(plate_number):
    plate_number = clean_ocr_text(plate_number)

    connection = get_connection()
    cursor = connection.execute(
        "DELETE FROM vehicles WHERE UPPER(plate_number)=?",
        (plate_number,)
    )
    connection.commit()
    deleted = cursor.rowcount
    connection.close()

    if deleted == 0:
        return jsonify({"error": "Vehicle not found"}), 404

    return jsonify({
        "message": "Vehicle details removed",
        "plate_number": plate_number
    })


# ==========================================
# TOLL GATES
# ==========================================
@app.route("/toll_gates", methods=["GET"])
def get_toll_gates():
    connection = get_connection()
    rows = connection.execute(
        "SELECT * FROM toll_gates ORDER BY id ASC"
    ).fetchall()
    connection.close()
    return jsonify([dict(r) for r in rows])


@app.route("/toll_gates", methods=["POST"])
def add_toll_gate():
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    if not name:
        return jsonify({"error": "name is required"}), 400

    connection = get_connection()
    try:
        connection.execute(
            """
            INSERT INTO toll_gates (name, highway, city, state, latitude, longitude, address, is_demo)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                data.get("highway"),
                data.get("city"),
                data.get("state"),
                data.get("latitude"),
                data.get("longitude"),
                data.get("address"),
                0,
            )
        )
        connection.commit()
    except sqlite3.IntegrityError:
        connection.close()
        return jsonify({"error": "Toll gate already exists"}), 409

    connection.close()
    return jsonify({"message": "Toll gate added", "name": name}), 201


# ==========================================
# ANALYTICS - DAILY
# ==========================================
@app.route("/analytics/daily", methods=["GET"])
def analytics_daily():
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT substr(detection_time, 1, 10) AS date,
               COUNT(*) AS total,
               SUM(CASE WHEN status='Recognized' THEN 1 ELSE 0 END) AS recognized,
               SUM(CASE WHEN status='Low Confidence' THEN 1 ELSE 0 END) AS low_confidence,
               SUM(CASE WHEN status IN ('Detection Failed','OCR Failed')
                        THEN 1 ELSE 0 END) AS failed
        FROM detections
        GROUP BY substr(detection_time, 1, 10)
        ORDER BY date
        """
    ).fetchall()
    connection.close()

    return jsonify([dict(row) for row in rows])



# ==========================================
# ANALYTICS - REPEATED
# ==========================================
@app.route("/analytics/repeated", methods=["GET"])
def analytics_repeated():
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT plate_number,
               COUNT(*) AS detection_count,
               MAX(detection_time) AS last_detected
        FROM detections
        WHERE plate_number NOT IN ('Not Recognized', 'Not Detected')
        GROUP BY plate_number
        HAVING COUNT(*) > 1
        ORDER BY detection_count DESC
        """
    ).fetchall()
    connection.close()

    return jsonify([dict(row) for row in rows])


# ==========================================
# ANALYTICS - WATCHLIST
# ==========================================
@app.route("/analytics/watchlist", methods=["GET"])
def analytics_watchlist():
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT plate_number,
               COUNT(*) AS match_count,
               MAX(detection_time) AS last_detected
        FROM detections
        WHERE watchlist_match = 1
        GROUP BY plate_number
        ORDER BY match_count DESC
        """
    ).fetchall()
    connection.close()

    return jsonify([dict(row) for row in rows])


# ==========================================
# MAP DETECTIONS
# ==========================================
@app.route("/map/detections", methods=["GET"])
def map_detections():
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT id, plate_number, ocr_confidence, status,
               alert, location, detection_time,
               latitude, longitude, plate_quality
        FROM detections
        WHERE latitude IS NOT NULL
          AND longitude IS NOT NULL
        ORDER BY id DESC
        """
    ).fetchall()
    connection.close()

    return jsonify([dict(row) for row in rows])


# ==========================================
# CSV REPORT
# ==========================================
@app.route("/report/csv", methods=["GET"])
def report_csv():
    state = request.args.get("state", "").strip().upper()
    rto = request.args.get("rto", "").strip().upper()

    query = "SELECT * FROM detections WHERE 1=1"
    params = []
    if state:
        query += " AND UPPER(state_code) = ?"
        params.append(state)
    if rto:
        rto_clean = re.sub(r"\s+", " ", rto).strip()
        query += " AND UPPER(rto_code) = ?"
        params.append(rto_clean)
    query += " ORDER BY id DESC"

    connection = get_connection()
    rows = connection.execute(query, params).fetchall()
    connection.close()

    output = io.StringIO()
    writer = csv.writer(output)

    headers = [
        "id", "image_name", "plate_number", "display_plate", "ocr_confidence",
        "best_method", "status", "validation_status", "location", "detection_time",
        "plate_quality", "alert", "watchlist_match", "watchlist_reason",
        "previous_detections", "state_code", "state_name", "rto_code", "rto_name",
        "toll_gate_name", "latitude", "longitude", "original_image", "plate_image",
        "is_demo"
    ]
    writer.writerow(headers)

    for row in rows:
        d = dict(row)
        writer.writerow([d.get(h, "") for h in headers])

    data = io.BytesIO(output.getvalue().encode("utf-8"))
    data.seek(0)

    return send_file(
        data,
        mimetype="text/csv",
        as_attachment=True,
        download_name="anpr_detection_report.csv"
    )


# ==========================================
# PDF REPORT
# ==========================================
@app.route("/report/pdf", methods=["GET"])
def report_pdf():
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import (
            SimpleDocTemplate,
            Table,
            TableStyle,
            Paragraph
        )
        from reportlab.lib.styles import getSampleStyleSheet
    except ImportError:
        return jsonify({
            "error": "PDF report requires reportlab. Install it with: pip install reportlab"
        }), 500

    state = request.args.get("state", "").strip().upper()
    rto = request.args.get("rto", "").strip().upper()

    query = """
        SELECT id, plate_number, display_plate, ocr_confidence, status,
               state_code, rto_code, location, detection_time, alert, is_demo
        FROM detections WHERE 1=1
    """
    params = []
    if state:
        query += " AND UPPER(state_code) = ?"
        params.append(state)
    if rto:
        rto_clean = re.sub(r"\s+", " ", rto).strip()
        query += " AND UPPER(rto_code) = ?"
        params.append(rto_clean)
    query += " ORDER BY id DESC"

    connection = get_connection()
    rows = connection.execute(query, params).fetchall()
    connection.close()

    pdf_buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=landscape(A4),
        rightMargin=20,
        leftMargin=20,
        topMargin=20,
        bottomMargin=20
    )

    styles = getSampleStyleSheet()
    title_text = "ANPR Detection Report"
    if state:
        title_text += f" — State: {state}"
    if rto:
        title_text += f" | RTO: {rto}"
    elements = [Paragraph(title_text, styles["Title"])]

    data = [[
        "ID", "Plate", "Confidence", "Status",
        "State", "RTO", "Location", "Detection Time", "Alert", "Demo?"
    ]]

    for row in rows:
        data.append([
            row["id"],
            row["display_plate"] or row["plate_number"],
            f'{row["ocr_confidence"]}%',
            row["status"],
            row["state_code"] or "--",
            row["rto_code"] or "--",
            row["location"],
            row["detection_time"],
            row["alert"],
            "YES" if row["is_demo"] else "No"
        ])

    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e40af")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("VALIGN", (0, 0), (-1, -1), "TOP")
    ]))

    elements.append(table)
    doc.build(elements)

    pdf_buffer.seek(0)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name="anpr_detection_report.pdf"
    )


# ==========================================
# EVIDENCE FILES
# ==========================================
@app.route("/evidence/<path:filename>", methods=["GET"])
def evidence_file(filename):
    full_path = os.path.abspath(
        os.path.join(EVIDENCE_DIR, filename)
    )

    evidence_root = os.path.abspath(EVIDENCE_DIR)

    if not full_path.startswith(evidence_root + os.sep):
        return jsonify({"error": "Invalid evidence path"}), 400

    if not os.path.isfile(full_path):
        return jsonify({"error": "Evidence file not found"}), 404

    return send_file(full_path)


# ==========================================
# INDIAN STATE & RTO INFO (for frontend dropdowns)
# ==========================================
@app.route("/indian_states", methods=["GET"])
def indian_states():
    return jsonify([
        {"code": code, "name": name}
        for code, name in sorted(INDIAN_STATES.items())
    ])


@app.route("/rto_codes", methods=["GET"])
def rto_codes():
    state = request.args.get("state", "").strip().upper()
    if state:
        filtered = {
            k: v for k, v in RTO_MAPPINGS.items()
            if k.startswith(state)
        }
    else:
        filtered = RTO_MAPPINGS

    result = []
    for code, name in sorted(filtered.items()):
        state_code = code[:2]
        rto_num = code[2:]
        display = f"{state_code} {int(rto_num):02d}"
        result.append({"code": display, "raw_code": code, "name": name, "state": state_code})

    return jsonify(result)


# ==========================================
# HOURLY TRAFFIC ANALYTICS
# ==========================================
@app.route("/analytics/hourly", methods=["GET"])
def analytics_hourly():
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT strftime('%H', detection_time) AS hour,
               COUNT(*) AS total,
               SUM(CASE WHEN status='Recognized' THEN 1 ELSE 0 END) AS recognized,
               SUM(CASE WHEN status='Low Confidence' THEN 1 ELSE 0 END) AS low_confidence,
               AVG(CASE WHEN status='Recognized' THEN ocr_confidence ELSE NULL END) AS avg_conf
        FROM detections
        WHERE detection_time IS NOT NULL
        GROUP BY hour
        ORDER BY hour
        """
    ).fetchall()
    connection.close()

    hour_labels = {
        "00": "12 AM", "01": "1 AM", "02": "2 AM", "03": "3 AM",
        "04": "4 AM", "05": "5 AM", "06": "6 AM", "07": "7 AM",
        "08": "8 AM", "09": "9 AM", "10": "10 AM", "11": "11 AM",
        "12": "12 PM", "13": "1 PM", "14": "2 PM", "15": "3 PM",
        "16": "4 PM", "17": "5 PM", "18": "6 PM", "19": "7 PM",
        "20": "8 PM", "21": "9 PM", "22": "10 PM", "23": "11 PM",
    }
    result = []
    for row in rows:
        h = row["hour"] or "00"
        result.append({
            "hour": h,
            "label": hour_labels.get(h, f"{h}:00"),
            "total": row["total"],
            "recognized": row["recognized"] or 0,
            "low_confidence": row["low_confidence"] or 0,
            "avg_confidence": round(row["avg_conf"] or 0, 1),
        })
    return jsonify(result)


# ==========================================
# OCR ACCURACY METRICS
# ==========================================
@app.route("/analytics/ocr-accuracy", methods=["GET"])
def analytics_ocr_accuracy():
    connection = get_connection()

    total = connection.execute(
        "SELECT COUNT(*) AS c FROM detections"
    ).fetchone()["c"]

    recognized = connection.execute(
        "SELECT COUNT(*) AS c FROM detections WHERE status='Recognized'"
    ).fetchone()["c"]

    low_conf = connection.execute(
        "SELECT COUNT(*) AS c FROM detections WHERE status='Low Confidence'"
    ).fetchone()["c"]

    failed = connection.execute(
        "SELECT COUNT(*) AS c FROM detections WHERE status IN ('Detection Failed','OCR Failed')"
    ).fetchone()["c"]

    avg_conf_row = connection.execute(
        "SELECT AVG(ocr_confidence) AS avg FROM detections WHERE status='Recognized' AND ocr_confidence > 0"
    ).fetchone()
    avg_conf = round(avg_conf_row["avg"] or 0, 1)

    duplicate_count = connection.execute(
        "SELECT COUNT(*) AS c FROM detections WHERE is_duplicate=1"
    ).fetchone()["c"]

    connection.close()

    success_rate = round((recognized / total * 100), 1) if total > 0 else 0.0

    return jsonify({
        "total_detections": total,
        "recognized_count": recognized,
        "low_confidence_count": low_conf,
        "failed_count": failed,
        "avg_confidence": avg_conf,
        "success_rate_pct": success_rate,
        "duplicate_suppressed": duplicate_count,
    })


# ==========================================
# CAMERAS / SOURCES LIST
# ==========================================
@app.route("/cameras", methods=["GET"])
def get_cameras():
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT camera_id,
               COUNT(*) AS detection_count,
               MAX(detection_time) AS last_seen
        FROM detections
        WHERE camera_id IS NOT NULL AND camera_id != ''
        GROUP BY camera_id
        ORDER BY last_seen DESC
        """
    ).fetchall()
    connection.close()

    result = []
    for row in rows:
        cid = row["camera_id"]
        result.append({
            "id": cid,
            "name": cid,
            "last_seen": row["last_seen"],
            "detection_count": row["detection_count"],
        })
    # Always include default cameras if no records yet
    known_ids = {r["id"] for r in result}
    defaults = [
        {"id": "WEBCAM-1", "name": "Webcam 1", "last_seen": None, "detection_count": 0},
        {"id": "CAM-01",   "name": "Camera 01", "last_seen": None, "detection_count": 0},
        {"id": "CAM-02",   "name": "Camera 02", "last_seen": None, "detection_count": 0},
    ]
    for d in defaults:
        if d["id"] not in known_ids:
            result.append(d)
    return jsonify(result)


# ==========================================
# LANE-WISE SUMMARY
# ==========================================
@app.route("/analytics/lane-summary", methods=["GET"])
def analytics_lane_summary():
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT lane_id,
               COUNT(*) AS total,
               SUM(CASE WHEN status='Recognized' THEN 1 ELSE 0 END) AS recognized,
               AVG(CASE WHEN status='Recognized' THEN ocr_confidence ELSE NULL END) AS avg_confidence
        FROM detections
        WHERE lane_id IS NOT NULL AND lane_id != ''
        GROUP BY lane_id
        ORDER BY total DESC
        """
    ).fetchall()
    connection.close()
    return jsonify([{
        "lane_id": row["lane_id"],
        "total": row["total"],
        "recognized": row["recognized"] or 0,
        "avg_confidence": round(row["avg_confidence"] or 0, 1),
    } for row in rows])


# ==========================================
# DATABASE BACKUP / RESTORE
# ==========================================
@app.route("/backup/export", methods=["GET"])
def backup_export():
    """Download the entire SQLite database as a backup file."""
    if not os.path.isfile(DATABASE):
        return jsonify({"error": "Database not found"}), 404
    return send_file(
        os.path.abspath(DATABASE),
        as_attachment=True,
        download_name="anpr_backup.db",
        mimetype="application/octet-stream"
    )


@app.route("/backup/restore", methods=["POST"])
def backup_restore():
    """Upload a .db file to restore the database. Creates a backup of current DB first."""
    if "db_file" not in request.files:
        return jsonify({"error": "No file uploaded. Field name must be 'db_file'"}), 400

    uploaded = request.files["db_file"]
    if not uploaded.filename:
        return jsonify({"error": "No filename"}), 400

    db_bytes = uploaded.read()
    if len(db_bytes) < 100:
        return jsonify({"error": "File too small to be a valid SQLite database"}), 400

    # Validate SQLite magic bytes
    if db_bytes[:16] != b"SQLite format 3\x00":
        return jsonify({"error": "Uploaded file is not a valid SQLite database"}), 400

    # Validate it has 'detections' table
    tmp_path = DATABASE + ".restore_tmp"
    with open(tmp_path, "wb") as f:
        f.write(db_bytes)

    try:
        check_conn = sqlite3.connect(tmp_path)
        tables = [r[0] for r in check_conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        check_conn.close()
        if "detections" not in tables:
            os.remove(tmp_path)
            return jsonify({"error": "Database does not contain a 'detections' table"}), 400
    except Exception as e:
        os.remove(tmp_path)
        return jsonify({"error": f"Could not read database: {str(e)}"}), 400

    # Back up current database
    import shutil
    backup_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{DATABASE}.bak_{backup_stamp}"
    if os.path.isfile(DATABASE):
        shutil.copy2(DATABASE, backup_path)

    # Replace database
    shutil.move(tmp_path, DATABASE)

    return jsonify({
        "message": "Database restored successfully",
        "backup_created": os.path.basename(backup_path),
        "tables_found": tables,
    })


# ==========================================
# UPDATE VEHICLE (PATCH)
# ==========================================
@app.route("/vehicles/<plate_number>", methods=["PATCH"])
def update_vehicle(plate_number):
    """Update vehicle profile fields."""
    plate_clean = clean_ocr_text(plate_number)
    data = request.get_json(silent=True) or {}

    allowed_fields = {
        "owner_name", "manufacturer_model", "fuel_type", "vehicle_color",
        "registration_status", "insurance_expiry", "fitness_expiry",
        "pollution_expiry", "notes", "purchase_condition",
        "previous_owners_count", "registration_date", "registration_year",
        "purchase_date", "previous_ownership_history",
    }

    updates = {k: v for k, v in data.items() if k in allowed_fields}
    if not updates:
        return jsonify({"error": "No valid fields provided"}), 400

    set_clause = ", ".join(f"{col} = ?" for col in updates)
    values = list(updates.values()) + [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), plate_clean]

    connection = get_connection()
    cursor = connection.execute(
        f"UPDATE vehicles SET {set_clause}, updated_time = ? WHERE UPPER(plate_number) = ?",
        values
    )
    connection.commit()
    affected = cursor.rowcount
    connection.close()

    if affected == 0:
        return jsonify({"error": "Vehicle not found"}), 404

    return jsonify({"message": "Vehicle updated", "plate_number": plate_clean})


# ==========================================
# REPEAT VEHICLE TRACKING (ENHANCED)
# ==========================================
@app.route("/analytics/repeat-vehicles", methods=["GET"])
def repeat_vehicles():
    """Vehicles detected more than once with first/last seen and visit counts."""
    connection = get_connection()
    rows = connection.execute(
        """
        SELECT plate_number, display_plate, state_code, rto_code,
               COUNT(*) AS detection_count,
               MIN(detection_time) AS first_seen,
               MAX(detection_time) AS last_seen,
               MAX(ocr_confidence) AS best_confidence,
               SUM(CASE WHEN watchlist_match=1 THEN 1 ELSE 0 END) AS watchlist_hits
        FROM detections
        WHERE plate_number NOT IN ('Not Detected', 'Not Recognized')
          AND plate_number IS NOT NULL
        GROUP BY plate_number
        HAVING COUNT(*) > 1
        ORDER BY detection_count DESC
        LIMIT 100
        """
    ).fetchall()
    connection.close()
    return jsonify([dict(r) for r in rows])



# ==========================================
# LOCATION-SPECIFIC HOURLY ANALYTICS
# ==========================================
@app.route("/analytics/hourly-by-location", methods=["GET"])
def analytics_hourly_by_location():
    """
    Hourly traffic breakdown filtered by optional toll_gate_id, toll_gate_name, lane_id,
    camera_id, date_from, and date_to. Returns 24-hour row array same shape as /analytics/hourly.
    Also returns metadata: available toll gates and date range.
    """
    toll_gate_id_f = request.args.get("toll_gate_id", "").strip()
    toll_gate_name_f = request.args.get("toll_gate_name", "").strip()
    lane_id_f = request.args.get("lane_id", "").strip()
    camera_id_f = request.args.get("camera_id", "").strip()
    date_from_f = request.args.get("date_from", "").strip()
    date_to_f = request.args.get("date_to", "").strip()

    where_clauses = ["detection_time IS NOT NULL"]
    params = []

    if toll_gate_id_f:
        try:
            where_clauses.append("toll_gate_id = ?")
            params.append(int(toll_gate_id_f))
        except ValueError:
            pass

    if toll_gate_name_f:
        where_clauses.append("LOWER(toll_gate_name) LIKE ?")
        params.append(f"%{toll_gate_name_f.lower()}%")

    if lane_id_f:
        where_clauses.append("lane_id = ?")
        params.append(lane_id_f)

    if camera_id_f:
        where_clauses.append("camera_id = ?")
        params.append(camera_id_f)

    if date_from_f:
        where_clauses.append("detection_time >= ?")
        params.append(date_from_f)

    if date_to_f:
        where_clauses.append("detection_time <= ?")
        params.append(date_to_f + " 23:59:59")

    where_sql = " AND ".join(where_clauses)

    connection = get_connection()

    # Hourly breakdown
    rows = connection.execute(
        f"""
        SELECT strftime('%H', detection_time) AS hour,
               COUNT(*) AS total,
               SUM(CASE WHEN status='Recognized' THEN 1 ELSE 0 END) AS recognized,
               SUM(CASE WHEN status='Low Confidence' THEN 1 ELSE 0 END) AS low_confidence,
               AVG(CASE WHEN status='Recognized' THEN ocr_confidence ELSE NULL END) AS avg_conf
        FROM detections
        WHERE {where_sql}
        GROUP BY hour
        ORDER BY hour
        """,
        params
    ).fetchall()

    # Toll gate list for the dropdown (distinct toll gates in DB)
    gate_rows = connection.execute(
        """
        SELECT DISTINCT toll_gate_id, toll_gate_name
        FROM detections
        WHERE toll_gate_id IS NOT NULL AND toll_gate_name IS NOT NULL AND toll_gate_name != ''
        ORDER BY toll_gate_name
        """
    ).fetchall()

    # Lane list
    lane_rows = connection.execute(
        """
        SELECT DISTINCT lane_id
        FROM detections
        WHERE lane_id IS NOT NULL AND lane_id != ''
        ORDER BY lane_id
        """
    ).fetchall()

    # Date range available
    dr = connection.execute(
        "SELECT MIN(date(detection_time)) AS min_d, MAX(date(detection_time)) AS max_d FROM detections"
    ).fetchone()

    connection.close()

    hour_labels = {
        "00": "12 AM", "01": "1 AM", "02": "2 AM", "03": "3 AM",
        "04": "4 AM", "05": "5 AM", "06": "6 AM", "07": "7 AM",
        "08": "8 AM", "09": "9 AM", "10": "10 AM", "11": "11 AM",
        "12": "12 PM", "13": "1 PM", "14": "2 PM", "15": "3 PM",
        "16": "4 PM", "17": "5 PM", "18": "6 PM", "19": "7 PM",
        "20": "8 PM", "21": "9 PM", "22": "10 PM", "23": "11 PM",
    }

    result_rows = []
    for row in rows:
        h = row["hour"] or "00"
        result_rows.append({
            "hour": h,
            "label": hour_labels.get(h, f"{h}:00"),
            "total": row["total"],
            "recognized": row["recognized"] or 0,
            "low_confidence": row["low_confidence"] or 0,
            "avg_confidence": round(row["avg_conf"] or 0, 1),
        })

    toll_gates_meta = [
        {"id": r["toll_gate_id"], "name": r["toll_gate_name"]}
        for r in gate_rows
    ]

    lanes_meta = [r["lane_id"] for r in lane_rows]

    return jsonify({
        "hourly": result_rows,
        "toll_gates": toll_gates_meta,
        "lanes": lanes_meta,
        "date_range": {
            "min": dr["min_d"] if dr else None,
            "max": dr["max_d"] if dr else None,
        },
        "filters_applied": {
            "toll_gate_id": toll_gate_id_f or None,
            "toll_gate_name": toll_gate_name_f or None,
            "lane_id": lane_id_f or None,
            "camera_id": camera_id_f or None,
            "date_from": date_from_f or None,
            "date_to": date_to_f or None,
        },
    })


# ==========================================
# START SERVER
# ==========================================
if __name__ == "__main__":
    init_database()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
