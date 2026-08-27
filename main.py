import os
import cv2
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ultralytics import YOLO


# ============================================================
# INFRA SIGHT CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = (
    BASE_DIR
    / "model"
    / "best.pt"
)
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "infrasight.db"

OUTPUT_DIR = BASE_DIR / "output"
INPUT_DIR = BASE_DIR / "data" / "input"

DASHBOARD_DIR = BASE_DIR / "dashboard"

DATABASE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="InfraSight API",
    description="AI-Powered Road Infrastructure Monitoring System",
    version="1.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/output",
    StaticFiles(directory=str(OUTPUT_DIR)),
    name="output"
)

app.mount(
    "/dashboard",
    StaticFiles(directory=str(DASHBOARD_DIR), html=True),
    name="dashboard"
)


# ============================================================
# YOLO MODEL
# ============================================================

print("=" * 60)
print("InfraSight starting...")
print("=" * 60)

if not MODEL_PATH.exists():

    print("WARNING: YOLO model not found:")
    print(MODEL_PATH)

    model = None

else:

    print("Loading YOLO model:")
    print(MODEL_PATH)

    try:

        model = YOLO(str(MODEL_PATH))

        print("YOLO model loaded successfully.")
        print("Classes:", model.names)

    except Exception as e:

        print("ERROR loading YOLO model:", e)

        model = None


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        str(DATABASE_PATH)
    )

    connection.row_factory = sqlite3.Row

    return connection


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

def initialize_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (

            detection_id INTEGER PRIMARY KEY AUTOINCREMENT,

            segment_id TEXT,

            defect_class TEXT,

            confidence REAL,

            severity TEXT,

            timestamp TEXT,

            frame_number INTEGER DEFAULT 0,

            evidence_path TEXT,

            image TEXT,

            persistence TEXT,

            persistence_score REAL,

            deterioration TEXT,

            deterioration_score REAL,

            priority TEXT,

            recommended_action TEXT

        )
    """)

    conn.commit()

    # --------------------------------------------------------
    # Repair / upgrade old database
    # --------------------------------------------------------

    cursor.execute(
        "PRAGMA table_info(detections)"
    )

    columns = [
        row["name"]
        for row in cursor.fetchall()
    ]

    required_columns = {

        "segment_id": "TEXT",

        "defect_class": "TEXT",

        "confidence": "REAL",

        "severity": "TEXT",

        "timestamp": "TEXT",

        "frame_number": "INTEGER DEFAULT 0",

        "evidence_path": "TEXT",

        "image": "TEXT",

        "persistence": "TEXT",

        "persistence_score": "REAL",

        "deterioration": "TEXT",

        "deterioration_score": "REAL",

        "priority": "TEXT",

        "recommended_action": "TEXT"
    }

    for column, datatype in required_columns.items():

        if column not in columns:

            cursor.execute(
                f"ALTER TABLE detections ADD COLUMN {column} {datatype}"
            )

    # --------------------------------------------------------
    # Repair detection_id if old database needs it
    # --------------------------------------------------------

    if "detection_id" not in columns:

        cursor.execute(
            "ALTER TABLE detections ADD COLUMN detection_id INTEGER"
        )

        cursor.execute("""
            UPDATE detections
            SET detection_id = rowid
            WHERE detection_id IS NULL
        """)

    conn.commit()
    conn.close()

    print("Database initialized successfully.")


initialize_database()


# ============================================================
# SEGMENT ID
# ============================================================

def generate_segment_id():

    return "segment_" + uuid.uuid4().hex[:6]


# ============================================================
# OUTPUT FILE NAME
# ============================================================

def generate_output_filename():

    now = datetime.now()

    timestamp = now.strftime(
        "%Y%m%d_%H%M%S"
    )

    random_id = uuid.uuid4().hex[:6]

    return (
        f"detection_{timestamp}_{random_id}.jpg"
    )


# ============================================================
# SEVERITY
# ============================================================

def calculate_severity(
    defect_class,
    confidence
):

    defect = str(
        defect_class
    ).upper()

    confidence = float(
        confidence
    )

    # --------------------------------------------------------
    # Known RDD-style classes
    # --------------------------------------------------------

    if defect == "D00":

        return "Low"

    if defect == "D10":

        return "Medium"

    if defect == "D20":

        return "High"

    if defect == "D40":

        return "High"

    if defect == "D50":

        return "High"

    # --------------------------------------------------------
    # Generic fallback
    # --------------------------------------------------------

    if confidence >= 0.75:

        return "High"

    if confidence >= 0.45:

        return "Medium"

    return "Low"


# ============================================================
# INTELLIGENCE:
# PERSISTENCE
# ============================================================

def calculate_persistence(
    conn,
    segment_id,
    defect_class
):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) AS count
        FROM detections
        WHERE segment_id = ?
          AND defect_class = ?
    """, (
        segment_id,
        defect_class
    ))

    row = cursor.fetchone()

    previous_count = int(
        row["count"] or 0
    )

    # --------------------------------------------------------
    # This detection itself becomes observation number +1.
    # --------------------------------------------------------

    observation_count = (
        previous_count + 1
    )

    # --------------------------------------------------------
    # Persistence score:
    #
    # 1 observation  = 0.00
    # 2 observations = 0.50
    # 3 observations = 0.75
    # 4+             = 1.00
    # --------------------------------------------------------

    if observation_count <= 1:

        score = 0.0
        label = "New"

    elif observation_count == 2:

        score = 0.50
        label = "Persistent"

    elif observation_count == 3:

        score = 0.75
        label = "Persistent"

    else:

        score = 1.0
        label = "Persistent"

    return (
        label,
        score,
        observation_count
    )


# ============================================================
# INTELLIGENCE:
# DETERIORATION
# ============================================================

def calculate_deterioration(
    conn,
    segment_id,
    defect_class,
    current_confidence
):

    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            confidence,
            timestamp
        FROM detections
        WHERE segment_id = ?
          AND defect_class = ?
        ORDER BY datetime(timestamp) DESC
        LIMIT 1
    """, (
        segment_id,
        defect_class
    ))

    previous = cursor.fetchone()

    # --------------------------------------------------------
    # No previous observation
    # --------------------------------------------------------

    if previous is None:

        return (
            "Stable",
            0.0,
            0.0
        )

    previous_confidence = float(
        previous["confidence"] or 0
    )

    current_confidence = float(
        current_confidence
    )

    change = (
        current_confidence
        - previous_confidence
    )

    # --------------------------------------------------------
    # Convert confidence change to score.
    #
    # Positive confidence growth means deterioration.
    # --------------------------------------------------------

    score = max(
        0.0,
        min(
            1.0,
            change / 0.50
        )
    )

    if change >= 0.20:

        label = "Rapid deterioration"

    elif change >= 0.05:

        label = "Deteriorating"

    elif change <= -0.05:

        label = "Improving"

    else:

        label = "Stable"

    return (
        label,
        score,
        change
    )


# ============================================================
# INTELLIGENCE:
# PRIORITY
# ============================================================

def calculate_priority(
    severity,
    persistence_score,
    deterioration_score
):

    severity = str(
        severity
    ).lower()

    persistence_score = float(
        persistence_score
    )

    deterioration_score = float(
        deterioration_score
    )

    severity_score = {

        "low": 0.25,

        "medium": 0.60,

        "high": 1.00

    }.get(
        severity,
        0.25
    )

    # --------------------------------------------------------
    # Weighted infrastructure risk score
    #
    # Severity       = 50%
    # Persistence    = 25%
    # Deterioration  = 25%
    # --------------------------------------------------------

    risk_score = (

        severity_score * 0.50

        + persistence_score * 0.25

        + deterioration_score * 0.25

    )

    if risk_score >= 0.80:

        return "Critical"

    if risk_score >= 0.60:

        return "High"

    if risk_score >= 0.35:

        return "Medium"

    return "Low"


# ============================================================
# INTELLIGENCE:
# RECOMMENDED ACTION
# ============================================================

def calculate_recommended_action(
    priority,
    severity,
    persistence,
    deterioration
):

    priority = str(
        priority
    ).lower()

    if priority == "critical":

        return (
            "Immediate inspection and urgent road repair"
        )

    if priority == "high":

        if str(deterioration).lower() in {
            "deteriorating",
            "rapid deterioration"
        }:

            return (
                "Prioritize repair and schedule field inspection"
            )

        if str(persistence).lower() == "persistent":

            return (
                "Schedule field inspection and maintenance"
            )

        return (
            "Schedule maintenance inspection"
        )

    if priority == "medium":

        return (
            "Monitor condition and plan maintenance"
        )

    return (
        "Continue routine monitoring"
    )


# ============================================================
# SAVE DETECTION
# ============================================================

def save_detection(

    segment_id,

    defect_class,

    confidence,

    severity,

    timestamp,

    frame_number,

    evidence_path,

    persistence,

    persistence_score,

    deterioration,

    deterioration_score,

    priority,

    recommended_action

):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO detections (

            segment_id,

            defect_class,

            confidence,

            severity,

            timestamp,

            frame_number,

            evidence_path,

            image,

            persistence,

            persistence_score,

            deterioration,

            deterioration_score,

            priority,

            recommended_action

        )

        VALUES (

            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?

        )
    """, (

        segment_id,

        defect_class,

        float(confidence),

        severity,

        timestamp,

        int(frame_number),

        evidence_path,

        evidence_path,

        persistence,

        float(persistence_score),

        deterioration,

        float(deterioration_score),

        priority,

        recommended_action

    ))

    detection_id = cursor.lastrowid

    conn.commit()

    conn.close()

    return detection_id


# ============================================================
# TIMELINE
# ============================================================

def get_detection_history(
    limit=100
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT

            detection_id,

            segment_id,

            defect_class,

            confidence,

            severity,

            timestamp,

            frame_number,

            evidence_path,

            image,

            persistence,

            persistence_score,

            deterioration,

            deterioration_score,

            priority,

            recommended_action

        FROM detections

        WHERE detection_id IS NOT NULL

        ORDER BY datetime(timestamp) DESC

        LIMIT ?
    """, (
        int(limit),
    ))

    rows = cursor.fetchall()

    conn.close()

    observations = []

    for row in rows:

        evidence = row["evidence_path"]

        observations.append({

            "detection_id":
                row["detection_id"],

            "segment_id":
                row["segment_id"],

            "defect_class":
                row["defect_class"],

            "confidence":
                float(
                    row["confidence"] or 0
                ),

            "severity":
                row["severity"] or "Low",

            "timestamp":
                row["timestamp"],

            "frame_number":
                row["frame_number"] or 0,

            "evidence_path":
                evidence,

            "image":
                row["image"] or evidence,

            "persistence":
                row["persistence"] or "New",

            "persistence_score":
                float(
                    row["persistence_score"] or 0
                ),

            "deterioration":
                row["deterioration"] or "Stable",

            "deterioration_score":
                float(
                    row["deterioration_score"] or 0
                ),

            "priority":
                row["priority"] or "Low",

            "recommended_action":
                row["recommended_action"]
                or "Continue routine monitoring",

            "message":
                "AI detection recorded"

        })

    return observations


# ============================================================
# YOLO DETECTION
# ============================================================

def run_yolo_detection(

    image_path,

    frame_number=0,

    requested_segment_id=None

):

    if model is None:

        raise RuntimeError(
            "YOLO model is not loaded."
        )

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise RuntimeError(
            "Unable to read image."
        )

    # --------------------------------------------------------
    # YOLO inference
    # --------------------------------------------------------

    results = model.predict(

        source=image,

        conf=0.10,

        verbose=False

    )

    result = results[0]

    detections = []

    timestamp = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use supplied segment for repeated observation.
    # Otherwise generate a new segment.
    # --------------------------------------------------------

    if requested_segment_id:

        segment_id = str(
            requested_segment_id
        )

    else:

        segment_id = generate_segment_id()

    # --------------------------------------------------------
    # Extract detections
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            names = result.names

            defect_class = names.get(
                cls_id,
                str(cls_id)
            )

            severity = calculate_severity(

                defect_class,

                confidence

            )

            detections.append({

                "defect_class":
                    defect_class,

                "confidence":
                    confidence,

                "severity":
                    severity

            })

    # --------------------------------------------------------
    # Annotated image
    # --------------------------------------------------------

    annotated = result.plot()

    output_filename = (
        generate_output_filename()
    )

    output_path = (
        OUTPUT_DIR / output_filename
    )

    cv2.imwrite(

        str(output_path),

        annotated

    )

    evidence_path = (
        f"/output/{output_filename}"
    )

    # --------------------------------------------------------
    # Calculate intelligence + save
    # --------------------------------------------------------

    saved_records = []

    for detection in detections:

        defect_class = (
            detection["defect_class"]
        )

        confidence = (
            detection["confidence"]
        )

        severity = (
            detection["severity"]
        )

        conn = get_connection()

        (
            persistence,
            persistence_score,
            observation_count

        ) = calculate_persistence(

            conn,

            segment_id,

            defect_class

        )

        (
            deterioration,
            deterioration_score,
            confidence_change

        ) = calculate_deterioration(

            conn,

            segment_id,

            defect_class,

            confidence

        )

        conn.close()

        priority = calculate_priority(

            severity,

            persistence_score,

            deterioration_score

        )

        recommended_action = (
            calculate_recommended_action(

                priority,

                severity,

                persistence,

                deterioration

            )
        )

        detection_id = save_detection(

            segment_id=segment_id,

            defect_class=defect_class,

            confidence=confidence,

            severity=severity,

            timestamp=timestamp,

            frame_number=frame_number,

            evidence_path=evidence_path,

            persistence=persistence,

            persistence_score=persistence_score,

            deterioration=deterioration,

            deterioration_score=deterioration_score,

            priority=priority,

            recommended_action=recommended_action

        )

        saved_records.append({

            "detection_id":
                detection_id,

            "segment_id":
                segment_id,

            "defect_class":
                defect_class,

            "confidence":
                confidence,

            "severity":
                severity,

            "timestamp":
                timestamp,

            "frame_number":
                frame_number,

            "evidence_path":
                evidence_path,

            "persistence":
                persistence,

            "persistence_score":
                persistence_score,

            "deterioration":
                deterioration,

            "deterioration_score":
                deterioration_score,

            "priority":
                priority,

            "recommended_action":
                recommended_action

        })

    return {

        "segment_id":
            segment_id,

        "output":
            evidence_path,

        "detections":
            saved_records,

        "detection_count":
            len(saved_records),

        "timestamp":
            datetime.now().isoformat()

    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "application":
            "InfraSight",

        "status":
            "running",

        "message":
            "AI-Powered Road Infrastructure Monitoring"

    }


# ============================================================
# DASHBOARD
# ============================================================

@app.get("/dashboard")
@app.get("/dashboard/")
def dashboard():

    index_file = (
        DASHBOARD_DIR / "index.html"
    )

    if not index_file.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "Dashboard index.html not found."

        )

    return FileResponse(
        index_file
    )


# ============================================================
# CAMERA PAGE
# ============================================================

@app.get("/dashboard/camera.html")
def camera_page():

    camera_file = (
        DASHBOARD_DIR / "camera.html"
    )

    if not camera_file.exists():

        raise HTTPException(

            status_code=404,

            detail=
                "camera.html not found."

        )

    return FileResponse(
        camera_file
    )


# ============================================================
# STATUS
# ============================================================

@app.get("/status")
def status():

    return {

        "status":
            "online",

        "model_loaded":
            model is not None,

        "model_path":
            str(MODEL_PATH),

        "database":
            str(DATABASE_PATH),

        "output_directory":
            str(OUTPUT_DIR)

    }


# ============================================================
# METRICS
# ============================================================

@app.get("/metrics")
def metrics():

    observations = (
        get_detection_history(10000)
    )

    total = len(
        observations
    )

    low = sum(

        1 for x in observations

        if str(
            x["severity"]
        ).lower() == "low"

    )

    medium = sum(

        1 for x in observations

        if str(
            x["severity"]
        ).lower() == "medium"

    )

    high = sum(

        1 for x in observations

        if str(
            x["severity"]
        ).lower() == "high"

    )

    critical = sum(

        1 for x in observations

        if str(
            x["priority"]
        ).lower() == "critical"

    )

    return {

        "total_detections":
            total,

        "total_records":
            total,

        "low_severity":
            low,

        "medium_severity":
            medium,

        "high_severity":
            high,

        "critical_priority":
            critical

    }


# ============================================================
# TIMELINE API
# ============================================================

@app.get("/timeline")
def timeline():

    observations = (
        get_detection_history(100)
    )

    return {

        "status":
            "Timeline available",

        "total_observations":
            len(observations),

        "observations":
            observations

    }


# ============================================================
# SINGLE DETECTION
# ============================================================

@app.get("/detection/{detection_id}")
def get_detection(
    detection_id: int
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
        SELECT

            detection_id,

            segment_id,

            defect_class,

            confidence,

            severity,

            timestamp,

            frame_number,

            evidence_path,

            image,

            persistence,

            persistence_score,

            deterioration,

            deterioration_score,

            priority,

            recommended_action

        FROM detections

        WHERE detection_id = ?

    """, (
        detection_id,
    ))

    row = cursor.fetchone()

    conn.close()

    if row is None:

        raise HTTPException(

            status_code=404,

            detail=
                "Detection not found."

        )

    return {

        "detection_id":
            row["detection_id"],

        "segment_id":
            row["segment_id"],

        "defect_class":
            row["defect_class"],

        "confidence":
            row["confidence"],

        "severity":
            row["severity"],

        "timestamp":
            row["timestamp"],

        "frame_number":
            row["frame_number"],

        "evidence_path":
            row["evidence_path"],

        "image":
            row["image"],

        "persistence":
            row["persistence"],

        "persistence_score":
            row["persistence_score"],

        "deterioration":
            row["deterioration"],

        "deterioration_score":
            row["deterioration_score"],

        "priority":
            row["priority"],

        "recommended_action":
            row["recommended_action"]

    }


# ============================================================
# UPLOAD IMAGE
# ============================================================

@app.post("/upload-image")
async def upload_image(

    file: UploadFile = File(...),

    segment_id: str = Form(None)

):

    if model is None:

        return JSONResponse(

            status_code=500,

            content={

                "success":
                    False,

                "message":
                    "YOLO model is not loaded."

            }

        )

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    allowed_extensions = {

        ".jpg",
        ".jpeg",
        ".png",
        ".webp"

    }

    extension = Path(

        file.filename or ""

    ).suffix.lower()

    if extension not in allowed_extensions:

        raise HTTPException(

            status_code=400,

            detail=
                "Please upload JPG, JPEG, PNG or WEBP image."

        )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    unique_name = (

        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        + "_"

        + uuid.uuid4().hex[:6]

        + extension

    )

    input_path = (
        INPUT_DIR / unique_name
    )

    contents = await file.read()

    with open(
        input_path,
        "wb"
    ) as f:

        f.write(contents)

    # --------------------------------------------------------
    # Run YOLO
    # --------------------------------------------------------

    try:

        detection_result = (
            run_yolo_detection(

                input_path,

                frame_number=0,

                requested_segment_id=
                    segment_id

            )
        )

    except Exception as e:

        print(
            "YOLO ERROR:",
            e
        )

        return JSONResponse(

            status_code=500,

            content={

                "success":
                    False,

                "message":
                    "YOLO detection failed.",

                "error":
                    str(e)

            }

        )

    detections = (
        detection_result["detections"]
    )

    # --------------------------------------------------------
    # No detection
    # --------------------------------------------------------

    if len(detections) == 0:

        return {

            "success":
                True,

            "message":
                "Road damage detection completed",

            "status":
                "No road damage detected",

            "detections":
                0,

            "detection_count":
                0,

            "count":
                0,

            "segment_id":
                detection_result["segment_id"],

            "detection_image":
                detection_result["output"],

            "image":
                detection_result["output"],

            "output":
                detection_result["output"],

            "timestamp":
                detection_result["timestamp"],

            "results":
                []

        }

    # --------------------------------------------------------
    # Detection found
    # --------------------------------------------------------

    return {

        "success":
            True,

        "message":
            "Road damage detection completed",

        "status":
            "Road damage detected",

        "detections":
            len(detections),

        "detection_count":
            len(detections),

        "count":
            len(detections),

        "segment_id":
            detection_result["segment_id"],

        "detection_image":
            detection_result["output"],

        "image":
            detection_result["output"],

        "output":
            detection_result["output"],

        "timestamp":
            detection_result["timestamp"],

        "results":
            detections

    }


# ============================================================
# OLD FRONTEND ALIAS
# ============================================================

@app.post("/detect")
async def detect_alias(

    file: UploadFile = File(...),

    segment_id: str = Form(None)

):

    return await upload_image(

        file=file,

        segment_id=segment_id

    )


# ============================================================
# CAMERA FRAME DETECTION
# ============================================================

@app.post("/detect-frame")
async def detect_frame(

    file: UploadFile = File(...),

    segment_id: str = Form(None)

):

    return await upload_image(

        file=file,

        segment_id=segment_id

    )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    print("=" * 60)

    print(
        "Starting InfraSight server..."
    )

    print(
        "Dashboard:"
    )

    print(
        "http://127.0.0.1:8000/dashboard/"
    )

    print()

    print(
        "API:"
    )

    print(
        "http://127.0.0.1:8000/docs"
    )

    print("=" * 60)

    uvicorn.run(

        app,

        host="127.0.0.1",

        port=8000,

        reload=False

    )