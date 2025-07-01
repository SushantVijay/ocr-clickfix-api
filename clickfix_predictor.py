import os
import re
import cv2
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import numpy as np
import tensorflow as tf
import psutil
import logging
from PIL import Image
import tempfile

# === Logging Setup ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# === Set CPU Affinity to 2 cores ===
try:
    process = psutil.Process(os.getpid())
    process.cpu_affinity([0, 1])
except AttributeError:
    pass  # Not supported on all OS

# === Force CPU Only ===
os.environ["CUDA_VISIBLE_DEVICES"] = ""
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# === Load Model ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "clickfix_mobilenetv1.keras")
model = tf.keras.models.load_model(MODEL_PATH)

labels = ['clickfix', 'legit']

# === Keyword Definitions ===
keyword_sequences = [['win + r'], ['windows + r'], ['+ r'], ['ctrl + v'], ['control + v'], ['enter']]
suspicious_keywords = [
    'win + r', 'windows', 'win', ' + r', 'windows + r', 'powershell', 'cmd',
    'control + v', 'ctrl + v', 'command prompt', 'paste the command',
    'run:', 'copy and paste', 'open powershell', 'execute'
]

# === Image Preprocessing for Model ===
def preprocess_image_for_model(image_path, target_size=(224, 224)):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# === OCR Extraction from Original Image ===
def extract_text(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        text = pytesseract.image_to_string(image).lower()
        logger.info("[Extracted OCR Text]: %s", text)
        return text
    except Exception as e:
        logger.error("[OCR Extraction Failed]: %s", e)
        return ""

# === Normalize and Match Keywords ===
def normalize_text(text):
    text = re.sub(r'\s*\+\s*', ' + ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def check_keywords(text):
    text = normalize_text(text)
    logger.info("[Normalized OCR Text]: %s", text)
    found = [kw for kw in suspicious_keywords if kw in text]
    matched_sequence = any(all(word in text for word in seq) for seq in keyword_sequences)
    logger.info("[Keywords Found]: %s | Matched Sequence: %s", found, matched_sequence)
    return bool(found or matched_sequence), found

# === System & Process Telemetry ===
def get_usage_stats():
    proc = psutil.Process(os.getpid())
    ram_mb = proc.memory_info().rss / (1024 * 1024)
    num_threads = proc.num_threads()

    try:
        cores_used = len(proc.cpu_affinity())
    except AttributeError:
        cores_used = "N/A"

    physical_cores = psutil.cpu_count(logical=False)
    logical_cores = psutil.cpu_count(logical=True)

    try:
        cpu_freq = psutil.cpu_freq().current
    except Exception:
        cpu_freq = "N/A"

    try:
        load_avg_1m = os.getloadavg()[0] if hasattr(os, "getloadavg") else "N/A"
    except:
        load_avg_1m = "N/A"

    return {
        "ram_usage_mb": round(ram_mb, 2),
        "num_threads": num_threads,
        "process_cores_used": cores_used,
        "physical_cores": physical_cores,
        "logical_cores": logical_cores,
        "cpu_freq_mhz": round(cpu_freq, 2) if isinstance(cpu_freq, (float, int)) else cpu_freq,
        "load_average_1m": round(load_avg_1m, 2) if isinstance(load_avg_1m, (float, int)) else load_avg_1m
    }

# === Accurate CPU Percent Using Time ===
def get_cpu_percent_for_prediction(fn, *args, **kwargs):
    process = psutil.Process(os.getpid())
    cpu_times_before = process.cpu_times()
    start_time = time.time()

    result = fn(*args, **kwargs)

    elapsed = time.time() - start_time
    cpu_times_after = process.cpu_times()

    cpu_time_used = (
        (cpu_times_after.user - cpu_times_before.user) +
        (cpu_times_after.system - cpu_times_before.system)
    )

    cpu_percent = (cpu_time_used / elapsed) * 100 if elapsed > 0 else 0.0
    return result, round(cpu_percent, 2)

# === Logging Block ===
def log_prediction_details(image_name, model_pred, confidence, final_label, keywords, cpu, duration):
    logger.info("\n" + "="*75)
    logger.info("📥 [Request] Image Received: %s", image_name)
    logger.info("🔍 [Model Prediction]: %s (Confidence: %.4f)", model_pred, confidence)
    logger.info("🔑 [OCR Keywords Matched]: %s", keywords if keywords else "None")
    logger.info("✅ [Final Classification]: %s", final_label)
    logger.info("🧠 [CPU Usage During Prediction]: %.2f%%", cpu)
    logger.info("⏱️ [Total Time Taken]: %.3f sec", duration)
    logger.info("="*60 + "\n")

# === Main Classification Logic ===
def classify_single_image(image_path):
    start_time = time.time()

    def prediction_block():
        image = preprocess_image_for_model(image_path)
        prediction = model.predict(image)[0]

        if len(prediction) == 1:
            confidence = float(prediction[0])
            predicted_label = 'clickfix' if confidence > 0.5 else 'legit'
        else:
            pred_idx = np.argmax(prediction)
            confidence = float(prediction[pred_idx])
            predicted_label = labels[pred_idx]

        text = extract_text(image_path)
        keyword_hit, matched_keywords = check_keywords(text)
        final_label = "clickfix" if keyword_hit else predicted_label

        return {
            "image": os.path.basename(image_path),
            "model_prediction": predicted_label,
            "confidence": round(confidence, 4),
            "keywords_matched": matched_keywords,
            "final_classification": final_label
        }

    prediction_result, cpu_used = get_cpu_percent_for_prediction(prediction_block)
    time_taken = round(time.time() - start_time, 3)

    stats = get_usage_stats()
    stats.update(prediction_result)
    stats["cpu_percent"] = cpu_used
    stats["time_taken_sec"] = time_taken

    log_prediction_details(
        image_name=prediction_result["image"],
        model_pred=prediction_result["model_prediction"],
        confidence=prediction_result["confidence"],
        final_label=prediction_result["final_classification"],
        keywords=prediction_result["keywords_matched"],
        cpu=cpu_used,
        duration=time_taken
    )

    return stats

# === Flask-compatible Wrapper ===
def analyze_image(image_file):
    logger.info("📡 [Incoming Request] Processing image file: %s", image_file.filename)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
        image_file.save(tmp.name)
        return classify_single_image(tmp.name)

# === System-Wide Metrics Endpoint ===
def get_system_metrics():
    virtual_mem = psutil.virtual_memory()
    boot_time = time.time() - psutil.boot_time()
    return {
        "system": {
            "uptime_seconds": round(boot_time),
            "cpu_percent": psutil.cpu_percent(interval=0.3),
            "ram_usage_percent": virtual_mem.percent,
            "available_ram_mb": round(virtual_mem.available / (1024 * 1024), 2),
            "used_ram_mb": round(virtual_mem.used / (1024 * 1024), 2),
            "total_ram_mb": round(virtual_mem.total / (1024 * 1024), 2),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "load_average_1m": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else "N/A"
        },
        "process": get_usage_stats()
    }
