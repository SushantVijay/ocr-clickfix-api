import os
import cv2
import time
import pytesseract
import numpy as np
import tensorflow as tf
import psutil
import re
import difflib
from PIL import Image

# === CPU Affinity ===
try:
    process = psutil.Process(os.getpid())
    process.cpu_affinity([0, 1])
    print("✅ CPU affinity set to cores: [0, 1]")
except AttributeError:
    print("⚠️ CPU affinity not supported on this OS")

# === GPU Setup ===
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        print(f"✅ GPU available: using {len(physical_devices)} GPU(s)")
    except:
        print("⚠️ Could not enable GPU memory growth.")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    print("💡 No GPU found. Using CPU only.")

# === Load Model ===
model = tf.keras.models.load_model("clickfix_mobilenetv1.keras")
labels = ['clickfix', 'legit']

# === Keyword Definitions ===
keyword_sequences = [['win + r'], ['windows + r'], ['+ r'], ['ctrl + v'], ['control + v'], ['enter']]
suspicious_keywords = [
    'win + r', 'windows + r', 'powershell', 'cmd',
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

# === OCR Preprocessing ===
def preprocess_image_for_ocr(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv2.THRESH_BINARY, 31, 2)
    return Image.fromarray(image)

# === Extract Text ===
def extract_text(image_path):
    try:
        image = preprocess_image_for_ocr(image_path)
        config = "--psm 6 --oem 3"
        text = pytesseract.image_to_string(image, config=config).lower()
        text = re.sub(r"\s+", " ", text)
        print(f"\n[OCR DEBUG] Extracted Text from {os.path.basename(image_path)}:\n{text}\n")
        return text
    except Exception as e:
        print(f"OCR error on {image_path}: {e}")
        return ""

# === Normalize and Match Keywords ===
def normalize_text(text):
    text = re.sub(r"[^a-z0-9 +]", "", text.lower())
    text = re.sub(r"\s+", " ", text)
    return text

def check_keywords(text):
    text = normalize_text(text)
    found = []

    for kw in suspicious_keywords:
        if kw in text:
            found.append(kw)
        else:
            # Fuzzy match in case of OCR errors
            matches = difflib.get_close_matches(kw, [text], n=1, cutoff=0.7)
            if matches:
                found.append(kw)

    matched_sequence = any(all(kw in text for kw in seq) for seq in keyword_sequences)

    print(f"[Keyword DEBUG] Matched keywords: {found}")
    return bool(found or matched_sequence), found

# === System Telemetry ===
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

# === CPU Usage Tracking Wrapper ===
def get_cpu_percent_for_prediction(fn, *args, **kwargs):
    process = psutil.Process(os.getpid())
    process.cpu_percent(interval=None)
    system_cpu_percent_before = psutil.cpu_percent(interval=None)

    start_time = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - start_time

    process_cpu = process.cpu_percent(interval=elapsed)
    system_cpu_after = psutil.cpu_percent(interval=0.2)

    cpu_final = process_cpu if process_cpu > 0 else system_cpu_after
    return result, round(cpu_final, 2)

# === Main Classifier ===
def classify_single_image(image_path):
    start_time = time.time()

    def prediction_block():
        image = preprocess_image_for_model(image_path)
        prediction = model.predict(image)[0]
        pred_idx = np.argmax(prediction)
        confidence = float(prediction[pred_idx])
        predicted_label = labels[pred_idx]

        text = extract_text(image_path)
        keyword_hit, matched = check_keywords(text)
        final_label = "clickfix" if keyword_hit else predicted_label

        return {
            "image": os.path.basename(image_path),
            "model_prediction": predicted_label,
            "confidence": round(confidence, 4),
            "keywords_matched": matched,
            "final_classification": final_label,
        }

    prediction_result, cpu_used = get_cpu_percent_for_prediction(prediction_block)

    stats = get_usage_stats()
    stats.update(prediction_result)
    stats["cpu_percent"] = cpu_used
    stats["time_taken_sec"] = round(time.time() - start_time, 3)

    return stats

# === System-Wide Telemetry ===
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
