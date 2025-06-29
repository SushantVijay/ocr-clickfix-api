# 📦ClickFix Detector API
A lightweight and optimized Flask API for detecting fake CAPTCHA-based phishing pages (ClickFix) using image classification and OCR keyword analysis. Designed to work efficiently even on systems with 2 vCPUs and 4 GB RAM.

# 🚀Features
✅ Image classification using a trained MobileNetV1 .keras model

🔍 OCR-based keyword matching for phishing patterns (e.g., win + r, powershell)

📊 System telemetry: RAM, CPU usage, threads, and core info

📈 /metrics endpoint for live performance insights

🖥️ Optimized for low-resource environments (CPU-only fallback)

🧠 Supports GPU if available (TensorFlow auto-detects)

🔐 Suitable for deployment behind browser extensions or SIEMs


# Clone the repo
git clone https://github.com/SushantVijay/ocr-clickfix-api.git

cd ocr-clickfix-api

# Install dependencies
pip install -r requirements.txt

# ⚙️Run Locally
python app.py


# Then test the API 
curl -X POST http://localhost:5000/predict -F image=@test/sample1.png

# 📡API Endpoints
/predict – POST

Input: multipart/form-data with image field

Output: JSON with prediction, confidence, OCR match, and system telemetry

# 🧠How It Works
Loads a Keras-trained image classification model (clickfix_mobilenetv1.keras)

Performs OCR with Tesseract on the input image

Checks for suspicious phishing-related keywords or sequences

Combines ML prediction and text analysis to produce a final classification

Logs system telemetry like CPU usage, memory usage, and execution time