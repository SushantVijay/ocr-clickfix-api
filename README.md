📦 ClickFix Detector API
A lightweight and optimized Flask API for detecting fake CAPTCHA-based phishing pages (ClickFix) using image classification and OCR keyword analysis. Designed to work efficiently even on systems with 2 vCPUs and 4 GB RAM.

🚀 Features
✅ Image classification using a trained MobileNetV1 .keras model
🔍 OCR-based keyword matching for phishing patterns (e.g., win + r, powershell)
📊 System telemetry: RAM, CPU usage, threads, and core info
📈 /metrics endpoint for live performance insights
🖥️ Optimized for low-resource environments (CPU-only fallback)
🧠 Supports GPU if available (TensorFlow auto-detects)
🔐 Suitable for deployment behind browser extensions or SIEMs

