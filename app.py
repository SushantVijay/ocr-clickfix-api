from flask import Flask, request, jsonify
from clickfix_predictor import analyze_image
import os

app = Flask(__name__)

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ClickFix API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    image = request.files['image']
    result = analyze_image(image)
    return jsonify(result)

@app.route("/metrics", methods=["GET"])
def metrics():
    from clickfix_predictor import get_system_metrics
    return jsonify(get_system_metrics())

if __name__ == "__main__":
    app.run(debug=True)
