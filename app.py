from flask import Flask, request, jsonify
import os
import tempfile
from clickfix_predictor import classify_single_image

app = Flask(__name__)

@app.route("/")
def health_check():
    return {"status": "ClickFix API is running"}

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
        file.save(temp_file.name)
        image_path = temp_file.name

    try:
        result = classify_single_image(image_path)
    except Exception as e:
        return jsonify({"error": f"Classification failed: {str(e)}"}), 500
    finally:
        os.remove(image_path)  # Clean up

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
