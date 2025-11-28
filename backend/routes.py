import os
from flask import Flask, request, jsonify
from backend.metrics import calculate_metrics

app = Flask(__name__)

# API route for file upload and metric calculation
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    # Save file temporarily
    upload_path = os.path.join('uploads', file.filename)
    file.save(upload_path)

    # Mock metric calculation
    metrics = calculate_metrics("uploads/" + file.filename)  # Pass file name to logic

    return jsonify({"metrics": metrics}), 200
