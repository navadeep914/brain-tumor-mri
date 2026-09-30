"""Python inference service that preserves the existing Streamlit model pipeline."""
from io import BytesIO
import base64
import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from common import load_gray, preprocess_small, load_models
from classification import classify
from original_mri import get_original
from tumor_highlight import make_highlight, make_segmentation
from feature_map import make_feature_map
from feature_map import make_feature_map

app = Flask(__name__)
CORS(app)

def image_data_uri(image):
    ok, encoded = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        return None
    return 'data:image/png;base64,' + base64.b64encode(encoded.tobytes()).decode('ascii')

@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'service': 'neurolens-python-inference'})

@app.post('/analyze')
def analyze():
    uploaded = request.files.get('file')
    if not uploaded:
        return jsonify({'error': 'An MRI image file is required.'}), 400
    gray = load_gray(uploaded.read())
    if gray is None:
        return jsonify({'error': 'The uploaded file could not be read as an image.'}), 400
    norm = preprocess_small(gray)
    result = classify(norm)
    original = get_original(gray)
    feature_map = make_feature_map(norm)
    is_tumor = result['label'] != 'notumor'
    highlighted, highlight_found = make_highlight(original) if is_tumor else (original, False)
    segmentation, segmentation_found = make_segmentation(original) if is_tumor else (original * 0, False)
    return jsonify({**result, 'tumor_detected': is_tumor, 'highlight_found': highlight_found, 'segmentation_found': segmentation_found, 'images': {'original': image_data_uri(original), 'highlight': image_data_uri(highlighted), 'segmentation': image_data_uri(segmentation), 'feature_map': image_data_uri(feature_map)}})

if __name__ == '__main__':
    load_models()
    app.run(host='0.0.0.0', port=5000, debug=False)
