"""
app.py
------
Flask web server for the Confidence-Aware OCR Dashboard.
Exposes REST endpoints consumed by the frontend.

Endpoints:
    GET  /            → Serve dashboard HTML
    POST /process     → Upload image, run pipeline, return JSON results
    GET  /lookup/<plate> → Query plate database
    GET  /history     → Return session history
    GET  /health      → Health check

Novelty additions:
    - Plate database lookup (flagged / expired vehicle alerting)
    - Session history persistence via SQLite
"""

import os
import io
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np

from pipeline import run_full_pipeline
from db import init_db, lookup_plate, save_session, get_history

app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)

app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'}

# Initialize database on startup
init_db()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'message': 'OCR server is running'})


@app.route('/process', methods=['POST'])
def process_image():
    """
    Accept an uploaded image and return full OCR pipeline results.

    Request:  multipart/form-data with field 'image'
    Response: JSON with base64 images, word data, stats, and novelty fields
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image field in request'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({
            'error': f'File type not supported. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Read image from upload stream
        img_bytes = file.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        # Run full pipeline (now returns novelty fields too)
        results = run_full_pipeline(pil_image)

        # ── Plate database lookup ─────────────────────────────────────────────
        plate_text = results.get('plate_text', '')
        db_record  = lookup_plate(plate_text) if plate_text else None
        results['db_record'] = db_record

        # ── Persist session history ───────────────────────────────────────────
        iq    = results.get('image_quality', {})
        stats = results.get('enhanced_stats', {})
        save_session(
            filename      = secure_filename(file.filename),
            plate_text    = plate_text,
            plate_matched = results.get('plate_matched', False),
            avg_confidence= stats.get('avg_confidence', 0),
            blur_type     = iq.get('blur_type', ''),
            sharpness     = iq.get('sharpness_score', 0),
            retry_count   = len(results.get('retry_trace', [])),
        )

        return jsonify({
            'success': True,
            'filename': secure_filename(file.filename),
            **results
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/lookup/<path:plate_text>', methods=['GET'])
def plate_lookup(plate_text):
    """Query the local plate database for a specific number."""
    record = lookup_plate(plate_text)
    if record:
        return jsonify({'found': True, 'record': record})
    return jsonify({'found': False, 'record': None})


@app.route('/history', methods=['GET'])
def history():
    """Return the last 50 session history records."""
    limit = min(int(request.args.get('limit', 50)), 200)
    rows  = get_history(limit)
    return jsonify({'history': rows})


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16 MB.'}), 413


if __name__ == '__main__':
    print("=" * 60)
    print("  Confidence-Aware OCR Dashboard  (+ Novelty Features)")
    print("  http://localhost:5000")
    print("  New endpoints: /lookup/<plate>  /history")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
