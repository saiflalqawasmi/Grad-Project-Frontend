import os
import pickle
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from PIL import Image

from inference import get_pipeline, FeatureExtractor, VECTORS_DIR, ROTATION_ANGLES
from stats_tracker import stats, Timer
from camera_feed import (
    connect_camera, disconnect_camera, generate_frames, get_frame_jpeg, is_connected,
)

app = Flask(__name__)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

CATALOG_IMG_DIR = Path(__file__).parent / "static" / "catalog"
CATALOG_IMG_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

_extractor = None


def get_extractor():
    """Lazy singleton for the DINOv2 FeatureExtractor (see inference.py) --
    only loads the extractor, not the full YOLO/SAM pipeline."""
    global _extractor
    if _extractor is None:
        _extractor = FeatureExtractor()
    return _extractor


@app.route("/")
def dashboard():
    return render_template('dashboard.html', camera_connected=is_connected())


@app.route("/products")
def products_page():
    return render_template('products.html')


@app.route("/livecart")
def livecart_page():
    return render_template('livecart.html', camera_connected=is_connected())


@app.route("/connect_camera", methods=["POST"])
def connect_camera_route():
    """
    Connects to a phone acting as an IP camera (e.g. the 'IP Webcam' Android
    app's MJPEG stream, usually http://<phone-ip>:8080/video). Called via
    fetch() from livecart.html so the page doesn't need a full reload.
    """
    url = request.form.get("camera_url")
    if not url and request.is_json:
        url = (request.get_json(silent=True) or {}).get("camera_url")
    if not url:
        return jsonify({"error": "camera_url is required"}), 400

    connected = connect_camera(url)
    return jsonify({"connected": connected, "camera_url": url if connected else None})


@app.route("/disconnect_camera", methods=["POST"])
def disconnect_camera_route():
    disconnect_camera()
    return jsonify({"connected": False})


@app.route("/video_feed")
def video_feed():
    if not is_connected():
        return jsonify({"error": "no camera connected"}), 400
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/products")
def api_products():
    """
    Reads the real catalog straight out of vectors/*.pkl -- no hardcoded
    product list. Each file's payload has item_id and the DINOv2 feature
    matrix (num_samples x embedding_dim) used by the matcher.
    """
    # تحديد مسار مجلد clean_data الرئيسي تلقائياً من مسار vectors
    clean_data_dir = VECTORS_DIR.parent / "clean_data"
    
    products = []
    for pkl_path in sorted(VECTORS_DIR.glob("*.pkl")):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        product_id = data["item_id"].replace("_vector", "")
        local_img_path = CATALOG_IMG_DIR / f"{product_id}.jpg"

        image_url = None
        if local_img_path.exists():
            image_url = f"/static/catalog/{product_id}.jpg"
        else:
            # البحث أوتوماتيكياً في مجلد clean_data عن أول صورة تخص المنتج ونسخها
            product_folder = clean_data_dir / product_id
            if product_folder.exists():
                images = sorted(list(product_folder.glob("*.jpg")) + list(product_folder.glob("*.png")))
                if images:
                    import shutil
                    shutil.copy(images[0], local_img_path)
                    image_url = f"/static/catalog/{product_id}.jpg"

        products.append({
            "id": product_id,
            "reference_samples": data.get("num_samples", data["features"].shape[0]),
            "embedding_dim": data["features"].shape[1],
            "image_url": image_url,
        })

    return jsonify({"products": products, "count": len(products)})

@app.route("/api/products/new", methods=["POST"])
def add_product():
    """
    Builds a real catalog entry from uploaded photos: runs each image
    through the same DINOv2 + 12-angle-rotation extraction used by the
    live matcher, saves it as a new vectors/<id>_vector.pkl, and stores
    the first photo as the catalog thumbnail.
    """
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no images uploaded (field name 'images')"}), 400

    existing_ids = [p.name.replace("_vector.pkl", "") for p in VECTORS_DIR.glob("*.pkl")]
    requested_id = request.form.get("product_id", "").strip()

    if requested_id:
        product_id = requested_id
        if product_id in existing_ids:
            return jsonify({"error": f"product id '{product_id}' already exists"}), 400
    else:
        numeric_ids = [int(i) for i in existing_ids if i.isdigit()]
        next_num = (max(numeric_ids) + 1) if numeric_ids else 1
        product_id = f"{next_num:03d}"

    try:
        extractor = get_extractor()
        features = []
        thumb_saved = False

        for file in files:
            ext = Path(file.filename).suffix.lower()
            if ext not in ALLOWED_EXTS:
                continue

            image = Image.open(file.stream).convert("RGB")

            if not thumb_saved:
                image.save(CATALOG_IMG_DIR / f"{product_id}.jpg")
                thumb_saved = True

            for angle in ROTATION_ANGLES:
                rotated = image.rotate(
                    angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white"
                )
                features.append(extractor.embed(rotated))

        if not features:
            return jsonify({"error": "no valid images in upload"}), 400

        features = np.array(features)
        payload = {
            "item_id": f"{product_id}_vector",
            "num_samples": features.shape[0],
            "features": features,
        }
        with open(VECTORS_DIR / f"{product_id}_vector.pkl", "wb") as f:
            pickle.dump(payload, f)

        return jsonify({
            "success": True,
            "product_id": product_id,
            "reference_samples": features.shape[0],
            "embedding_dim": features.shape[1],
            "image_url": f"/static/catalog/{product_id}.jpg" if thumb_saved else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/capture_frame")
def capture_frame():
    """
    Grabs a single current frame from the connected phone camera as a JPEG.
    Used by the 'Scan Now' button in live-camera mode: the frontend fetches
    this once, then re-posts the bytes to /predict — so /predict only ever
    has to deal with 'here is one image', whether it came from an upload
    or from the live phone stream.
    """
    if not is_connected():
        return jsonify({"error": "no camera connected"}), 400

    jpeg_bytes = get_frame_jpeg()
    if jpeg_bytes is None:
        return jsonify({"error": "failed to read frame from camera"}), 500

    return Response(jpeg_bytes, mimetype="image/jpeg")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a multipart/form-data upload under the field name 'image',
    runs it through YOLO -> MobileSAM -> ResNet50 matcher, records the
    result into the running stats tracker, and returns the detections
    as JSON.
    """
    if "image" not in request.files:
        return jsonify({"error": "no file field named 'image'"}), 400

    file = request.files["image"]
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTS:
        return jsonify({"error": f"unsupported file type: {ext}"}), 400

    temp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    file.save(temp_path)

    try:
        with Timer() as t:
            result = get_pipeline().predict(temp_path)
        stats.record(result["products"], t.elapsed_ms)
        result["inference_ms"] = round(t.elapsed_ms, 1)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)


@app.route("/stats", methods=["GET"])
def get_stats():
    """Aggregated KPIs the dashboard polls to fill in real numbers."""
    return jsonify(stats.snapshot())


if __name__ == "__main__":
    # Warm the pipeline at boot so the first request isn't slow, and so
    # missing model files fail fast instead of on first upload.
    get_pipeline()
    app.run(debug=True, threaded=True)