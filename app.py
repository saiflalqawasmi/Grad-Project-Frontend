import os
import pickle
import uuid
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required, current_user
from PIL import Image

from inference import get_pipeline, FeatureExtractor, VECTORS_DIR, ROTATION_ANGLES
from stats_tracker import stats, Timer
import catalog
import db
import auth
from auth import admin_required, manager_or_admin_required
from camera_feed import (
    connect_camera, disconnect_camera, generate_frames, get_frame_jpeg, is_connected,
)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-this-secret-key")
db.init_db()
auth.init_db()
auth.login_manager.init_app(app)

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


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(current_user.home_url())

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            flash("Username and password are required.", "danger")
            return render_template("login.html")

        user = auth.get_user_by_username(username)
        if user is None or not user.check_password(password):
            flash("Invalid username or password.", "danger")
            return render_template("login.html")

        login_user(user)
        next_url = request.args.get("next")
        return redirect(next_url or user.home_url())

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/admin")
@admin_required
def admin():
    return render_template("admin.html", users=auth.list_users())


@app.route("/admin/users/create", methods=["POST"])
@admin_required
def create_user():
    username = request.form.get("username", "").strip()
    email = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "")
    role = request.form.get("role", "cashier")

    if not username or not email or not password:
        flash("All fields are required.", "danger")
        return redirect(url_for("admin"))

    if role not in auth.ROLES:
        flash("Invalid role.", "danger")
        return redirect(url_for("admin"))

    if auth.find_conflict(username, email):
        flash("Username or email already exists.", "danger")
        return redirect(url_for("admin"))

    auth.create_user(username, email, password, role)
    flash(f'User "{username}" created successfully.', "success")
    return redirect(url_for("admin"))


@app.route("/admin/users/<int:user_id>/role", methods=["POST"])
@admin_required
def change_user_role(user_id):
    user = auth.get_user_by_id(user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    if user.id == current_user.id:
        return jsonify({"error": "You cannot change your own role"}), 400

    new_role = (request.get_json(silent=True) or {}).get("role")
    if new_role not in auth.ROLES:
        return jsonify({"error": "Invalid role"}), 400

    auth.update_role(user.id, new_role)
    return jsonify({"success": True, "user_id": user.id, "role": new_role})


@app.route("/admin/users/<int:user_id>/delete", methods=["POST"])
@admin_required
def delete_user_route(user_id):
    user = auth.get_user_by_id(user_id)
    if user is None:
        return jsonify({"error": "User not found"}), 404
    if user.id == current_user.id:
        return jsonify({"error": "You cannot delete yourself"}), 400

    auth.delete_user(user.id)
    return jsonify({"success": True, "message": "User deleted successfully"})


@app.route("/")
@manager_or_admin_required
def dashboard():
    return render_template('dashboard.html', camera_connected=is_connected())


@app.route("/forecast")
@manager_or_admin_required
def forecast_page():
    return render_template('forecast.html')


@app.route("/products")
@manager_or_admin_required
def products_page():
    return render_template('products.html')


@app.route("/livecart")
@login_required
def livecart_page():
    return render_template('livecart.html', camera_connected=is_connected())


@app.route("/connect_camera", methods=["POST"])
@login_required
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
    if connected:
        stats.log_event(f"CAMERA_CONNECT: linked to {url}", level="success")
    else:
        stats.log_event(f"CAMERA_CONNECT_FAILED: could not reach {url}", level="danger")
    return jsonify({"connected": connected, "camera_url": url if connected else None})


@app.route("/disconnect_camera", methods=["POST"])
@login_required
def disconnect_camera_route():
    disconnect_camera()
    stats.log_event("CAMERA_DISCONNECT: stream closed", level="warning")
    return jsonify({"connected": False})


@app.route("/video_feed")
@login_required
def video_feed():
    if not is_connected():
        return jsonify({"error": "no camera connected"}), 400
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/products")
@manager_or_admin_required
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
            "name": catalog.get_name(product_id),
            "price": catalog.get_price(product_id),
            "reference_samples": data.get("num_samples", data["features"].shape[0]),
            "embedding_dim": data["features"].shape[1],
            "image_url": image_url,
        })

    return jsonify({"products": products, "count": len(products)})


@app.route("/api/products/<product_id>/meta", methods=["POST"])
@manager_or_admin_required
def update_product_meta(product_id):
    """
    Sets the real-world name/price for a catalog product. This is the only
    place price data enters the system -- the vision pipeline never
    produces one -- so it's what makes the dashboard's revenue and
    forecast-revenue figures real rather than placeholder numbers.
    """
    existing_ids = [p.name.replace("_vector.pkl", "") for p in VECTORS_DIR.glob("*.pkl")]
    if product_id not in existing_ids:
        return jsonify({"error": f"unknown product id '{product_id}'"}), 404

    payload = request.get_json(silent=True) or request.form
    name = payload.get("name")
    price = payload.get("price")

    if price is not None:
        try:
            price = float(price)
            if price < 0:
                raise ValueError
        except (TypeError, ValueError):
            return jsonify({"error": "price must be a non-negative number"}), 400

    entry = catalog.set_meta(product_id, price=price, name=name)
    return jsonify({"product_id": product_id, **entry})

@app.route("/api/products/new", methods=["POST"])
@admin_required
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
@login_required
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
@login_required
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

        for p in result["products"]:
            p["name"] = catalog.get_name(p["product"])
            p["price"] = catalog.get_price(p["product"])

        stats.record(result["products"], t.elapsed_ms)
        result["inference_ms"] = round(t.elapsed_ms, 1)
        result["transaction_revenue"] = round(sum(p["price"] for p in result["products"]), 2)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.remove(temp_path)


@app.route("/stats", methods=["GET"])
@manager_or_admin_required
def get_stats():
    """Aggregated KPIs the dashboard polls to fill in real numbers."""
    return jsonify(stats.snapshot())


@app.route("/api/predictions", methods=["GET"])
@manager_or_admin_required
def get_predictions():
    """
    AI forecast: best-selling items + suggested reorder quantities, derived
    from live scan volume (see StatsTracker.predictions for the method).
    """
    return jsonify(stats.predictions())


@app.route("/api/timeseries", methods=["GET"])
@manager_or_admin_required
def get_timeseries():
    """Real hourly/daily scan+revenue+confidence series for the dashboard's trend charts."""
    return jsonify(stats.timeseries())


@app.route("/api/checkout", methods=["POST"])
@login_required
def checkout():
    """
    Finalizes the current Live Cart into a real, persisted transaction row.
    Prices are always looked up server-side from the catalog -- a client
    could send any price it wants in the request body, so one from the
    client is never trusted; only product_id + quantity are read from it.
    """
    payload = request.get_json(silent=True) or {}
    cart_items = payload.get("items") or []

    known_ids = {p.name.replace("_vector.pkl", "") for p in VECTORS_DIR.glob("*.pkl")}

    priced_items = []
    for entry in cart_items:
        product_id = str(entry.get("product_id", "")).strip()
        try:
            quantity = int(entry.get("quantity", 0))
        except (TypeError, ValueError):
            quantity = 0
        if not product_id or quantity <= 0 or product_id not in known_ids:
            continue
        priced_items.append({
            "product_id": product_id,
            "product_name": catalog.get_name(product_id),
            "unit_price": catalog.get_price(product_id),
            "quantity": quantity,
        })

    if not priced_items:
        return jsonify({"error": "no valid items in cart"}), 400

    txn = db.create_transaction(priced_items)
    stats.log_event(
        f"CHECKOUT_OK: transaction #{txn['id']}, {txn['item_count']} item(s), ${txn['total']:.2f}",
        level="success",
    )
    return jsonify(txn)


@app.route("/api/transactions", methods=["GET"])
@manager_or_admin_required
def get_transactions():
    """Recent completed transactions, most recent first."""
    limit = request.args.get("limit", default=50, type=int)
    return jsonify({"transactions": db.list_transactions(limit=limit)})


# Add these imports at the top if not already present
import os
import hashlib
from werkzeug.utils import secure_filename

@app.route('/products/add-scan')
@login_required
def add_scan_page():
    return render_template('add_scan.html', camera_connected=is_connected())

@app.route('/api/products/<product_id>/delete', methods=['POST'])
@login_required
def delete_product(product_id):
    try:
        # Delete from vectors directory
        vectors_dir = 'vectors'
        deleted_count = 0
        
        # Find and delete all files related to this product
        for filename in os.listdir(vectors_dir):
            if filename.startswith(f"{product_id}_") or filename == f"{product_id}.pkl":
                filepath = os.path.join(vectors_dir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    deleted_count += 1
        
        # Also delete from uploads if exists
        uploads_dir = 'uploads'
        if os.path.exists(uploads_dir):
            for filename in os.listdir(uploads_dir):
                if filename.startswith(f"{product_id}_"):
                    filepath = os.path.join(uploads_dir, filename)
                    if os.path.isfile(filepath):
                        os.remove(filepath)
        
        # Remove from database if you have a products table
        try:
            from db import get_db
            db = get_db()
            db.execute('DELETE FROM products WHERE id = ?', (product_id,))
            db.commit()
        except:
            pass  # Database might not exist
        
        return {'success': True, 'deleted_files': deleted_count}
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/api/capture-and-check', methods=['POST'])
@login_required
def capture_and_check():
    """Capture a frame and check if it's a good angle for vector building"""
    try:
        # Capture frame from camera
        from camera_feed import capture_frame
        frame = capture_frame()
        
        if frame is None:
            return {'error': 'Could not capture frame'}, 500
        
        # Run inference to check confidence
        from inference import predict
        result = predict(frame)
        
        # Check if we have a good detection
        products = result.get('products', [])
        good_detection = False
        best_confidence = 0
        best_bbox = None
        
        for p in products:
            if p.get('product') != 'unknown' and p.get('confidence', 0) > 0.85:
                good_detection = True
                if p.get('confidence', 0) > best_confidence:
                    best_confidence = p.get('confidence', 0)
                    best_bbox = p.get('bbox')
        
        # Save the frame temporarily
        import cv2
        import uuid
        temp_id = str(uuid.uuid4())[:8]
        temp_filename = f"temp_{temp_id}.jpg"
        temp_path = os.path.join('uploads', temp_filename)
        cv2.imwrite(temp_path, frame)
        
        return {
            'success': True,
            'good_angle': good_detection,
            'confidence': best_confidence,
            'bbox': best_bbox,
            'temp_image': temp_filename
        }
    except Exception as e:
        return {'error': str(e)}, 500
    
    
    
@app.route('/api/products/new-from-captures', methods=['POST'])
@login_required
def add_product_from_captures():
    """Build vectors from captured live images"""
    try:
        temp_images = request.form.getlist('temp_images')
        
        if not temp_images:
            return {'error': 'No images provided'}, 400
        
        # Generate new product ID
        import os
        vectors_dir = 'vectors'
        existing_ids = []
        for filename in os.listdir(vectors_dir):
            if filename.endswith('.pkl'):
                try:
                    pid = int(filename.split('_')[0])
                    existing_ids.append(pid)
                except:
                    pass
        
        new_id = max(existing_ids, default=0) + 1
        product_id = f"{new_id:03d}"
        
        # Process each captured image
        from pipeline import build_product_vectors
        image_paths = [os.path.join('uploads', img) for img in temp_images]
        
        # Build vectors
        build_product_vectors(product_id, image_paths)
        
        # Clean up temp images
        for img_path in image_paths:
            if os.path.exists(img_path):
                os.remove(img_path)
        
        return {
            'success': True,
            'product_id': product_id,
            'reference_samples': len(temp_images)
        }
    except Exception as e:
        return {'error': str(e)}, 500


if __name__ == "__main__":
    # Warm the pipeline at boot so the first request isn't slow, and so
    # missing model files fail fast instead of on first upload.
    get_pipeline()
    app.run(debug=True, threaded=True)