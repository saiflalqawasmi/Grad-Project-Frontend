# What's actually needed to run this

```
frontend/
  app.py              # Flask app: "/" dashboard + "/predict" inference endpoint
  inference.py         # loads models once, exposes ProductPipeline
  models/
    product_detector.pt   # trained YOLO detector (required)
    mobile_sam.pt          # MobileSAM segmenter (required)
  vectors/
    001_vector.pkl ... 014_vector.pkl   # product catalog embeddings (required)
  templates/
    dashboard.html
```

`yolo11n.pt` is **not** included — it's only the base checkpoint used when
retraining the detector (`TRAIN_YOLO = True` in the notebook) and plays no
role at serving time.

ResNet50 weights are pulled automatically by torchvision on first run and
cached (`~/.cache/torch/hub/checkpoints`) — no file to ship for that unless
you want fully offline/air-gapped startup, in which case pre-download and
point `torchvision.models.resnet50` at a local weights path.

## Install

```
pip install flask ultralytics torch torchvision opencv-python pillow numpy
```

## Run

```
python app.py
```

The pipeline loads once at startup (`get_pipeline()` in `app.py`), so a
missing model file fails immediately instead of on the first request.

## Endpoint

`POST /predict` — multipart form, field name `image`.

```
curl -X POST -F "image=@test.jpg" http://localhost:5000/predict
```

```json
{
  "count": 2,
  "products": [
    {"bbox": [x1, y1, x2, y2], "product": "003", "confidence": 0.87},
    {"bbox": [x1, y1, x2, y2], "product": "011", "confidence": 0.71}
  ]
}
```

`dashboard.html` currently has no JS wiring this endpoint up (it's static
placeholder data) — next step is a fetch() call from the dashboard that
POSTs an uploaded/captured image and renders `products` into the KPI cards.
