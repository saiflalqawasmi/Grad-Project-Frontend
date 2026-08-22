from pathlib import Path
import pickle
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from ultralytics import YOLO, SAM


BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"  
VECTORS_DIR = BASE_DIR / "vectors"  
DETECTOR_PATH = MODELS_DIR / "product_detector.pt"
SAM_PATH = MODELS_DIR / "mobile_sam.pt"
ROTATION_ANGLES = [-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90, 180]

class FeatureExtractor:

    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1]).to(self.device)
        self.model.eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def embed(self, pil_image):
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor).squeeze()
        return vec.cpu().numpy()


class ProductMatcher:

    def __init__(self, extractor: FeatureExtractor, vectors_dir: Path):
        self.extractor = extractor
        self.catalog = self._load_catalog(vectors_dir)

    @staticmethod
    def _load_catalog(vectors_dir: Path):
        catalog = {}
        for pkl_path in vectors_dir.glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            features = data["features"]
            features = features / np.clip(
                np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None
            )
            catalog[data["item_id"]] = features
        if not catalog:
            raise FileNotFoundError(f"No *.pkl catalog vectors found in {vectors_dir}")
        return catalog

    def identify(self, crop_pil, min_similarity=0.55, min_margin=0.02):
        query_features = []
        for angle in ROTATION_ANGLES:
            rotated = crop_pil.rotate(
                angle, resample=Image.Resampling.BICUBIC, expand=True, fillcolor="white"
            )
            vec = self.extractor.embed(rotated)
            vec = vec / max(np.linalg.norm(vec), 1e-12)
            query_features.append(vec)
        query_features = np.array(query_features)

        scores = {}
        for item_id, ref_features in self.catalog.items():
            sims = query_features @ ref_features.T
            best_per_angle = sims.max(axis=1)
            keep = max(3, int(np.ceil(len(best_per_angle) * 0.75)))
            scores[item_id] = float(np.sort(best_per_angle)[-keep:].mean())

        ranking = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        best_item, best_score = ranking[0]
        second_score = ranking[1][1] if len(ranking) > 1 else 0.0
        margin = best_score - second_score

        if best_score < min_similarity or margin < min_margin:
            return "unknown", best_score, scores
        return best_item.replace("_vector", ""), best_score, scores


class ProductPipeline:
    """Ties detector + SAM + matcher together. Instantiate once, reuse per request."""

    def __init__(self):
        if not DETECTOR_PATH.exists():
            raise FileNotFoundError(f"Missing {DETECTOR_PATH}")
        if not SAM_PATH.exists():
            raise FileNotFoundError(f"Missing {SAM_PATH}")

        self.detector = YOLO(str(DETECTOR_PATH))
        self.sam = SAM(str(SAM_PATH))
        self.extractor = FeatureExtractor()
        self.matcher = ProductMatcher(self.extractor, VECTORS_DIR)

    def predict(self, image_path, conf=0.40, iou=0.30):
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img_h, img_w = image.shape[:2]

        detections = self.detector.predict(
            source=str(image_path), conf=conf, iou=iou, save=False, verbose=False
        )
        boxes = detections[0].boxes
        if len(boxes) == 0:
            return {"products": [], "count": 0}

        box_prompts = boxes.xyxy.cpu().numpy().tolist()
        seg_results = self.sam.predict(
            source=str(image_path), bboxes=box_prompts, device="cpu", verbose=False
        )
        masks = seg_results[0].masks.data.cpu().numpy()

        results = []
        for box, mask in zip(boxes, masks):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int).tolist()

            mask = cv2.resize(mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST) > 0.5
            isolated = np.full_like(image, 255)
            isolated[mask] = image[mask]

            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue

            crop = isolated[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)

            item_id, confidence, _ = self.matcher.identify(crop_pil)

            results.append({
                "bbox": [x1, y1, x2, y2],
                "product": item_id,
                "confidence": round(confidence, 4),
            })

        return {"products": results, "count": len(results)}


# Loaded once when the Flask app starts, not per-request.
pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        pipeline = ProductPipeline()
    return pipeline
