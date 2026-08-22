"""
Turns real inference results into the numbers the dashboard displays.

There's no labeled validation set in this project, so there is no ground
truth to compute a true accuracy/precision/recall against. What we CAN
report honestly from live traffic is detection confidence — that's what
"accuracy" below actually means. If you later get labeled test data, swap
`overall_accuracy` for a real mAP/precision figure from
`YOLO(...).val(data=...)` instead of this running average.
"""

import threading
import time
from collections import defaultdict


class StatsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._all_confidences = []          # every detection's confidence, ever
        self._inference_times_ms = []       # wall time per /predict call
        self._last_scan_confidences = []    # confidences from the most recent call
        self._scan_count = 0                # number of /predict calls (a "scan")
        self._per_product = defaultdict(lambda: {"scans": 0, "confidences": []})

    def record(self, products, elapsed_ms):
        with self._lock:
            self._scan_count += 1
            self._inference_times_ms.append(elapsed_ms)
            self._last_scan_confidences = [p["confidence"] for p in products]

            for p in products:
                self._all_confidences.append(p["confidence"])
                entry = self._per_product[p["product"]]
                entry["scans"] += 1
                entry["confidences"].append(p["confidence"])

    def snapshot(self):
        with self._lock:
            overall = _pct_mean(self._all_confidences)
            last_scan = _pct_mean(self._last_scan_confidences)
            avg_speed = int(sum(self._inference_times_ms) / len(self._inference_times_ms)) \
                if self._inference_times_ms else None

            high_conf = _share(self._all_confidences, lambda c: c > 0.90)
            low_conf = _share(self._all_confidences, lambda c: c < 0.80)

            products = []
            for name, data in self._per_product.items():
                products.append({
                    "name": name,
                    "scans": data["scans"],
                    "accuracy": _pct_mean(data["confidences"]),
                })
            products.sort(key=lambda p: p["scans"], reverse=True)

            return {
                "overall_accuracy": overall,
                "last_scan_confidence": last_scan,
                "inference_speed_ms": avg_speed,
                "scans_today": self._scan_count,
                "high_confidence_pct": high_conf,
                "low_confidence_pct": low_conf,
                "products": products[:8],
                "has_data": self._scan_count > 0,
            }


def _pct_mean(values):
    if not values:
        return None
    return round(100 * sum(values) / len(values), 1)


def _share(values, predicate):
    if not values:
        return None
    return round(100 * sum(1 for v in values if predicate(v)) / len(values), 1)


class Timer:
    """Small context manager so app.py doesn't need to import `time` itself."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


stats = StatsTracker()
