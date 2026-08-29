"""
Turns real inference results into the numbers the dashboard displays.

There's no labeled validation set in this project, so there is no ground
truth to compute a true accuracy/precision/recall against. What we CAN
report honestly from live traffic is detection confidence — that's what
"accuracy" below actually means. If you later get labeled test data, swap
`overall_accuracy` for a real mAP/precision figure from
`YOLO(...).val(data=...)` instead of this running average.

Revenue and product names come from `catalog.py` (a small persisted
price/name side-car file) -- the detector itself has no notion of price,
so every dollar figure here traces back to whatever was actually entered
on the Product Catalog page, not a guess.
"""

import math
import threading
import time
from collections import defaultdict, deque

import numpy as np

import catalog

MAX_EVENTS = 25


class StatsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._all_confidences = []          # every detection's confidence, ever
        self._inference_times_ms = []       # wall time per /predict call
        self._scan_count = 0                # number of /predict calls (a "scan")
        self._revenue_total = 0.0           # sum of catalog price at time of each detection
        self._last_scan_items = []          # rich detail from the most recent /predict call
        self._per_product = defaultdict(lambda: {"scans": 0, "confidences": [], "timestamps": []})
        self._events = deque(maxlen=MAX_EVENTS)   # real telemetry log, newest last

    def record(self, products, elapsed_ms):
        with self._lock:
            now = time.time()
            self._scan_count += 1
            self._inference_times_ms.append(elapsed_ms)

            last_items = []
            call_revenue = 0.0

            for p in products:
                self._all_confidences.append(p["confidence"])
                entry = self._per_product[p["product"]]
                entry["scans"] += 1
                entry["confidences"].append(p["confidence"])
                entry["timestamps"].append(now)

                price = catalog.get_price(p["product"])
                call_revenue += price
                last_items.append({
                    "product_id": p["product"],
                    "name": catalog.get_name(p["product"]),
                    "confidence": p["confidence"],
                    "price": price,
                    "image_url": f"/static/catalog/{p['product']}.jpg",
                })

            self._revenue_total += call_revenue
            self._last_scan_items = last_items
            self._log(
                f"MODEL_INFER_OK: {len(products)} item(s) detected, "
                f"{elapsed_ms:.0f}ms, ${call_revenue:.2f} this scan",
                level="info",
            )

    def log_event(self, message, level="info"):
        """Lets app.py record real, non-inference telemetry (camera connect/
        disconnect, etc.) into the same feed shown on the dashboard."""
        with self._lock:
            self._log(message, level)

    def _log(self, message, level="info"):
        # caller must already hold self._lock
        self._events.append({
            "timestamp": time.time(),
            "message": message,
            "level": level,
        })

    def snapshot(self):
        with self._lock:
            overall = _pct_mean(self._all_confidences)
            last_scan = _pct_mean([i["confidence"] for i in self._last_scan_items])
            avg_speed = int(sum(self._inference_times_ms) / len(self._inference_times_ms)) \
                if self._inference_times_ms else None

            high_conf = _share(self._all_confidences, lambda c: c > 0.90)
            low_conf = _share(self._all_confidences, lambda c: c < 0.80)

            total_items = len(self._all_confidences)
            avg_basket_size = round(total_items / self._scan_count, 1) if self._scan_count else None

            products = []
            for product_id, data in self._per_product.items():
                products.append({
                    "name": catalog.get_name(product_id),
                    "scans": data["scans"],
                    "accuracy": _pct_mean(data["confidences"]),
                })
            products.sort(key=lambda p: p["scans"], reverse=True)

            events = [
                {
                    "time": time.strftime("%H:%M:%S", time.localtime(e["timestamp"])),
                    "message": e["message"],
                    "level": e["level"],
                }
                for e in reversed(self._events)
            ]

            return {
                "overall_accuracy": overall,
                "last_scan_confidence": last_scan,
                "inference_speed_ms": avg_speed,
                "scans_today": self._scan_count,
                "high_confidence_pct": high_conf,
                "low_confidence_pct": low_conf,
                "revenue_today": round(self._revenue_total, 2),
                "avg_basket_size": avg_basket_size,
                "products": products[:8],
                "last_scan_items": self._last_scan_items,
                "events": events,
                "has_data": self._scan_count > 0,
            }

    def predictions(self, window_days=14, forecast_days=7, lead_time_days=7, safety_factor=0.25):
        """
        Lightweight "what sells best / how much to reorder" forecast.

        There's no separate POS/sales table in this project -- a detection at
        checkout IS the closest thing we have to a sale, so scan volume is
        used as the demand signal. For each product we bucket its scan
        timestamps from the last `window_days` into whole days, fit a simple
        linear trend (numpy.polyfit) across those daily counts, and project
        it forward `forecast_days` to size a reorder quantity (with a lead
        time + safety-stock buffer). With fewer than 2 days of data we fall
        back to a flat daily-average projection instead of a trend line.

        This is a heuristic, not a trained ML model -- it's honest about
        that in the API response (`method`). If real sales/inventory data
        becomes available later, swap this out for a proper time-series
        model trained on that instead.
        """
        with self._lock:
            now = time.time()
            day_seconds = 86400.0
            window_start = now - window_days * day_seconds
            span_days = max(1, math.ceil((now - window_start) / day_seconds))

            rows = []
            for product_id, data in self._per_product.items():
                ts = [t for t in data["timestamps"] if t >= window_start]
                if not ts:
                    continue

                buckets = defaultdict(int)
                for t in ts:
                    day_idx = int((t - window_start) // day_seconds)
                    buckets[day_idx] += 1

                total_recent = sum(buckets.values())
                distinct_days = len(buckets)

                if distinct_days >= 2:
                    xs = np.array(sorted(buckets.keys()), dtype=float)
                    ys = np.array([buckets[x] for x in sorted(buckets.keys())], dtype=float)
                    slope, intercept = np.polyfit(xs, ys, 1)
                    projected_daily = slope * span_days + intercept
                    # don't let a steep downward fit go negative / to zero outright
                    projected_daily = max(projected_daily, (total_recent / span_days) * 0.25)
                else:
                    slope = 0.0
                    projected_daily = total_recent / span_days

                projected_daily = max(projected_daily, 0.0)
                predicted_demand = projected_daily * forecast_days
                reorder_qty = math.ceil(
                    projected_daily * (forecast_days + lead_time_days) * (1 + safety_factor)
                )

                if slope > 0.15:
                    trend = "rising"
                elif slope < -0.15:
                    trend = "falling"
                else:
                    trend = "steady"

                price = catalog.get_price(product_id)

                rows.append({
                    "name": catalog.get_name(product_id),
                    "scans_recent": total_recent,
                    "avg_confidence": _pct_mean(data["confidences"]),
                    "predicted_demand": round(predicted_demand, 1),
                    "suggested_reorder_qty": max(reorder_qty, 1),
                    "predicted_revenue": round(predicted_demand * price, 2),
                    "trend": trend,
                })

            rows.sort(key=lambda r: r["scans_recent"], reverse=True)

            return {
                "method": "scan_volume_linear_trend",
                "window_days": window_days,
                "forecast_days": forecast_days,
                "lead_time_days": lead_time_days,
                "items": rows[:10],
                "top_seller": rows[0]["name"] if rows else None,
                "has_data": len(rows) > 0,
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