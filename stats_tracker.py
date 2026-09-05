"""
Turns real inference results into the numbers the dashboard displays.

There's no labeled validation set in this project, so there is no ground
truth to compute a true accuracy/precision/recall against. What we CAN
report honestly from live traffic is detection confidence — that's what
"accuracy" below actually means. If you later get labeled test data, swap
`overall_accuracy` for a real mAP/precision figure from
`YOLO(...).val(data=...)` instead of this running average.

Revenue and product names come from `catalog.py` (a small persisted
price/name side-car file) for pricing detected items, but the actual
"revenue" figures shown on the dashboard come from `db.py` -- the SQLite
transactions table -- which only gets a row when a cashier presses Pay in
Live Cart. A detection is not a sale; a completed checkout is. This module
still tracks per-scan telemetry (confidence, latency, per-product scan
counts) since none of that requires a completed transaction to be
meaningful.
"""

import math
import threading
import time
from collections import defaultdict, deque

import numpy as np

import catalog
import db

MAX_EVENTS = 25
MAX_SCAN_EVENTS = 5000


class StatsTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._all_confidences = []          # every detection's confidence, ever
        self._inference_times_ms = []       # wall time per /predict call
        self._scan_count = 0                # number of /predict calls (a "scan")
        self._last_scan_items = []          # rich detail from the most recent /predict call
        self._per_product = defaultdict(lambda: {"scans": 0, "confidences": [], "timestamps": []})
        self._events = deque(maxlen=MAX_EVENTS)          # real telemetry log, newest last
        self._scan_events = deque(maxlen=MAX_SCAN_EVENTS)  # one entry per /predict call, for trend charts

    def record(self, products, elapsed_ms):
        with self._lock:
            now = time.time()
            self._scan_count += 1
            self._inference_times_ms.append(elapsed_ms)

            last_items = []
            scanned_value = 0.0  # catalog value of items detected this scan -- NOT revenue

            for p in products:
                self._all_confidences.append(p["confidence"])
                entry = self._per_product[p["product"]]
                entry["scans"] += 1
                entry["confidences"].append(p["confidence"])
                entry["timestamps"].append(now)

                price = catalog.get_price(p["product"])
                scanned_value += price
                last_items.append({
                    "product_id": p["product"],
                    "name": catalog.get_name(p["product"]),
                    "confidence": p["confidence"],
                    "price": price,
                    "image_url": f"/static/catalog/{p['product']}.jpg",
                })

            self._last_scan_items = last_items
            self._scan_events.append({
                "timestamp": now,
                "items": len(products),
                "avg_confidence": (sum(p["confidence"] for p in products) / len(products)) if products else None,
            })
            self._log(
                f"MODEL_INFER_OK: {len(products)} item(s) detected, "
                f"{elapsed_ms:.0f}ms, ${scanned_value:.2f} catalog value scanned",
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
            unique_products_scanned = len(self._per_product)

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

        # Revenue is read from the transactions DB (completed checkouts),
        # outside the stats lock since it's an independent data source.
        today_start = _start_of_today()
        revenue_info = db.revenue_since(today_start)
        top_products_by_revenue = db.revenue_by_product(limit=5)
        all_time = db.all_time_summary()

        return {
            "overall_accuracy": overall,
            "last_scan_confidence": last_scan,
            "inference_speed_ms": avg_speed,
            "scans_today": self._scan_count,
            "high_confidence_pct": high_conf,
            "low_confidence_pct": low_conf,
            "revenue_today": revenue_info["revenue"],
            "transactions_today": revenue_info["transaction_count"],
            "avg_basket_size": avg_basket_size,
            "total_items_detected": total_items,
            "unique_products_scanned": unique_products_scanned,
            "all_time_revenue": all_time["revenue"],
            "all_time_transactions": all_time["transaction_count"],
            "avg_transaction_value": all_time["avg_transaction_value"],
            "products": products[:8],
            "top_products_by_revenue": top_products_by_revenue,
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

    def timeseries(self, hours=24, days=7):
        """
        Real hour-by-hour and day-by-day series for the trend charts on the
        dashboard. Scan volume and confidence come from per-scan telemetry
        (no synthetic/sample data); revenue comes from the transactions DB
        (completed checkouts only) bucketed into the same windows. Buckets
        with no activity simply report 0 -- they are not backfilled or
        smoothed.
        """
        with self._lock:
            now = time.time()
            hour_seconds = 3600.0
            day_seconds = 86400.0

            hour_start = now - hours * hour_seconds
            hourly = [{"scans": 0} for _ in range(hours)]

            day_start = now - days * day_seconds
            daily = [{"scans": 0, "confidences": []} for _ in range(days)]

            for e in self._scan_events:
                if e["timestamp"] >= hour_start:
                    idx = min(hours - 1, int((e["timestamp"] - hour_start) // hour_seconds))
                    hourly[idx]["scans"] += e["items"]

                if e["timestamp"] >= day_start:
                    idx = min(days - 1, int((e["timestamp"] - day_start) // day_seconds))
                    daily[idx]["scans"] += e["items"]
                    if e["avg_confidence"] is not None:
                        daily[idx]["confidences"].append(e["avg_confidence"])

            has_scan_data = len(self._scan_events) > 0

        # Revenue comes from completed transactions, bucketed the same way,
        # outside the stats lock since it's an independent data source.
        hourly_revenue = [0.0] * hours
        daily_revenue = [0.0] * days
        transaction_totals = db.all_transaction_totals()
        for txn_time, total in transaction_totals:
            if txn_time >= hour_start:
                idx = min(hours - 1, int((txn_time - hour_start) // hour_seconds))
                hourly_revenue[idx] += total
            if txn_time >= day_start:
                idx = min(days - 1, int((txn_time - day_start) // day_seconds))
                daily_revenue[idx] += total

        hourly_labels = [
            time.strftime("%H:00", time.localtime(hour_start + i * hour_seconds))
            for i in range(hours)
        ]
        daily_labels = [
            time.strftime("%a", time.localtime(day_start + i * day_seconds))
            for i in range(days)
        ]

        return {
            "hourly": {
                "labels": hourly_labels,
                "scans": [b["scans"] for b in hourly],
                "revenue": [round(v, 2) for v in hourly_revenue],
            },
            "daily": {
                "labels": daily_labels,
                "scans": [b["scans"] for b in daily],
                "revenue": [round(v, 2) for v in daily_revenue],
                "avg_confidence": [
                    round(100 * sum(b["confidences"]) / len(b["confidences"]), 1)
                    if b["confidences"] else None
                    for b in daily
                ],
            },
            "has_data": has_scan_data or len(transaction_totals) > 0,
        }


def _pct_mean(values):
    if not values:
        return None
    return round(100 * sum(values) / len(values), 1)


def _share(values, predicate):
    if not values:
        return None
    return round(100 * sum(1 for v in values if predicate(v)) / len(values), 1)


def _start_of_today():
    """Unix timestamp for local midnight -- the boundary for 'revenue today'."""
    now = time.localtime()
    start = time.struct_time((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, now.tm_isdst))
    return time.mktime(start)


class Timer:
    """Small context manager so app.py doesn't need to import `time` itself."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000


stats = StatsTracker()