"""
Persisted price/name metadata for products, keyed by product_id (the same
id used by vectors/<id>_vector.pkl and static/catalog/<id>.jpg).

The vision pipeline only ever produces a product_id + confidence -- it has
no concept of price. Revenue is a business fact, not something a detector
can infer, so it has to come from somewhere real: this is a small JSON
side-car file that the Product Catalog page reads and writes. Until a
price is set for a product it defaults to 0, which naturally zeroes out
its contribution to revenue instead of guessing a number.
"""
import json
import threading
from pathlib import Path

CATALOG_META_PATH = Path(__file__).parent / "catalog_meta.json"
_lock = threading.Lock()


def _load():
    if not CATALOG_META_PATH.exists():
        return {}
    try:
        with open(CATALOG_META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data):
    with open(CATALOG_META_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def get_all():
    with _lock:
        return _load()


def get_entry(product_id):
    with _lock:
        return _load().get(product_id, {})


def get_price(product_id):
    with _lock:
        entry = _load().get(product_id, {})
        try:
            return float(entry.get("price") or 0)
        except (TypeError, ValueError):
            return 0.0


def get_name(product_id):
    with _lock:
        entry = _load().get(product_id, {})
        return entry.get("name") or product_id


def set_meta(product_id, price=None, name=None):
    """Update (or create) a product's price/name. Only overwrites fields
    that are explicitly passed (not None)."""
    with _lock:
        meta = _load()
        entry = meta.get(product_id, {})
        if price is not None:
            entry["price"] = float(price)
        if name is not None:
            entry["name"] = name.strip() or entry.get("name")
        meta[product_id] = entry
        _save(meta)
        return entry