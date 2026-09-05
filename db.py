"""
SQLite-backed transaction ledger.

This is the only durable, authoritative source of "money actually taken"
in the project. Everywhere else (StatsTracker) is an in-memory proxy built
from scan/detection volume, and resets whenever the server restarts. A row
only lands here when the cashier presses Pay in Live Cart and /api/checkout
successfully creates it -- so revenue figures sourced from this module
reflect completed checkouts, not items that merely passed in front of the
camera.

Uses Python's built-in sqlite3 (no new dependency, no separate DB server
to run) and persists to scandesk.db next to this file.
"""
import sqlite3
import threading
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "scandesk.db"
_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                total REAL NOT NULL,
                item_count INTEGER NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transaction_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transaction_id INTEGER NOT NULL REFERENCES transactions(id),
                product_id TEXT NOT NULL,
                product_name TEXT NOT NULL,
                unit_price REAL NOT NULL,
                quantity INTEGER NOT NULL,
                line_total REAL NOT NULL
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_txn_items_txn_id ON transaction_items(transaction_id)"
        )
        conn.commit()


def create_transaction(items):
    """
    items: list of {"product_id", "product_name", "unit_price", "quantity"}.
    Prices/names must already be resolved server-side from the catalog by
    the caller -- this function trusts whatever it's handed, so never pass
    it a client-supplied price directly.
    """
    if not items:
        raise ValueError("cannot create a transaction with no items")

    now = time.time()
    total = round(sum(i["unit_price"] * i["quantity"] for i in items), 2)
    item_count = sum(i["quantity"] for i in items)

    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO transactions (created_at, total, item_count) VALUES (?, ?, ?)",
            (now, total, item_count),
        )
        transaction_id = cur.lastrowid
        for i in items:
            line_total = round(i["unit_price"] * i["quantity"], 2)
            conn.execute(
                """INSERT INTO transaction_items
                   (transaction_id, product_id, product_name, unit_price, quantity, line_total)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (transaction_id, i["product_id"], i["product_name"], i["unit_price"], i["quantity"], line_total),
            )
        conn.commit()

    return {
        "id": transaction_id,
        "created_at": now,
        "total": total,
        "item_count": item_count,
        "items": [
            {**i, "line_total": round(i["unit_price"] * i["quantity"], 2)}
            for i in items
        ],
    }


def list_transactions(limit=50):
    with _lock, _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM transactions ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        result = []
        for row in rows:
            items = conn.execute(
                "SELECT * FROM transaction_items WHERE transaction_id = ?", (row["id"],)
            ).fetchall()
            result.append({
                "id": row["id"],
                "created_at": row["created_at"],
                "total": row["total"],
                "item_count": row["item_count"],
                "items": [dict(i) for i in items],
            })
        return result


def revenue_since(timestamp):
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(total), 0) as revenue, COUNT(*) as count "
            "FROM transactions WHERE created_at >= ?",
            (timestamp,),
        ).fetchone()
        return {"revenue": round(row["revenue"], 2), "transaction_count": row["count"]}


def all_time_summary():
    """Lifetime totals -- distinct from revenue_since(today), which resets
    daily. Used for the dashboard's all-time stats, which are real
    historical figures pulled straight from the transactions table."""
    info = revenue_since(0)
    count = info["transaction_count"]
    revenue = info["revenue"]
    return {
        "revenue": revenue,
        "transaction_count": count,
        "avg_transaction_value": round(revenue / count, 2) if count else 0.0,
    }


def all_transaction_totals():
    """Raw (created_at, total) pairs for every transaction -- used by
    StatsTracker.timeseries() to bucket revenue into the same hourly/daily
    windows as scan activity."""
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT created_at, total FROM transactions").fetchall()
    return [(r["created_at"], r["total"]) for r in rows]


def revenue_by_product(limit=None):
    with _lock, _connect() as conn:
        rows = conn.execute(
            """SELECT product_id, product_name, SUM(line_total) as revenue, SUM(quantity) as units
               FROM transaction_items
               GROUP BY product_id
               ORDER BY revenue DESC"""
        ).fetchall()
    result = [
        {"product_id": r["product_id"], "name": r["product_name"], "revenue": round(r["revenue"], 2), "units": r["units"]}
        for r in rows
    ]
    return result[:limit] if limit else result