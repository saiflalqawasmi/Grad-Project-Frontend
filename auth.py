"""
Authentication and role-based access control.

Ported from the auth-enabled branch of this project. Kept deliberately in
the same style as db.py (raw sqlite3, no ORM) so this doesn't drag in
Flask-SQLAlchemy just for a five-column users table -- only Flask-Login
is added as a new dependency, for session handling.

Three roles: admin, manager, cashier.
  - admin    : full access -- manage users, add new catalog products.
  - manager  : dashboard, product catalog, stats/forecast/transactions.
  - cashier  : Live Cart only (scan, camera, checkout).
"""
import sqlite3
import threading
from functools import wraps
from pathlib import Path

from flask import redirect, url_for, flash, request, jsonify
from flask_login import LoginManager, UserMixin, current_user
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = Path(__file__).parent / "users.db"
_lock = threading.Lock()

ROLES = ("admin", "manager", "cashier")

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please log in to access this page."
login_manager.login_message_category = "danger"


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the users table and seeds a default admin account the
    first time the app runs against an empty database."""
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'cashier'
            )
        """)
        conn.commit()
        row = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()
        if row["c"] == 0:
            conn.execute(
                "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
                ("admin", "admin@scandesk.local", generate_password_hash("admin123"), "admin"),
            )
            conn.commit()
            print(
                "[auth] no users found -- seeded default account "
                "admin / admin123 (change this password immediately)."
            )


class User(UserMixin):
    """Thin wrapper around a users-table row for Flask-Login."""

    def __init__(self, row):
        self.id = row["id"]
        self.username = row["username"]
        self.email = row["email"]
        self.password_hash = row["password_hash"]
        self.role = row["role"]

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def is_admin(self):
        return self.role == "admin"

    def is_manager(self):
        return self.role == "manager"

    def is_cashier(self):
        return self.role == "cashier"

    def home_url(self):
        """Where to send this user right after login."""
        if self.is_admin():
            return url_for("admin")
        if self.is_manager():
            return url_for("dashboard")
        return url_for("livecart_page")


def _row_to_user(row):
    return User(row) if row else None


def get_user_by_id(user_id):
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return _row_to_user(row)


def get_user_by_username(username):
    with _lock, _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    return _row_to_user(row)


def list_users():
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM users ORDER BY id DESC").fetchall()
    return [User(r) for r in rows]


def find_conflict(username, email):
    """Returns the existing user blocking this username/email, if any."""
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?", (username, email)
        ).fetchone()
    return _row_to_user(row)


def create_user(username, email, password, role):
    with _lock, _connect() as conn:
        cur = conn.execute(
            "INSERT INTO users (username, email, password_hash, role) VALUES (?, ?, ?, ?)",
            (username, email, generate_password_hash(password), role),
        )
        conn.commit()
        return cur.lastrowid


def update_role(user_id, role):
    with _lock, _connect() as conn:
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (role, user_id))
        conn.commit()


def delete_user(user_id):
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()


@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(int(user_id))


@login_manager.unauthorized_handler
def unauthorized():
    flash("Please log in to access this page.", "danger")
    return redirect(url_for("login", next=request.path))


def _forbidden(message):
    if request.path.startswith("/api/") or request.is_json:
        return jsonify({"error": message}), 403
    flash(message, "danger")
    return redirect(current_user.home_url())


def admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            return login_manager.unauthorized()
        if not current_user.is_admin():
            return _forbidden("Admin access required.")
        return view(*args, **kwargs)
    return wrapped


def manager_or_admin_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            return login_manager.unauthorized()
        if current_user.role not in ("admin", "manager"):
            return _forbidden("Manager or Admin access required.")
        return view(*args, **kwargs)
    return wrapped
