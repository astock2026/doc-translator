"""
SQLite database module for DocTranslator user management.
"""
import sqlite3
import os
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "doc_translator.db")


def get_db():
    """Get a database connection with row_factory set."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL,
            email       TEXT    UNIQUE NOT NULL,
            password_hash TEXT  NOT NULL,
            is_authorized INTEGER NOT NULL DEFAULT 0,
            balance     INTEGER NOT NULL DEFAULT 0,
            created_at  TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


# ── User CRUD ──────────────────────────────────────────────────────────

def create_user(name, email, password):
    """Create a new user. Returns (user_id, None) on success, (None, error_msg) on failure."""
    conn = get_db()
    try:
        password_hash = generate_password_hash(password)
        now = datetime.utcnow().isoformat()
        cursor = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email, password_hash, now),
        )
        conn.commit()
        return cursor.lastrowid, None
    except sqlite3.IntegrityError:
        return None, "An account with this email already exists."
    finally:
        conn.close()


def get_user_by_email(email):
    """Get a user by email. Returns dict or None."""
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_user_by_id(user_id):
    """Get a user by ID. Returns dict or None."""
    conn = get_db()
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def verify_password(user, password):
    """Check if the given password matches the user's hash."""
    return check_password_hash(user["password_hash"], password)


def get_all_users():
    """Get all users ordered by creation date (newest first)."""
    conn = get_db()
    rows = conn.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def set_authorized(user_id, authorized=True):
    """Toggle a user's authorization status."""
    conn = get_db()
    conn.execute(
        "UPDATE users SET is_authorized = ? WHERE id = ?",
        (1 if authorized else 0, user_id),
    )
    conn.commit()
    conn.close()


def update_balance(user_id, amount):
    """Add (or subtract) balance for a user."""
    conn = get_db()
    conn.execute(
        "UPDATE users SET balance = balance + ? WHERE id = ?",
        (amount, user_id),
    )
    conn.commit()
    conn.close()
