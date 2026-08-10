"""
PostgreSQL database module for DocTranslator user management.
Uses Render's free PostgreSQL tier (DATABASE_URL env var).
"""
import os
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone
from werkzeug.security import generate_password_hash, check_password_hash

DATABASE_URL = os.environ.get("DATABASE_URL", "")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set.")


def _connect():
    """Get a database connection."""
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """Create tables if they don't exist."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id            SERIAL PRIMARY KEY,
                    name          TEXT NOT NULL,
                    email         TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    is_authorized BOOLEAN NOT NULL DEFAULT FALSE,
                    balance       INTEGER NOT NULL DEFAULT 0,
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        conn.commit()
    finally:
        conn.close()


# ── User CRUD ──────────────────────────────────────────────────────────

def create_user(name, email, password):
    """Create a new user. Returns (user_id, None) on success, (None, error_msg) on failure."""
    conn = _connect()
    try:
        password_hash = generate_password_hash(password)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (%s, %s, %s) RETURNING id",
                (name, email, password_hash),
            )
            user_id = cur.fetchone()[0]
        conn.commit()
        return user_id, None
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return None, "An account with this email already exists."
    finally:
        conn.close()


def get_user_by_email(email):
    """Get a user by email. Returns dict or None."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id):
    """Get a user by ID. Returns dict or None."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def verify_password(user, password):
    """Check if the given password matches the user's hash."""
    return check_password_hash(user["password_hash"], password)


def get_all_users():
    """Get all users ordered by creation date (newest first)."""
    conn = _connect()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM users ORDER BY created_at DESC")
            rows = cur.fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def set_authorized(user_id, authorized=True):
    """Toggle a user's authorization status."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET is_authorized = %s WHERE id = %s",
                (authorized, user_id),
            )
        conn.commit()
    finally:
        conn.close()


def update_balance(user_id, amount):
    """Add (or subtract) balance for a user."""
    conn = _connect()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE users SET balance = balance + %s WHERE id = %s",
                (amount, user_id),
            )
        conn.commit()
    finally:
        conn.close()
