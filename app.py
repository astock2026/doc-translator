"""
Document Translator — Skill-quality bilingual .docx translation
==============================================================
Three modes:
  1. Manual two-step:  extract → (user translates) → insert
  2. Auto pipeline:     extract → LLM translate → CMC verify → insert
  3. Verification only: verify translations.json against CMC glossary

Powered by the bilingual-docx-translate skill pipeline.
LLM translation uses OpenAI-compatible API (DeepSeek, OpenAI, etc.)
with the full CMC/GMP terminology glossary.
"""
import os
import sys
import json
import subprocess
import shutil
import uuid
import logging
from functools import wraps
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, send_file, session, redirect, url_for
)
from werkzeug.utils import secure_filename

import db

SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB
app.config["UPLOAD_FOLDER"] = str(Path(__file__).parent / "uploads")
app.config["OUTPUT_FOLDER"] = str(Path(__file__).parent / "outputs")

app.config["TEMPLATES_AUTO_RELOAD"] = True  # re-read templates from disk every request

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# LLM config from environment
# LLM_PROVIDER: "openai" (DeepSeek/OpenAI/Groq) or "gemini" (Google Gemini)
_provider = os.environ.get("LLM_PROVIDER", "openai").lower()
_api_url = os.environ.get("LLM_API_URL") or os.environ.get("LLM_API_BASE")
if _provider == "gemini":
    _default_model = "gemini-3.6-flash"
else:
    _default_model = "deepseek-chat"
LLM_CONFIG = {
    "provider": _provider,
    "api_base": _api_url or "https://api.deepseek.com/v1",
    "api_key": os.environ.get("LLM_API_KEY", ""),
    "model": os.environ.get("LLM_MODEL", _default_model),
}
LLM_AVAILABLE = bool(LLM_CONFIG["api_key"])

# ── Session & Security ─────────────────────────────────────────────────
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24).hex())

# Session security
app.config["SESSION_COOKIE_SECURE"] = True
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# ── Email (SMTP) Config ────────────────────────────────────────────────
# SendGrid email config (replaces SMTP — Outlook/Hotmail basic auth is dead)
SENDGRID_API_KEY = os.environ.get("SENDGRID_API_KEY", "")
SENDGRID_ENABLED = bool(SENDGRID_API_KEY)
SENDGRID_FROM = os.environ.get("SENDGRID_FROM_EMAIL", "adam_j_cheng@hotmail.com")
NOTIFY_EMAIL = os.environ.get("NOTIFY_EMAIL", "adam_j_cheng@hotmail.com")

# Admin password for /admin dashboard
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# ── Initialize Database ────────────────────────────────────────────────
db.init_db()
logger.info("Database initialized.")


def run_script(script_name, *args, timeout=120):
    """Run a Python script from the scripts/ directory."""
    script_path = SCRIPTS_DIR / script_name
    cmd = [sys.executable, str(script_path)] + list(args)
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout.strip():
        logger.info(result.stdout.strip())
    if result.returncode != 0:
        logger.error(f"Script {script_name} failed:\nSTDERR: {result.stderr}")
        raise RuntimeError(result.stderr or result.stdout or f"Script exited with code {result.returncode}")
    return result.stdout


# ═══════════════════════════════════════════════════════════════════════
#  Email & Auth Helpers
# ═══════════════════════════════════════════════════════════════════════

def send_email(subject, body):
    """Send an email notification to the admin via SendGrid API."""
    if not SENDGRID_ENABLED:
        logger.warning(f"SendGrid not configured (SENDGRID_API_KEY empty). Would have sent: {subject}")
        return False

    logger.info(f"Attempting to send email via SendGrid: from={SENDGRID_FROM} to={NOTIFY_EMAIL}")
    try:
        from sendgrid import SendGridAPIClient
        from sendgrid.helpers.mail import Mail

        message = Mail(
            from_email=SENDGRID_FROM,
            to_emails=NOTIFY_EMAIL,
            subject=f"[DocTranslator] {subject}",
            plain_text_content=body,
        )
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)

        if 200 <= response.status_code < 300:
            logger.info(f"Email sent successfully: {subject} (status {response.status_code})")
            return True
        else:
            logger.error(f"SendGrid returned status {response.status_code}: {response.body}")
            return False
    except Exception as e:
        logger.error(f"Failed to send email: {type(e).__name__}: {e}")
        return False


def login_required(f):
    """Decorator: require user to be logged in (session['user_id'] must exist)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Please log in first."}), 401
        return f(*args, **kwargs)
    return decorated


COST_PER_TRANSLATION = 29  # CNY


def authorized_required(f):
    """Decorator: require user to be logged in, authorized, and have sufficient balance."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Please log in first."}), 401
        user = db.get_user_by_id(session["user_id"])
        if not user:
            session.clear()
            return jsonify({"error": "Account not found."}), 401
        if not user["is_authorized"]:
            return jsonify({
                "error": "Your account is not yet authorized. Please complete payment and wait for approval."
            }), 403
        if user["balance"] < COST_PER_TRANSLATION:
            return jsonify({
                "error": f"Insufficient balance. Each translation costs ¥{COST_PER_TRANSLATION}. "
                         f"Your current balance is ¥{user['balance']}. Please top up on the payment page."
            }), 402
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════════
#  Pages
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", llm_available=LLM_AVAILABLE)


@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/login")
def login():
    return render_template("login.html")


@app.route("/payment")
def payment():
    return render_template("payment.html")


# ═══════════════════════════════════════════════════════════════════════
#  Auth API
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/signup", methods=["POST"])
def api_signup():
    """Register a new user. Expects JSON: {name, email, password}."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid request. Send JSON."}), 400

    name = (data.get("name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = (data.get("password") or "").strip()

    if not name:
        return jsonify({"error": "Name is required."}), 400
    if not email or "@" not in email:
        return jsonify({"error": "A valid email is required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    user_id, error = db.create_user(name, email, password)
    if error:
        return jsonify({"error": error}), 409

    # Log the user in immediately
    session["user_id"] = user_id
    session["user_name"] = name
    session["user_email"] = email

    # Send notification to admin
    send_email(
        f"New signup: {name}",
        f"A new user has signed up on DocTranslator.\n\n"
        f"Name:  {name}\n"
        f"Email: {email}\n"
        f"Time:  {db.get_user_by_id(user_id)['created_at']}\n\n"
        f"Go to the admin panel to authorize this user:\n"
        f"  (your domain)/admin\n",
    )

    logger.info(f"New signup: {name} <{email}>")
    return jsonify({
        "success": True,
        "message": f"Welcome, {name}! Your account has been created. "
                    "Please complete payment to activate translation access.",
        "user": {"name": name, "email": email, "is_authorized": False, "balance": 0},
    }), 201


@app.route("/api/login", methods=["POST"])
def api_login():
    """Log in with email + password. Expects JSON or form data."""
    if request.is_json:
        data = request.get_json()
        email = (data.get("email") or "").strip().lower()
        password = data.get("password", "")
    else:
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    user = db.get_user_by_email(email)
    if not user or not db.verify_password(user, password):
        return jsonify({"error": "Invalid email or password."}), 401

    session["user_id"] = user["id"]
    session["user_name"] = user["name"]
    session["user_email"] = user["email"]

    return jsonify({
        "success": True,
        "user": {
            "name": user["name"],
            "email": user["email"],
            "is_authorized": bool(user["is_authorized"]),
            "balance": user["balance"],
            "cost_per_translation": COST_PER_TRANSLATION,
            "can_translate": bool(user["is_authorized"] and user["balance"] >= COST_PER_TRANSLATION),
        },
    })


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Clear the session."""
    session.clear()
    return jsonify({"success": True})


@app.route("/api/me")
def api_me():
    """Return the currently logged-in user, or null if not logged in."""
    if "user_id" not in session:
        return jsonify({"user": None})

    user = db.get_user_by_id(session["user_id"])
    if not user:
        session.clear()
        return jsonify({"user": None})

    return jsonify({
        "user": {
            "name": user["name"],
            "email": user["email"],
            "is_authorized": bool(user["is_authorized"]),
            "balance": user["balance"],
            "cost_per_translation": COST_PER_TRANSLATION,
            "can_translate": bool(user["is_authorized"] and user["balance"] >= COST_PER_TRANSLATION),
        },
    })


@app.route("/api/payment-confirm", methods=["POST"])
@login_required
def api_payment_confirm():
    """User claims they have paid. Sends email to admin with details."""
    data = request.get_json(silent=True) or {}
    amount = data.get("amount", 0)

    user = db.get_user_by_id(session["user_id"])
    if not user:
        return jsonify({"error": "Account not found."}), 404

    send_email(
        f"Payment claim: {user['name']} — ¥{amount}",
        f"A user claims they have paid via WeChat.\n\n"
        f"Name:   {user['name']}\n"
        f"Email:  {user['email']}\n"
        f"Amount: ¥{amount}\n\n"
        f"Please verify the payment in WeChat, then authorize this user from the admin panel:\n"
        f"  (your domain)/admin\n",
    )

    logger.info(f"Payment confirm: {user['name']} <{user['email']}> amount=¥{amount}")
    return jsonify({
        "success": True,
        "message": "Payment confirmation sent! We will verify and activate your account shortly.",
    })


# ═══════════════════════════════════════════════════════════════════════
#  Admin
# ═══════════════════════════════════════════════════════════════════════

@app.route("/admin", methods=["GET", "POST"])
def admin():
    """Admin dashboard — password-protected."""
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == ADMIN_PASSWORD:
            session["is_admin"] = True
        else:
            return render_template("admin.html", error="Incorrect password.", users=None)

    if not session.get("is_admin"):
        return render_template("admin.html", error=None, users=None)

    users = db.get_all_users()
    return render_template("admin.html", error=None, users=users)


@app.route("/api/admin/authorize", methods=["POST"])
def admin_authorize():
    """Toggle a user's authorization. Admin only."""
    if not session.get("is_admin"):
        return jsonify({"error": "Admin access required."}), 403

    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    authorized = data.get("authorized", True)

    user = db.get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    db.set_authorized(user_id, authorized)
    logger.info(f"Admin: {'authorized' if authorized else 'deauthorized'} user {user['name']} <{user['email']}>")
    return jsonify({"success": True})


@app.route("/api/admin/balance", methods=["POST"])
def admin_balance():
    """Adjust a user's balance. Admin only."""
    if not session.get("is_admin"):
        return jsonify({"error": "Admin access required."}), 403

    data = request.get_json(silent=True) or {}
    user_id = data.get("user_id")
    amount = data.get("amount", 0)

    user = db.get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found."}), 404

    db.update_balance(user_id, amount)
    updated = db.get_user_by_id(user_id)
    logger.info(f"Admin: adjusted balance for {user['name']} by {amount:+d}. New balance: {updated['balance']}")
    return jsonify({"success": True, "balance": updated["balance"]})


@app.route("/api/extract", methods=["POST"])
@authorized_required
def extract():
    """Upload .docx → extract Chinese paragraph/table text → return JSON."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".docx"):
        return jsonify({"error": "Only .docx files are supported"}), 400

    job_id = uuid.uuid4().hex[:8]
    work_dir = Path(app.config["UPLOAD_FOLDER"]) / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_path = work_dir / secure_filename(file.filename)
        file.save(str(input_path))

        content_path = work_dir / "content.json"
        run_script("extract_paragraphs.py", str(input_path), "--output", str(content_path))

        with open(content_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        para_count = len(content.get("paragraphs", []))
        table_count = len(content.get("tables", []))
        cell_count = sum(
            len(row.get("cells", []))
            for t in content.get("tables", [])
            for row in t.get("rows", [])
        )

        return jsonify({
            "success": True,
            "job_id": job_id,
            "content": content,
            "stats": {
                "paragraphs": para_count,
                "tables": table_count,
                "table_cells": cell_count,
            },
        })

    except Exception as e:
        logger.exception("Extract failed")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(str(work_dir), ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
#  Step 2: Insert translations
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/insert", methods=["POST"])
@authorized_required
def insert():
    """Upload .docx + translations.json → insert translations → return bilingual .docx."""
    if "file" not in request.files:
        return jsonify({"error": "No .docx file uploaded"}), 400
    if "translations" not in request.files:
        return jsonify({"error": "No translations.json file uploaded"}), 400

    docx_file = request.files["file"]
    trans_file = request.files["translations"]

    if not docx_file.filename or not docx_file.filename.lower().endswith(".docx"):
        return jsonify({"error": "Invalid .docx file"}), 400
    if not trans_file.filename or not trans_file.filename.lower().endswith(".json"):
        return jsonify({"error": "Invalid translations file (must be .json)"}), 400

    job_id = uuid.uuid4().hex[:8]
    work_dir = Path(app.config["UPLOAD_FOLDER"]) / job_id
    output_dir = Path(app.config["OUTPUT_FOLDER"]) / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        input_path = work_dir / secure_filename(docx_file.filename)
        docx_file.save(str(input_path))

        trans_path = work_dir / "translations.json"
        trans_file.save(str(trans_path))

        with open(trans_path, "r", encoding="utf-8") as f:
            trans_data = json.load(f)
        if "paragraphs" not in trans_data:
            return jsonify({"error": "translations.json must have 'paragraphs' key"}), 400

        base_name = docx_file.filename.rsplit(".", 1)[0]
        output_path = output_dir / f"{base_name}_Bilingual.docx"
        run_script(
            "insert_translations_safe.py",
            str(input_path),
            str(trans_path),
            "--output", str(output_path),
        )

        logger.info(f"Insert complete: {output_path}")

        # Deduct balance
        db.update_balance(session["user_id"], -COST_PER_TRANSLATION)
        updated_user = db.get_user_by_id(session["user_id"])
        logger.info(
            f"Charged ¥{COST_PER_TRANSLATION} to {updated_user['name']}. "
            f"Remaining balance: ¥{updated_user['balance']}"
        )

        para_done = sum(1 for p in trans_data.get("paragraphs", []) if p.get("translation", "").strip())
        cell_done = sum(
            1 for t in trans_data.get("tables", [])
            for r in t.get("rows", [])
            for c in r.get("cells", [])
            if c.get("translation", "").strip()
        )

        return send_file(
            str(output_path),
            as_attachment=True,
            download_name=f"{base_name}_Bilingual.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

    except Exception as e:
        logger.exception("Insert failed")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(str(work_dir), ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
#  CMC Terminology Verification
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/verify", methods=["POST"])
@authorized_required
def verify():
    """Upload translations.json [+ optional content.json] → verify CMC/GMP terminology → return report."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".json"):
        return jsonify({"error": "Only .json files are supported"}), 400

    job_id = uuid.uuid4().hex[:8]
    work_dir = Path(app.config["UPLOAD_FOLDER"]) / job_id
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        trans_path = work_dir / "translations.json"
        file.save(str(trans_path))

        # Optional content.json for Chinese text cross-reference
        content_arg = []
        if "content" in request.files:
            content_file = request.files["content"]
            if content_file.filename and content_file.filename.lower().endswith(".json"):
                content_path = work_dir / "content.json"
                content_file.save(str(content_path))
                content_arg = ["--content", str(content_path)]

        report_path = work_dir / "report.json"
        run_script(
            "verify_cmc.py", str(trans_path), *content_arg, "--output", str(report_path),
            timeout=30,
        )

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        return jsonify({
            "success": True,
            "report": report,
        })

    except Exception as e:
        logger.exception("Verification failed")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(str(work_dir), ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
#  Full Auto Pipeline: Extract → LLM Translate → CMC Verify → Insert
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/translate", methods=["POST"])
@authorized_required
def translate():
    """One-click: upload .docx → LLM translate → CMC verify → download bilingual .docx."""
    if not LLM_AVAILABLE:
        return jsonify({
            "error": "LLM not configured. Set LLM_API_KEY environment variable.\n"
                     "Get a free key at https://platform.deepseek.com/api_keys"
        }), 503

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if not file.filename or not file.filename.lower().endswith(".docx"):
        return jsonify({"error": "Only .docx files are supported"}), 400

    job_id = uuid.uuid4().hex[:8]
    work_dir = Path(app.config["UPLOAD_FOLDER"]) / job_id
    output_dir = Path(app.config["OUTPUT_FOLDER"]) / job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        base_name = secure_filename(file.filename).rsplit(".", 1)[0]
        input_path = work_dir / secure_filename(file.filename)
        file.save(str(input_path))

        # Phase 1: Extract
        logger.info("[Phase 1/4] Extracting content...")
        content_path = work_dir / "content.json"
        run_script("extract_paragraphs.py", str(input_path), "--output", str(content_path))

        with open(content_path, "r", encoding="utf-8") as f:
            content = json.load(f)

        para_count = len(content.get("paragraphs", []))
        cell_count = sum(
            len(row.get("cells", []))
            for t in content.get("tables", [])
            for row in t.get("rows", [])
        )

        # Phase 2: LLM Translate (longer timeout for API calls)
        logger.info(f"[Phase 2/4] Translating {para_count} paragraphs + {cell_count} cells via LLM...")
        trans_path = work_dir / "translations.json"
        run_script(
            "translate_llm.py", str(content_path), "--output", str(trans_path),
            timeout=600,  # 10 min for LLM translation
        )

        # Phase 3: CMC Verification
        logger.info("[Phase 3/4] Verifying CMC/GMP terminology...")
        report_path = work_dir / "report.json"
        run_script(
            "verify_cmc.py", str(trans_path),
            "--content", str(content_path),
            "--output", str(report_path),
            timeout=30,
        )
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        # Phase 4: Insert
        logger.info("[Phase 4/4] Inserting translations into document...")
        output_path = output_dir / f"{base_name}_Bilingual.docx"
        run_script(
            "insert_translations_safe.py",
            str(input_path),
            str(trans_path),
            "--output", str(output_path),
        )

        logger.info(f"Pipeline complete: {output_path}")

        # Return file + verification report
        response = send_file(
            str(output_path),
            as_attachment=True,
            download_name=f"{base_name}_Bilingual.docx",
            mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
        response.headers["X-Verification-Score"] = str(report["score"])
        response.headers["X-Verification-Status"] = report["status"]
        response.headers["X-Verification-Issues"] = str(report["issues_found"])
        response.headers["X-Stats"] = json.dumps({
            "paragraphs": para_count,
            "table_cells": cell_count,
            "translated": para_count + cell_count,
        })
        return response

    except Exception as e:
        logger.exception("Translate pipeline failed")
        return jsonify({"error": str(e)}), 500
    finally:
        shutil.rmtree(str(work_dir), ignore_errors=True)


# ═══════════════════════════════════════════════════════════════════════
#  Pipeline Status
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/status")
def status():
    return jsonify({
        "llm_available": LLM_AVAILABLE,
        "llm_provider": LLM_CONFIG["provider"],
        "llm_model": LLM_CONFIG["model"],
        "scripts": {
            "extract": os.path.exists(SCRIPTS_DIR / "extract_paragraphs.py"),
            "insert": os.path.exists(SCRIPTS_DIR / "insert_translations_safe.py"),
            "translate_llm": os.path.exists(SCRIPTS_DIR / "translate_llm.py"),
            "verify_cmc": os.path.exists(SCRIPTS_DIR / "verify_cmc.py"),
        },
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
