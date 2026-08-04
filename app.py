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
from pathlib import Path

from flask import (
    Flask, render_template, request, jsonify, send_file
)
from werkzeug.utils import secure_filename

SCRIPTS_DIR = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB
app.config["UPLOAD_FOLDER"] = str(Path(__file__).parent / "uploads")
app.config["OUTPUT_FOLDER"] = str(Path(__file__).parent / "outputs")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# LLM config from environment
LLM_CONFIG = {
    "api_base": os.environ.get("LLM_API_BASE", "https://api.deepseek.com/v1"),
    "api_key": os.environ.get("LLM_API_KEY", ""),
    "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
}
LLM_AVAILABLE = bool(LLM_CONFIG["api_key"])


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
#  Pages
# ═══════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", llm_available=LLM_AVAILABLE)


# ═══════════════════════════════════════════════════════════════════════
#  Step 1: Extract Chinese content
# ═══════════════════════════════════════════════════════════════════════

@app.route("/api/extract", methods=["POST"])
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
