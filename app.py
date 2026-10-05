"""
MediSort web UI — a thin Flask wrapper around main.run_pipeline().

Nothing about the pipeline changes here: this just gives you a browser
page to upload a photo and pick a document type, instead of typing
commands in a terminal, and renders the structured result as a readable
report layout instead of raw JSON.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
import re
from flask import Flask, render_template, request

from main import run_pipeline


def _flag_test_result(result_str, range_str):
    """
    Best-effort check of whether a lab test result falls outside its
    normal range, for highlighting in the UI (e.g. red for abnormal).

    This is a simple heuristic, not a clinical interpretation: it extracts
    the first number from the result and compares it to the range. It does
    not reliably handle a negative-to-positive range written without
    spaces (e.g. "-3-3" meaning "-3 to 3") — that specific format can be
    misread. Good enough to highlight the common case, not meant to be
    authoritative.
    """
    if not result_str or not range_str:
        return None
    try:
        value = float(re.search(r"-?\d+\.?\d*", result_str).group())
    except (ValueError, AttributeError):
        return None

    range_lower = range_str.lower()
    if "up to" in range_lower:
        m = re.search(r"-?\d+\.?\d*", range_str)
        if not m:
            return None
        return "high" if value > float(m.group()) else "normal"

    nums = re.findall(r"-?\d+\.?\d*", range_str)
    if len(nums) >= 2:
        low, high = float(nums[0]), float(nums[1])
        if value < low:
            return "low"
        if value > high:
            return "high"
        return "normal"
    return None

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    doc_type = request.form.get("doc_type", "prescription")
    image_file = request.files.get("image")

    if not image_file or image_file.filename == "":
        return render_template("index.html", error="Please choose a photo first.")

    image_path = os.path.join(UPLOAD_DIR, image_file.filename)
    image_file.save(image_path)

    try:
        result = run_pipeline(image_path, doc_type)
        error = None
        # UI-only enrichment: flag abnormal lab results for highlighting.
        # This doesn't touch the pipeline's own output shape — it's added
        # here, at the display layer, not in main.py.
        if doc_type == "lab_report" and result.get("tests"):
            for test in result["tests"]:
                test["flag"] = _flag_test_result(
                    test.get("result"), test.get("normal_range")
                )
    except Exception as exc:  # noqa: BLE001 — surfaced to the user as-is
        result = None
        error = str(exc)

    return render_template(
        "result.html",
        result=result,
        doc_type=doc_type,
        error=error,
        image_filename=image_file.filename,
    )


if __name__ == "__main__":
    app.run(debug=True)
