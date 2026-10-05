
"""
MediSort pipeline — end to end, structured-JSON architecture.

    image  ->  VLM (asked for structured JSON, schema per doc_type)  ->  fields
            ->  expiry_checker (medicine_strip only)
            ->  drug_lookup (medicine_strip + prescription)
            ->  final structured JSON

No NER stage: every document type now gets structured fields directly
from the VLM (see vlm_extractor.py for why — the clinical NER model used
earlier in this project was trained on a different kind of text than
prescriptions/packaging/lab reports, causing repeated mislabeling).

Date handling differs by document type on purpose:
  - medicine_strip -> real EXPIRED/VALID expiry status (the only type
                      where "expiry" is a meaningful packaging concept)
  - lab_report     -> just the report date, no expiry judgment
  - prescription   -> the date as written, no expiry judgment

All models are pretrained and called via the Hugging Face Inference API —
nothing is trained or fine-tuned in this pipeline.

Usage:
    export HF_TOKEN=hf_xxx
    python main.py path/to/photo.jpg prescription
    python main.py path/to/photo.jpg lab_report
    python main.py path/to/photo.jpg medicine_strip
"""

import json
import sys

from vlm_extractor import extract_structured
from expiry_checker import check_expiry
from drug_lookup import lookup_drug_usage


def _dedupe_by_key(items: list, key_fn) -> list:
    """Remove duplicate dicts from a list based on a key function, keeping order."""
    seen = set()
    deduped = []
    for item in items:
        key = key_fn(item)
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


def _run_medicine_strip(fields: dict) -> dict:
    expiry_info = check_expiry(
        [d for d in [fields.get("mfg_date"), fields.get("exp_date")] if d],
        exp_date_hint=fields.get("exp_date"),
    )
    active_ingredients = fields.get("active_ingredients") or []
    drug_usage = [lookup_drug_usage(name) for name in active_ingredients]

    return {
        "document_type": "medicine_strip",
        "raw_text": fields.get("_raw_response", ""),
        "brand_name": fields.get("brand_name"),
        "drugs": [{"text": n, "score": None} for n in active_ingredients],
        "dosages": [{"text": fields["strength"], "score": None}] if fields.get("strength") else [],
        "manufacturer": fields.get("manufacturer"),
        "batch_no": fields.get("batch_no"),
        "drug_usage_info": drug_usage,
        "expiry": expiry_info,
    }


def _run_prescription(fields: dict) -> dict:
    medications = fields.get("medications") or []
    # The VLM occasionally repeats an item it saw more than once visually —
    # dedupe by drug name, same precaution as the medicine-strip line
    # repetition issue we fixed earlier.
    medications = _dedupe_by_key(medications, lambda m: (m.get("name") or "").strip().lower())

    drug_usage = [
        lookup_drug_usage(m["name"]) for m in medications if m.get("name")
    ]

    return {
        "document_type": "prescription",
        "raw_text": fields.get("_raw_response", ""),
        "patient_name": fields.get("patient_name"),
        "doctor_name": fields.get("doctor_name"),
        "medications": medications,
        "drug_usage_info": drug_usage,
        "dates": {
            "dates_found": [fields["date"]] if fields.get("date") else [],
            "status": "NOT_APPLICABLE",
        },
    }


def _run_lab_report(fields: dict) -> dict:
    tests = fields.get("tests") or []

    return {
        "document_type": "lab_report",
        "raw_text": fields.get("_raw_response", ""),
        "patient_name": fields.get("patient_name"),
        "tests": tests,
        "dates": {
            "report_date": fields.get("report_date"),
            "dates_found": [fields["report_date"]] if fields.get("report_date") else [],
        },
    }


_HANDLERS = {
    "medicine_strip": _run_medicine_strip,
    "prescription": _run_prescription,
    "lab_report": _run_lab_report,
}


def run_pipeline(image_path: str, doc_type: str = "prescription") -> dict:
    fields = extract_structured(image_path, doc_type)

    # Safety check: the VLM was asked to judge for itself whether the image
    # actually matches the requested document type, rather than just
    # complying with whatever type we told it to expect. If it says no,
    # stop here instead of building a result full of fields the model may
    # have fabricated to fit a schema that doesn't actually apply.
    if fields.get("matches_expected_document_type") is False:
        return {
            "document_type": doc_type,
            "matches_expected_document_type": False,
            "mismatch_reason": fields.get("mismatch_reason"),
            "raw_text": fields.get("_raw_response", ""),
        }

    handler = _HANDLERS.get(doc_type, _run_prescription)
    return handler(fields)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path> [prescription|lab_report|medicine_strip]")
        sys.exit(1)

    img_path = sys.argv[1]
    document_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"

    result = run_pipeline(img_path, document_type)
    print(json.dumps(result, indent=2))
