
"""
Stage 1: Vision-Language Model extraction — structured JSON for every
document type.

This module asks the VLM to read a photographed document and return
structured JSON directly, with a schema tailored to the document type
(prescription / lab report / medicine strip), instead of returning plain
text for a separate NER model to re-interpret.

Why no NER stage anymore: every NER failure encountered in this project
(brand name mislabeled as drug, BID/TID not recognized, date fragments
tagged as duration, sub-word splitting) traced back to one root cause —
the clinical NER model was trained on narrative clinical notes, not on
packaging labels or prescription shorthand. The VLM, on the other hand,
consistently read the actual text correctly. So instead of patching the
mismatched NER model call by call, this version removes it: the VLM that
already reads correctly is simply asked to hand back the structure too.

No model is trained or fine-tuned here — this only calls a hosted
pretrained model through the Hugging Face Inference API.
"""

import base64
import json
import os
from huggingface_hub import InferenceClient

_HF_TOKEN = os.environ.get("HF_TOKEN")
# Qwen3-VL-30B-A3B-Instruct via featherless-ai — confirmed working in this
# project. (Novita's deployment of this model was flagged unreliable by
# Hugging Face's own health check; featherless-ai has been solid.)
_VLM_MODEL = "Qwen/Qwen3-VL-30B-A3B-Instruct"
_PROVIDER = "featherless-ai"


def _image_to_data_url(image_path: str) -> str:
    """Read a local image file and encode it as a base64 data URL."""
    ext = image_path.rsplit(".", 1)[-1].lower()
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


# Shared rules appended to every schema below, so the anti-hallucination
# behavior we tuned earlier applies uniformly across document types.
#
# "matches_expected_document_type" / "mismatch_reason" exist because, in
# testing, simply TELLING the model "this is a lab report" made it comply
# with that framing even when the photo clearly wasn't one (it fabricated
# plausible-looking lab data from an unrelated image rather than objecting).
# Explicitly giving it permission to disagree — asking it to judge the
# match rather than assuming it — meaningfully changes that behavior.
_COMMON_RULES = """
Rules:
- Transcribe every value exactly as printed/written, character by
  character. Do not autocorrect a drug or brand name into a different,
  more familiar-sounding real drug name, even if it looks unusual or like
  a typo to you — many real drug names are unfamiliar words.
- If a field is genuinely not visible or not present, use null for that
  field rather than guessing.
- FIRST, before filling in any other field, judge for yourself whether
  the image actually matches the document type described above. Set
  "matches_expected_document_type" to true or false accordingly. If
  false, set "mismatch_reason" to a brief plain-language description of
  what the image actually appears to show instead, and still return null
  or empty values for the other fields rather than inventing data to fit
  the requested shape.
- Respond with ONLY the JSON object — no markdown code fences, no
  explanation before or after.
"""

_JSON_SCHEMAS = {
    "medicine_strip": {
        "instructions": """This is a photo of a medicine strip or box.
Read it and respond with ONLY a valid JSON object in exactly this shape:

{
  "matches_expected_document_type": true or false,
  "mismatch_reason": string or null,
  "brand_name": string or null,
  "active_ingredients": [string, ...],
  "strength": string or null,
  "manufacturer": string or null,
  "batch_no": string or null,
  "mfg_date": string or null,
  "exp_date": string or null
}
"""
        + _COMMON_RULES
        + """
- "active_ingredients" are the actual medicinal compound names (e.g.
  "Aceclofenac", "Paracetamol") — NOT the brand/product name printed in
  large logo-style text at the top of the strip. Brand names and active
  ingredient names are usually printed as two separate lines; do not
  confuse one for the other.
- Dates are usually printed as MM/YYYY. Keep that exact format.
- If a brand name or logo is printed repeatedly across multiple tablet
  pockets on the strip, mention it only ONCE — do not repeat it per
  pocket.""",
        "empty": {
            "matches_expected_document_type": False,
            "mismatch_reason": "Could not parse model response",
            "brand_name": None,
            "active_ingredients": [],
            "strength": None,
            "manufacturer": None,
            "batch_no": None,
            "mfg_date": None,
            "exp_date": None,
        },
    },
    "prescription": {
        "instructions": """This is a photo of a doctor's prescription, possibly handwritten.
Read it and respond with ONLY a valid JSON object in exactly this shape:

{
  "matches_expected_document_type": true or false,
  "mismatch_reason": string or null,
  "patient_name": string or null,
  "doctor_name": string or null,
  "date": string or null,
  "medications": [
    {
      "name": string,
      "dosage": string or null,
      "frequency": string or null,
      "duration": string or null
    }
  ]
}
"""
        + _COMMON_RULES
        + """
- "frequency" is dosing shorthand like BID, TID, QD, OD, PRN, or a
  plain-language frequency if that's what's written — keep it exactly as
  written, don't expand abbreviations into full words.
- "duration" is how long to take the medication (e.g. "7 days", "2
  weeks") — only fill this if explicitly stated on the prescription;
  otherwise null. Do not confuse it with the prescription's date field.
- If handwriting is unclear, make your best reading rather than leaving
  it blank, but never substitute a different, more familiar drug name for
  what's actually written.""",
        "empty": {
            "matches_expected_document_type": False,
            "mismatch_reason": "Could not parse model response",
            "patient_name": None,
            "doctor_name": None,
            "date": None,
            "medications": [],
        },
    },
    "lab_report": {
        "instructions": """This is a photo of a medical lab report.
Read it and respond with ONLY a valid JSON object in exactly this shape:

{
  "matches_expected_document_type": true or false,
  "mismatch_reason": string or null,
  "patient_name": string or null,
  "report_date": string or null,
  "tests": [
    {
      "name": string,
      "result": string or null,
      "normal_range": string or null,
      "units": string or null
    }
  ]
}
"""
        + _COMMON_RULES
        + """
- Include every row of every test table exactly as printed.
- "report_date" is the date the test was taken/reported — not any other
  date that might appear on the document.""",
        "empty": {
            "matches_expected_document_type": False,
            "mismatch_reason": "Could not parse model response",
            "patient_name": None,
            "report_date": None,
            "tests": [],
        },
    },
}


def extract_structured(image_path: str, doc_type: str = "prescription") -> dict:
    """
    Call the pretrained VLM (via HF Inference API) on one image and get
    back structured fields directly, using a schema tailored to doc_type.

    Args:
        image_path: path to a local image file (jpg/png).
        doc_type: one of "prescription", "lab_report", "medicine_strip".

    Returns:
        A dict matching the schema for that doc_type, plus a
        "_raw_response" key holding the VLM's unparsed text (kept for
        debugging if JSON parsing ever fails).
    """
    if not _HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Get a token at https://huggingface.co/settings/tokens"
        )

    schema = _JSON_SCHEMAS.get(doc_type, _JSON_SCHEMAS["prescription"])
    image_url = _image_to_data_url(image_path)
    client = InferenceClient(model=_VLM_MODEL, provider=_PROVIDER, token=_HF_TOKEN)

    completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": schema["instructions"]},
                ],
            }
        ],
        max_tokens=1024,
    )

    raw_response = completion.choices[0].message.content.strip()

    # Models sometimes wrap JSON in markdown code fences even when told
    # not to — strip those defensively before parsing.
    cleaned = raw_response
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        parsed = dict(schema["empty"])

    parsed["_raw_response"] = raw_response
    return parsed


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python vlm_extractor.py <image_path> [doc_type]")
        sys.exit(1)

    path = sys.argv[1]
    d_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"
    print(json.dumps(extract_structured(path, d_type), indent=2))
