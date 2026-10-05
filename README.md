# MediSort — AI Prescription & Medical Document Reader

Photograph a prescription, lab report, or medicine strip/box — MediSort reads it and returns structured data: drug names, dosages, frequencies, dates, and an expiry flag for medicine packaging.

Built entirely on **pretrained models called via API** — no model training, no local model weights, no GPU required.

## How it works

```
Photo  →  Vision-Language Model (Qwen3-VL-30B-A3B-Instruct, via Hugging Face Inference API)
       →  structured JSON, with a schema tailored to the document type
       →  expiry logic (medicine strips only) + drug usage lookup (OpenFDA + Wikipedia)
       →  rendered as a report in the web UI
```

A single vision-language model call reads the image and returns structured fields directly — there's no separate OCR step and no separate entity-extraction model. An earlier version of this project used a general clinical NER model as a second stage, but that model was trained on narrative clinical notes and repeatedly mislabeled packaging brand names, prescription shorthand (BID/TID/QD), and date fragments. Asking the VLM — which was already reading the text correctly — to hand back the structure directly removed that whole class of error.

**Built-in safety check:** every response includes a `matches_expected_document_type` field. The model is explicitly asked to judge whether the photo actually matches the document type selected, rather than blindly complying with whatever type it's told — this stops it from fabricating a lab report's worth of data from, say, a photo of a blister pack.

## Features

- **Medicine strips/boxes** — brand name, active ingredients, strength, manufacturer, batch number, and an EXPIRED/VALID flag computed from the printed dates
- **Prescriptions** (including handwritten) — patient/doctor info, medications with dosage/frequency/duration, grouped per drug
- **Lab reports** — patient info, report date, every test row (name/result/normal range/units), with abnormal results highlighted in the UI
- **Drug usage lookup** — OpenFDA (US) with a regional name-alias table (e.g. Paracetamol ↔ Acetaminophen) and a Wikipedia fallback for international drug names
- **Document-type mismatch detection** — flags when the photo doesn't match the selected type instead of fabricating data
- Simple web UI (Flask) to upload a photo and view the result as a readable report, instead of raw JSON

## Tech stack

- **Vision-Language Model:** `Qwen/Qwen3-VL-30B-A3B-Instruct`, via Hugging Face Inference API (`featherless-ai` provider)
- **Backend:** Python, Flask
- **Drug info:** OpenFDA public API + Wikipedia REST API
- All models accessed via API — nothing trained or fine-tuned

## Setup

1. Clone this repo and install dependencies:
   ```
   pip install -r requirements.txt
   pip install flask
   ```
2. Get a free Hugging Face token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (create it as a **Fine-grained** token with "Make calls to Inference Providers" enabled)
3. Set it as an environment variable:
   ```
   export HF_TOKEN=hf_xxxxxxxxxxxx        # macOS/Linux
   $env:HF_TOKEN="hf_xxxxxxxxxxxx"        # Windows PowerShell
   ```

## Run

**Web UI:**
```
python app.py
```
Then open `http://127.0.0.1:5000` in your browser.

**Command line (prints JSON, also saves to `last_result.json`):**
```
python main.py path/to/photo.jpg prescription
python main.py path/to/photo.jpg lab_report
python main.py path/to/photo.jpg medicine_strip
```

## Project structure

| File | Role |
|---|---|
| `vlm_extractor.py` | Calls the VLM, holds the per-document-type JSON schemas and prompts |
| `main.py` | Orchestrates the pipeline per document type, builds the final result |
| `expiry_checker.py` | Pure date logic — EXPIRED/VALID status for medicine packaging |
| `drug_lookup.py` | OpenFDA + Wikipedia drug-usage lookup, with regional name aliases |
| `app.py` | Flask web app |
| `templates/`, `static/` | Web UI |
| `ner_extractor.py` | Earlier NER-based approach, kept for reference — no longer used in the pipeline (see "How it works" above) |

## Known limitations

- **Handwriting accuracy isn't perfect.** The VLM can occasionally misread a drug name as a different, similarly-spelled real drug — this is a model accuracy ceiling, not something prompting alone fully solves. For real clinical use, a human-confirmation step before trusting extracted drug names is strongly recommended.
- **Drug usage lookup isn't exhaustive.** OpenFDA is US-centric; Wikipedia is the fallback but won't have every obscure regional brand.
- **This is a prototype, not a certified medical device.** All output is informational only — see the disclaimer in the UI.

## Real-world use cases

Pharmacies (quick expiry checks on stock), elder care (verifying medication schedules from handwritten prescriptions), clinics (digitizing lab report data).
