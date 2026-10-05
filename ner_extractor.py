# # """
# # Stage 2: Clinical NER extraction.

# # Takes the raw text produced by the VLM (Stage 1) and labels the specific
# # medical entities inside it — DRUG, STRENGTH/DOSAGE, FREQUENCY, DURATION,
# # FORM, and any dates — using a pretrained clinical NER model via the HF
# # Inference API. No training happens here either.
# # """

# # import os
# # import re
# # from huggingface_hub import InferenceClient

# # # blaze999/Medical-NER (a hosted copy of Clinical-AI-Apollo/Medical-NER) —
# # # unlike Posos/ClinicalNER, this one is confirmed in Hugging Face's own docs
# # # as deployed on the "hf-inference" provider, which is the ONLY provider
# # # that serves classic token-classification (NER) models. Third-party
# # # providers (Novita, Together, etc.) only serve chat/VLM-style models —
# # # that's why Stage 1 uses provider="novita" but this stage must use
# # # provider="hf-inference" instead.
# # _HF_TOKEN = os.environ.get("HF_TOKEN")
# # _NER_MODEL = "blaze999/Medical-NER"

# # # Simple date pattern to catch expiry/mfg dates the NER model's entity
# # # schema doesn't cover as a labeled field the way we need it — this is why
# # # date extraction is handled separately below rather than via the NER model.
# # _DATE_PATTERN = re.compile(
# #     r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b"
# # )


# # def extract_entities(text: str) -> dict:
# #     """
# #     Run the pretrained clinical NER model on raw text (via HF Inference API).

# #     Returns:
# #         {
# #           "drugs": [{"text": "...", "score": 0.98}, ...],
# #           "dosages": [...],
# #           "frequencies": [...],
# #           "durations": [...],
# #           "forms": [...],
# #           "dates_found": ["12/03/2027", ...],
# #         }
# #     """
# #     if not _HF_TOKEN:
# #         raise RuntimeError(
# #             "HF_TOKEN environment variable is not set. "
# #             "Get a token at https://huggingface.co/settings/tokens"
# #         )

# #     client = InferenceClient(provider="hf-inference", token=_HF_TOKEN)
# #     raw_entities = client.token_classification(text, model=_NER_MODEL)

# #     buckets = {
# #         "drugs": [],
# #         "dosages": [],
# #         "frequencies": [],
# #         "durations": [],
# #         "forms": [],
# #     }

# #     # blaze999/Medical-NER uses the 41-class MACCROBAT-style label set —
# #     # "MEDICATION" is its drug-name label (not "DRUG"), and it has no
# #     # separate FORM label, so that bucket stays empty from this model.
# #     label_map = {
# #         "MEDICATION": "drugs",
# #         "DOSAGE": "dosages",
# #         "FREQUENCY": "frequencies",
# #         "DURATION": "durations",
# #     }

# #     for ent in raw_entities:
# #         # huggingface_hub returns entity_group / word / score depending on
# #         # provider response shape; handle both key styles defensively.
# #         label = (ent.get("entity_group") or ent.get("entity") or "").upper()
# #         word = ent.get("word") or ent.get("text", "")
# #         score = float(ent.get("score", 0.0))

# #         bucket = label_map.get(label)
# #         if bucket:
# #             buckets[bucket].append({"text": word, "score": round(score, 3)})

# #     buckets["dates_found"] = _DATE_PATTERN.findall(text)
# #     return buckets


# # if __name__ == "__main__":
# #     import sys
# #     import json

# #     if len(sys.argv) < 2:
# #         print("Usage: python ner_extractor.py \"<raw text from VLM>\"")
# #         sys.exit(1)

# #     result = extract_entities(sys.argv[1])
# #     print(json.dumps(result, indent=2))


# """
# Stage 2: Clinical NER extraction.

# Takes the raw text produced by the VLM (Stage 1) and labels the specific
# medical entities inside it — DRUG, STRENGTH/DOSAGE, FREQUENCY, DURATION,
# FORM, and any dates — using a pretrained clinical NER model via the HF
# Inference API. No training happens here either.
# """

# import os
# import re
# from huggingface_hub import InferenceClient

# # blaze999/Medical-NER (a hosted copy of Clinical-AI-Apollo/Medical-NER) —
# # unlike Posos/ClinicalNER, this one is confirmed in Hugging Face's own docs
# # as deployed on the "hf-inference" provider, which is the ONLY provider
# # that serves classic token-classification (NER) models. Third-party
# # providers (Novita, Together, etc.) only serve chat/VLM-style models —
# # that's why Stage 1 uses provider="novita" but this stage must use
# # provider="hf-inference" instead.
# _HF_TOKEN = os.environ.get("HF_TOKEN")
# _NER_MODEL = "blaze999/Medical-NER"

# # Date patterns seen on real medicine packaging: full dates AND the far
# # more common month/year-only format ("05/2024", "08-2025"). The earlier
# # version of this pattern only matched full dates, which is why it missed
# # every date on an actual medicine strip photo.
# _DATE_PATTERN = re.compile(
#     r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{4})\b"
# )

# # Looks for a date within a short distance of an EXP/EXPIRY or MFG label —
# # this is far more reliable than guessing "the later date must be expiry",
# # since it directly uses the label printed on the packaging.
# _EXP_LABELED_PATTERN = re.compile(
#     r"EXP(?:IRY)?.{0,15}?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{4})",
#     re.IGNORECASE,
# )
# _MFG_LABELED_PATTERN = re.compile(
#     r"(?:MFG|MFD|MANUFACTURED?).{0,15}?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{4})",
#     re.IGNORECASE,
# )


# def _merge_fragments(raw_entities: list, text: str) -> list:
#     """
#     Merge sub-word fragments (e.g. "Ace" + "clof" + "enac", tagged
#     B-MEDICATION/I-MEDICATION) back into whole-word entities, using each
#     fragment's character start/end offset in the original text rather than
#     relying on the server to have aggregated them already.
#     """
#     # Sort by position in the text so adjacent fragments end up next to
#     # each other regardless of the order the API returned them in.
#     sortable = [e for e in raw_entities if e.get("start") is not None]
#     sortable.sort(key=lambda e: e["start"])
 
#     merged = []
#     for ent in sortable:
#         label = (ent.get("entity_group") or ent.get("entity") or "").upper()
#         label = label.removeprefix("B-").removeprefix("I-")
#         start, end = ent["start"], ent["end"]
#         score = float(ent.get("score", 0.0))
 
#         # Merge into the previous entity if same label and directly
#         # adjacent (allowing a 1-character gap for a space that isn't
#         # part of either token, e.g. "Ace" + "clofenac" with no gap, or
#         # "100" + "mg" with none either).
#         if merged and merged[-1]["label"] == label and start - merged[-1]["end"] <= 1:
#             merged[-1]["end"] = end
#             merged[-1]["scores"].append(score)
#         else:
#             merged.append({"label": label, "start": start, "end": end, "scores": [score]})
 
#     return [
#         {
#             "entity_group": m["label"],
#             "word": text[m["start"] : m["end"]],
#             "score": sum(m["scores"]) / len(m["scores"]),
#         }
#         for m in merged
#     ]



# def extract_entities(text: str) -> dict:
#     """
#     Run the pretrained clinical NER model on raw text (via HF Inference API).

#     Returns:
#         {
#           "drugs": [{"text": "...", "score": 0.98}, ...],
#           "dosages": [...],
#           "frequencies": [...],
#           "durations": [...],
#           "forms": [...],
#           "dates_found": ["12/03/2027", ...],
#         }
#     """
#     if not _HF_TOKEN:
#         raise RuntimeError(
#             "HF_TOKEN environment variable is not set. "
#             "Get a token at https://huggingface.co/settings/tokens"
#         )

#     client = InferenceClient(provider="hf-inference", token=_HF_TOKEN)
#     raw_entities = client.token_classification(text, model=_NER_MODEL,aggregation_strategy="simple")
#     raw_entities = _merge_fragments(raw_entities, text)



#     buckets = {
#         "drugs": [],
#         "dosages": [],
#         "frequencies": [],
#         "durations": [],
#         "forms": [],
#     }

#     # blaze999/Medical-NER uses the 41-class MACCROBAT-style label set —
#     # "MEDICATION" is its drug-name label (not "DRUG"), and it has no
#     # separate FORM label, so that bucket stays empty from this model.
#     label_map = {
#         "MEDICATION": "drugs",
#         "DOSAGE": "dosages",
#         "FREQUENCY": "frequencies",
#         "DURATION": "durations",
#     }

#     for ent in raw_entities:
#         # huggingface_hub returns entity_group / word / score depending on
#         # provider response shape; handle both key styles defensively.
#         label = (ent.get("entity_group") or ent.get("entity") or "").upper()
#         word = ent.get("word") or ent.get("text", "")
#         score = float(ent.get("score", 0.0))

#         bucket = label_map.get(label)
#         if bucket:
#             buckets[bucket].append({"text": word, "score": round(score, 3)})

#     buckets["dates_found"] = _DATE_PATTERN.findall(text)

#     exp_match = _EXP_LABELED_PATTERN.search(text)
#     mfg_match = _MFG_LABELED_PATTERN.search(text)
#     buckets["exp_date_hint"] = exp_match.group(1) if exp_match else None
#     buckets["mfg_date_hint"] = mfg_match.group(1) if mfg_match else None

#     return buckets


# if __name__ == "__main__":
#     import sys
#     import json

#     if len(sys.argv) < 2:
#         print("Usage: python ner_extractor.py \"<raw text from VLM>\"")
#         sys.exit(1)

#     result = extract_entities(sys.argv[1])
#     print(json.dumps(result, indent=2))


"""
Stage 2: Clinical NER extraction.

Takes the raw text produced by the VLM (Stage 1) and labels the specific
medical entities inside it — DRUG, STRENGTH/DOSAGE, FREQUENCY, DURATION,
FORM, and any dates — using a pretrained clinical NER model via the HF
Inference API. No training happens here either.
"""

import os
import re
from huggingface_hub import InferenceClient

# blaze999/Medical-NER (a hosted copy of Clinical-AI-Apollo/Medical-NER) —
# unlike Posos/ClinicalNER, this one is confirmed in Hugging Face's own docs
# as deployed on the "hf-inference" provider, which is the ONLY provider
# that serves classic token-classification (NER) models. Third-party
# providers (Novita, Together, etc.) only serve chat/VLM-style models —
# that's why Stage 1 uses provider="novita" but this stage must use
# provider="hf-inference" instead.
_HF_TOKEN = os.environ.get("HF_TOKEN")
_NER_MODEL = "blaze999/Medical-NER"

# Date patterns seen on real medicine packaging: full dates AND the far
# more common month/year-only format ("05/2024", "08-2025"). The earlier
# version of this pattern only matched full dates, which is why it missed
# every date on an actual medicine strip photo.
_DATE_PATTERN = re.compile(
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|\d{1,2}[/-]\d{4})\b"
)

# Looks for a date within a short distance of an EXP/EXPIRY or MFG label —
# this is far more reliable than guessing "the later date must be expiry",
# since it directly uses the label printed on the packaging.
_EXP_LABELED_PATTERN = re.compile(
    r"EXP(?:IRY)?.{0,15}?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{4})",
    re.IGNORECASE,
)
_MFG_LABELED_PATTERN = re.compile(
    r"(?:MFG|MFD|MANUFACTURED?).{0,15}?(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}[/-]\d{4})",
    re.IGNORECASE,
)

# Standard prescription frequency shorthand (Latin abbreviations, e.g.
# "BID" = twice daily, "TID" = three times daily, "QD" = once daily). The
# clinical NER model was trained on prose clinical notes, not this kind of
# shorthand, so it doesn't reliably recognize these — pattern-matching them
# directly is more reliable than trusting the NER model's FREQUENCY label
# for prescriptions specifically.
_FREQUENCY_PATTERN = re.compile(
    r"\b(OD|BID|BD|TID|TDS|QID|QDS|QD|PRN|STAT|HS|AC|PC)\b", re.IGNORECASE
)


def _merge_fragments(raw_entities: list, text: str) -> list:
    """
    Merge sub-word fragments (e.g. "Ace" + "clof" + "enac", tagged
    B-MEDICATION/I-MEDICATION) back into whole-word entities, using each
    fragment's character start/end offset in the original text rather than
    relying on the server to have aggregated them already.
    """
    # Sort by position in the text so adjacent fragments end up next to
    # each other regardless of the order the API returned them in.
    sortable = [e for e in raw_entities if e.get("start") is not None]
    sortable.sort(key=lambda e: e["start"])

    merged = []
    for ent in sortable:
        label = (ent.get("entity_group") or ent.get("entity") or "").upper()
        label = label.removeprefix("B-").removeprefix("I-")
        start, end = ent["start"], ent["end"]
        score = float(ent.get("score", 0.0))

        # Merge into the previous entity if same label and directly
        # adjacent (allowing a 1-character gap for a space that isn't
        # part of either token, e.g. "Ace" + "clofenac" with no gap, or
        # "100" + "mg" with none either).
        if merged and merged[-1]["label"] == label and start - merged[-1]["end"] <= 1:
            merged[-1]["end"] = end
            merged[-1]["scores"].append(score)
        else:
            merged.append({"label": label, "start": start, "end": end, "scores": [score]})

    return [
        {
            "entity_group": m["label"],
            "word": text[m["start"] : m["end"]],
            "score": sum(m["scores"]) / len(m["scores"]),
        }
        for m in merged
    ]


def extract_entities(text: str) -> dict:
    """
    Run the pretrained clinical NER model on raw text (via HF Inference API).

    Returns:
        {
          "drugs": [{"text": "...", "score": 0.98}, ...],
          "dosages": [...],
          "frequencies": [...],
          "durations": [...],
          "forms": [...],
          "dates_found": ["12/03/2027", ...],
        }
    """
    if not _HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Get a token at https://huggingface.co/settings/tokens"
        )

    client = InferenceClient(provider="hf-inference", token=_HF_TOKEN)
    # aggregation_strategy is requested, but this provider route doesn't
    # reliably honor it in practice (confirmed by testing) — so we merge
    # sub-word fragments back into whole words ourselves below, using each
    # fragment's character offsets in the original text rather than trusting
    # the server to do it.
    raw_entities = client.token_classification(
        text, model=_NER_MODEL, aggregation_strategy="simple"
    )
    raw_entities = _merge_fragments(raw_entities, text)

    buckets = {
        "drugs": [],
        "dosages": [],
        "frequencies": [],
        "durations": [],
        "forms": [],
    }

    # blaze999/Medical-NER uses the 41-class MACCROBAT-style label set —
    # "MEDICATION" is its drug-name label (not "DRUG"), and it has no
    # separate FORM label, so that bucket stays empty from this model.
    label_map = {
        "MEDICATION": "drugs",
        "DOSAGE": "dosages",
        "FREQUENCY": "frequencies",
        "DURATION": "durations",
    }

    for ent in raw_entities:
        # huggingface_hub returns entity_group / word / score depending on
        # provider response shape; handle both key styles defensively.
        label = (ent.get("entity_group") or ent.get("entity") or "").upper()
        word = ent.get("word") or ent.get("text", "")
        score = float(ent.get("score", 0.0))

        bucket = label_map.get(label)
        if bucket and word.strip():
            buckets[bucket].append({"text": word.strip(), "score": round(score, 3)})

    buckets["dates_found"] = _DATE_PATTERN.findall(text)

    # Prescription frequency shorthand (BID/TID/QD/etc.) via direct pattern
    # match rather than the NER model's FREQUENCY label, which doesn't
    # reliably recognize this shorthand (see comment on _FREQUENCY_PATTERN).
    freq_matches = _FREQUENCY_PATTERN.findall(text)
    if freq_matches:
        buckets["frequencies"] = [
            {"text": f.upper(), "score": None} for f in dict.fromkeys(freq_matches)
        ]

    # The NER model has repeatedly mislabeled fragments of DATES as
    # "DURATION" (e.g. part of "09-11-12" tagged as duration) — a real
    # duration phrase ("7 days", "2 weeks") looks nothing like a date
    # fragment, so drop anything that's just digits/dashes with no unit.
    buckets["durations"] = [
        d for d in buckets["durations"]
        if not re.fullmatch(r"[\d\s/-]+", d["text"])
    ]

    exp_match = _EXP_LABELED_PATTERN.search(text)
    mfg_match = _MFG_LABELED_PATTERN.search(text)
    buckets["exp_date_hint"] = exp_match.group(1) if exp_match else None
    buckets["mfg_date_hint"] = mfg_match.group(1) if mfg_match else None

    return buckets


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) < 2:
        print("Usage: python ner_extractor.py \"<raw text from VLM>\"")
        sys.exit(1)

    result = extract_entities(sys.argv[1])
    print(json.dumps(result, indent=2))