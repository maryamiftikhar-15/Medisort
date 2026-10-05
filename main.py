# # # # # # # """
# # # # # # # MediSort pipeline — end to end.

# # # # # # #     image  ->  VLM (Qwen2.5-VL)  ->  raw text
# # # # # # #             ->  NER (Posos/ClinicalNER)  ->  drug/dosage/frequency/duration
# # # # # # #             ->  expiry_checker  ->  expiry status
# # # # # # #             ->  drug_lookup  ->  plain-language usage info
# # # # # # #             ->  final structured JSON

# # # # # # # All models are pretrained and called via the Hugging Face Inference API —
# # # # # # # nothing is trained or fine-tuned in this pipeline.

# # # # # # # Usage:
# # # # # # #     export HF_TOKEN=hf_xxx
# # # # # # #     python main.py path/to/photo.jpg prescription
# # # # # # # """

# # # # # # # import json
# # # # # # # import sys

# # # # # # # from vlm_extractor import extract_text_from_image
# # # # # # # from ner_extractor import extract_entities
# # # # # # # from expiry_checker import check_expiry
# # # # # # # from drug_lookup import lookup_drug_usage


# # # # # # # def run_pipeline(image_path: str, doc_type: str = "prescription") -> dict:
# # # # # # #     # Stage 1: read the image
# # # # # # #     raw_text = extract_text_from_image(image_path, doc_type)

# # # # # # #     # Stage 2: label the medical entities in that text
# # # # # # #     entities = extract_entities(raw_text)

# # # # # # #     # Stage 3: figure out expiry from any dates found
# # # # # # #     expiry_info = check_expiry(entities.get("dates_found", []))

# # # # # # #     # Stage 4 (optional): look up what each identified drug is used for
# # # # # # #     drug_usage = [
# # # # # # #         lookup_drug_usage(drug["text"]) for drug in entities.get("drugs", [])
# # # # # # #     ]

# # # # # # #     return {
# # # # # # #         "document_type": doc_type,
# # # # # # #         "raw_text": raw_text,
# # # # # # #         "drugs": entities.get("drugs", []),
# # # # # # #         "dosages": entities.get("dosages", []),
# # # # # # #         "frequencies": entities.get("frequencies", []),
# # # # # # #         "durations": entities.get("durations", []),
# # # # # # #         "forms": entities.get("forms", []),
# # # # # # #         "drug_usage_info": drug_usage,
# # # # # # #         "expiry": expiry_info,
# # # # # # #     }


# # # # # # # if __name__ == "__main__":
# # # # # # #     if len(sys.argv) < 2:
# # # # # # #         print("Usage: python main.py <image_path> [prescription|lab_report|medicine_strip]")
# # # # # # #         sys.exit(1)

# # # # # # #     img_path = sys.argv[1]
# # # # # # #     document_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"

# # # # # # #     result = run_pipeline(img_path, document_type)
# # # # # # #     print(json.dumps(result, indent=2))



# # # # # # """
# # # # # # MediSort pipeline — end to end.

# # # # # #     image  ->  VLM (Qwen2.5-VL)  ->  raw text
# # # # # #             ->  NER (Posos/ClinicalNER)  ->  drug/dosage/frequency/duration
# # # # # #             ->  expiry_checker  ->  expiry status
# # # # # #             ->  drug_lookup  ->  plain-language usage info
# # # # # #             ->  final structured JSON

# # # # # # All models are pretrained and called via the Hugging Face Inference API —
# # # # # # nothing is trained or fine-tuned in this pipeline.

# # # # # # Usage:
# # # # # #     export HF_TOKEN=hf_xxx
# # # # # #     python main.py path/to/photo.jpg prescription
# # # # # # """

# # # # # # import json
# # # # # # import sys

# # # # # # from vlm_extractor import extract_text_from_image
# # # # # # from ner_extractor import extract_entities
# # # # # # from expiry_checker import check_expiry
# # # # # # from drug_lookup import lookup_drug_usage


# # # # # # def run_pipeline(image_path: str, doc_type: str = "prescription") -> dict:
# # # # # #     # Stage 1: read the image
# # # # # #     raw_text = extract_text_from_image(image_path, doc_type)

# # # # # #     # Stage 2: label the medical entities in that text
# # # # # #     entities = extract_entities(raw_text)

# # # # # #     # Stage 3: figure out expiry — prefer the EXP-labeled date if the NER
# # # # # #     # stage found one, since that's more reliable than guessing from a
# # # # # #     # plain list of dates.
# # # # # #     expiry_info = check_expiry(
# # # # # #         entities.get("dates_found", []),
# # # # # #         exp_date_hint=entities.get("exp_date_hint"),
# # # # # #     )

# # # # # #     # Stage 4 (optional): look up what each identified drug is used for
# # # # # #     drug_usage = [
# # # # # #         lookup_drug_usage(drug["text"]) for drug in entities.get("drugs", [])
# # # # # #     ]

# # # # # #     return {
# # # # # #         "document_type": doc_type,
# # # # # #         "raw_text": raw_text,
# # # # # #         "drugs": entities.get("drugs", []),
# # # # # #         "dosages": entities.get("dosages", []),
# # # # # #         "frequencies": entities.get("frequencies", []),
# # # # # #         "durations": entities.get("durations", []),
# # # # # #         "forms": entities.get("forms", []),
# # # # # #         "drug_usage_info": drug_usage,
# # # # # #         "expiry": expiry_info,
# # # # # #     }


# # # # # # if __name__ == "__main__":
# # # # # #     if len(sys.argv) < 2:
# # # # # #         print("Usage: python main.py <image_path> [prescription|lab_report|medicine_strip]")
# # # # # #         sys.exit(1)

# # # # # #     img_path = sys.argv[1]
# # # # # #     document_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"

# # # # # #     result = run_pipeline(img_path, document_type)
# # # # # #     print(json.dumps(result, indent=2))



# # # # # """
# # # # # MediSort pipeline — end to end.

# # # # #     image  ->  VLM (Qwen2.5-VL)  ->  raw text
# # # # #             ->  NER (Posos/ClinicalNER)  ->  drug/dosage/frequency/duration
# # # # #             ->  expiry_checker  ->  expiry status
# # # # #             ->  drug_lookup  ->  plain-language usage info
# # # # #             ->  final structured JSON

# # # # # All models are pretrained and called via the Hugging Face Inference API —
# # # # # nothing is trained or fine-tuned in this pipeline.

# # # # # Usage:
# # # # #     export HF_TOKEN=hf_xxx
# # # # #     python main.py path/to/photo.jpg prescription
# # # # # """

# # # # # import json
# # # # # import sys

# # # # # from vlm_extractor import extract_text_from_image
# # # # # from ner_extractor import extract_entities
# # # # # from expiry_checker import check_expiry
# # # # # from drug_lookup import lookup_drug_usage


# # # # # def _dedupe_lines(text: str) -> str:
# # # # #     """
# # # # #     Remove duplicate lines while preserving order.

# # # # #     VLMs sometimes repeat a line once per visual occurrence (e.g. a brand
# # # # #     name printed once per tablet pocket on a strip) instead of once
# # # # #     overall. That repetition adds no information but can overwhelm the
# # # # #     downstream NER model's input length, causing it to mis-tokenize real
# # # # #     content — so it's cleaned up here before NER ever sees the text.
# # # # #     """
# # # # #     seen = set()
# # # # #     deduped = []
# # # # #     for line in text.split("\n"):
# # # # #         if line.strip() and line in seen:
# # # # #             continue
# # # # #         seen.add(line)
# # # # #         deduped.append(line)
# # # # #     return "\n".join(deduped)


# # # # # def run_pipeline(image_path: str, doc_type: str = "prescription") -> dict:
# # # # #     # Stage 1: read the image
# # # # #     raw_text = extract_text_from_image(image_path, doc_type)
# # # # #     raw_text = _dedupe_lines(raw_text)

# # # # #     # Stage 2: label the medical entities in that text
# # # # #     entities = extract_entities(raw_text)

# # # # #     # Stage 3: figure out expiry — prefer the EXP-labeled date if the NER
# # # # #     # stage found one, since that's more reliable than guessing from a
# # # # #     # plain list of dates.
# # # # #     expiry_info = check_expiry(
# # # # #         entities.get("dates_found", []),
# # # # #         exp_date_hint=entities.get("exp_date_hint"),
# # # # #     )

# # # # #     # Stage 4 (optional): look up what each identified drug is used for
# # # # #     drug_usage = [
# # # # #         lookup_drug_usage(drug["text"]) for drug in entities.get("drugs", [])
# # # # #     ]

# # # # #     return {
# # # # #         "document_type": doc_type,
# # # # #         "raw_text": raw_text,
# # # # #         "drugs": entities.get("drugs", []),
# # # # #         "dosages": entities.get("dosages", []),
# # # # #         "frequencies": entities.get("frequencies", []),
# # # # #         "durations": entities.get("durations", []),
# # # # #         "forms": entities.get("forms", []),
# # # # #         "drug_usage_info": drug_usage,
# # # # #         "expiry": expiry_info,
# # # # #     }


# # # # # if __name__ == "__main__":
# # # # #     if len(sys.argv) < 2:
# # # # #         print("Usage: python main.py <image_path> [prescription|lab_report|medicine_strip]")
# # # # #         sys.exit(1)

# # # # #     img_path = sys.argv[1]
# # # # #     document_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"

# # # # #     result = run_pipeline(img_path, document_type)
# # # # #     print(json.dumps(result, indent=2))



# # # # """
# # # # MediSort pipeline — end to end.

# # # #     image  ->  VLM (Qwen2.5-VL)  ->  raw text
# # # #             ->  NER (Posos/ClinicalNER)  ->  drug/dosage/frequency/duration
# # # #             ->  expiry_checker  ->  expiry status
# # # #             ->  drug_lookup  ->  plain-language usage info
# # # #             ->  final structured JSON

# # # # All models are pretrained and called via the Hugging Face Inference API —
# # # # nothing is trained or fine-tuned in this pipeline.

# # # # Usage:
# # # #     export HF_TOKEN=hf_xxx
# # # #     python main.py path/to/photo.jpg prescription
# # # # """

# # # # """
# # # # MediSort pipeline — end to end.

# # # # For prescriptions/lab reports (free-form text):
# # # #     image  ->  VLM  ->  raw text  ->  NER  ->  drug/dosage/frequency/duration
# # # #             ->  expiry_checker  ->  drug_lookup  ->  final structured JSON

# # # # For medicine strips/boxes (highly structured, predictable layout):
# # # #     image  ->  VLM (asked for structured JSON directly)  ->  fields
# # # #             ->  expiry_checker  ->  drug_lookup  ->  final structured JSON

# # # # Medicine strips skip the NER stage entirely: NER was mislabeling the brand
# # # # name as the active ingredient (it's trained on clinical narrative notes,
# # # # not product packaging), while the VLM was already reading the text
# # # # correctly — so we now have the VLM hand back the structure directly for
# # # # this document type instead of routing through a mismatched NER model.

# # # # All models are pretrained and called via the Hugging Face Inference API —
# # # # nothing is trained or fine-tuned in this pipeline.

# # # # Usage:
# # # #     export HF_TOKEN=hf_xxx
# # # #     python main.py path/to/photo.jpg prescription
# # # #     python main.py path/to/photo.jpg medicine_strip
# # # # """

# # # # import json
# # # # import sys

# # # # from vlm_extractor import extract_text_from_image, extract_structured_medicine_strip
# # # # from ner_extractor import extract_entities
# # # # from expiry_checker import check_expiry
# # # # from drug_lookup import lookup_drug_usage


# # # # def _dedupe_lines(text: str) -> str:
# # # #     """
# # # #     Remove duplicate lines while preserving order.

# # # #     VLMs sometimes repeat a line once per visual occurrence (e.g. a brand
# # # #     name printed once per tablet pocket on a strip) instead of once
# # # #     overall. That repetition adds no information but can overwhelm the
# # # #     downstream NER model's input length, causing it to mis-tokenize real
# # # #     content — so it's cleaned up here before NER ever sees the text.
# # # #     """
# # # #     seen = set()
# # # #     deduped = []
# # # #     for line in text.split("\n"):
# # # #         if line.strip() and line in seen:
# # # #             continue
# # # #         seen.add(line)
# # # #         deduped.append(line)
# # # #     return "\n".join(deduped)


# # # # def _run_medicine_strip_pipeline(image_path: str) -> dict:
# # # #     fields = extract_structured_medicine_strip(image_path)

# # # #     expiry_info = check_expiry(
# # # #         [d for d in [fields.get("mfg_date"), fields.get("exp_date")] if d],
# # # #         exp_date_hint=fields.get("exp_date"),
# # # #     )

# # # #     drug_usage = [
# # # #         lookup_drug_usage(name) for name in fields.get("active_ingredients") or []
# # # #     ]

# # # #     return {
# # # #         "document_type": "medicine_strip",
# # # #         "raw_text": fields.get("_raw_response", ""),
# # # #         "brand_name": fields.get("brand_name"),
# # # #         "drugs": [{"text": n, "score": None} for n in fields.get("active_ingredients") or []],
# # # #         "dosages": [{"text": fields["strength"], "score": None}] if fields.get("strength") else [],
# # # #         "frequencies": [],
# # # #         "durations": [],
# # # #         "forms": [],
# # # #         "manufacturer": fields.get("manufacturer"),
# # # #         "batch_no": fields.get("batch_no"),
# # # #         "drug_usage_info": drug_usage,
# # # #         "expiry": expiry_info,
# # # #     }


# # # # def _run_free_text_pipeline(image_path: str, doc_type: str) -> dict:
# # # #     # Stage 1: read the image
# # # #     raw_text = extract_text_from_image(image_path, doc_type)
# # # #     raw_text = _dedupe_lines(raw_text)

# # # #     # Stage 2: label the medical entities in that text
# # # #     entities = extract_entities(raw_text)

# # # #     # Stage 3: figure out expiry — prefer the EXP-labeled date if the NER
# # # #     # stage found one, since that's more reliable than guessing from a
# # # #     # plain list of dates.
# # # #     expiry_info = check_expiry(
# # # #         entities.get("dates_found", []),
# # # #         exp_date_hint=entities.get("exp_date_hint"),
# # # #     )

# # # #     # Stage 4 (optional): look up what each identified drug is used for
# # # #     drug_usage = [
# # # #         lookup_drug_usage(drug["text"]) for drug in entities.get("drugs", [])
# # # #     ]

# # # #     return {
# # # #         "document_type": doc_type,
# # # #         "raw_text": raw_text,
# # # #         "drugs": entities.get("drugs", []),
# # # #         "dosages": entities.get("dosages", []),
# # # #         "frequencies": entities.get("frequencies", []),
# # # #         "durations": entities.get("durations", []),
# # # #         "forms": entities.get("forms", []),
# # # #         "drug_usage_info": drug_usage,
# # # #         "expiry": expiry_info,
# # # #     }


# # # # def run_pipeline(image_path: str, doc_type: str = "prescription") -> dict:
# # # #     if doc_type == "medicine_strip":
# # # #         return _run_medicine_strip_pipeline(image_path)
# # # #     return _run_free_text_pipeline(image_path, doc_type)


# # # # if __name__ == "__main__":
# # # #     if len(sys.argv) < 2:
# # # #         print("Usage: python main.py <image_path> [prescription|lab_report|medicine_strip]")
# # # #         sys.exit(1)

# # # #     img_path = sys.argv[1]
# # # #     document_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"

# # # #     result = run_pipeline(img_path, document_type)
# # # #     print(json.dumps(result, indent=2))


# # # """
# # # MediSort pipeline — end to end.

# # # For prescriptions/lab reports (free-form text):
# # #     image  ->  VLM  ->  raw text  ->  NER  ->  drug/dosage/frequency/duration
# # #             ->  date handling (see below)  ->  drug_lookup  ->  final JSON

# # # For medicine strips/boxes (highly structured, predictable layout):
# # #     image  ->  VLM (asked for structured JSON directly)  ->  fields
# # #             ->  expiry_checker  ->  drug_lookup  ->  final structured JSON

# # # Medicine strips skip the NER stage entirely: NER was mislabeling the brand
# # # name as the active ingredient (it's trained on clinical narrative notes,
# # # not product packaging), while the VLM was already reading the text
# # # correctly — so we now have the VLM hand back the structure directly for
# # # this document type instead of routing through a mismatched NER model.

# # # Date handling differs by document type on purpose: "expiry" is only a
# # # meaningful concept for physical medicine packaging. A lab report's date is
# # # when the test was taken, and a prescription's date is when it was written
# # # — neither of those is an "expiry", so applying EXPIRED/VALID logic to them
# # # would be actively misleading (e.g. a 2011 lab report showing as "EXPIRED"
# # # is nonsensical, not useful). So:
# # #   - medicine_strip -> real EXPIRED/VALID expiry status
# # #   - lab_report     -> just the report date, no expiry judgment
# # #   - prescription   -> dates shown as-is, no expiry judgment (prescriptions
# # #                        don't carry a packaging-style expiry date)

# # # All models are pretrained and called via the Hugging Face Inference API —
# # # nothing is trained or fine-tuned in this pipeline.

# # # Usage:
# # #     export HF_TOKEN=hf_xxx
# # #     python main.py path/to/photo.jpg prescription
# # #     python main.py path/to/photo.jpg lab_report
# # #     python main.py path/to/photo.jpg medicine_strip
# # # """

# # # import json
# # # import sys

# # # from vlm_extractor import extract_text_from_image, extract_structured_medicine_strip
# # # from ner_extractor import extract_entities
# # # from expiry_checker import check_expiry
# # # from drug_lookup import lookup_drug_usage


# # # def _dedupe_lines(text: str) -> str:
# # #     """
# # #     Remove duplicate lines while preserving order.

# # #     VLMs sometimes repeat a line once per visual occurrence (e.g. a brand
# # #     name printed once per tablet pocket on a strip) instead of once
# # #     overall. That repetition adds no information but can overwhelm the
# # #     downstream NER model's input length, causing it to mis-tokenize real
# # #     content — so it's cleaned up here before NER ever sees the text.
# # #     """
# # #     seen = set()
# # #     deduped = []
# # #     for line in text.split("\n"):
# # #         if line.strip() and line in seen:
# # #             continue
# # #         seen.add(line)
# # #         deduped.append(line)
# # #     return "\n".join(deduped)


# # # def _run_medicine_strip_pipeline(image_path: str) -> dict:
# # #     fields = extract_structured_medicine_strip(image_path)

# # #     expiry_info = check_expiry(
# # #         [d for d in [fields.get("mfg_date"), fields.get("exp_date")] if d],
# # #         exp_date_hint=fields.get("exp_date"),
# # #     )

# # #     drug_usage = [
# # #         lookup_drug_usage(name) for name in fields.get("active_ingredients") or []
# # #     ]

# # #     return {
# # #         "document_type": "medicine_strip",
# # #         "raw_text": fields.get("_raw_response", ""),
# # #         "brand_name": fields.get("brand_name"),
# # #         "drugs": [{"text": n, "score": None} for n in fields.get("active_ingredients") or []],
# # #         "dosages": [{"text": fields["strength"], "score": None}] if fields.get("strength") else [],
# # #         "frequencies": [],
# # #         "durations": [],
# # #         "forms": [],
# # #         "manufacturer": fields.get("manufacturer"),
# # #         "batch_no": fields.get("batch_no"),
# # #         "drug_usage_info": drug_usage,
# # #         "expiry": expiry_info,
# # #     }


# # # def _run_free_text_pipeline(image_path: str, doc_type: str) -> dict:
# # #     # Stage 1: read the image
# # #     raw_text = extract_text_from_image(image_path, doc_type)
# # #     raw_text = _dedupe_lines(raw_text)

# # #     # Stage 2: label the medical entities in that text
# # #     entities = extract_entities(raw_text)

# # #     # Stage 3: date handling — differs by document type on purpose (see
# # #     # module docstring). Only medicine packaging genuinely has an
# # #     # "expiry"; a lab report's date is just when the test was taken.
# # #     dates_found = entities.get("dates_found", [])
# # #     if doc_type == "lab_report":
# # #         date_info = {
# # #             "report_date": dates_found[0] if dates_found else None,
# # #             "dates_found": dates_found,
# # #         }
# # #     else:
# # #         # prescription: show any dates found, but don't claim an expiry
# # #         # judgment unless the text explicitly labeled one as EXP/EXPIRY.
# # #         exp_hint = entities.get("exp_date_hint")
# # #         if exp_hint:
# # #             date_info = check_expiry(dates_found, exp_date_hint=exp_hint)
# # #         else:
# # #             date_info = {"dates_found": dates_found, "status": "NOT_APPLICABLE"}

# # #     # Stage 4 (optional): look up what each identified drug is used for
# # #     drug_usage = [
# # #         lookup_drug_usage(drug["text"]) for drug in entities.get("drugs", [])
# # #     ]

# # #     return {
# # #         "document_type": doc_type,
# # #         "raw_text": raw_text,
# # #         "drugs": entities.get("drugs", []),
# # #         "dosages": entities.get("dosages", []),
# # #         "frequencies": entities.get("frequencies", []),
# # #         "durations": entities.get("durations", []),
# # #         "forms": entities.get("forms", []),
# # #         "drug_usage_info": drug_usage,
# # #         "dates": date_info,
# # #     }


# # # def run_pipeline(image_path: str, doc_type: str = "prescription") -> dict:
# # #     if doc_type == "medicine_strip":
# # #         return _run_medicine_strip_pipeline(image_path)
# # #     return _run_free_text_pipeline(image_path, doc_type)


# # # if __name__ == "__main__":
# # #     if len(sys.argv) < 2:
# # #         print("Usage: python main.py <image_path> [prescription|lab_report|medicine_strip]")
# # #         sys.exit(1)

# # #     img_path = sys.argv[1]
# # #     document_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"

# # #     result = run_pipeline(img_path, document_type)
# # #     print(json.dumps(result, indent=2))



# # """
# # MediSort pipeline — end to end.

# # For prescriptions/lab reports (free-form text):
# #     image  ->  VLM  ->  raw text  ->  NER  ->  drug/dosage/frequency/duration
# #             ->  date handling (see below)  ->  drug_lookup  ->  final JSON

# # For medicine strips/boxes (highly structured, predictable layout):
# #     image  ->  VLM (asked for structured JSON directly)  ->  fields
# #             ->  expiry_checker  ->  drug_lookup  ->  final structured JSON

# # Medicine strips skip the NER stage entirely: NER was mislabeling the brand
# # name as the active ingredient (it's trained on clinical narrative notes,
# # not product packaging), while the VLM was already reading the text
# # correctly — so we now have the VLM hand back the structure directly for
# # this document type instead of routing through a mismatched NER model.

# # Date handling differs by document type on purpose: "expiry" is only a
# # meaningful concept for physical medicine packaging. A lab report's date is
# # when the test was taken, and a prescription's date is when it was written
# # — neither of those is an "expiry", so applying EXPIRED/VALID logic to them
# # would be actively misleading (e.g. a 2011 lab report showing as "EXPIRED"
# # is nonsensical, not useful). So:
# #   - medicine_strip -> real EXPIRED/VALID expiry status
# #   - lab_report     -> just the report date, no expiry judgment
# #   - prescription   -> dates shown as-is, no expiry judgment (prescriptions
# #                        don't carry a packaging-style expiry date)

# # All models are pretrained and called via the Hugging Face Inference API —
# # nothing is trained or fine-tuned in this pipeline.

# # Usage:
# #     export HF_TOKEN=hf_xxx
# #     python main.py path/to/photo.jpg prescription
# #     python main.py path/to/photo.jpg lab_report
# #     python main.py path/to/photo.jpg medicine_strip
# # """

# # import json
# # import sys

# # from vlm_extractor import extract_text_from_image, extract_structured_medicine_strip
# # from ner_extractor import extract_entities
# # from expiry_checker import check_expiry
# # from drug_lookup import lookup_drug_usage


# # def _dedupe_lines(text: str) -> str:
# #     """
# #     Remove duplicate lines while preserving order.

# #     VLMs sometimes repeat a line once per visual occurrence (e.g. a brand
# #     name printed once per tablet pocket on a strip) instead of once
# #     overall. That repetition adds no information but can overwhelm the
# #     downstream NER model's input length, causing it to mis-tokenize real
# #     content — so it's cleaned up here before NER ever sees the text.
# #     """
# #     seen = set()
# #     deduped = []
# #     for line in text.split("\n"):
# #         if line.strip() and line in seen:
# #             continue
# #         seen.add(line)
# #         deduped.append(line)
# #     return "\n".join(deduped)


# # def _run_medicine_strip_pipeline(image_path: str) -> dict:
# #     fields = extract_structured_medicine_strip(image_path)

# #     expiry_info = check_expiry(
# #         [d for d in [fields.get("mfg_date"), fields.get("exp_date")] if d],
# #         exp_date_hint=fields.get("exp_date"),
# #     )

# #     drug_usage = [
# #         lookup_drug_usage(name) for name in fields.get("active_ingredients") or []
# #     ]

# #     return {
# #         "document_type": "medicine_strip",
# #         "raw_text": fields.get("_raw_response", ""),
# #         "brand_name": fields.get("brand_name"),
# #         "drugs": [{"text": n, "score": None} for n in fields.get("active_ingredients") or []],
# #         "dosages": [{"text": fields["strength"], "score": None}] if fields.get("strength") else [],
# #         "frequencies": [],
# #         "durations": [],
# #         "forms": [],
# #         "manufacturer": fields.get("manufacturer"),
# #         "batch_no": fields.get("batch_no"),
# #         "drug_usage_info": drug_usage,
# #         "expiry": expiry_info,
# #     }


# # def _run_free_text_pipeline(image_path: str, doc_type: str) -> dict:
# #     # Stage 1: read the image
# #     raw_text = extract_text_from_image(image_path, doc_type)
# #     raw_text = _dedupe_lines(raw_text)

# #     # Stage 2: label the medical entities in that text
# #     entities = extract_entities(raw_text)

# #     # Stage 3: date handling — differs by document type on purpose (see
# #     # module docstring). Only medicine packaging genuinely has an
# #     # "expiry"; a lab report's date is just when the test was taken.
# #     dates_found = entities.get("dates_found", [])
# #     if doc_type == "lab_report":
# #         date_info = {
# #             "report_date": dates_found[0] if dates_found else None,
# #             "dates_found": dates_found,
# #         }
# #     else:
# #         # prescription: show any dates found, but don't claim an expiry
# #         # judgment unless the text explicitly labeled one as EXP/EXPIRY.
# #         exp_hint = entities.get("exp_date_hint")
# #         if exp_hint:
# #             date_info = check_expiry(dates_found, exp_date_hint=exp_hint)
# #         else:
# #             date_info = {"dates_found": dates_found, "status": "NOT_APPLICABLE"}

# #     # Stage 4 (optional): look up what each identified drug is used for
# #     drug_usage = [
# #         lookup_drug_usage(drug["text"]) for drug in entities.get("drugs", [])
# #     ]

# #     return {
# #         "document_type": doc_type,
# #         "raw_text": raw_text,
# #         "drugs": entities.get("drugs", []),
# #         "dosages": entities.get("dosages", []),
# #         "frequencies": entities.get("frequencies", []),
# #         "durations": entities.get("durations", []),
# #         "forms": entities.get("forms", []),
# #         "drug_usage_info": drug_usage,
# #         "dates": date_info,
# #     }


# # def run_pipeline(image_path: str, doc_type: str = "prescription") -> dict:
# #     if doc_type == "medicine_strip":
# #         return _run_medicine_strip_pipeline(image_path)
# #     return _run_free_text_pipeline(image_path, doc_type)


# # if __name__ == "__main__":
# #     if len(sys.argv) < 2:
# #         print("Usage: python main.py <image_path> [prescription|lab_report|medicine_strip]")
# #         sys.exit(1)

# #     img_path = sys.argv[1]
# #     document_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"

# #     result = run_pipeline(img_path, document_type)
# #     print(json.dumps(result, indent=2))

    

# """
# MediSort pipeline — end to end, structured-JSON architecture.

#     image  ->  VLM (asked for structured JSON, schema per doc_type)  ->  fields
#             ->  expiry_checker (medicine_strip only)
#             ->  drug_lookup (medicine_strip + prescription)
#             ->  final structured JSON

# No NER stage: every document type now gets structured fields directly
# from the VLM (see vlm_extractor.py for why — the clinical NER model used
# earlier in this project was trained on a different kind of text than
# prescriptions/packaging/lab reports, causing repeated mislabeling).

# Date handling differs by document type on purpose:
#   - medicine_strip -> real EXPIRED/VALID expiry status (the only type
#                       where "expiry" is a meaningful packaging concept)
#   - lab_report     -> just the report date, no expiry judgment
#   - prescription   -> the date as written, no expiry judgment

# All models are pretrained and called via the Hugging Face Inference API —
# nothing is trained or fine-tuned in this pipeline.

# Usage:
#     export HF_TOKEN=hf_xxx
#     python main.py path/to/photo.jpg prescription
#     python main.py path/to/photo.jpg lab_report
#     python main.py path/to/photo.jpg medicine_strip
# """

# import json
# import sys

# from vlm_extractor import extract_structured
# from expiry_checker import check_expiry
# from drug_lookup import lookup_drug_usage


# def _dedupe_by_key(items: list, key_fn) -> list:
#     """Remove duplicate dicts from a list based on a key function, keeping order."""
#     seen = set()
#     deduped = []
#     for item in items:
#         key = key_fn(item)
#         if key and key not in seen:
#             seen.add(key)
#             deduped.append(item)
#     return deduped


# def _run_medicine_strip(fields: dict) -> dict:
#     expiry_info = check_expiry(
#         [d for d in [fields.get("mfg_date"), fields.get("exp_date")] if d],
#         exp_date_hint=fields.get("exp_date"),
#     )
#     active_ingredients = fields.get("active_ingredients") or []
#     drug_usage = [lookup_drug_usage(name) for name in active_ingredients]

#     return {
#         "document_type": "medicine_strip",
#         "raw_text": fields.get("_raw_response", ""),
#         "brand_name": fields.get("brand_name"),
#         "drugs": [{"text": n, "score": None} for n in active_ingredients],
#         "dosages": [{"text": fields["strength"], "score": None}] if fields.get("strength") else [],
#         "manufacturer": fields.get("manufacturer"),
#         "batch_no": fields.get("batch_no"),
#         "drug_usage_info": drug_usage,
#         "expiry": expiry_info,
#     }


# def _run_prescription(fields: dict) -> dict:
#     medications = fields.get("medications") or []
#     # The VLM occasionally repeats an item it saw more than once visually —
#     # dedupe by drug name, same precaution as the medicine-strip line
#     # repetition issue we fixed earlier.
#     medications = _dedupe_by_key(medications, lambda m: (m.get("name") or "").strip().lower())

#     drug_usage = [
#         lookup_drug_usage(m["name"]) for m in medications if m.get("name")
#     ]

#     return {
#         "document_type": "prescription",
#         "raw_text": fields.get("_raw_response", ""),
#         "patient_name": fields.get("patient_name"),
#         "doctor_name": fields.get("doctor_name"),
#         "medications": medications,
#         "drug_usage_info": drug_usage,
#         "dates": {
#             "dates_found": [fields["date"]] if fields.get("date") else [],
#             "status": "NOT_APPLICABLE",
#         },
#     }


# def _run_lab_report(fields: dict) -> dict:
#     tests = fields.get("tests") or []

#     return {
#         "document_type": "lab_report",
#         "raw_text": fields.get("_raw_response", ""),
#         "patient_name": fields.get("patient_name"),
#         "tests": tests,
#         "dates": {
#             "report_date": fields.get("report_date"),
#             "dates_found": [fields["report_date"]] if fields.get("report_date") else [],
#         },
#     }


# _HANDLERS = {
#     "medicine_strip": _run_medicine_strip,
#     "prescription": _run_prescription,
#     "lab_report": _run_lab_report,
# }


# def run_pipeline(image_path: str, doc_type: str = "prescription") -> dict:
#     fields = extract_structured(image_path, doc_type)
#     handler = _HANDLERS.get(doc_type, _run_prescription)
#     return handler(fields)


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python main.py <image_path> [prescription|lab_report|medicine_strip]")
#         sys.exit(1)

#     img_path = sys.argv[1]
#     document_type = sys.argv[2] if len(sys.argv) > 2 else "prescription"

#     result = run_pipeline(img_path, document_type)
#     print(json.dumps(result, indent=2))





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