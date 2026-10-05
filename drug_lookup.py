"""
Optional Stage 4: Drug usage lookup.

The NER/VLM stage only tells us a word IS a drug name — it has no medical
knowledge of what that drug treats. This module looks the drug name up to
get a plain-language description.

Two sources, tried in order:
  1. OpenFDA — US drug label database. Good when it hits, but it's
     US-centric: it often doesn't recognize international/generic names
     used outside the US (e.g. "Paracetamol", common in Pakistan/India/UK,
     is called "Acetaminophen" in US labeling — same drug, different name).
     A small alias table below catches the common cases.
  2. Wikipedia's summary API — a general-purpose fallback that works for
     virtually any real drug name regardless of region, since it isn't
     tied to one country's regulatory naming.

This is informational only, not medical advice — see the disclaimer
appended to every result.
"""

import requests

_OPENFDA_URL = "https://api.fda.gov/drug/label.json"
_WIKIPEDIA_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"
_DISCLAIMER = "Informational only — always follow your doctor's or pharmacist's instructions."

# Common regional name differences for the same active ingredient. OpenFDA
# only recognizes the US-side name, so we try both directions.
_NAME_ALIASES = {
    "paracetamol": "acetaminophen",
    "acetaminophen": "paracetamol",
    "aceclofenac": "aceclofenac",  # no US equivalent name difference known
    "salbutamol": "albuterol",
    "albuterol": "salbutamol",
    "frusemide": "furosemide",
    "furosemide": "frusemide",
    "adrenaline": "epinephrine",
    "epinephrine": "adrenaline",
}


def _query_openfda(drug_name: str) -> str | None:
    """Return a purpose string from OpenFDA, or None if not found."""
    params = {
        "search": f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
        "limit": 1,
    }
    try:
        resp = requests.get(_OPENFDA_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    results = data.get("results", [])
    if not results:
        return None

    record = results[0]
    purpose_field = record.get("purpose") or record.get("indications_and_usage")
    return purpose_field[0] if purpose_field else None


def _query_wikipedia(drug_name: str) -> str | None:
    """Return a general-description string from Wikipedia, or None if not found."""
    try:
        resp = requests.get(
            _WIKIPEDIA_SUMMARY_URL.format(drug_name.replace(" ", "_")), timeout=10
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None

    extract = data.get("extract")
    # Wikipedia disambiguation/missing pages return an extract too, but it's
    # not useful here — skip anything that doesn't look like real content.
    if not extract or data.get("type") == "disambiguation":
        return None
    return extract


def lookup_drug_usage(drug_name: str) -> dict:
    """
    Look up a plain-language description of what a drug is used for.

    Returns:
        {
          "drug_name": "Paracetamol",
          "purpose": "..." or None if not found anywhere,
          "source": "openfda" | "openfda (as Acetaminophen)" | "wikipedia" | None,
          "disclaimer": "..."
        }
    """
    # 1. Try the name exactly as given.
    purpose = _query_openfda(drug_name)
    source = "openfda" if purpose else None

    # 2. Try a known regional alias (e.g. Paracetamol -> Acetaminophen).
    alias = _NAME_ALIASES.get(drug_name.lower())
    if not purpose and alias:
        purpose = _query_openfda(alias)
        source = f"openfda (as {alias.title()})" if purpose else None

    # 3. Fall back to Wikipedia, which isn't tied to US naming conventions.
    if not purpose:
        purpose = _query_wikipedia(drug_name)
        source = "wikipedia" if purpose else None

    return {
        "drug_name": drug_name,
        "purpose": purpose,
        "source": source,
        "disclaimer": _DISCLAIMER,
    }


if __name__ == "__main__":
    import json
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Amoxicillin"
    print(json.dumps(lookup_drug_usage(name), indent=2))


if __name__ == "__main__":
    import json
    import sys

    name = sys.argv[1] if len(sys.argv) > 1 else "Amoxicillin"
    print(json.dumps(lookup_drug_usage(name), indent=2))