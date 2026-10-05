"""
Stage 3: Expiry checking.

Pure date logic — no model involved. Takes the raw date strings found by
the NER stage and figures out which one is most likely the expiry date,
then compares it to today.
"""

import calendar
from datetime import date
from dateutil import parser as date_parser

# Medicine packaging almost always prints month/year only (e.g. "05/2024",
# "08/2024"), not a full day/month/year date. dateutil can't reliably parse
# a bare two-part "MM/YYYY" string on its own, so it's handled explicitly
# here rather than relying only on the general parser below.
import re

_MONTH_YEAR_PATTERN = re.compile(r"^(\d{1,2})[/-](\d{4})$")


def _parse_one_date(raw: str) -> date | None:
    """
    Parse a single raw date string into a date object.

    Handles two shapes:
      - Full dates ("12/03/2027", "2027-03-12") via dateutil.
      - Month/year-only strings ("05/2024") as printed on medicine
        packaging — treated as the LAST day of that month, since that's
        the standard convention for expiry dates on pharma packaging.
    """
    my_match = _MONTH_YEAR_PATTERN.match(raw.strip())
    if my_match:
        month, year = int(my_match.group(1)), int(my_match.group(2))
        if 1 <= month <= 12:
            last_day = calendar.monthrange(year, month)[1]
            return date(year, month, last_day)
        return None

    try:
        return date_parser.parse(raw, dayfirst=True).date()
    except (ValueError, OverflowError):
        return None


def parse_dates(date_strings: list[str]) -> list[date]:
    """Convert raw date strings into date objects, skipping unparsable ones."""
    parsed = []
    for s in date_strings:
        d = _parse_one_date(s)
        if d is not None:
            parsed.append(d)
    return parsed


def check_expiry(date_strings: list[str], exp_date_hint: str | None = None) -> dict:
    """
    Args:
        date_strings: raw date strings found in the document (e.g. from
            medicine strip text — "03/2024", "03/2027", or full dates).
        exp_date_hint: optional raw date string that was found specifically
            next to an "EXP"/"EXPIRY" label in the text (see
            ner_extractor.extract_entities). When available, this is far
            more reliable than guessing from a plain list of dates, so it
            takes priority over the max-of-all-dates heuristic below.

    Returns:
        {
          "dates_found": [...],
          "likely_expiry_date": "2027-03-31" or None,
          "status": "EXPIRED" | "VALID" | "UNKNOWN",
          "days_remaining": int or None,
        }
    """
    likely_expiry = None

    if exp_date_hint:
        likely_expiry = _parse_one_date(exp_date_hint)

    if likely_expiry is None:
        parsed_dates = parse_dates(date_strings)
        if parsed_dates:
            # Fallback heuristic when no labeled EXP date was found:
            # the expiry date is almost always the LATER of the dates
            # printed (mfg date comes before expiry date).
            likely_expiry = max(parsed_dates)

    if likely_expiry is None:
        return {
            "dates_found": date_strings,
            "likely_expiry_date": None,
            "status": "UNKNOWN",
            "days_remaining": None,
        }

    today = date.today()
    days_remaining = (likely_expiry - today).days
    status = "EXPIRED" if days_remaining < 0 else "VALID"

    return {
        "dates_found": date_strings,
        "likely_expiry_date": likely_expiry.isoformat(),
        "status": status,
        "days_remaining": days_remaining,
    }


if __name__ == "__main__":
    import json

    demo = check_expiry(["05/2024", "05/2025"], exp_date_hint="05/2025")
    print(json.dumps(demo, indent=2))