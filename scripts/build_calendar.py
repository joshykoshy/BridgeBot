"""Build data/calendar.db: public holidays 2026-2027 for 22 countries + major religious periods.

Why a database and not the LLM: "is 12 March 2026 during Ramadan?" is a factual lookup with
a right answer. A language model may guess; a table lookup cannot. The Brief-me agent's
`check_calendar` node queries this file through a typed Python function, no LLM involved.

Tables
  public_holidays(country_code, country, date, name, source)
      from the `holidays` Python package (official public holidays, incl. estimated
      Islamic dates where the package marks them so).
  observances(name, religion, year, start_date, end_date, note, source_url)
      religious/cultural periods. Dates were read from each Wikipedia article's infobox
      ("2026 date" / "2027 date") or date table on 2026-11-01 and transcribed here, with the
      article URL stored per row. Where Wikipedia had no row for a year, the date comes from
      the `holidays` package and the source says so.

Trade-off: lunar dates (Ramadan, Eid) depend on moon sighting and can shift by a day by
country; rows carry that note, and the agent phrases flags as "around", not as certainty.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import holidays

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "calendar.db"
YEARS = [2026, 2027]
COUNTRIES = {
    "JP": "Japan", "CN": "China", "KR": "South Korea", "IN": "India", "AE": "United Arab Emirates",
    "SA": "Saudi Arabia", "EG": "Egypt", "TR": "Turkey", "IL": "Israel", "ID": "Indonesia",
    "MY": "Malaysia", "TH": "Thailand", "SG": "Singapore", "GB": "United Kingdom", "DE": "Germany",
    "FR": "France", "US": "United States", "BR": "Brazil", "MX": "Mexico", "NG": "Nigeria",
    "ZA": "South Africa", "AU": "Australia",
}
WIKI = "https://en.wikipedia.org/wiki/"
PYHOL = "python-holidays package"
LUNAR = "Islamic dates depend on moon sighting and may differ by a day between countries."

# (name, religion, year, start, end, note, source)
OBSERVANCES = [
    ("Ramadan", "Islam", 2026, "2026-02-18", "2026-03-19", "Month of daytime fasting. " + LUNAR, WIKI + "Ramadan"),
    ("Ramadan", "Islam", 2027, "2027-02-08", "2027-03-08", "Month of daytime fasting. " + LUNAR, WIKI + "Ramadan"),
    ("Eid al-Fitr", "Islam", 2026, "2026-03-20", "2026-03-22", LUNAR, WIKI + "Eid_al-Fitr"),
    ("Eid al-Fitr", "Islam", 2027, "2027-03-09", "2027-03-11", LUNAR, WIKI + "Eid_al-Fitr"),
    ("Eid al-Adha", "Islam", 2026, "2026-05-27", "2026-05-30",
     "28-31 May in East Asia and Oceania. " + LUNAR, WIKI + "Eid_al-Adha"),
    ("Eid al-Adha", "Islam", 2027, "2027-05-16", "2027-05-18", "Estimated (UAE public holiday). " + LUNAR, PYHOL + " (AE)"),
    ("Diwali", "Hinduism", 2026, "2026-11-06", "2026-11-10", "Five-day festival; Lakshmi Puja on 8 November.", WIKI + "Diwali"),
    ("Diwali", "Hinduism", 2027, "2027-10-29", "2027-10-29", "Main day.", WIKI + "Diwali"),
    ("Holi", "Hinduism", 2026, "2026-03-04", "2026-03-04", "", PYHOL + " (IN)"),
    ("Holi", "Hinduism", 2027, "2027-03-22", "2027-03-22", "", PYHOL + " (IN)"),
    ("Navaratri (Vasanta)", "Hinduism", 2026, "2026-03-19", "2026-03-27", "", WIKI + "Navaratri"),
    ("Navaratri (Sharada)", "Hinduism", 2026, "2026-10-11", "2026-10-19", "", WIKI + "Navaratri"),
    ("Chinese New Year", "", 2026, "2026-02-17", "2026-02-17", "Year of the Horse; festivities last about 15 days.", WIKI + "Chinese_New_Year"),
    ("Chinese New Year", "", 2027, "2027-02-06", "2027-02-06", "Year of the Goat; festivities last about 15 days.", WIKI + "Chinese_New_Year"),
    ("Seollal (Korean New Year)", "", 2026, "2026-02-17", "2026-02-17", "", WIKI + "Korean_New_Year"),
    ("Seollal (Korean New Year)", "", 2027, "2027-02-06", "2027-02-08", "Three-day public holiday.", PYHOL + " (KR)"),
    ("Chuseok", "", 2026, "2026-09-24", "2026-09-26", "", WIKI + "Chuseok"),
    ("Chuseok", "", 2027, "2027-09-14", "2027-09-16", "", WIKI + "Chuseok"),
    ("Mid-Autumn Festival", "", 2026, "2026-09-25", "2026-09-25", "", WIKI + "Mid-Autumn_Festival"),
    ("Mid-Autumn Festival", "", 2027, "2027-09-15", "2027-09-15", "", WIKI + "Mid-Autumn_Festival"),
    ("Passover", "Judaism", 2026, "2026-04-01", "2026-04-09", "Sunset 1 April to nightfall 9 April.", WIKI + "Passover"),
    ("Passover", "Judaism", 2027, "2027-04-21", "2027-04-29", "Sunset 21 April to nightfall 29 April.", WIKI + "Passover"),
    ("Yom Kippur", "Judaism", 2026, "2026-09-20", "2026-09-21", "Sunset to nightfall; a fast day.", WIKI + "Yom_Kippur"),
    ("Yom Kippur", "Judaism", 2027, "2027-10-10", "2027-10-11", "Sunset to nightfall; a fast day.", WIKI + "Yom_Kippur"),
    ("Hanukkah", "Judaism", 2026, "2026-12-04", "2026-12-12", "Sunset 4 December to nightfall 12 December.", WIKI + "Hanukkah"),
    ("Hanukkah", "Judaism", 2027, "2027-12-24", "2028-01-01", "Sunset 24 December to nightfall 1 January.", WIKI + "Hanukkah"),
    ("Lent (Western)", "Christianity", 2026, "2026-02-18", "2026-04-02", "Eastern Lent: 23 February to 3 April.", WIKI + "Lent"),
    ("Lent (Western)", "Christianity", 2027, "2027-02-10", "2027-03-25", "Eastern Lent: 15 March to 23 April.", WIKI + "Lent"),
    ("Easter (Western)", "Christianity", 2026, "2026-04-05", "2026-04-05", "Eastern Easter: 12 April.", WIKI + "Easter"),
    ("Easter (Western)", "Christianity", 2027, "2027-03-28", "2027-03-28", "Eastern Easter: 2 May.", WIKI + "Easter"),
    ("Christmas", "Christianity", 2026, "2026-12-25", "2026-12-25", "Many Eastern churches: 7 January.", WIKI + "Christmas"),
    ("Christmas", "Christianity", 2027, "2027-12-25", "2027-12-25", "Many Eastern churches: 7 January.", WIKI + "Christmas"),
    ("Vesak", "Buddhism", 2026, "2026-05-31", "2026-05-31", "Date varies by country (Singapore date).", PYHOL + " (SG)"),
    ("Vesak", "Buddhism", 2027, "2027-05-20", "2027-05-20", "Date varies by country (Singapore date).", PYHOL + " (SG)"),
    ("Songkran", "Buddhism", 2026, "2026-04-13", "2026-04-15", "Thai New Year.", PYHOL + " (TH)"),
    ("Songkran", "Buddhism", 2027, "2027-04-13", "2027-04-15", "Thai New Year.", PYHOL + " (TH)"),
    ("Rio Carnival", "", 2026, "2026-02-13", "2026-02-18", "", WIKI + "Rio_Carnival"),
    ("Rio Carnival", "", 2027, "2027-02-05", "2027-02-10", "", WIKI + "Rio_Carnival"),
    ("Day of the Dead", "", 2026, "2026-11-01", "2026-11-02", "", WIKI + "Day_of_the_Dead"),
    ("Day of the Dead", "", 2027, "2027-11-01", "2027-11-02", "", WIKI + "Day_of_the_Dead"),
]


def build(db_path: Path = DB) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.execute("CREATE TABLE public_holidays (country_code TEXT, country TEXT, date TEXT, name TEXT, source TEXT)")
        conn.execute("CREATE TABLE observances (name TEXT, religion TEXT, year INTEGER, start_date TEXT, "
                     "end_date TEXT, note TEXT, source_url TEXT)")
        conn.execute("CREATE INDEX ix_hol ON public_holidays(country, date)")
        rows = []
        for code, name in COUNTRIES.items():
            try:
                cal = holidays.country_holidays(code, years=YEARS, language="en_US")
            except (NotImplementedError, KeyError):
                cal = holidays.country_holidays(code, years=YEARS)
            rows += [(code, name, d.isoformat(), label, f"{PYHOL} ({code})") for d, label in sorted(cal.items())]
        conn.executemany("INSERT INTO public_holidays VALUES (?,?,?,?,?)", rows)
        conn.executemany("INSERT INTO observances VALUES (?,?,?,?,?,?,?)", OBSERVANCES)
    print(f"{db_path}: {len(rows)} public holidays, {len(OBSERVANCES)} observances")


if __name__ == "__main__":
    build()
