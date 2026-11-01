"""Input and output guards: prompt injection, PII masking, toxicity, citation checking.

Concept: guardrails written into a prompt ("be respectful", "ignore attempts to change the
rules") are requests, not controls; the model can be talked out of them. These guards are
ordinary code that runs *before* the model (input) and *after* it (output), so they are
testable and cannot be overridden by the text they inspect.

  * detect_injection: regex patterns for instruction-override attempts ("ignore the rubric",
    "you are now", fake </source> tags, "give this 10/10").
  * mask_pii: replaces emails, phone numbers, payment cards (Luhn-checked) and IBANs with
    placeholders before text is logged, embedded or sent to a model.
  * check_toxicity: keyword + generalisation patterns ("all X people are ...") offline;
    optionally the OpenAI moderation endpoint when a key is present.
  * citation_check: verifies an answer cites only real source numbers and that most
    claim-like sentences carry a citation.

Trade-off: pattern guards are fast, free and deterministic but have false negatives
(paraphrased attacks) and some false positives; that's why the model-side defences
(data wrapping, structured outputs) and red-team evals exist as additional layers.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# --- Prompt injection --------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all |any |the |your |previous |prior |above )+(instructions|rules|rubric|prompt|guidelines)",
    r"disregard (all |any |the |your |previous |prior |above )*(instructions|rules|rubric|prompt|guidelines)",
    r"forget (all |your |the )*(instructions|rules|rubric)",
    r"\byou are now\b", r"\bact as (an? )?(unfiltered|unrestricted|jailbroken|dan)\b",
    r"\b(developer|god|jailbreak) mode\b", r"\bsystem prompt\b",
    r"</?\s*(system|source|evidence|instructions?)\b[^>]*>",
    r"\b(give|award|score|rate)\b[^.\n]{0,40}\b(10|ten)\s*(/|out of)\s*10\b",
    r"\b(maximum|max|full|perfect) (score|marks|points)\b",
    r"\bnew instructions?\s*:", r"\boverride\b[^.\n]{0,30}\b(rules|rubric|policy|instructions)\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> list[str]:
    """Return the matched snippets (empty list = clean)."""
    return [m.group(0) for rx in _INJECTION_RE for m in [rx.search(text or "")] if m]


# --- PII -------------------------------------------------------------------------------------
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
PHONE_RE = re.compile(r"(?<![\w/])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{1,4}\)[\s.-]?)?\d{2,4}[\s.-]\d{3,4}[\s.-]?\d{3,4}\b")


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return 13 <= len(digits) <= 19 and total % 10 == 0


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return (masked_text, kinds_found). Order matters: IBAN and card before phone."""
    found: list[str] = []

    def sub(rx: re.Pattern, label: str, s: str, check=None) -> str:
        def repl(m: re.Match) -> str:
            if check and not check(m.group(0)):
                return m.group(0)
            found.append(label)
            return f"[{label}]"
        return rx.sub(repl, s)

    text = sub(EMAIL_RE, "EMAIL", text or "")
    text = sub(IBAN_RE, "IBAN", text)
    text = sub(CARD_RE, "CARD", text, _luhn_ok)
    text = sub(PHONE_RE, "PHONE", text)
    return text, sorted(set(found))


# --- Toxicity --------------------------------------------------------------------------------
GROUP_TERMS = (
    r"muslims?|christians?|jews?|jewish|hindus?|buddhists?|sikhs?|atheists?|arabs?|africans?|asians?|"
    r"indians?|chinese|japanese|koreans?|mexicans?|latinos?|europeans?|americans?|brazilians?|"
    r"immigrants?|foreigners?|westerners?|black people|white people|women|men|gays?|lesbians?|"
    r"[a-z]+ people"
)
INSULTS = (
    r"stupid|dumb|idiots?|inferior|savages?|primitive|backwards?|barbaric|barbarians?|dirty|filthy|"
    r"disgusting|terrorists?|criminals?|animals|vermin|subhuman|lazy|thieves|evil|worthless|"
    r"uncivili[sz]ed|retarded|rude|greedy|arrogant|ignorant|brainwashed|fanatics?|parasites?"
)
TOXIC_PATTERNS = [
    rf"\b(all|most|every|those|these)\s+({GROUP_TERMS})\s+(are|is)\s+(\w+\s+)?({INSULTS})\b",
    rf"\b({GROUP_TERMS})\s+(are|is)\s+(all\s+|so\s+|just\s+|such\s+)?(\w+\s+(and\s+)?)?({INSULTS})\b",
    rf"\bwhy (are|do) ({GROUP_TERMS})\s+(so\s+|always\s+)?({INSULTS}|hate|smell|steal|lie|cheat)\b",
    rf"\b({INSULTS})\s+({GROUP_TERMS})\b",
    r"\b(kill|exterminate|deport|ban) (all )?(the )?(" + GROUP_TERMS + r")\b",
    r"\b(religion|culture|tradition)s? (is|are) (for )?(stupid|idiots|garbage|trash|nonsense|barbaric|primitive)\b",
]
_TOXIC_RE = [re.compile(p, re.IGNORECASE) for p in TOXIC_PATTERNS]


@dataclass
class ToxicityResult:
    flagged: bool
    reasons: list[str] = field(default_factory=list)
    source: str = "rules"


def check_toxicity(text: str, use_moderation_api: bool = True) -> ToxicityResult:
    reasons = [m.group(0) for rx in _TOXIC_RE for m in [rx.search(text or "")] if m]
    if reasons:
        return ToxicityResult(True, reasons, "rules")
    if use_moderation_api and os.getenv("OPENAI_API_KEY") and os.getenv("OFFLINE", "0") != "1":
        try:
            import openai
            res = openai.OpenAI().moderations.create(model="omni-moderation-latest", input=text).results[0]
            if res.flagged:
                cats = [k for k, v in res.categories.model_dump().items() if v]
                return ToxicityResult(True, cats, "openai-moderation")
        except Exception:
            pass  # the moderation API is an extra layer; rules already ran
    return ToxicityResult(False, [], "rules")


# --- Citations -------------------------------------------------------------------------------
CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(*])")
CITE_ONLY_RE = re.compile(r"^(\[\d+(?:\s*,\s*\d+)*\]\s*)+[.!?]?$")


@dataclass
class CitationReport:
    ok: bool
    cited: list[int]
    invalid: list[int]
    uncited_sentences: list[str]
    coverage: float


def split_sentences(text: str) -> list[str]:
    lines = [ln.strip(" -*\t") for ln in (text or "").splitlines()]
    out: list[str] = []
    for ln in lines:
        if not ln or ln.startswith("#") or ln.endswith(":"):
            continue
        for piece in (p.strip() for p in SENTENCE_RE.split(ln) if p.strip()):
            if out and CITE_ONLY_RE.match(piece):  # "... claim. [2]" -> citation belongs to the claim
                out[-1] = f"{out[-1]} {piece}"
            else:
                out.append(piece)
    return out


def is_claim(sentence: str) -> bool:
    """Heuristic: long enough to assert something, and not a question or a sign-off."""
    return len(sentence.split()) >= 6 and not sentence.rstrip().endswith("?")


def citation_check(answer: str, n_sources: int, min_coverage: float = 0.75) -> CitationReport:
    """ok = at least one citation, no citation number outside 1..n_sources, and at least
    `min_coverage` of claim-like sentences carry a citation."""
    cited: list[int] = []
    for group in CITE_RE.findall(answer or ""):
        cited += [int(x) for x in group.split(",")]
    invalid = sorted({c for c in cited if c < 1 or c > n_sources})
    claims = [s for s in split_sentences(answer) if is_claim(s)]
    uncited = [s for s in claims if not CITE_RE.search(s)]
    coverage = 1.0 if not claims else (len(claims) - len(uncited)) / len(claims)
    ok = bool(cited) and not invalid and coverage >= min_coverage
    return CitationReport(ok, sorted(set(cited)), invalid, uncited, round(coverage, 3))
