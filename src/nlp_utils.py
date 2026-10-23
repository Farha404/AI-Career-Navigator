"""
NLP utilities — exact port of the preprocessing functions from final_project.ipynb.

These functions are the single source of truth for text cleaning, skill extraction,
education/seniority/experience parsing, and role canonicalization.
They must never diverge from the notebook implementation.
"""

import re
import datetime
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

# ──────────────────────────────────────────────────────────────────────────────
# Text cleaning  (cell 42 of final_project.ipynb)
# ──────────────────────────────────────────────────────────────────────────────

STOP_WORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """Lowercase, keep a-z0-9+#/\s, collapse whitespace. (notebook cell 42)"""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#/\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list:
    return clean_text(text).split()


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline: clean → tokenize → remove stopwords. (cell 42)"""
    return " ".join(remove_stopwords(tokenize(text)))


# ──────────────────────────────────────────────────────────────────────────────
# Education extraction  (cell 54)
# ──────────────────────────────────────────────────────────────────────────────

EDU_RANK = {
    "High School": 0,
    "BA": 1,
    "BSc": 1,
    "MBA": 2,
    "MSc": 2,
    "PhD": 3,
}


def extract_education(text: str) -> str:
    """Extract highest detected education level. (cell 54)"""
    raw = str(text)
    low = raw.lower()
    found = []

    if re.search(r"\bph\.?d\b|doctorate|doctoral", low):
        found.append("PhD")
    if re.search(r"\bmba\b|master of business", low):
        found.append("MBA")
    if re.search(
        r"\bm\.?sc\b|master of science|(?<!scrum )master's|\bmasters\b|master of (?!business)",
        low,
    ):
        found.append("MSc")
    if re.search(
        r"\bb\.?sc\b|bachelor of science|bachelor(?! of arts)|undergraduate degree",
        low,
    ):
        found.append("BSc")
    if re.search(r"\bB\.?A\b", raw) or "bachelor of arts" in low:
        found.append("BA")
    if re.search(r"high school|secondary school|thanaweya", low):
        found.append("High School")

    return max(found, key=EDU_RANK.get) if found else "Unknown"


# ──────────────────────────────────────────────────────────────────────────────
# Years of experience extraction  (cell 52)
# ──────────────────────────────────────────────────────────────────────────────


# Existing imports in nlp_utils.py already include re and datetime.

_MONTHS = {
    name.lower(): number
    for number, names in enumerate(
        [
            ("jan", "january"), ("feb", "february"), ("mar", "march"),
            ("apr", "april"), ("may",), ("jun", "june"),
            ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"),
            ("oct", "october"), ("nov", "november"), ("dec", "december"),
        ],
        start=1,
    )
    for name in names
}

_MONTH_PATTERN = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|"
    r"Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_DATE_TOKEN = (
    rf"(?:{_MONTH_PATTERN}\.?\s+\d{{4}}|"
    rf"\d{{4}}[-/]\d{{1,2}}|\d{{1,2}}[/-]\d{{4}}|\d{{4}})"
)
_DATE_RANGE_RE = re.compile(
    rf"(?P<start>{_DATE_TOKEN})\s*(?:-|to|–|—|through)\s*"
    rf"(?P<end>{_DATE_TOKEN}|present|current|now)",
    re.IGNORECASE,
)

_WORK_HEADINGS = {
    "experience", "work experience", "professional experience", "employment",
    "employment history", "work history", "career history", "professional history",
    "internships", "internship experience", "freelance experience",
}
_NONWORK_HEADINGS = {
    "education", "education and training", "academic background", "academic history",
    "certifications", "certificates", "licenses and certifications", "courses",
    "coursework", "training", "projects", "academic projects", "personal projects",
    "awards", "publications",
}

_JOB_TITLE_RE = re.compile(
    r"\b(?:engineer|developer|analyst|scientist|manager|intern|consultant|"
    r"designer|assistant|coordinator|specialist|administrator|technician|"
    r"teacher|researcher|accountant|architect|director|lead|officer|"
    r"representative|operator|editor|nurse|doctor|professor|instructor|"
    r"executive|recruiter|marketing|sales|support|technologist)\b",
    re.IGNORECASE,
)
_EMPLOYMENT_RE = re.compile(
    r"\b(?:worked|working|employed|employment|job|position|role|responsibilit(?:y|ies)|"
    r"full[- ]?time|part[- ]?time|freelance|self[- ]employed|intern(?:ship)?|"
    r"contract(?:or)?|company|employer|client|staff|promoted|managed|developed|"
    r"built|designed|led|delivered|maintained|supported|served)\b",
    re.IGNORECASE,
)
_NONWORK_CONTEXT_RE = re.compile(
    r"\b(?:university|college|school|bachelor|master|phd|doctorate|degree|"
    r"certification|certificate|course|coursework|bootcamp|graduat(?:ed|ion)|"
    r"academic|student|semester|thesis|diploma)\b",
    re.IGNORECASE,
)


def _normalize_heading(line: str) -> str:
    line = re.sub(r"^[\s#*•|\-]+|[\s:*#•|\-]+$", "", line.lower())
    return re.sub(r"\s+", " ", line).strip()


def _parse_resume_date(token: str, today: datetime.date) -> tuple[int, int]:
    """Return (year, month). Year-only dates are interpreted as January."""
    value = token.strip().lower().rstrip(".")

    if value in {"present", "current", "now"}:
        return today.year, today.month

    year_month = re.fullmatch(r"(\d{4})[-/](\d{1,2})", value)
    if year_month:
        year, month = int(year_month.group(1)), int(year_month.group(2))
        if not 1 <= month <= 12:
            raise ValueError("invalid month")
        return year, month

    month_year = re.fullmatch(r"(\d{1,2})[/-](\d{4})", value)
    if month_year:
        month, year = int(month_year.group(1)), int(month_year.group(2))
        if not 1 <= month <= 12:
            raise ValueError("invalid month")
        return year, month

    month_year = re.fullmatch(
        rf"({_MONTH_PATTERN})\.?\s+(\d{{4}})", value, re.IGNORECASE
    )
    if month_year:
        month_name = month_year.group(1).lower().rstrip(".")
        return int(month_year.group(2)), _MONTHS[month_name]

    if re.fullmatch(r"\d{4}", value):
        return int(value), 1

    raise ValueError(f"unsupported date token: {token}")


def _date_range_months(
    start_token: str,
    end_token: str,
    today: datetime.date,
) -> tuple[int, int] | None:
    try:
        start_year, start_month = _parse_resume_date(start_token, today)
        end_year, end_month = _parse_resume_date(end_token, today)
    except ValueError:
        return None

    start_index = start_year * 12 + start_month - 1
    end_index = end_year * 12 + end_month - 1

    if end_token.strip().lower() in {"present", "current", "now"}:
        end_index += 1  # Include the current month.
    elif not re.fullmatch(r"\d{4}", end_token.strip()):
        end_index += 1  # Month/year end dates include the named month.

    if end_index < start_index or end_index - start_index > 40 * 12:
        return None
    if end_index == start_index:
        end_index += 1

    return start_index, end_index


def extract_years_experience(
    text: str,
    today: datetime.date | None = None,
) -> int:
    """Count unique months in dated work roles, excluding education and training.

    Internships, freelance, contract, and part-time work count when the entry
    has role or employment evidence. Concurrent roles count once. This function
    intentionally does not trust free-text totals such as "5 years of experience".
    """
    today = today or datetime.date.today()
    lines = str(text or "").splitlines()
    if not lines:
        return 0

    section_by_line = []
    section = "unknown"
    for line in lines:
        heading = _normalize_heading(line)
        if heading in _WORK_HEADINGS:
            section = "work"
        elif heading in _NONWORK_HEADINGS:
            section = "nonwork"
        section_by_line.append(section)

    intervals = []
    for line_index, line in enumerate(lines):
        ranges = list(_DATE_RANGE_RE.finditer(line))
        if not ranges:
            continue

        nearby_indices = range(
            max(0, line_index - 2),
            min(len(lines), line_index + 3),
        )
        context = " ".join(
            lines[i]
            for i in nearby_indices
            if section_by_line[i] == section_by_line[line_index]
        )

        if section_by_line[line_index] == "nonwork":
            continue
        if _NONWORK_CONTEXT_RE.search(context):
            continue
        if not (_JOB_TITLE_RE.search(context) or _EMPLOYMENT_RE.search(context)):
            continue

        for match in ranges:
            interval = _date_range_months(
                match.group("start"),
                match.group("end"),
                today,
            )
            if interval:
                intervals.append(interval)

    if not intervals:
        return 0

    # Union intervals so overlapping or duplicated jobs do not add months twice.
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    total_months = sum(end - start for start, end in merged)
    return min(int(total_months / 12 + 0.5), 40)


# ──────────────────────────────────────────────────────────────────────────────
# Seniority extraction  (cell 56)
# ──────────────────────────────────────────────────────────────────────────────


def years_to_seniority(years: int) -> str:
    """Map years of experience to seniority level. (cell 56)"""
    if years < 2:
        return "Junior"
    if years < 6:
        return "Mid"
    return "Senior"


def extract_seniority(text: str, years: int = None) -> str:
    """Extract seniority from text keywords; fall back to years. (cell 56)"""
    low = str(text).lower()

    if re.search(
        r"\b(senior|sr\.?|lead|principal|staff|director|vp|head of)\b",
        low,
    ):
        return "Senior"
    if re.search(
        r"\b(junior|jr\.?|intern|internship|trainee|entry[- ]level|graduate)\b",
        low,
    ):
        return "Junior"
    if re.search(r"\b(mid[- ]level|intermediate)\b", low):
        return "Mid"

    return years_to_seniority(years) if years is not None else "Mid"


# ──────────────────────────────────────────────────────────────────────────────
# Skill extraction  (cells 46-50)
# These functions need the SKILL_VOCAB / SKILL_ALIASES from the saved artifacts.
# They are wired up at import time in artifacts.py.
# ──────────────────────────────────────────────────────────────────────────────


def build_skill_extractor(skill_vocab: list, skill_aliases: dict):
    """
    Build and return an extract_skills() function from the saved vocabulary.
    Mirrors the notebook's cells 48-50 exactly.
    """
    alias_to_skill = {skill.lower(): skill for skill in skill_vocab}
    for alias, target in skill_aliases.items():
        if target in skill_vocab:
            alias_to_skill[alias.lower()] = target

    alias_patterns_sorted = sorted(
        alias_to_skill.keys(), key=len, reverse=True
    )

    alias_regex = [
        (
            re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)"),
            alias_to_skill[alias],
        )
        for alias in alias_patterns_sorted
    ]

    def extract_skills(text: str) -> list:
        """Return sorted list of canonical skills found in text. (cell 50)"""
        text = str(text).lower()
        found = []
        for pattern, skill in alias_regex:
            if pattern.search(text):
                found.append(skill)
                text = pattern.sub(" ", text)
        return sorted(set(found))

    return extract_skills


# ──────────────────────────────────────────────────────────────────────────────
# Role canonicalization  (cell 58)
# Requires KNOWN_ROLES from the saved artifacts.
# ──────────────────────────────────────────────────────────────────────────────


def build_role_canonicalizer(known_roles: list):
    """
    Build and return canonical_role() and extract_job_titles() from saved roles.
    Mirrors the notebook's cell 58 exactly.
    """
    role_vec = TfidfVectorizer(ngram_range=(1, 2))
    role_matrix = role_vec.fit_transform(
        [clean_text(role) for role in known_roles]
    )

    def canonical_role(title: str, threshold: float = 0.35) -> str:
        if not str(title).strip():
            return "Other"
        query = role_vec.transform([clean_text(title)])
        similarities = (role_matrix @ query.T).toarray().ravel()
        best_idx = similarities.argmax()
        if similarities[best_idx] >= threshold:
            return known_roles[best_idx]
        return "Other"

    def extract_job_titles(text: str) -> list:
        low = str(text).lower()
        hits = [
            (low.find(role.lower()), role)
            for role in known_roles
            if role.lower() in low
        ]
        return [role for _, role in sorted(hits)]

    return canonical_role, extract_job_titles


# ──────────────────────────────────────────────────────────────────────────────
# Industry extraction
# Identifies candidate / job industry domain from text and tags.
# ──────────────────────────────────────────────────────────────────────────────

KNOWN_INDUSTRIES = [
    ("FinTech", [r"\bfintech\b", r"\bfinancial\s+technology\b"]),
    ("Finance", [r"\bfinance\b", r"\bfinancial\b", r"\bbanking\b", r"\baccounting\b", r"\baudit\b"]),
    ("Healthcare", [r"\bhealthcare\b", r"\bhealth\b", r"\bmedical\b", r"\bpharma\b", r"\bbiotech\b", r"\bclinical\b"]),
    ("E-Commerce", [r"\be-?commerce\b", r"\bonline\s+retail\b", r"\bmarketplace\b", r"\bshop\b"]),
    ("Gaming", [r"\bgaming\b", r"\bgames?\s+development\b", r"\bvideo\s+games?\b"]),
    ("Technology", [r"\btechnology\b", r"\bsoftware\b", r"\btech\b", r"\bit\s+services\b", r"\bcloud\b", r"\bcybersecurity\b"]),
    ("Retail", [r"\bretail\b", r"\bfashion\b", r"\bconsumer\s+goods\b", r"\bfmcg\b"]),
    ("Education", [r"\beducation\b", r"\bedtech\b", r"\blearning\b", r"\be-?learning\b", r"\buniversity\b"]),
    ("Automotive", [r"\bautomotive\b", r"\bvehicles?\b", r"\bmobility\b", r"\bcars?\b"]),
    ("Consulting", [r"\bconsulting\b", r"\badvisory\b", r"\bmanagement\s+consulting\b"]),
    ("Telecommunications", [r"\btelecommunications?\b", r"\btelecom\b", r"\bnetwork\s+provider\b"]),
    ("Manufacturing", [r"\bmanufacturing\b", r"\bindustrial\b", r"\bfactory\b", r"\bproduction\b"]),
    ("Media", [r"\bmedia\b", r"\bpublishing\b", r"\bbroadcasting\b", r"\bentertainment\b"]),
    ("Human Resources", [r"\bhuman\s+resources\b", r"\bhr\b", r"\brecruiting\b", r"\btalent\s+acquisition\b"]),
]


def extract_industry(text: str, tags: list[str] = None) -> str:
    """Extract recognized industry domain from text and optional tags."""
    combined = (str(text) + " " + " ".join(tags or [])).lower()
    for ind_name, patterns in KNOWN_INDUSTRIES:
        for p in patterns:
            if re.search(p, combined):
                return ind_name
    return "Unknown"

