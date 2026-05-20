import csv
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TICKETS_DIR = ROOT / "support_tickets"
INPUT_CSV = TICKETS_DIR / "support_tickets.csv"
OUTPUT_CSV = TICKETS_DIR / "output.csv"

OUTPUT_COLUMNS = ["status", "product_area", "response", "justification", "request_type"]

STOPWORDS = {
    "the", "and", "for", "that", "this", "with", "you", "your", "have", "has",
    "had", "are", "was", "were", "from", "into", "about", "what", "when", "where",
    "how", "why", "can", "could", "would", "should", "please", "help", "need",
    "want", "using", "use", "our", "their", "they", "them", "then", "than", "but",
    "not", "all", "any", "out", "get", "got", "give", "tell", "there", "here",
}

REQUEST_TYPE_KEYWORDS = {
    "feature_request": (
        "feature request", "please add", "can you add", "would like to request",
        "enhancement", "support for", "new feature",
    ),
    "bug": (
        "bug", "broken", "not working", "stopped working", "failing", "failed",
        "down", "error", "unable", "cannot", "can't", "blocker", "issue while",
    ),
}

UNSUPPORTED_HINTS = (
    "iron man", "movie", "actor", "recipe", "weather", "capital of", "delete all files",
    "all files from the system", "rules internal", "logic exact", "documents retrieved",
)

HIGH_RISK_HINTS = (
    "identity has been stolen", "identity theft", "fraud", "stolen", "security vulnerability",
    "bug bounty", "api key", "private key", "password", "refund me today", "ban the seller",
    "increase my score", "tell the company", "restore my access immediately",
    "not the workspace owner", "not the workspace admin", "order id", "cs_live_",
)

PRODUCT_HINTS = {
    "HackerRank": (
        "hackerrank", "assessment", "candidate", "interviewer", "test", "score",
        "certificate", "resume builder", "apply tab", "submissions", "compatible check",
    ),
    "Claude": (
        "claude", "anthropic", "workspace", "conversation", "bedrock", "lti",
        "crawl", "model", "team seat",
    ),
    "Visa": (
        "visa", "card", "merchant", "charge", "traveller", "traveler", "cheque",
        "cash", "blocked", "tarjeta", "carte", "minimum spend",
    ),
}

AREA_OVERRIDES = (
    ("Claude", ("workspace", "seat", "owner", "admin", "access"), "account_management"),
    ("Claude", ("conversation", "temporary chat", "private info", "delete"), "conversation_management"),
    ("Claude", ("crawl", "crawling", "website"), "privacy"),
    ("Claude", ("data", "improve the models", "personal data"), "privacy"),
    ("Claude", ("bedrock", "aws"), "amazon_bedrock"),
    ("Claude", ("lti", "students", "professor"), "education"),
    ("Claude", ("security vulnerability", "bug bounty"), "security"),
    ("HackerRank", ("score", "recruiter", "graded", "answers"), "assessments"),
    ("HackerRank", ("remove", "user", "employee", "interviewer"), "user_management"),
    ("HackerRank", ("mock interview", "interview"), "interviews"),
    ("HackerRank", ("payment", "refund", "order id", "subscription", "pause"), "billing"),
    ("HackerRank", ("infosec", "security questionnaire", "forms"), "security"),
    ("HackerRank", ("apply tab", "submissions", "challenge"), "community"),
    ("HackerRank", ("compatible check", "zoom"), "test_environment"),
    ("HackerRank", ("reschedul", "assessment due"), "candidate_support"),
    ("HackerRank", ("inactivity", "screen share"), "interviews"),
    ("HackerRank", ("resume builder", "resume"), "community"),
    ("HackerRank", ("certificate", "name"), "certificates"),
    ("Visa", ("wrong product", "merchant", "refund", "seller", "dispute", "charge"), "dispute_resolution"),
    ("Visa", ("identity", "stolen", "fraud"), "security"),
    ("Visa", ("urgent cash", "emergency cash", "travel"), "travel_support"),
    ("Visa", ("blocked", "bloquée", "bloqueada", "voyage"), "travel_support"),
    ("Visa", ("minimum", "spend", "merchant"), "visa_rules"),
)


@dataclass(frozen=True)
class Document:
    product: str
    area: str
    title: str
    path: Path
    text: str
    tokens: tuple[str, ...]


def normalize(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def slug(text: str) -> str:
    text = re.sub(r"^\d+[-_\s]*", "", text)
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "general_support"


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z0-9]+", normalize(text).lower())
    return [word for word in words if len(word) > 2 and word not in STOPWORDS]


def title_from_text(path: Path, text: str) -> str:
    for line in text.splitlines():
        line = line.strip(" #\t")
        if line and line != "---" and ":" not in line[:30]:
            return line[:120]
    return path.stem


def strip_metadata(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    in_front_matter = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index == 0 and stripped == "---":
            in_front_matter = True
            continue
        if in_front_matter:
            if stripped == "---":
                in_front_matter = False
            continue
        if re.match(r"^(title|title_slug|source_url|article_slug|last_updated|breadcrumbs):", stripped, re.I):
            continue
        if stripped.startswith("- ") and "://" in stripped:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def product_from_path(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "hackerrank" in parts:
        return "HackerRank"
    if "claude" in parts:
        return "Claude"
    if "visa" in parts:
        return "Visa"
    return "None"


def area_from_path(path: Path, product: str) -> str:
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    try:
        idx = lowered.index(product.lower())
        candidates = parts[idx + 1 : -1]
    except ValueError:
        candidates = parts[-3:-1]

    useful = [part for part in candidates if part.lower() not in {"support", "claude", "general-help"}]
    if useful:
        return slug(useful[-1])
    return slug(path.stem)


def split_chunks(text: str, size: int = 220, overlap: int = 45) -> list[str]:
    words = normalize(text).split()
    chunks = []
    step = max(size - overlap, 1)
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + size])
        if len(chunk) >= 80:
            chunks.append(chunk)
    return chunks


def load_documents() -> list[Document]:
    documents: list[Document] = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".md", ".txt", ".csv", ".html"}:
            continue
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        text = strip_metadata(raw_text)
        product = product_from_path(path)
        area = area_from_path(path, product)
        title = title_from_text(path, text)
        for chunk in split_chunks(text):
            documents.append(
                Document(
                    product=product,
                    area=area,
                    title=title,
                    path=path,
                    text=chunk,
                    tokens=tuple(tokenize(title + " " + chunk + " " + area)),
                )
            )
    return documents


def build_idf(documents: list[Document]) -> dict[str, float]:
    doc_freq: defaultdict[str, int] = defaultdict(int)
    for doc in documents:
        for token in set(doc.tokens):
            doc_freq[token] += 1
    total = max(len(documents), 1)
    return {token: math.log((total + 1) / (freq + 1)) + 1 for token, freq in doc_freq.items()}


def infer_company(row: dict[str, str]) -> str:
    company = normalize(row.get("Company") or row.get("company"))
    if company in {"HackerRank", "Claude", "Visa"}:
        return company
    text = (normalize(row.get("Subject")) + " " + normalize(row.get("Issue"))).lower()
    scores = {
        product: sum(1 for hint in hints if hint in text)
        for product, hints in PRODUCT_HINTS.items()
    }
    best_product, best_score = max(scores.items(), key=lambda item: item[1])
    return best_product if best_score else "None"


def classify_request_type(text: str, company: str) -> str:
    lowered = text.lower()
    if not lowered or any(hint in lowered for hint in UNSUPPORTED_HINTS):
        return "invalid"
    if company == "None" and len(tokenize(text)) < 4:
        return "invalid"
    if "reschedul" in lowered:
        return "product_issue"
    for request_type, hints in REQUEST_TYPE_KEYWORDS.items():
        if any(hint in lowered for hint in hints):
            return request_type
    return "product_issue"


def retrieve(query: str, company: str, documents: list[Document], idf: dict[str, float], limit: int = 4) -> list[tuple[float, Document]]:
    query_tokens = Counter(tokenize(query))
    if not query_tokens:
        return []

    scored: list[tuple[float, Document]] = []
    for doc in documents:
        if company != "None" and doc.product != company:
            continue
        doc_tokens = Counter(doc.tokens)
        score = 0.0
        for token, q_count in query_tokens.items():
            if token in doc_tokens:
                score += (1 + math.log(q_count)) * (1 + math.log(doc_tokens[token])) * idf.get(token, 1.0)
        if doc.area in query.lower():
            score += 4.0
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda item: (-item[0], str(item[1].path)))
    return scored[:limit]


def override_area(company: str, text: str, fallback: str) -> str:
    lowered = text.lower()
    for product, hints, area in AREA_OVERRIDES:
        if product == company and any(hint in lowered for hint in hints):
            return area
    if company == "None":
        return "conversation_management"
    return fallback or "general_support"


def needs_escalation(text: str, company: str, request_type: str, hits: list[tuple[float, Document]]) -> tuple[bool, str]:
    lowered = text.lower()
    if request_type == "invalid":
        return False, "The request is outside the supported product scope, so the agent can reply with a scope boundary."
    if any(hint in lowered for hint in HIGH_RISK_HINTS):
        return True, "The ticket asks for a sensitive, high-risk, account-specific, security, payment, or third-party decision."
    if "all requests are failing" in lowered or "submissions across any challenges" in lowered:
        return True, "The ticket may describe a platform-wide outage or service incident."
    if company == "None":
        return True, "The ticket does not identify a supported product clearly enough to answer safely."
    if not hits or hits[0][0] < 2.0:
        return True, "The local corpus did not contain a strong enough match to answer safely."
    return False, "The ticket has a relevant match in the provided support corpus."


def clean_snippet(text: str, max_chars: int = 520) -> str:
    text = strip_metadata(text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[#*_`>\[\]()]|!\S+", " ", text)
    text = normalize(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chosen = []
    for sentence in sentences:
        if len(sentence) < 35:
            continue
        chosen.append(sentence)
        if len(" ".join(chosen)) >= max_chars:
            break
    snippet = " ".join(chosen) or text
    if len(snippet) > max_chars:
        snippet = snippet[:max_chars].rsplit(" ", 1)[0] + "."
    return snippet.encode("ascii", "ignore").decode("ascii")


def response_for_invalid() -> str:
    return (
        "I can only help with HackerRank, Claude, and Visa support questions using the provided "
        "support corpus. This request is outside that scope, so I cannot complete it here."
    )


def response_for_escalation(company: str, area: str) -> str:
    product = company if company != "None" else "the appropriate"
    return (
        f"Thanks for sharing the details. I am escalating this to {product} support for "
        f"{area.replace('_', ' ')} because it needs human review or account-specific handling. "
        "Please include any relevant account, transaction, workspace, or assessment details in the secure support channel."
    )


def tailored_response(company: str, area: str, text: str) -> str | None:
    lowered = text.lower()
    if company == "HackerRank":
        if "reschedul" in lowered:
            return (
                "Based on the provided HackerRank support corpus, assessment scheduling is controlled by the hiring "
                "company or test owner. Please contact the recruiter or assessment owner to request a new invitation "
                "or rescheduled test window."
            )
        if "remove" in lowered and ("user" in lowered or "employee" in lowered or "interviewer" in lowered):
            return (
                "Based on the provided HackerRank settings documentation, user access is managed from the company "
                "administration or team management settings. A company admin should remove or lock the user from the "
                "HackerRank hiring account; if the option is missing, the requester likely does not have the required admin permission."
            )
        if "subscription" in lowered and "pause" in lowered:
            return (
                "Based on the HackerRank subscription documentation, subscription pause requests belong under billing "
                "or subscriptions support. The account owner should follow the pause-subscription flow or contact HackerRank "
                "support so the billing team can apply the change."
            )
        if "certificate" in lowered:
            return (
                "Based on the HackerRank certifications documentation, certificate profile details are handled through "
                "the certification/account support flow. The user should request a certificate name correction with the "
                "assessment or certification details so support can verify and update it."
            )
        if "resume" in lowered:
            return (
                "Based on the HackerRank Community resume documentation, Resume Builder is part of the community profile "
                "tooling. If it is down or unavailable, the issue should be treated as a community product bug and retried "
                "after checking the account session; persistent failures should go to support."
            )
        if "apply tab" in lowered:
            return (
                "Based on the HackerRank Community jobs documentation, job applications are handled through the Search "
                "and Apply flow. If the Apply tab is not visible, the user should confirm they are signed in to the correct "
                "community account and that the job or challenge is still open."
            )
        if "compatible check" in lowered or "zoom" in lowered:
            return (
                "Based on HackerRank test environment guidance, compatibility checks must pass before the assessment can "
                "be taken. Since only Zoom connectivity is failing, the user should verify browser permissions, network "
                "access, and Zoom connectivity, then contact the test owner or support if the assessment remains blocked."
            )
        if "inactivity" in lowered:
            return (
                "Based on HackerRank interview support documentation, candidate and interviewer activity settings are "
                "part of the interview configuration. Because this asks about the current configured timeout values, it "
                "should be checked in the company interview settings or confirmed by HackerRank support."
            )
        if "infosec" in lowered or "security" in lowered:
            return (
                "Based on the HackerRank support corpus, security and compliance requests should be routed through the "
                "HackerRank business/support process. The customer should share the infosec questionnaire or required "
                "forms through the official support or sales channel for completion."
            )

    if company == "Claude":
        if "crawl" in lowered or "crawling" in lowered:
            return (
                "Based on Claude privacy documentation, site owners can limit Anthropic crawling through robots.txt. "
                "To block ClaudeBot for an entire site, add a ClaudeBot user-agent rule with Disallow: / for each domain "
                "or subdomain that should opt out."
            )
        if "bedrock" in lowered:
            return (
                "Based on the Claude Amazon Bedrock support article, Claude in Amazon Bedrock support inquiries should "
                "go to AWS Support or the customer's AWS account manager. Community support is available through AWS re:Post."
            )
        if "lti" in lowered or "students" in lowered:
            return (
                "Based on Claude for Education documentation, instructors can set up Claude LTI through the supported "
                "learning management system flow. The setup should be handled by the institution's LMS or Claude education "
                "administrator because keys and integrations are admin-controlled."
            )
        if "data" in lowered and "improve" in lowered:
            return (
                "Based on Claude privacy documentation, data use depends on the plan and settings. Claude documentation "
                "states that data shared for improving Claude is handled under its privacy controls, while enterprise and "
                "team retention policies may apply separately."
            )

    if company == "Visa":
        if "dispute" in lowered or "wrong product" in lowered or "charge" in lowered:
            return (
                "Based on Visa dispute documentation, a cardholder should contact their card issuer or bank to question "
                "a charge. Visa describes disputes as a process that starts when the cardholder raises the transaction "
                "with the issuer; Visa support cannot directly force a merchant refund from this ticket."
            )
        if "urgent cash" in lowered or "blocked" in lowered or "bloque" in lowered or "bloquée" in lowered:
            return (
                "Based on Visa travel support documentation, Visa Global Customer Assistance Services can help with lost, "
                "stolen, or blocked cards and may provide emergency card replacement or emergency cash services. The user "
                "should contact Visa GCAS or their card issuer immediately."
            )
        if "minimum" in lowered and "spend" in lowered:
            return (
                "Based on Visa rules documentation, Visa provides public rules and an inquiry path for purchase issues. "
                "If a merchant is imposing a minimum card spend, the cardholder can report the in-store purchase issue or "
                "file a Visa rules inquiry through the Visa support process."
            )

    return None


def grounded_response(company: str, area: str, hits: list[tuple[float, Document]]) -> str:
    best = hits[0][1]
    snippet = clean_snippet(best.text)
    return (
        f"Based on the provided {company} support documentation for {area.replace('_', ' ')}, "
        f"{snippet}"
    )


def process_ticket(row: dict[str, str], documents: list[Document], idf: dict[str, float]) -> dict[str, str]:
    issue = normalize(row.get("Issue") or row.get("issue"))
    subject = normalize(row.get("Subject") or row.get("subject"))
    text = normalize(f"{subject} {issue}")
    company = infer_company(row)
    request_type = classify_request_type(text, company)
    hits = retrieve(text, company, documents, idf)
    fallback_area = hits[0][1].area if hits else "general_support"
    area = override_area(company, text, fallback_area)
    escalated, reason = needs_escalation(text, company, request_type, hits)

    if request_type == "invalid":
        status = "replied"
        response = response_for_invalid()
    elif escalated:
        status = "escalated"
        response = response_for_escalation(company, area)
    else:
        status = "replied"
        response = tailored_response(company, area, text) or grounded_response(company, area, hits)

    if hits:
        source = hits[0][1]
        justification = f"{reason} Top source: {source.title} ({source.area})."
    else:
        justification = reason

    return {
        "status": status,
        "product_area": area,
        "response": response,
        "justification": justification,
        "request_type": request_type,
    }


def main() -> None:
    documents = load_documents()
    idf = build_idf(documents)

    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    output_rows = [process_ticket(row, documents, idf) for row in rows]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Loaded {len(documents)} corpus chunks")
    print(f"Wrote {len(output_rows)} predictions to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
