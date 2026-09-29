"""Authored synthetic CFPB-schema fixture generator.

Everything here is fictional. Company names are obviously fictional, identifiers are
prefixed ``SYN-CMP-``, and narratives are built from generic phrase templates that
carry no names, numbers, addresses or other personal data.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .fields import CFPB_COLUMNS, NARRATIVE

FIXTURE_VERSION = "v1"
GENERATOR = "tools/make_fixture.py"
START_DATE = date(2024, 7, 1)
MONTHS = 24

PRODUCTS: dict[str, dict[str, list[str]]] = {
    "Credit reporting": {
        "sub_product": ["Credit reporting", "Other personal consumer report"],
        "issue": [
            "Incorrect information on your report",
            "Problem with a company's investigation into an existing problem",
            "Improper use of your report",
        ],
    },
    "Debt collection": {
        "sub_product": ["Credit card debt", "Medical debt", "Other debt"],
        "issue": [
            "Attempts to collect debt not owed",
            "Written notification about debt",
            "Communication tactics",
        ],
    },
    "Mortgage": {
        "sub_product": ["Conventional home mortgage", "FHA mortgage"],
        "issue": [
            "Trouble during payment process",
            "Struggling to pay mortgage",
            "Applying for a mortgage or refinancing",
        ],
    },
    "Credit card": {
        "sub_product": ["General-purpose credit card", "Store credit card"],
        "issue": [
            "Problem with a purchase shown on your statement",
            "Fees or interest",
            "Closing your account",
        ],
    },
    "Checking or savings account": {
        "sub_product": ["Checking account", "Savings account"],
        "issue": [
            "Managing an account",
            "Problem with a company charging your account",
            "Closing an account",
        ],
    },
    "Student loan": {
        "sub_product": ["Federal student loan servicing", "Private student loan"],
        "issue": ["Dealing with your lender or servicer", "Struggling to repay your loan"],
    },
    "Vehicle loan or lease": {
        "sub_product": ["Loan", "Lease"],
        "issue": ["Managing the loan or lease", "Problems at the end of the loan or lease"],
    },
    "Money transfer": {
        "sub_product": ["Domestic transfer", "Mobile or digital wallet"],
        "issue": ["Fraud or scam", "Money was not available when promised"],
    },
}
PRODUCT_WEIGHTS = [0.38, 0.17, 0.10, 0.10, 0.09, 0.06, 0.05, 0.05]

# Narrative phrase templates per product. Deliberately generic; no slots for names,
# amounts, dates or account identifiers.
TEMPLATES: dict[str, list[str]] = {
    "Credit reporting": [
        "There is an entry on my credit report that I do not recognise.",
        "I disputed an item on my report and the investigation result did not change anything.",
        "My report shows an account as late although I paid it on time.",
        "A hard inquiry appears on my report that I did not authorise.",
        "The reporting agency has not corrected the balance after my dispute.",
    ],
    "Debt collection": [
        "A collector keeps contacting me about a debt that is not mine.",
        "I asked for written validation of the debt and never received it.",
        "The collection agency calls several times a day and at work.",
        "I was told the debt was paid but it is still being collected.",
        "The collector would not tell me the original creditor.",
    ],
    "Mortgage": [
        "My mortgage payment was applied late even though it was sent before the due date.",
        "The servicer added an escrow shortage that was not explained.",
        "I applied for a loan modification and received no decision for months.",
        "My refinancing application was closed without a clear reason.",
        "The servicer did not credit my extra principal payment.",
    ],
    "Credit card": [
        "A charge on my statement is for a purchase I did not make.",
        "I was charged an annual fee that was not disclosed when I opened the card.",
        "The interest rate on my card increased without notice.",
        "I closed my card and the company still reports it as open.",
        "A refund from a merchant was never posted to my card.",
    ],
    "Checking or savings account": [
        "The bank charged overdraft fees in an order that increased the total fees.",
        "A debit was taken from my account that I did not authorise.",
        "My savings account was closed and the remaining balance was not returned.",
        "A deposit was placed on hold longer than the bank said it would be.",
        "I could not access my online banking for several days.",
    ],
    "Student loan": [
        "My student loan servicer misapplied my payments across loans.",
        "I requested an income driven plan and the servicer lost the paperwork.",
        "The servicer reported my loan as delinquent during an approved deferment.",
        "My loan balance went up after payments were made.",
        "I cannot reach anyone at the servicer about my repayment options.",
    ],
    "Vehicle loan or lease": [
        "The lender charged a late fee for a vehicle payment that was on time.",
        "After returning my leased vehicle I was billed for damage I did not cause.",
        "The dealer added products to my auto loan that I did not agree to.",
        "My vehicle title was not released after the loan was paid off.",
        "The payoff amount quoted was different from the amount later demanded.",
    ],
    "Money transfer": [
        "I sent a transfer through the app and the recipient never received it.",
        "I was tricked into sending money and the provider refused to investigate.",
        "The transfer was marked complete but the funds were not available.",
        "My account was frozen after a transfer with no explanation.",
        "The provider reversed a transfer without telling me why.",
    ],
}
# Sentences shared across every product; these add realistic overlap and make the
# label harder to read off a single phrase.
GENERIC: list[str] = [
    "I have contacted the company more than once about this.",
    "I would like this resolved as soon as possible.",
    "I am submitting this complaint because nothing has changed.",
    "Please review my case.",
    "I have kept copies of my correspondence.",
    "This has caused me a lot of stress.",
    "Each time I call I am given a different answer.",
    "I was promised a call back that never came.",
    "I want the record corrected and a written confirmation.",
    "The company has not responded to my letters.",
    "I am asking for someone to look into this properly.",
    "I do not know what else to do.",
]
# Ambiguous sentences that fit several products. A share of narratives use one of
# these instead of a product-specific sentence, so the label is not always readable
# from the text and abstention has something to do.
AMBIGUOUS: dict[str, list[str]] = {
    "fees": ["I was charged a fee that was never explained to me.", "The fees on my account keep changing."],
    "payment": ["My payment was not applied to my account.", "A payment I made is missing from my records."],
    "service": ["Customer service could not explain what happened.", "I was transferred between departments without an answer."],
    "report": ["This was reported to the credit bureaus incorrectly.", "My credit was affected by their mistake."],
}
AMBIGUOUS_BY_PRODUCT: dict[str, list[str]] = {
    "Credit reporting": ["report", "service"],
    "Debt collection": ["report", "service", "payment"],
    "Mortgage": ["fees", "payment", "service"],
    "Credit card": ["fees", "payment", "service", "report"],
    "Checking or savings account": ["fees", "payment", "service"],
    "Student loan": ["payment", "service", "report"],
    "Vehicle loan or lease": ["fees", "payment", "report"],
    "Money transfer": ["payment", "service", "fees"],
}
AMBIGUOUS_RATE = 0.3
# Optional time phrases slotted at the front of a sentence to vary the surface form.
TIME_PHRASES = ["", "", "Last month ", "Recently ", "A few weeks ago ", "Earlier this year ", "Some time ago "]

COMPANIES = [
    "Demo Bank One",
    "Demo Bank Two",
    "Northwind Demo Credit",
    "Contoso Demo Lending",
    "Fabrikam Demo Card Services",
    "Example Collections Demo",
]
STATES = ["CA", "TX", "FL", "NY", "PA", "IL", "OH", "GA", "NC", "MI", "NJ", "VA", "WA", "AZ", "MA"]
SUBMITTED_VIA = ["Web", "Phone", "Referral", "Postal mail"]
COMPANY_RESPONSE = [
    "Closed with explanation",
    "Closed with non-monetary relief",
    "Closed with monetary relief",
    "In progress",
]

# Trend bump: this theme triples in volume during the last six months so that the
# trend module has something to find on the fixture.
BUMP_PRODUCT = "Money transfer"
BUMP_ISSUE = "Fraud or scam"
BUMP_START_MONTH = 21
BUMP_EXTRA_RATE = 0.06  # share of complaints in bump months redirected to the theme

EMPTY_RATE = 0.05
EXACT_DUP_RATE = 0.03
NEAR_DUP_RATE = 0.03


def _month_start(offset: int) -> date:
    year = START_DATE.year + (START_DATE.month - 1 + offset) // 12
    month = (START_DATE.month - 1 + offset) % 12 + 1
    return date(year, month, 1)


def _random_date(rng: np.random.Generator, month_offset: int) -> date:
    start = _month_start(month_offset)
    end = _month_start(month_offset + 1)
    span = (end - start).days
    return start + timedelta(days=int(rng.integers(0, span)))


def _with_time(rng: np.random.Generator, sentence: str) -> str:
    prefix = str(rng.choice(TIME_PHRASES))
    if not prefix:
        return sentence
    return prefix + sentence[0].lower() + sentence[1:]


def _narrative(rng: np.random.Generator, product: str) -> str:
    own = TEMPLATES[product]
    n_own = int(rng.integers(1, 4))
    picks = [_with_time(rng, str(t)) for t in rng.choice(own, size=n_own, replace=False)]
    if rng.random() < AMBIGUOUS_RATE:
        kind = str(rng.choice(AMBIGUOUS_BY_PRODUCT[product]))
        picks[0] = str(rng.choice(AMBIGUOUS[kind]))
    n_generic = int(rng.choice([0, 1, 1, 2]))
    if n_generic:
        picks.extend(str(g) for g in rng.choice(GENERIC, size=n_generic, replace=False))
    return " ".join(picks)


def _near_duplicate(rng: np.random.Generator, text: str) -> str:
    # Append one generic sentence; shingle overlap stays high.
    extra = str(rng.choice(GENERIC))
    return f"{text} {extra}"


def generate_fixture(rows: int = 4000, seed: int = 42) -> pd.DataFrame:
    """Return a synthetic CFPB-schema frame with ``rows`` records over 24 months."""
    if rows < 50:
        raise ValueError("rows must be at least 50 so every product class is present")
    rng = np.random.default_rng(seed)
    products = list(PRODUCTS)
    weights = np.asarray(PRODUCT_WEIGHTS, dtype=float)
    weights = weights / weights.sum()

    records: list[dict[str, object]] = []
    for i in range(rows):
        month = int(rng.integers(0, MONTHS))
        product = str(rng.choice(products, p=weights))
        spec = PRODUCTS[product]
        issue = str(rng.choice(spec["issue"]))
        # Volume bump for one theme in the last months.
        if month >= BUMP_START_MONTH and rng.random() < BUMP_EXTRA_RATE:
            product, issue = BUMP_PRODUCT, BUMP_ISSUE
            spec = PRODUCTS[product]
        sub_product = str(rng.choice(spec["sub_product"]))
        narrative = "" if rng.random() < EMPTY_RATE else _narrative(rng, product)
        records.append(
            {
                "complaint_id": f"SYN-CMP-{i + 1:06d}",
                "date_received": _random_date(rng, month).isoformat(),
                "product": product,
                "sub_product": sub_product,
                "issue": issue,
                "sub_issue": f"{issue} (general)",
                "company": str(rng.choice(COMPANIES)),
                "state": str(rng.choice(STATES)),
                "submitted_via": str(rng.choice(SUBMITTED_VIA, p=[0.75, 0.12, 0.08, 0.05])),
                "company_response": str(rng.choice(COMPANY_RESPONSE, p=[0.7, 0.15, 0.1, 0.05])),
                "timely": "Yes" if rng.random() < 0.95 else "No",
                "consumer_disputed": str(rng.choice(["N/A", "No", "Yes"], p=[0.7, 0.2, 0.1])),
                NARRATIVE: narrative,
            }
        )
    frame = pd.DataFrame.from_records(records)

    # Inject exact and near duplicates. A target row copies the narrative of another
    # row with the SAME product, so duplication never contradicts the label.
    non_empty = frame.index[frame[NARRATIVE] != ""].to_numpy()
    n_exact = int(round(EXACT_DUP_RATE * rows))
    n_near = int(round(NEAR_DUP_RATE * rows))
    targets = rng.choice(non_empty, size=n_exact + n_near, replace=False)
    by_product = {p: frame.index[(frame["product"] == p) & (frame[NARRATIVE] != "")].to_numpy() for p in products}
    for k, dst in enumerate(targets):
        pool = by_product[str(frame.at[dst, "product"])]
        pool = pool[pool != dst]
        if len(pool) == 0:
            continue
        src = int(rng.choice(pool))
        text = str(frame.at[src, NARRATIVE])
        frame.at[dst, NARRATIVE] = text if k < n_exact else _near_duplicate(rng, text)

    frame = frame.sort_values("date_received", kind="stable").reset_index(drop=True)
    frame["complaint_id"] = [f"SYN-CMP-{i + 1:06d}" for i in range(len(frame))]
    return frame[CFPB_COLUMNS + [NARRATIVE]]


def fixture_card(seed: int, rows: int) -> str:
    return (
        "# FIXTURE.md\n\n"
        "Authored synthetic demonstration data. Fictional identifiers. Not derived from any "
        "person, company, account or dataset. Generated by `tools/make_fixture.py` seed "
        f"`{seed}`, version {FIXTURE_VERSION}.\n\n"
        f"- File: `complaints_synthetic.csv` ({rows} rows, {MONTHS} months from {START_DATE.isoformat()})\n"
        "- Schema: CFPB Consumer Complaint Database column names (see DATA_CARD.md)\n"
        "- Identifiers: `SYN-CMP-000001` style, never real complaint IDs\n"
        "- Companies: obviously fictional demo names such as `Demo Bank One`\n"
        "- Narratives: assembled from generic phrase templates, no personal data\n"
        f"- Deliberate properties: about {int(EMPTY_RATE * 100)}% empty narratives, about "
        f"{int(EXACT_DUP_RATE * 100)}% exact and {int(NEAR_DUP_RATE * 100)}% near-duplicate narratives, "
        f"and a volume increase for `{BUMP_PRODUCT} / {BUMP_ISSUE}` from month {BUMP_START_MONTH + 1}\n"
        "- Class balance is skewed on purpose so that imbalance handling can be demonstrated\n\n"
        "Regenerate: `python tools/make_fixture.py --rows 4000 --seed 42`\n"
    )


def write_fixture(frame: pd.DataFrame, out_dir: Path, seed: int) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "complaints_synthetic.csv"
    card_path = out_dir / "FIXTURE.md"
    frame.to_csv(csv_path, index=False, lineterminator="\n")
    card_path.write_text(fixture_card(seed, len(frame)), encoding="utf-8")
    return {"csv": csv_path, "card": card_path}
