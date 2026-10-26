from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"


@dataclass(frozen=True)
class Finding:
    claim: str
    verdict: Verdict
    rule_citation: str
    explanation: str
    evidence: dict = field(default_factory=dict)
    confidence: float = 1.0
