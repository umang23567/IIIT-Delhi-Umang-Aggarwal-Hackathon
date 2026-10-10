import re
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel

class EventClassificationResult(BaseModel):
    event_type: str
    confidence: float
    evidence_text: str
    all_matched_events: List[str]

class EventTaxonomyClassifier:
    """
    Classifies financial text into structured risk event taxonomy classes:
    - Credit Event
    - Geopolitical
    - Macroeconomic
    - Regulatory/Legal
    - M&A
    - Product Launch
    - Cybersecurity
    - Supply-Chain Disruption
    Extracts evidence text spans and confidence metrics.
    """
    TAXONOMY_RULES = {
        "Credit Event": [
            r"\bdowngrade[ds]?\b", r"\bdefault[s]?\b", r"\bdebt\b", r"\bliquidity\b",
            r"\bjunk bond[s]?\b", r"\brating cut\b", r"\binsolven[ct]\b", r"\bcredit spread\b",
            r"\bbankrupt(cy)?\b", r"\bdistressed debt\b", r"\bcash burn\b"
        ],
        "Geopolitical": [
            r"\bwar\b", r"\bconflict\b", r"\bsanction[s]?\b", r"\bmilitary\b",
            r"\bdrone strike[s]?\b", r"\bblockade\b", r"\bchokepoint[s]?\b", r"\bmaritime\b",
            r"\bgeopolitical\b", r"\bmissile[s]?\b", r"\btariff[s]?\b"
        ],
        "Macroeconomic": [
            r"\binflation\b", r"\binterest rate[s]?\b", r"\bfederal reserve\b", r"\bfed\b",
            r"\bhike[s]?\b", r"\brate cut[s]?\b", r"\bcpi\b", r"\bgdp\b",
            r"\byield curve\b", r"\bcentral bank\b", r"\bunemployment\b"
        ],
        "Regulatory/Legal": [
            r"\bregulat(ory|ors?)\b", r"\bfine[s]?\b", r"\bpenalt(y|ies)\b", r"\bconsent order\b",
            r"\blawsuit[s]?\b", r"\bsec\b", r"\bdpj\b", r"\bdoj\b", r"\bantitrust\b",
            r"\bcompliance\b", r"\baudit[s]?\b"
        ],
        "M&A": [
            r"\bacqui(re[ds]?|sition)\b", r"\bmerger[s]?\b", r"\btakeover\b", r"\bbuyout\b",
            r"\bhostile bid\b", r"\bdivestiture\b", r"\bspin-?off\b"
        ],
        "Cybersecurity": [
            r"\bcyber(attack|security)?\b", r"\bransomware\b", r"\bdata breach\b",
            r"\bhack(ed|er)?\b", r"\bvulnerability\b", r"\bmalware\b"
        ],
        "Supply-Chain Disruption": [
            r"\bsupply chain\b", r"\bbottleneck[s]?\b", r"\bshortage[s]?\b", r"\bstrike[s]?\b",
            r"\bmanufacturing halt\b", r"\bfactory shutdown\b", r"\blogistics\b"
        ],
        "Product Launch": [
            r"\bunveil[s]?\b", r"\blaunch(ed|es)?\b", r"\bnext-gen\b", r"\bchip[s]?\b",
            r"\barchitecture\b", r"\bflagship\b", r"\bproduct\b", r"\bcolor options\b"
        ]
    }

    def classify(self, text: str) -> EventClassificationResult:
        lowered = text.lower()
        event_scores: Dict[str, float] = {}
        event_evidence: Dict[str, str] = {}

        for event, patterns in self.TAXONOMY_RULES.items():
            matches = []
            for pattern in patterns:
                m = re.search(pattern, lowered)
                if m:
                    matches.append(m.group(0))
            if matches:
                # Score based on number and prominence of matched keywords
                event_scores[event] = min(0.99, 0.60 + 0.10 * len(matches))
                
                # Extract snippet around first match
                first_match = re.search(rf"[^.?!]*\b{re.escape(matches[0])}\b[^.?!]*[.?!]", text, re.IGNORECASE)
                evidence = first_match.group(0).strip() if first_match else text[:120]
                event_evidence[event] = evidence

        if not event_scores:
            return EventClassificationResult(
                event_type="General Financial News",
                confidence=0.50,
                evidence_text=text[:100],
                all_matched_events=[]
            )

        # Sort by highest score
        sorted_events = sorted(event_scores.items(), key=lambda x: x[1], reverse=True)
        top_event, top_score = sorted_events[0]

        return EventClassificationResult(
            event_type=top_event,
            confidence=round(top_score, 4),
            evidence_text=event_evidence[top_event],
            all_matched_events=[e[0] for e in sorted_events]
        )
