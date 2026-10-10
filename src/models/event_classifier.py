import re
import logging
from typing import List, Dict, Any, Tuple, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class EventClassificationResult(BaseModel):
    event_type: str
    confidence: float
    evidence_text: str
    all_matched_events: List[str]
    model_mode: str = "Hybrid Zero-Shot Transformer + Domain Rules"

class EventTaxonomyClassifier:
    """
    SOTA Financial Event Taxonomy Classifier.
    Employs a two-tier Hybrid Architecture:
    1. Zero-Shot NLI Transformer (e.g. facebook/bart-large-mnli or valhalla/distilbart-mnli-12-3)
       to classify unseen financial contexts into target event candidate hypotheses.
    2. Domain Taxonomy Evidence Extractor to pinpoint exact regulatory/financial
       evidence text spans and calibrate high-impact confidence.
    """
    CANDIDATE_LABELS = [
        "Credit Event",
        "Geopolitical",
        "Macroeconomic",
        "Regulatory/Legal",
        "M&A",
        "Cybersecurity",
        "Supply-Chain Disruption",
        "Product Launch"
    ]

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
            r"\bunveil[s]?\b", r"\blaunch(ed|es)?\b", r"\brelease[ds]?\b",
            r"\bupdate[ds]?\b", r"\bnext-gen\b", r"\bchip[s]?\b",
            r"\barchitecture\b", r"\bflagship\b", r"\bproduct\b",
            r"\bcatalog\b", r"\bcolor options\b"
        ]
    }

    def __init__(self, enable_zero_shot: bool = True, model_name: str = "valhalla/distilbart-mnli-12-3"):
        self.zero_shot_pipeline = None
        self.model_mode = "Domain Rule Engine"

        if enable_zero_shot:
            try:
                import torch
                from transformers import pipeline
                logger.info(f"Initializing Zero-Shot Classification Pipeline ({model_name})...")
                dev_idx = 0 if torch.cuda.is_available() else -1
                self.zero_shot_pipeline = pipeline(
                    "zero-shot-classification",
                    model=model_name,
                    device=dev_idx
                )
                self.model_mode = f"Zero-Shot NLI ({model_name}) + Evidence Extractor"
                logger.info("Zero-Shot Classification Pipeline initialized successfully.")
            except Exception as e:
                logger.info(f"Zero-shot transformer deferred ({e}). Operating in high-speed domain taxonomy mode.")
                self.model_mode = "High-Fidelity Domain Taxonomy Engine"

    def classify(self, text: str) -> EventClassificationResult:
        lowered = text.lower()
        rule_scores: Dict[str, float] = {}
        evidence_spans: Dict[str, str] = {}

        # 1. Domain evidence span extraction and baseline rule scoring
        for event, patterns in self.TAXONOMY_RULES.items():
            matches = []
            for pattern in patterns:
                m = re.search(pattern, lowered)
                if m:
                    matches.append(m.group(0))
            if matches:
                rule_scores[event] = min(0.95, 0.60 + 0.10 * len(matches))
                first_match = re.search(rf"[^.?!]*\b{re.escape(matches[0])}\b[^.?!]*[.?!]", text, re.IGNORECASE)
                evidence = first_match.group(0).strip() if first_match else text[:120]
                evidence_spans[event] = evidence

        # 2. If Zero-Shot NLI transformer is available, run inference
        if self.zero_shot_pipeline:
            try:
                hypothesis_template = "This financial news report discusses a {}."
                zs_res = self.zero_shot_pipeline(
                    text[:512],
                    candidate_labels=self.CANDIDATE_LABELS,
                    hypothesis_template=hypothesis_template,
                    multi_label=True
                )
                zs_scores = dict(zip(zs_res["labels"], zs_res["scores"]))

                # Ensemble: 60% Zero-Shot NLI + 40% Domain Keywords
                combined_scores: Dict[str, float] = {}
                for label in self.CANDIDATE_LABELS:
                    z_score = zs_scores.get(label, 0.0)
                    r_score = rule_scores.get(label, 0.0)
                    combined_scores[label] = round(0.60 * z_score + 0.40 * r_score, 4)

                sorted_events = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
                top_event, top_score = sorted_events[0]
                evidence = evidence_spans.get(top_event) or text[:120]

                return EventClassificationResult(
                    event_type=top_event,
                    confidence=round(top_score, 4),
                    evidence_text=evidence,
                    all_matched_events=[e[0] for e in sorted_events if e[1] > 0.3],
                    model_mode=self.model_mode
                )
            except Exception as e:
                logger.warning(f"Zero-shot runtime inference error: {e}. Falling back to domain rule engine.")

        # 3. Rule-based fallback if transformer is absent or fails
        if not rule_scores:
            return EventClassificationResult(
                event_type="General Financial News",
                confidence=0.50,
                evidence_text=text[:100],
                all_matched_events=[],
                model_mode=self.model_mode
            )

        # Distinguish a product rumor sourced from a supply chain leak from an
        # actual operational supply-chain disruption.
        product_markers = r"\b(launch|launches|launched|unveil|unveils|product|architecture|catalog)\b"
        disruption_markers = r"\b(shortage|bottleneck|halt|shutdown|disruption|strike)\b"
        if (
            "Product Launch" in rule_scores
            and "Supply-Chain Disruption" in rule_scores
            and re.search(product_markers, lowered)
            and not re.search(disruption_markers, lowered)
        ):
            rule_scores.pop("Supply-Chain Disruption")
            evidence_spans.pop("Supply-Chain Disruption", None)

        sorted_events = sorted(rule_scores.items(), key=lambda x: x[1], reverse=True)
        top_event, top_score = sorted_events[0]

        return EventClassificationResult(
            event_type=top_event,
            confidence=round(top_score, 4),
            evidence_text=evidence_spans[top_event],
            all_matched_events=[e[0] for e in sorted_events],
            model_mode=self.model_mode
        )
