import logging
from typing import Dict, Any, Tuple
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class SentimentResult(BaseModel):
    positive_prob: float
    neutral_prob: float
    negative_prob: float
    sentiment_score: float # scalar s = P(pos) - P(neg) in [-1.0, 1.0]
    sentiment_label: str   # POSITIVE, NEUTRAL, NEGATIVE
    confidence: float
    model_version: str

class FinBERTSentimentEngine:
    """
    SOTA Financial Sentiment Engine using ProsusAI/finbert.
    Provides probability vector [P(pos), P(neu), P(neg)] and scalar score s = P(pos) - P(neg).
    Includes Loughran-McDonald fallback if weights or device is offline.
    """
    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str = "auto", enable_transformer: bool = True):
        self.device = device
        self.pipeline = None
        self.is_loaded = False
        self.model_version = "ProsusAI/finbert-v1.0"

        if enable_transformer:
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                
                logger.info(f"Loading {self.MODEL_NAME} tokenizer and weights...")
                tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
                model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
                
                dev_idx = 0 if torch.cuda.is_available() else (-1)
                self.pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True, device=dev_idx)
                self.is_loaded = True
                logger.info("FinBERT successfully loaded into memory.")
            except Exception as e:
                logger.warning(f"Transformer model initialization deferred or encountered issue: {e}. Active fallback ready.")
                self.model_version = "LoughranMcDonald-Lexicon-v2.0"

    def analyze(self, text: str) -> SentimentResult:
        if self.is_loaded and self.pipeline:
            try:
                # Truncate text to 512 tokens
                results = self.pipeline(text[:1500])[0]
                # FinBERT returns labels: positive, negative, neutral
                prob_map = {item["label"].lower(): item["score"] for item in results}
                p_pos = float(prob_map.get("positive", 0.0))
                p_neg = float(prob_map.get("negative", 0.0))
                p_neu = float(prob_map.get("neutral", 0.0))

                s_score = round(p_pos - p_neg, 4)
                
                # Determine label
                if p_pos > p_neg and p_pos > p_neu:
                    label = "POSITIVE"
                    conf = p_pos
                elif p_neg > p_pos and p_neg > p_neu:
                    label = "NEGATIVE"
                    conf = p_neg
                else:
                    label = "NEUTRAL"
                    conf = p_neu

                return SentimentResult(
                    positive_prob=round(p_pos, 4),
                    neutral_prob=round(p_neu, 4),
                    negative_prob=round(p_neg, 4),
                    sentiment_score=s_score,
                    sentiment_label=label,
                    confidence=round(conf, 4),
                    model_version=self.model_version
                )
            except Exception as e:
                logger.error(f"Error during FinBERT inference: {e}. Falling back to domain lexicon.")

        # Domain-adapted Financial Lexicon Fallback (Loughran-McDonald baseline)
        return self._lexicon_fallback(text)

    def _lexicon_fallback(self, text: str) -> SentimentResult:
        import re
        tokens = re.findall(r"\w+", text.lower())
        
        # Domain positive words
        pos_words = {
            "growth", "record", "profit", "surpassed", "outperformed", "beat", "surged",
            "upgrade", "breakthrough", "strong", "resilient", "expansion", "positive",
            "dividend", "higher", "gain", "upside", "soared", "robust", "innovation"
        }
        # Domain negative words
        neg_words = {
            "downgrade", "downgraded", "loss", "plunge", "dropped", "slump", "crisis",
            "default", "insolvent", "bankruptcy", "liquidity", "penalties", "fine",
            "investigation", "recession", "strike", "halt", "warning", "burn", "turmoil",
            "deficit", "debt", "selloff", "weak", "cut", "lawsuit", "inflation", "war"
        }

        pos_count = sum(1 for w in tokens if w in pos_words)
        neg_count = sum(1 for w in tokens if w in neg_words)
        total = pos_count + neg_count

        if total == 0:
            return SentimentResult(
                positive_prob=0.15,
                neutral_prob=0.70,
                negative_prob=0.15,
                sentiment_score=0.0,
                sentiment_label="NEUTRAL",
                confidence=0.70,
                model_version=self.model_version
            )

        p_pos = pos_count / (total + 2.0)
        p_neg = neg_count / (total + 2.0)
        p_neu = 2.0 / (total + 2.0)
        s_score = round(p_pos - p_neg, 4)

        if p_pos > p_neg and s_score > 0.15:
            label = "POSITIVE"
            conf = p_pos
        elif p_neg > p_pos and s_score < -0.15:
            label = "NEGATIVE"
            conf = p_neg
        else:
            label = "NEUTRAL"
            conf = p_neu

        return SentimentResult(
            positive_prob=round(p_pos, 4),
            neutral_prob=round(p_neu, 4),
            negative_prob=round(p_neg, 4),
            sentiment_score=s_score,
            sentiment_label=label,
            confidence=round(conf, 4),
            model_version=self.model_version
        )
