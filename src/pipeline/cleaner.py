import re
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class RawArticle(BaseModel):
    article_id: str
    source_type: str
    source_name: str
    url: str
    title: str
    body: str
    published_at: str
    fetched_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    content_hash: str = ""

    def compute_hash(self) -> str:
        content = f"{self.title.strip().lower()}|{self.body.strip().lower()}"
        return hashlib.md5(content.encode("utf-8")).hexdigest()

class CleanArticle(BaseModel):
    article_id: str
    cleaned_title: str
    cleaned_body: str
    cleaned_text: str
    duplicate_group_id: str
    is_duplicate: bool = False
    source_type: str
    source_name: str
    url: str
    published_at: str

class ArticleCleaner:
    """
    Cleans raw HTML/text, standardizes whitespace and timestamps,
    and performs exact MD5 and near-duplicate title clustering.
    """
    def __init__(self):
        self.seen_hashes: Dict[str, str] = {} # hash -> primary article_id
        self.seen_titles: Dict[str, str] = {} # normalized title token set -> primary article_id

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # Remove HTML tags
        clean = re.sub(r"<[^>]+>", " ", text)
        # Remove URLs
        clean = re.sub(r"http\S+|www\.\S+", " ", clean)
        # Normalize whitespace
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    def process(self, raw: RawArticle) -> CleanArticle:
        cleaned_title = self.clean_text(raw.title)
        cleaned_body = self.clean_text(raw.body)
        combined_text = f"{cleaned_title}. {cleaned_body}" if cleaned_body else cleaned_title

        # Compute content hash
        content_hash = raw.content_hash or raw.compute_hash()

        # Check exact duplicate
        is_duplicate = False
        duplicate_group_id = raw.article_id

        if content_hash in self.seen_hashes:
            is_duplicate = True
            duplicate_group_id = self.seen_hashes[content_hash]
        else:
            self.seen_hashes[content_hash] = raw.article_id

            # Simple near-duplicate check based on normalized title tokens
            title_tokens = frozenset(re.findall(r"\w{4,}", cleaned_title.lower()))
            for existing_tokens, parent_id in self.seen_titles.items():
                if len(title_tokens) > 3 and len(existing_tokens) > 3:
                    overlap = len(title_tokens.intersection(existing_tokens)) / max(len(title_tokens), len(existing_tokens))
                    if overlap > 0.75:
                        is_duplicate = True
                        duplicate_group_id = parent_id
                        break
            if not is_duplicate:
                self.seen_titles[title_tokens] = raw.article_id

        return CleanArticle(
            article_id=raw.article_id,
            cleaned_title=cleaned_title,
            cleaned_body=cleaned_body,
            cleaned_text=combined_text,
            duplicate_group_id=duplicate_group_id,
            is_duplicate=is_duplicate,
            source_type=raw.source_type,
            source_name=raw.source_name,
            url=raw.url,
            published_at=raw.published_at
        )
