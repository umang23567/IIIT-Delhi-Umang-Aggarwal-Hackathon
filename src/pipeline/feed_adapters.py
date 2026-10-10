import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import feedparser
import requests
from src.pipeline.cleaner import RawArticle

logger = logging.getLogger(__name__)

class FeedAdapter:
    """Base interface for all news/social feeds."""
    def fetch_articles(self) -> List[RawArticle]:
        raise NotImplementedError

class ReplayAdapter(FeedAdapter):
    """Deterministic scenario replayer from curated dataset."""
    def __init__(self, filepath: str = "data/demo/replay_feed.json"):
        self.filepath = filepath

    def fetch_articles(self) -> List[RawArticle]:
        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        articles = []
        for item in data:
            articles.append(RawArticle(
                article_id=item["feed_id"],
                source_type=item["source_type"],
                source_name=item["source_name"],
                url=item["url"],
                title=item["title"],
                body=item["raw_text"],
                published_at=item["published_at"]
            ))
        return articles

class RSSFeedAdapter(FeedAdapter):
    """Live RSS adapter polling financial news feeds."""
    DEFAULT_FEEDS = [
        ("CNBC Markets", "https://search.cnbc.com/rs/search/view.html?partnerId=2000&keywords=markets&sort=date"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex"),
        ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories")
    ]

    def __init__(self, feed_urls: Optional[List[tuple]] = None):
        self.feeds = feed_urls or self.DEFAULT_FEEDS

    def fetch_articles(self) -> List[RawArticle]:
        articles = []
        for name, url in self.feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:8]: # cap to 8 latest per source
                    article_id = f"RSS_{abs(hash(entry.link)) % 10000000}"
                    title = entry.get("title", "")
                    body = entry.get("summary", "") or entry.get("description", "")
                    pub = entry.get("published", "") or datetime.utcnow().isoformat() + "Z"
                    articles.append(RawArticle(
                        article_id=article_id,
                        source_type="RSS Financial News",
                        source_name=name,
                        url=entry.get("link", ""),
                        title=title,
                        body=body,
                        published_at=pub
                    ))
            except Exception as e:
                logger.warning(f"Failed to fetch RSS from {name}: {e}")
        return articles
