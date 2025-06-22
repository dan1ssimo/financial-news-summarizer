import os
from datetime import datetime

import feedparser
import pandas as pd
from newspaper import Article
from newspaper.exceptions import ArticleException
from tqdm import tqdm


def fetch_article_text(url: str) -> str:
    """
    Fetch article text from url
    Args:
        url: str
    Returns:
        str: article text
    """
    a = Article(url)
    try:
        a.download()
        a.parse()
    except ArticleException as e:
        print(f"Error downloading or parsing article: {e}")
        return None
    return a.text


def get_yahoo_news_rss(symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Get Yahoo News RSS feed
    Args:
        symbol: str
        start: str
        end: str
    Returns:
        pd.DataFrame: news dataframe
    """
    url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
    feed = feedparser.parse(url)
    news = []
    for entry in tqdm(feed.entries):
        pub_date = datetime(*entry.published_parsed[:6])
        if start and pub_date < datetime.fromisoformat(start):
            continue
        if end and pub_date > datetime.fromisoformat(end):
            continue

        extracted_text = fetch_article_text(entry.link)
        if not extracted_text or len(extracted_text) < 1900:
            extracted_text = entry.summary

        source = entry.link.split("/")[2]

        news.append(
            {
                "source": source,
                "link": entry.link,
                "extracted_text": extracted_text,
                "published": pub_date.strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

    df = pd.DataFrame(news)
    if os.path.exists("yahoo_news_tsla.csv"):
        df.to_csv("data/yahoo_news_tsla.csv", index=False, mode="a", header=False)
    else:
        df.to_csv("data/yahoo_news_tsla.csv", index=False)
    print(f"Saved {len(df)} news to yahoo_news.csv")


if __name__ == "__main__":
    if os.path.exists("data/yahoo_news_tsla.csv"):
        start_date = pd.read_csv("data/yahoo_news_tsla.csv")["published"].max()
        news = get_yahoo_news_rss("TSLA", start=start_date)
    else:
        news = get_yahoo_news_rss("TSLA")
