import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class SocialSentimentFetcher:
    """Fetch and aggregate sentiment metrics from Twitter and Reddit."""

    def __init__(self, twitter_client, reddit_client):
        self.twitter_client = twitter_client
        self.reddit_client = reddit_client
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def fetch(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Returns DataFrame of aggregated sentiment metrics per hour between start and end."""
        # Crear intervalo horario
        time_index = pd.date_range(start=start, end=end, freq="1H")
        records = []
        for current_start in time_index:
            window_end = current_start + pd.Timedelta(hours=1)
            # Inicializar métricas
            tweet_volume = 0
            retweet_count = 0
            like_count = 0
            sentiment_scores = []
            # Procesar Twitter
            if self.twitter_client:
                try:
                    tweets = self.twitter_client.search(
                        query="bitcoin", since=current_start, until=window_end
                    )
                except Exception:
                    tweets = []
                for tweet in tweets:
                    tweet_volume += 1
                    retweet_count += getattr(tweet, "retweet_count", 0)
                    like_count += getattr(tweet, "favorite_count", 0)
                    text = getattr(tweet, "text", "")
                    score = self.sentiment_analyzer.polarity_scores(text)["compound"]
                    sentiment_scores.append(score)
            # Procesar Reddit
            if self.reddit_client:
                try:
                    posts = self.reddit_client.search(
                        query="bitcoin", start=current_start, end=window_end
                    )
                except Exception:
                    posts = []
                for post in posts:
                    text = " ".join(
                        [getattr(post, "title", ""), getattr(post, "selftext", "")]
                    )
                    score = self.sentiment_analyzer.polarity_scores(text)["compound"]
                    sentiment_scores.append(score)
            # Promedio de sentimiento
            sentiment_score = (
                float(np.mean(sentiment_scores)) if sentiment_scores else 0.0
            )
            records.append(
                {
                    "timestamp": current_start,
                    "tweet_volume": tweet_volume,
                    "retweet_count": retweet_count,
                    "like_count": like_count,
                    "sentiment_score": sentiment_score,
                }
            )
        return pd.DataFrame(records)
