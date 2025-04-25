import pandas as pd
import pytest

from src.data.sentiment_data import SocialSentimentFetcher


def test_social_sentiment_fetcher_returns_hourly_data():
    start = pd.Timestamp("2025-02-01 00:00:00")
    end = pd.Timestamp("2025-02-01 03:00:00")
    fetcher = SocialSentimentFetcher(twitter_client=None, reddit_client=None)
    df = fetcher.fetch(start, end)
    assert isinstance(df, pd.DataFrame)
    expected_columns = [
        "timestamp",
        "tweet_volume",
        "retweet_count",
        "like_count",
        "sentiment_score",
    ]
    assert list(df.columns) == expected_columns
    expected_index = pd.date_range(start=start, end=end, freq="1H")
    assert len(df) == len(expected_index)
    pd.testing.assert_series_equal(
        df["timestamp"], pd.Series(expected_index), check_names=False
    )
