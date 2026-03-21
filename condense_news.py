import pandas as pd
import json
import os

rows = []
for file in os.listdir("scraped_news_sentiment"):
    with open(f"scraped_news_sentiment/{file}", encoding='utf-8') as f:
        data = json.load(f)
    for date, articles in data.items():
        for article in articles:
            rows.append({
                "date": date,
                "company": article.get("company"),
                "combined_sentiment": article.get("combined_sentiment"),
            })

df = pd.DataFrame(rows)
with open("sentiment_condensed.json", "w", encoding="utf-8") as f:
    df.to_json(f, index=False)