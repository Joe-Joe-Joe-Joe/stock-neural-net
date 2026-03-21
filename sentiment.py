import json
import os

from transformers import pipeline

# Example usage:
# analyze_json(
#         "news_Tesla_2025_01_01.json",
#         "news_Tesla_2025_01_01_with_sentiment.json"
#     )

sentiment = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def convert_score(label, confidence):
    """
    Convert sentiment label and confidence to a numerical score.

    Args:
        label (str): The sentiment label ('positive', 'negative', 'neutral').
        confidence (float): The confidence score from the model.

    Returns:
        float: A score between 1 and 10.
    """
    label = label.lower()

    if label == "negative":
        return (1 - confidence) * 5
    elif label == "neutral":
        return 5
    elif label == "positive":
        return 5 + (confidence * 5)
    else:
        return 5


def get_sentiment_score(text):
    """
    Analyze the sentiment of a text by chunking it if necessary and averaging scores.

    Args:
        text (str): The text to analyze.

    Returns:
        float or None: The average sentiment score, or None if no chunks.
    """
    max_len = 512
    chunks = [text[i:i+max_len] for i in range(0, len(text), max_len)]

    scores = []
    for chunk in chunks:
        result = sentiment(chunk)[0]
        score = convert_score(result["label"], result["score"])
        scores.append(score)

    return sum(scores) / len(scores) if scores else None


def analyze_json(input_file, output_file):
    """
    Analyze sentiment in a JSON file containing news articles and add sentiment scores.

    Args:
        input_file (str): Path to the input JSON file.
        output_file (str): Path to the output JSON file with added sentiment data.
    """
    
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for date, articles in data.items():
        for article in articles:

            title = article.get("title", "")
            if title:
                result = sentiment(title[:512])[0]
                title_score = convert_score(
                    result["label"], result["score"]
                )
                article["title_sentiment"] = title_score
            else:
                title_score = None
                article["title_sentiment"] = None

            text = article.get("text", "")
            if text:
                text_score = get_sentiment_score(text)
                article["text_sentiment"] = text_score
            else:
                text_score = None
                article["text_sentiment"] = None

            if title_score is not None and text_score is not None:
                combined = 0.7 * title_score + 0.3 * text_score
            elif title_score is not None:
                combined = title_score
            elif text_score is not None:
                combined = text_score
            else:
                combined = None

            article["combined_sentiment"] = combined

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved: {output_file}")

def preprocess_news():
    for file in os.listdir("scraped_news"):
        analyze_json(r"scraped_news/"+file, "scraped_news_sentiment/"+file)
        print(f"finished {file}!")

if __name__ == "__main__":
    preprocess_news()

