from amazon_sentiment_project.utils.reporting import save_report
from transformers import pipeline
import pandas as pd

# modelo de sentimiento
sentiment_model = pipeline("sentiment-analysis")

def analyze_sentiment(data: pd.DataFrame) -> pd.DataFrame:
    sentiments = []

    for text in data["translated_text"]:
        try:
            result = sentiment_model(text)
            sentiments.append(result[0]["label"])
        except:
            sentiments.append("UNKNOWN")

    data["sentiment"] = sentiments
    return data

# # #
def analyze_sentiment(data):
    sentiments = []
    positive = 0
    negative = 0

    for text in data["translated_text"]:
        try:
            result = sentiment_model(text)
            label = result[0]["label"]
            sentiments.append(label)

            if label == "POSITIVE":
                positive += 1
            else:
                negative += 1

        except:
            sentiments.append("UNKNOWN")

    data["sentiment"] = sentiments

    # 📊 REPORTE
    report = f"""
    SENTIMENT REPORT
    ----------------------
    Positivos: {positive}
    Negativos: {negative}
    Total: {len(data)}
    """

    save_report("sentiment_report", report)

    return data