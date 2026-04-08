from transformers import pipeline
import pandas as pd
from amazon_sentiment_project.utils.reporting import save_report

# Cargar modelo de traducción
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

def translate_text(data: pd.DataFrame) -> pd.DataFrame:
    """
    Traduce textos al inglés
    """
    translated_texts = []

    for text in data["review_text"]:
        try:
            result = translator(text)
            translated_texts.append(result[0]["translation_text"])
        except Exception as e:
            # Manejo de errores
            translated_texts.append("ERROR")

    data["translated_text"] = translated_texts
    return data

# Reporte
def translate_text(data, params):
    translated = []
    errors = 0

    for text in data["review_text"]:
        try:
            result = translator(text)
            translated.append(result[0]["translation_text"])
        except:
            translated.append(params["translation"]["error_value"])
            errors += 1

    data["translated_text"] = translated

    # 📊 REPORTE
    report = f"""
    TRANSLATION REPORT
    ----------------------
    Total textos: {len(data)}
    Errores: {errors}
    Exitos: {len(data) - errors}
    """

    save_report("translation_report", report)

    return data