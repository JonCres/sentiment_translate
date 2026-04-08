from amazon_sentiment_project.utils.reporting import save_report
import pandas as pd

def validate_translation(data: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Valida la calidad de la traducción
    """

    def is_valid(text):
        if text == params["translation"]["error_value"]:
            return False
        if len(text) < params["validation"]["min_length"]:
            return False
        return True

    data["is_valid_translation"] = data["translated_text"].apply(is_valid)

    return data



# # #
def validate_translation(data, params):
    valid = 0
    invalid = 0

    def check(text):
        if text == params["translation"]["error_value"]:
            return False
        if len(text) < params["validation"]["min_length"]:
            return False
        return True

    results = []

    for text in data["translated_text"]:
        is_valid = check(text)
        results.append(is_valid)

        if is_valid:
            valid += 1
        else:
            invalid += 1

    data["is_valid_translation"] = results

    # 📊 REPORTE
    report = f"""
    VALIDATION REPORT
    ----------------------
    Validos: {valid}
    Invalidos: {invalid}
    """

    save_report("validation_report", report)

    return data