import pandas as pd
import logging
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ai_core.pipelines.data_processing.nodes import detect_languages

logging.basicConfig(level=logging.INFO)


def test_language_detection():
    # Sample data
    data = pd.DataFrame(
        {
            "Interaction_ID": ["1", "2", "3"],
            "Interaction_Payload": [
                "Hello, I love this product!",
                "Hola, me encanta este producto!",
                "Bonjour, j'aime ce produit!",
            ],
        }
    )

    config = {
        "language_detection": {
            "enabled": True,
            "model_name": "papluca/xlm-roberta-base-language-detection",
            "batch_size": 2,
            "fallback_lang": "en",
        }
    }

    print("Testing with Transformer model...")
    result = detect_languages(data.copy(), config)
    print("\nResults with Transformer:")
    print(result[["Interaction_Payload", "detected_language", "language_score"]])

    # Test fallback
    config_fallback = {
        "language_detection": {
            "enabled": True,
            "model_name": None,  # Force fallback
            "fallback_lang": "en",
        }
    }

    print("\nTesting with fallback (langdetect)...")
    result_fallback = detect_languages(data.copy(), config_fallback)
    print("\nResults with Fallback:")
    print(
        result_fallback[["Interaction_Payload", "detected_language", "language_score"]]
    )


if __name__ == "__main__":
    test_language_detection()
