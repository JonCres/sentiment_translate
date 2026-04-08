# ============================================================
# main.py  — Punto de entrada del proyecto
# ============================================================
#
# Ejecutar con:
#   python main.py
#
# ¿Qué hace?
#   1. Configura el logging para ver mensajes en consola
#   2. Crea un DataFrame de ejemplo (simulando Amazon reviews)
#   3. Ejecuta el pipeline de traducción
#   4. Valida la calidad de las traducciones
#   5. Muestra el reporte final
#
# En producción, reemplazas el DataFrame de ejemplo por:
#   df = pd.read_csv("data/raw/amazon_reviews.csv")
# ============================================================

import logging
import pandas as pd
import sys
import os

# ── Configurar logging (mensajes en consola) ─────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),              # Mostrar en consola
        logging.FileHandler("translation.log", mode="w")  # Y guardar en archivo
    ]
)

logger = logging.getLogger(__name__)


def crear_dataset_ejemplo() -> pd.DataFrame:
    """
    Crea un pequeño DataFrame de ejemplo para probar el pipeline.

    En tu proyecto real reemplaza esto con:
        df = pd.read_csv("data/raw/amazon_reviews.csv")
    """
    datos = {
        "review_id": [1, 2, 3, 4, 5, 6],
        "review_body": [
            "Este producto es excelente, lo recomiendo a todos.",    # Español
            "Ce produit est vraiment bien, je suis satisfait.",       # Francés
            "Dieses Produkt ist sehr gut und ich bin zufrieden.",     # Alemán
            "This product is amazing and works perfectly!",           # Inglés (ya OK)
            "このproductは非常に良いです、とても満足しています。",            # Japonés
            "Este produto é ótimo, chegou rápido e funciona bem.",    # Portugués
        ],
        "star_rating": [5, 4, 5, 5, 4, 5],
        "marketplace":  ["ES", "FR", "DE", "US", "JP", "BR"],
    }
    return pd.DataFrame(datos)


def main():
    """
    Función principal que orquesta todo el proceso.
    """
    logger.info("=" * 60)
    logger.info("INICIO DEL PROCESO DE TRADUCCIÓN DE AMAZON REVIEWS")
    logger.info("=" * 60)

    # ── Importar módulos del proyecto ─────────────────────────
    from src.translation.pipeline    import TranslationPipeline
    from src.validation.quality_checker import QualityChecker
    from src.validation.error_handler  import ErrorHandler

    # ── PASO 1: Preparar datos ────────────────────────────────
    logger.info("\n[PASO 1] Cargando dataset...")

    # Para pruebas usamos datos de ejemplo
    # Para producción: df = pd.read_csv("data/raw/amazon_reviews.csv")
    df = crear_dataset_ejemplo()

    logger.info("Dataset cargado: %d reviews", len(df))
    print("\n--- Vista previa del dataset original ---")
    print(df[["review_id", "review_body", "marketplace"]].to_string(index=False))

    # ── PASO 2: Ejecutar pipeline de traducción ───────────────
    logger.info("\n[PASO 2] Iniciando pipeline de traducción...")

    pipeline = TranslationPipeline()

    try:
        df_traducido = pipeline.run(df, text_column="review_body")

        print("\n--- Vista previa de las traducciones ---")
        cols = ["review_id", "review_body", "review_en", "star_rating"]
        print(df_traducido[cols].to_string(index=False))

    except Exception as e:
        logger.error("Error en el pipeline de traducción: %s", e)
        raise

    # ── PASO 3: Validar calidad ───────────────────────────────
    logger.info("\n[PASO 3] Validando calidad de las traducciones...")

    checker = QualityChecker()
    df_validado = checker.check_dataframe(df_traducido)

    # ── PASO 4: Mostrar reporte ───────────────────────────────
    logger.info("\n[PASO 4] Generando reporte final...")

    reporte_calidad = checker.quality_report(df_validado)
    reporte_errores = pipeline.handler.get_error_report()

    print("\n" + "=" * 50)
    print("REPORTE DE CALIDAD DE TRADUCCIONES")
    print("=" * 50)
    for clave, valor in reporte_calidad.items():
        print(f"  {clave:<35}: {valor}")

    print("\n" + "=" * 50)
    print("REPORTE DE ERRORES")
    print("=" * 50)
    for clave, valor in reporte_errores.items():
        print(f"  {clave:<35}: {valor}")

    # ── PASO 5: Guardar resultado ─────────────────────────────
    output_dir = "data/translated"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reviews_translated.csv")

    df_validado.to_csv(output_path, index=False)
    logger.info("\nArchivo guardado en: %s", output_path)

    print("\n" + "=" * 50)
    print(f"✓ Proceso completado. Archivo guardado en: {output_path}")
    print("=" * 50)

    return df_validado


# ── Ejecutar solo si este archivo es el punto de entrada ─────
if __name__ == "__main__":
    df_resultado = main()