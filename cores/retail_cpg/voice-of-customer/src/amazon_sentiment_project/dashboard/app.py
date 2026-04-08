import streamlit as st
import pandas as pd
import os

st.title("Amazon Sentiment Dashboard")

# 📂 Ruta del archivo final generado por Kedro
file_path = "data/03_primary/amazon_reviews.csv"

# Validar existencia
if not os.path.exists(file_path):
    st.error("No se encontró el archivo. Ejecuta primero Kedro.")
    st.stop()

# Leer datos
data = pd.read_csv(file_path)

st.subheader("Datos generales")
st.write(data.head())

# 📊 Sentimientos
st.subheader("Distribución de Sentimientos")

sentiment_counts = data["sentiment"].value_counts()
st.bar_chart(sentiment_counts)

# 📊 Traducciones válidas
if "is_valid_translation" in data.columns:
    st.subheader("Validación de traducciones")
    validation_counts = data["is_valid_translation"].value_counts()
    st.bar_chart(validation_counts)

# 📊 Texto original vs traducido
st.subheader("Ejemplos")
st.write(data[["review_text", "translated_text", "sentiment"]].head())