import polars as pl
import pandas as pd
import yaml
import logging
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
import hdbscan
from umap import UMAP
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_debug():
    try:
        # Load params
        with open("conf/base/parameters.yml", "r") as f:
            params = yaml.safe_load(f)

        tm_params = params.get("topic_modeling", {})
        model_params = tm_params.get("model", {})
        reduction_params = tm_params.get("reduction", {})
        clustering_params = tm_params.get("clustering", {})

        # Load data
        df = pl.read_delta("data/05_model_input/feedback_with_churn.delta").to_pandas()
        logger.info(f"Data shape: {df.shape}")

        docs = df["feedback_text_masked"].fillna("").astype(str).tolist()
        logger.info(f"Docs length: {len(docs)}")

        if not docs:
            logger.error("DOCS IS EMPTY")
            return

        # Embeddings
        logger.info("Generating embeddings...")
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = embedding_model.encode(docs, show_progress_bar=True)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # UMAP
        logger.info("Fitting UMAP...")
        umap_model = UMAP(
            n_neighbors=reduction_params.get("n_neighbors", 15),
            n_components=reduction_params.get("n_components", 5),
            min_dist=reduction_params.get("min_dist", 0.0),
            random_state=42,
        )
        # We simulate what BERTopic does
        umap_embeddings = umap_model.fit_transform(embeddings)
        logger.info(f"UMAP embeddings shape: {umap_embeddings.shape}")

        # HDBSCAN
        logger.info("Fitting HDBSCAN...")
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=clustering_params.get("min_cluster_size", 50),
            metric=clustering_params.get("metric", "euclidean"),
            cluster_selection_method=clustering_params.get(
                "cluster_selection_method", "eom"
            ),
            prediction_data=True,
        )
        hdbscan_model.fit(umap_embeddings)
        logger.info("HDBSCAN success")

    except Exception as e:
        logger.error(f"DEBUG FAILED: {e}", exc_info=True)


if __name__ == "__main__":
    test_debug()
