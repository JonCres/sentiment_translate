import pandas as pd
import numpy as np
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
import hdbscan
from umap import UMAP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_synthetic_data(n=200):
    topics = {
        "Cost Savings": [
            "We managed to reduce costs by 20% using this tool.",
            "The platform is great for cost optimization.",
            "Saving money is our priority and this helps.",
            "Reduced operational expenses significantly.",
            "Cost-effective solution for our department."
        ],
        "AI Efficiency": [
            "The AI models are very efficient.",
            "Automating workflows with AI has saved us a lot of time.",
            "The efficiency of the AI engine is impressive.",
            "Our team is more productive thanks to the AI features.",
            "Speed of processing has increased with the new AI tools."
        ],
        "Customer Support": [
            "The support team is very responsive.",
            "Helpdesk solved my issue in minutes.",
            "Excellent customer service and support.",
            "I had a problem with my account and support fixed it quickly.",
            "The documentation is good but the live support is even better."
        ],
        "Product Bugs": [
            "Found a bug in the dashboard.",
            "The application crashes when I try to export data.",
            "There are some glitches in the user interface.",
            "The API returns 500 errors occasionally.",
            "I'm experiencing latency issues with the reporting module."
        ]
    }
    
    data = []
    for topic_name, examples in topics.items():
        for _ in range(n // len(topics)):
            text = np.random.choice(examples)
            # Add some noise
            if np.random.random() > 0.8:
                text += " Also the service is okay."
            data.append({"feedback_text_masked": text, "true_topic": topic_name})
            
    return pd.DataFrame(data)

def run_tm(df, min_topic_size=10, nr_topics="auto"):
    docs = df["feedback_text_masked"].tolist()
    
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, random_state=42)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=min_topic_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
    
    representation_model = [KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.3)]
    
    topic_model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        nr_topics=nr_topics,
        min_topic_size=min_topic_size,
        representation_model=representation_model,
        verbose=True
    )
    
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics

if __name__ == "__main__":
    df = generate_synthetic_data(400)
    logger.info("Running Topic Modeling with current parameters (approx)...")
    model, topics = run_tm(df, min_topic_size=20, nr_topics="auto")
    print(model.get_topic_info())
    
    logger.info("Running Topic Modeling with smaller min_topic_size...")
    model2, topics2 = run_tm(df, min_topic_size=10, nr_topics="auto")
    print(model2.get_topic_info())
