import logging
import pandas as pd
from typing import Dict, Any, List, Tuple
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sentence_transformers import SentenceTransformer
import hdbscan
from umap import UMAP
import os
from groq import Groq
from utils import MultiDeviceManager, load_model, clear_device_cache, get_device

logger = logging.getLogger(__name__)

# Try to import HuggingFace transformers for SLM-based labeling
try:
    import torch
    from transformers import pipeline

    SLM_AVAILABLE = True
except ImportError:
    SLM_AVAILABLE = False
    logger.warning(
        "Transformers not properly installed. SLM-based topic labeling will not be available."
    )

# Global variable to cache the loaded SLM model and tokenizer
_cached_slm_model = None
_cached_slm_tokenizer = None
_cached_slm_pipeline = None


# Common words to exclude from topic names
STOP_WORDS_FOR_LABELS = {
    "product",
    "review",
    "customer",
    "would",
    "could",
    "really",
    "very",
    "just",
    "also",
    "even",
    "much",
    "thing",
    "things",
    "make",
    "made",
    "lot",
    "many",
    "some",
    "like",
    "one",
    "two",
    "go",
    "going",
    "get",
    "got",
    "use",
    "used",
    "using",
    "good",
    "great",
    "nice",
    "well",
    "client",
    "feedback",
    "service",
}


def generate_user_friendly_topic_name(
    topic_words: List[Tuple[str, float]], top_n_words: int = 4, max_length: int = 50
) -> str:
    """Generate a user-friendly topic name from BERTopic keywords."""
    if not topic_words:
        return "Miscellaneous"

    filtered_words = []
    for word, score in topic_words:
        clean_word = word.strip().lower()
        if (
            clean_word in STOP_WORDS_FOR_LABELS
            or len(clean_word) < 3
            or clean_word.isdigit()
        ):
            continue
        filtered_words.append(clean_word)
        if len(filtered_words) >= top_n_words:
            break

    if not filtered_words:
        filtered_words = [w[0].strip().lower() for w in topic_words[:3]]

    formatted_words = [word.title() for word in filtered_words]
    if len(formatted_words) == 1:
        topic_name = formatted_words[0]
    elif len(formatted_words) == 2:
        topic_name = f"{formatted_words[0]} & {formatted_words[1]}"
    else:
        topic_name = ", ".join(formatted_words[:-1]) + f" & {formatted_words[-1]}"

    if len(topic_name) > max_length:
        topic_name = topic_name[: max_length - 3] + "..."

    return topic_name


def load_slm_model(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Load and cache the Small Language Model using MultiDeviceManager."""
    global \
        _cached_slm_model, \
        _cached_slm_tokenizer, \
        _cached_slm_pipeline, \
        _cached_slm_manager

    if _cached_slm_pipeline is not None:
        return _cached_slm_model, _cached_slm_tokenizer, _cached_slm_pipeline

    if not SLM_AVAILABLE:
        return None, None, None

    try:
        # Load using unified utility
        model, tokenizer, manager = load_model(
            model_name=model_name, torch_dtype=torch.float16, training=False
        )

        _cached_slm_model = model
        _cached_slm_tokenizer = tokenizer
        _cached_slm_manager = manager

        _cached_slm_pipeline = pipeline(
            "text-generation",
            model=_cached_slm_model,
            tokenizer=_cached_slm_tokenizer,
        )
        return _cached_slm_model, _cached_slm_tokenizer, _cached_slm_pipeline
    except Exception as e:
        logger.error(f"Failed to load SLM model {model_name}: {e}")
        return None, None, None


def generate_slm_topic_label(
    topic_words: List[Tuple[str, float]],
    representative_docs: List[str],
    slm_params: Dict[str, Any],
) -> str:
    """Generate a human-readable topic label using a Small Language Model."""
    if not SLM_AVAILABLE:
        return None

    model_name = slm_params.get("model", "microsoft/Phi-3-mini-4k-instruct")
    clear_device_cache()
    model, tokenizer, pipe = load_slm_model(model_name)
    if pipe is None:
        return None

    words_str = ", ".join([w[0] for w in topic_words[:10]])
    docs_str = ""
    if representative_docs:
        docs_str = "\n\nSample feedback from this topic:\n"
        for i, doc in enumerate(representative_docs[:3], 1):
            doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
            docs_str += f'{i}. "{doc_preview}"\n'

    messages = [
        {
            "role": "system",
            "content": "You are a topic labeling expert. Generate concise, descriptive labels for client feedback topics. Only output the label, nothing else.",
        },
        {
            "role": "user",
            "content": f"Based on these keywords: {words_str}{docs_str}\n\nGenerate a short label (2-5 words). Topic Label:",
        },
    ]

    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=slm_params.get("max_tokens", 50),
                temperature=slm_params.get("temperature", 0.3),
                do_sample=slm_params.get("temperature", 0.3) > 0,
                pad_token_id=tokenizer.eos_token_id,
            )
        label = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        ).strip()
        return label.split("\n")[0].strip().strip("\"'")
    except Exception as e:
        logger.error(f"Error calling SLM: {e}")
        return None


def generate_groq_topic_label(
    topic_words: List[Tuple[str, float]],
    representative_docs: List[str],
    llm_params: Dict[str, Any],
) -> str:
    """Generate a topic label using Groq Cloud API."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found. Falling back to keyword naming.")
        return None

    try:
        client = Groq(api_key=api_key)
        words_str = ", ".join([w[0] for w in topic_words[:10]])
        docs_str = ""
        if representative_docs:
            docs_str = "\n\nSample feedback:\n"
            for i, doc in enumerate(representative_docs[:3], 1):
                doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
                docs_str += f'{i}. "{doc_preview}"\n'

        prompt = f"""
        You are a topic labeling expert. Generate a concise, descriptive label (2-5 words) for a cluster of client feedback.
        Only output the label, nothing else.

        Keywords: {words_str}{docs_str}

        Topic Label:
        """

        completion = client.chat.completions.create(
            model=llm_params.get("model", "llama-3.1-8b-instant"),
            messages=[{"role": "user", "content": prompt}],
            temperature=llm_params.get("temperature", 0.1),
            max_tokens=llm_params.get("max_tokens", 50),
        )
        return completion.choices[0].message.content.strip().strip("\"'")
    except Exception as e:
        logger.error(f"Groq API error for topic labeling: {e}")
        return None


def train_topic_model(
    data: pd.DataFrame,
    parameters: Dict[str, Any],
) -> Any:
    """Train BERTopic model with explicit UMAP and HDBSCAN."""
    if not parameters.get("enabled", True):
        logger.info("Topic Modeling is disabled by configuration.")
        return BERTopic()

    model_params = parameters.get("model", {})
    reduction_params = parameters.get("reduction", {})
    clustering_params = parameters.get("clustering", {})

    if data is None or data.empty:
        logger.warning("Input data for topic modeling is empty. Skipping.")
        return BERTopic()

    logger.info(f"Initializing Topic Discovery for {len(data)} records...")

    # Data Quality Check: Ensure text column exists and has content
    text_col = "feedback_text_masked"
    if text_col not in data.columns:
        logger.error(
            f"Missing text column '{text_col}'. Available: {data.columns.tolist()}"
        )
        return BERTopic()

    docs = data[text_col].fillna("").astype(str).tolist()
    # Basic filtering to avoid embedding empty noise
    docs = [d for d in docs if len(d.strip()) > 2]

    logger.info(f"Docs after filtering: {len(docs)}")

    if not docs:
        logger.warning(
            "No valid text documents found after filtering. Skipping Topic Modeling."
        )
        return BERTopic()

    logger.info(f"Processing {len(docs)} text documents for topic discovery.")

    try:
        device_str = get_device(purpose="Topic model embeddings")
        logger.info(f"Using device: {device_str}")
        clear_device_cache()
        embedding_model = SentenceTransformer(
            model_params.get("embedding_model", "all-MiniLM-L6-v2"), device=device_str
        )

        # Explicit UMAP
        umap_model = UMAP(
            n_neighbors=reduction_params.get("n_neighbors", 15),
            n_components=reduction_params.get("n_components", 5),
            min_dist=reduction_params.get("min_dist", 0.0),
            random_state=42,
        )

        # Explicit HDBSCAN
        hdbscan_min_cluster = min(
            clustering_params.get("min_cluster_size", 50), len(docs) // 3
        )
        logger.info(f"HDBSCAN min_cluster_size: {hdbscan_min_cluster}")

        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=max(2, hdbscan_min_cluster),
            metric=clustering_params.get("metric", "euclidean"),
            cluster_selection_method=clustering_params.get(
                "cluster_selection_method", "eom"
            ),
            core_dist_n_jobs=clustering_params.get("core_dist_n_jobs", 1),
            prediction_data=True,
        )

        # Representation Models
        representation_models = [
            KeyBERTInspired(),
            MaximalMarginalRelevance(diversity=0.3),
        ]

        # BERTopic Assembly
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            nr_topics=model_params.get("nr_topics", "auto"),
            min_topic_size=max(
                2, min(model_params.get("min_topic_size", 50), len(docs) // 3)
            ),
            calculate_probabilities=model_params.get("calculate_probabilities", True),
            representation_model=representation_models,
            verbose=True,
        )

        # Fitting
        logger.info("Starting topic_model.fit_transform(docs)...")
        topics, probs = topic_model.fit_transform(docs)
        logger.info("Successfully finished topic_model.fit_transform(docs)")

        # Reduce Outliers
        # Map -1 (noise) to the nearest topic to minimize "Uncategorized"
        try:
            logger.info("Reducing outliers...")
            new_topics = topic_model.reduce_outliers(docs, topics)
            topic_model.update_topics(docs, topics=new_topics)
            logger.info("Outlier reduction complete.")
        except Exception as e:
            logger.warning(f"Failed to reduce outliers: {e}")

        # Labeling (SLM + Fallback)
        llm_params = model_params.get("llm", {})
        topic_labels = {}
        topic_info = topic_model.get_topic_info()

        for _, row in topic_info.iterrows():
            t_id = row["Topic"]
            if t_id == -1:
                topic_labels[t_id] = "Uncategorized"
                continue

            t_words = topic_model.get_topic(t_id)
            label = None
            if llm_params.get("enabled", True):
                provider = llm_params.get("provider", "local")
                rep_docs = row.get("Representative_Docs", [])

                if provider == "groq":
                    label = generate_groq_topic_label(t_words, rep_docs, llm_params)
                elif SLM_AVAILABLE:
                    label = generate_slm_topic_label(t_words, rep_docs, llm_params)

            if not label:
                label = generate_user_friendly_topic_name(
                    t_words,
                    top_n_words=model_params.get("naming", {}).get("top_n_words", 4),
                    max_length=model_params.get("naming", {}).get(
                        "max_label_length", 50
                    ),
                )
            topic_labels[t_id] = label
            logger.info(f"Topic {t_id}: {label}")

        topic_model.set_topic_labels(topic_labels)
        return topic_model

    except Exception as e:
        logger.error(f"Topic modeling failed: {e}", exc_info=True)
        return BERTopic()


def assign_topics(
    data: pd.DataFrame, topic_model: Any, parameters: Dict[str, Any]
) -> pd.DataFrame:
    """Assign topics to the feedback data."""
    if (
        not parameters.get("enabled", True)
        or topic_model is None
        or data is None
        or data.empty
    ):
        if data is not None:
            data["Topic_ID"] = -1
            data["Topic_Name"] = "Uncategorized"
        return data

    logger.info("Assigning topics to feedback...")
    docs = data["feedback_text_masked"].fillna("").astype(str).tolist()

    try:
        topics, _ = topic_model.transform(docs)

        # Reduce outliers in assignment
        try:
            topics = topic_model.reduce_outliers(docs, topics)
        except Exception as e:
            logger.warning(f"Failed to reduce outliers in assignment: {e}")

        data["Topic_ID"] = topics
        topic_map = (
            topic_model.get_topic_info().set_index("Topic")["CustomName"].to_dict()
        )
        data["Topic_Name"] = data["Topic_ID"].map(topic_map)
    except Exception as e:
        logger.error(f"Topic assignment failed: {e}")
        data["Topic_ID"] = -1
        data["Topic_Name"] = "Uncategorized"

    return data
