import polars as pl
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple, Optional
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
import hdbscan
import re
import os
from groq import Groq
from dotenv import load_dotenv
from utils import MultiDeviceManager, load_model, generate_text, clear_device_cache

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import HuggingFace transformers for SLM-based labeling
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch

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

# Topic label templates for generating user-friendly names
TOPIC_LABEL_TEMPLATES = {
    "default": "{keyword_phrase}",
    "with_category": "{category}: {keyword_phrase}",
}

# Common words to exclude from topic names (they don't add meaning)
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
}


def generate_user_friendly_topic_name(
    topic_words: List[Tuple[str, float]], top_n_words: int = 4, max_length: int = 50
) -> str:
    """Generate a user-friendly topic name from BERTopic keywords.

    Transforms cryptic topic names like "0_product_quality_issue" into
    readable labels like "Product Quality Issues".

    Args:
        topic_words: List of (word, score) tuples from BERTopic
        top_n_words: Number of top words to use for the name
        max_length: Maximum length of the generated name

    Returns:
        A human-readable topic name
    """
    if not topic_words:
        return "Miscellaneous"

    # Extract words, filtering out stop words and keeping meaningful terms
    filtered_words = []
    for word, score in topic_words:
        # Clean the word
        clean_word = word.strip().lower()

        # Skip stop words and very short words
        if clean_word in STOP_WORDS_FOR_LABELS or len(clean_word) < 3:
            continue

        # Skip numeric-only words
        if clean_word.isdigit():
            continue

        filtered_words.append(clean_word)

        if len(filtered_words) >= top_n_words:
            break

    if not filtered_words:
        # Fallback: use original words if all were filtered
        filtered_words = [w[0].strip().lower() for w in topic_words[:3]]

    # Create the topic name
    # Strategy: Capitalize each word and join with " & " for 2 words,
    # or ", " for more, ending with " & " before the last word
    formatted_words = [word.title() for word in filtered_words]

    if len(formatted_words) == 1:
        topic_name = formatted_words[0]
    elif len(formatted_words) == 2:
        topic_name = f"{formatted_words[0]} & {formatted_words[1]}"
    else:
        # Join first n-1 words with commas, last with &
        topic_name = ", ".join(formatted_words[:-1]) + f" & {formatted_words[-1]}"

    # Truncate if too long
    if len(topic_name) > max_length:
        topic_name = topic_name[: max_length - 3] + "..."

    return topic_name


def create_topic_label_mapping(
    topic_model: BERTopic, naming_params: Optional[Dict[str, Any]] = None
) -> Dict[int, str]:
    """Create a mapping of topic IDs to user-friendly names.

    Args:
        topic_model: Trained BERTopic model
        naming_params: Optional parameters for customizing naming behavior

    Returns:
        Dictionary mapping topic ID to friendly name
    """
    naming_params = naming_params or {}
    top_n_words = naming_params.get("top_n_words", 4)
    max_length = naming_params.get("max_label_length", 50)

    topic_labels = {}

    # Get all topic IDs (excluding -1 for outliers)
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]

        if topic_id == -1:
            topic_labels[topic_id] = "Uncategorized / Outliers"
            continue

        # Get the top words for this topic
        topic_words = topic_model.get_topic(topic_id)

        if topic_words:
            friendly_name = generate_user_friendly_topic_name(
                topic_words, top_n_words=top_n_words, max_length=max_length
            )
            topic_labels[topic_id] = friendly_name
        else:
            topic_labels[topic_id] = f"Topic {topic_id}"

    return topic_labels


def apply_user_friendly_topic_names(
    topic_model: BERTopic, naming_params: Optional[Dict[str, Any]] = None
) -> BERTopic:
    """Apply user-friendly topic names to a BERTopic model.

    Uses BERTopic's set_topic_labels method to replace default
    names with human-readable alternatives.

    Args:
        topic_model: Trained BERTopic model
        naming_params: Optional parameters for naming customization

    Returns:
        The same model with updated topic labels
    """
    topic_labels = create_topic_label_mapping(topic_model, naming_params)

    # Log the mapping for visibility
    logger.info("Generated user-friendly topic names:")
    for topic_id, label in sorted(topic_labels.items()):
        if topic_id != -1:
            logger.info(f"  Topic {topic_id}: {label}")

    # Apply the custom labels to the model
    topic_model.set_topic_labels(topic_labels)

    return topic_model


def extract_ctfidf_info(
    topic_model: BERTopic,
    topic_id: int,
    docs: List[str],
    top_n_words: int = 10,
    top_n_docs: int = 3,
) -> Dict[str, Any]:
    """Extract c-TF-IDF top words and representative documents for a topic.

    Uses BERTopic's c-TF-IDF representation to get the most important
    words for a topic, along with sample documents that belong to that topic.

    Args:
        topic_model: Trained BERTopic model
        topic_id: The topic ID to extract information for
        docs: List of all documents used for training
        top_n_words: Number of top words to extract
        top_n_docs: Number of representative documents to extract

    Returns:
        Dictionary containing top_words, word_scores, and representative_docs
    """
    # Get top words from c-TF-IDF representation
    topic_words = topic_model.get_topic(topic_id)

    if not topic_words:
        return {"top_words": [], "word_scores": [], "representative_docs": []}

    # Extract words and scores (limited to top_n)
    words = [word for word, score in topic_words[:top_n_words]]
    scores = [score for word, score in topic_words[:top_n_words]]

    # Get representative documents for this topic
    representative_docs = []
    try:
        # Get topic info which contains representative docs
        topic_info = topic_model.get_topic_info()
        topic_row = topic_info[topic_info["Topic"] == topic_id]

        if not topic_row.empty and "Representative_Docs" in topic_row.columns:
            rep_docs = topic_row["Representative_Docs"].iloc[0]
            if rep_docs and isinstance(rep_docs, list):
                representative_docs = rep_docs[:top_n_docs]
    except Exception as e:
        logger.debug(f"Could not extract representative docs for topic {topic_id}: {e}")

    # If no representative docs found, try to get them from document-topic mapping
    if not representative_docs:
        try:
            # Get document info and filter by topic
            doc_info = topic_model.get_document_info(docs)
            topic_docs = doc_info[doc_info["Topic"] == topic_id]

            if not topic_docs.empty:
                # Sort by probability if available, otherwise just take first few
                if "Probability" in topic_docs.columns:
                    topic_docs = topic_docs.nlargest(top_n_docs, "Probability")
                else:
                    topic_docs = topic_docs.head(top_n_docs)

                representative_docs = topic_docs["Document"].tolist()
        except Exception as e:
            logger.debug(f"Could not extract docs from document info: {e}")

    return {
        "top_words": words,
        "word_scores": scores,
        "representative_docs": representative_docs,
    }


def load_slm_model(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    """Load and cache the Small Language Model for topic labeling using MultiDeviceManager."""
    global \
        _cached_slm_model, \
        _cached_slm_tokenizer, \
        _cached_slm_pipeline, \
        _cached_slm_manager

    # Return cached model if already loaded
    if _cached_slm_pipeline is not None:
        return _cached_slm_model, _cached_slm_tokenizer, _cached_slm_pipeline

    if not SLM_AVAILABLE:
        logger.warning("Transformers not available for SLM loading")
        return None, None, None

    try:
        # Load using the unified load_model utility
        model, tokenizer, manager = load_model(
            model_name=model_name, torch_dtype=torch.float16, training=False
        )

        _cached_slm_model = model
        _cached_slm_tokenizer = tokenizer
        _cached_slm_manager = manager

        # Create text generation pipeline
        _cached_slm_pipeline = pipeline(
            "text-generation",
            model=_cached_slm_model,
            tokenizer=_cached_slm_tokenizer,
        )

        logger.info(
            f"Successfully loaded SLM: {model_name} on {manager.device_type.name}"
        )
        return _cached_slm_model, _cached_slm_tokenizer, _cached_slm_pipeline

    except Exception as e:
        logger.error(f"Failed to load SLM model {model_name}: {e}")
        return None, None, None


def generate_slm_topic_label(
    topic_info: Dict[str, Any], slm_params: Dict[str, Any]
) -> str:
    """Generate a human-readable topic label using a Small Language Model.

    Sends the c-TF-IDF top words and representative documents to a local
    SLM (e.g., Phi-3 or Phi-4) to generate a concise, descriptive label.

    Args:
        topic_info: Dictionary with top_words, word_scores, representative_docs
        slm_params: Parameters for SLM configuration (model, temperature, etc.)

    Returns:
        A human-readable topic label generated by the SLM
    """
    if not SLM_AVAILABLE:
        logger.warning(
            "Transformers not available, falling back to keyword-based labels"
        )
        return None

    # Extract parameters
    model_name = slm_params.get("model", "microsoft/Phi-3-mini-4k-instruct")
    temperature = slm_params.get("temperature", 0.3)
    max_tokens = slm_params.get("max_tokens", 50)

    # Load the model (uses cache if already loaded)
    model, tokenizer, pipe = load_slm_model(model_name)

    if pipe is None:
        logger.warning(
            "SLM pipeline not available, falling back to keyword-based labels"
        )
        return None

    # Prepare the prompt
    top_words = topic_info.get("top_words", [])
    representative_docs = topic_info.get("representative_docs", [])

    if not top_words:
        return None

    # Build context for the SLM
    words_str = ", ".join(top_words[:10])
    docs_str = ""
    if representative_docs:
        docs_str = "\n\nSample reviews from this topic:\n"
        for i, doc in enumerate(representative_docs[:3], 1):
            # Truncate long documents
            doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
            docs_str += f'{i}. "{doc_preview}"\n'

    # Create chat messages for the model
    messages = [
        {
            "role": "system",
            "content": "You are a topic labeling expert. Generate concise, descriptive labels for customer review topics. Only output the label, nothing else. Keep labels between 2-5 words.",
        },
        {
            "role": "user",
            "content": f"""Based on the following keywords and sample reviews from a customer review topic, generate a short, descriptive label (2-5 words) that captures the main theme.

Keywords (from c-TF-IDF analysis): {words_str}
{docs_str}
Important: 
- The label should be concise and professional
- Focus on the customer experience theme (e.g., "Product Quality Issues", "Shipping Delays", "Great Customer Service")
- Only output the label, nothing else

Topic Label:""",
        },
    ]

    try:
        # Use direct generation instead of pipeline to avoid DynamicCache compatibility issues
        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # Generator args
        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
            # Explicitly disable caching to avoid 'DynamicCache' errors with some model versions
            "use_cache": False,
        }

        with torch.no_grad():
            outputs = model.generate(input_ids, **gen_kwargs)

        # Decode output (skip input tokens)
        generated_tokens = outputs[0][input_ids.shape[1] :]
        label = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Clean up the label
        label = label.strip().strip("\"'")
        label = re.sub(r"\s+", " ", label)

        # Remove any trailing explanations (take first line only)
        label = label.split("\n")[0].strip()

        # Ensure label is not too long
        if len(label) > 60:
            label = label[:57] + "..."

        return label if label else None
    except Exception as e:
        logger.error(f"Error calling SLM for topic labeling: {e}")
        return None


def generate_groq_topic_label(
    topic_info: Dict[str, Any], llm_params: Dict[str, Any]
) -> str:
    """Generate a topic label using Groq Cloud API.

    Args:
        topic_info: Dictionary with top_words, word_scores, representative_docs
        llm_params: Parameters for LLM configuration (model, API key, etc.)

    Returns:
        A human-readable topic label generated by Groq
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not found in environment. Falling back.")
        return None

    model_name = llm_params.get("model", "llama-3.1-8b-instant")
    temperature = llm_params.get("temperature", 0.1)
    max_tokens = llm_params.get("max_tokens", 50)

    try:
        client = Groq(api_key=api_key)

        top_words = topic_info.get("top_words", [])
        representative_docs = topic_info.get("representative_docs", [])

        words_str = ", ".join(top_words[:10])
        docs_str = ""
        if representative_docs:
            docs_str = "\n\nSample reviews:\n"
            for i, doc in enumerate(representative_docs[:3], 1):
                doc_preview = doc[:200] + "..." if len(doc) > 200 else doc
                docs_str += f'{i}. "{doc_preview}"\n'

        prompt = f"""
        You are a topic labeling expert. Generate a concise, descriptive label (2-5 words) for a cluster of customer feedback.
        Only output the label, nothing else.

        Keywords: {words_str}{docs_str}

        Topic Label:
        """

        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        label = completion.choices[0].message.content.strip().strip("\"'")
        return label.split("\n")[0].strip()
    except Exception as e:
        logger.error(f"Groq API error for topic labeling: {e}")
        return None


# Keep backward compatibility alias
def generate_llm_topic_label(
    topic_info: Dict[str, Any], llm_params: Dict[str, Any]
) -> str:
    """Backward-compatible alias for generate_slm_topic_label.

    This function now uses a local Small Language Model instead of Groq.
    """
    return generate_slm_topic_label(topic_info, llm_params)


def create_llm_topic_labels(
    topic_model: BERTopic, docs: List[str], llm_params: Dict[str, Any]
) -> Dict[int, str]:
    """Generate SLM-based labels for all topics in the model.

    Iterates through all topics, extracts c-TF-IDF information,
    and uses a local Small Language Model to generate human-readable labels.
    Falls back to keyword-based labels if SLM fails.

    Args:
        topic_model: Trained BERTopic model
        docs: List of documents used for training
        llm_params: SLM configuration parameters

    Returns:
        Dictionary mapping topic IDs to SLM-generated labels
    """
    logger.info("Generating topic labels using Small Language Model...")

    topic_labels = {}
    topic_info = topic_model.get_topic_info()

    # Parameters for extraction
    top_n_words = llm_params.get("top_n_words", 10)
    top_n_docs = llm_params.get("top_n_docs", 3)

    for _, row in topic_info.iterrows():
        topic_id = row["Topic"]

        if topic_id == -1:
            topic_labels[topic_id] = "Uncategorized / Outliers"
            continue

        # Extract c-TF-IDF information
        ctfidf_info = extract_ctfidf_info(
            topic_model, topic_id, docs, top_n_words=top_n_words, top_n_docs=top_n_docs
        )

        # Generate LLM label based on provider
        provider = llm_params.get("provider", "local")
        llm_label = None

        if provider == "groq":
            llm_label = generate_groq_topic_label(ctfidf_info, llm_params)
        elif SLM_AVAILABLE:
            llm_label = generate_slm_topic_label(ctfidf_info, llm_params)

        if llm_label:
            topic_labels[topic_id] = llm_label
            logger.info(f"  Topic {topic_id}: {llm_label} ({provider.upper()})")
        else:
            # Fallback to keyword-based labeling
            topic_words = topic_model.get_topic(topic_id)
            fallback_label = generate_user_friendly_topic_name(
                topic_words, top_n_words=4, max_length=50
            )
            topic_labels[topic_id] = fallback_label
            logger.info(f"  Topic {topic_id}: {fallback_label} (fallback)")

    return topic_labels


def apply_llm_topic_labels(
    topic_model: BERTopic, docs: List[str], llm_params: Optional[Dict[str, Any]] = None
) -> BERTopic:
    """Apply SLM-generated labels to a BERTopic model.

    Uses a local Small Language Model (e.g., Phi-3, Phi-4) to generate
    intelligent, human-readable labels based on c-TF-IDF keywords and
    representative documents.

    Args:
        topic_model: Trained BERTopic model
        docs: List of documents used for training
        llm_params: Optional SLM configuration parameters

    Returns:
        The model with updated topic labels
    """
    llm_params = llm_params or {}

    # Check if SLM labeling is enabled
    if not llm_params.get("enabled", True):
        logger.info("SLM labeling disabled, using keyword-based labels")
        return apply_user_friendly_topic_names(topic_model, llm_params)

    # Check if transformers is available for SLM
    if not SLM_AVAILABLE:
        logger.warning(
            "Transformers not available for SLM. Falling back to keyword-based labels."
        )
        return apply_user_friendly_topic_names(topic_model, llm_params)

    # Generate LLM labels
    topic_labels = create_llm_topic_labels(topic_model, docs, llm_params)

    # Apply labels to the model
    topic_model.set_topic_labels(topic_labels)

    logger.info(f"Applied LLM-generated labels to {len(topic_labels)} topics")

    return topic_model


def train_topic_model(
    df: pl.DataFrame,
    model_params: Dict[str, Any],
    reduction_params: Dict[str, Any],
    clustering_params: Dict[str, Any],
) -> Dict[str, Any]:
    """Train BERTopic model with SLM-generated topic labels.

    Uses a local Small Language Model (e.g., Phi-3, Phi-4) to generate
    human-readable topic labels based on c-TF-IDF keywords and representative
    documents. Falls back to keyword-based labels if SLM is not available.

    Args:
        df: DataFrame containing review_text column
        model_params: BERTopic model parameters including:
            - 'naming': keyword-based labeling settings
            - 'llm': SLM-based labeling settings (model, temperature, etc.)
        reduction_params: UMAP dimensionality reduction parameters
        clustering_params: HDBSCAN clustering parameters

    Returns:
        Dictionary containing the trained model, topics, and probabilities
    """
    logger.info("Training topic model...")

    # Prepare data
    docs = df["review_text"].to_list()

    # Initialize HDBSCAN
    hdbscan_model = hdbscan.HDBSCAN(**clustering_params, prediction_data=True)

    # Configure representation models for better topic quality
    # KeyBERTInspired provides better keyword extraction
    # MMR ensures diverse keywords (reduces redundancy)
    representation_models = [KeyBERTInspired(), MaximalMarginalRelevance(diversity=0.3)]

    # Initialize BERTopic with enhanced representation
    topic_model = BERTopic(
        language="english",
        calculate_probabilities=model_params["calculate_probabilities"],
        verbose=True,
        nr_topics=model_params["nr_topics"],
        min_topic_size=model_params["min_topic_size"],
        n_gram_range=(1, 2),
        hdbscan_model=hdbscan_model,
        representation_model=representation_models,
    )

    # Fit model
    topics, probs = topic_model.fit_transform(docs)

    # Get topic info before applying custom names
    topic_info = topic_model.get_topic_info()
    logger.info(f"Found {len(topic_info) - 1} topics")

    # Check if SLM labeling is configured
    llm_params = model_params.get("llm", {})
    use_llm = llm_params.get("enabled", False)

    if use_llm:
        # Use SLM-based labeling with local model
        logger.info("Using SLM-based topic labeling with local model")
        topic_model = apply_llm_topic_labels(topic_model, docs, llm_params)
    else:
        # Fall back to keyword-based labeling
        logger.info("Using keyword-based topic labeling")
        naming_params = model_params.get("naming", {})
        topic_model = apply_user_friendly_topic_names(topic_model, naming_params)

    return {"model": topic_model, "topics": topics, "probs": probs}


def predict_topics(
    df: pl.DataFrame, topic_model_artifact: Dict[str, Any]
) -> pl.DataFrame:
    """Predict topics for reviews using user-friendly topic names.

    Args:
        df: DataFrame containing review_text column
        topic_model_artifact: Dictionary with trained BERTopic model

    Returns:
        DataFrame with topic assignments and user-friendly topic names
    """
    logger.info("Predicting topics...")

    topic_model = topic_model_artifact["model"]
    docs = df["review_text"].to_list()

    # Transform
    topics, probs = topic_model.transform(docs)

    # Add to dataframe
    df_result = df.with_columns(pl.Series(name="topic", values=topics))

    # Add topic names - prioritize custom labels if available
    topic_info = topic_model.get_topic_info()

    # Check if custom labels are available (created by set_topic_labels)
    # topic_info is a pandas DataFrame from BERTopic
    if "CustomName" in topic_info.columns:
        # Use the user-friendly custom names
        topic_names = {
            row["Topic"]: row["CustomName"] for _, row in topic_info.iterrows()
        }
        logger.info("Using user-friendly custom topic names")
    else:
        # Fallback to original names
        topic_names = {row["Topic"]: row["Name"] for _, row in topic_info.iterrows()}
        logger.info("Using default topic names")

    # Map topic names
    df_result = df_result.with_columns(
        pl.col("topic").replace(topic_names, default=None).alias("topic_name")
    )

    if probs is not None:
        df_result = df_result.with_columns(
            pl.Series(name="topic_probs", values=probs.tolist())
        )

    return df_result


def create_topic_summary(
    df: pl.DataFrame, topic_model_artifact: Dict[str, Any]
) -> Dict[str, Any]:
    """Create topic modeling summary with user-friendly names."""
    logger.info("Creating topic summary...")

    topic_model = topic_model_artifact["model"]

    # Get topic info with custom labels if available (Pandas DF)
    topic_info = topic_model.get_topic_info()

    # Use CustomName if available for cleaner output
    if "CustomName" in topic_info.columns:
        topic_info["DisplayName"] = topic_info["CustomName"]
    else:
        topic_info["DisplayName"] = topic_info["Name"]

    # Top topics with user-friendly names
    top_topics_data = topic_info.head(10)[["Topic", "Count", "DisplayName"]].copy()
    top_topics_data = top_topics_data.rename(columns={"DisplayName": "Name"})
    top_topics = top_topics_data.to_dict("records")

    # Topic distribution
    topic_dist_df = df["topic_name"].value_counts()
    topic_dist = {
        row["topic_name"]: row["count"] for row in topic_dist_df.iter_rows(named=True)
    }

    # Topics by sentiment (if available)
    topics_by_sentiment = {}
    if "sentiment" in df.columns:
        # Group by topic_name and sentiment, count
        sent_by_topic = df.group_by(["topic_name", "sentiment"]).len()
        # Calculate topic totals
        topic_totals = df.group_by("topic_name").len().rename({"len": "total"})
        # Join
        sent_by_topic = sent_by_topic.join(topic_totals, on="topic_name")
        # Calculate fraction
        sent_by_topic = sent_by_topic.with_columns(
            (pl.col("len") / pl.col("total")).alias("fraction")
        )

        # Build nested dictionary
        for row in sent_by_topic.iter_rows(named=True):
            t = row["topic_name"]
            s = row["sentiment"]
            f = row["fraction"]
            if t not in topics_by_sentiment:
                topics_by_sentiment[t] = {}
            topics_by_sentiment[t][s] = round(f, 3)

    summary = {
        "total_topics": len(topic_info) - 1,
        "top_topics": top_topics,
        "topic_distribution": topic_dist,
        "topics_by_sentiment": topics_by_sentiment,
        "timestamp": datetime.now().isoformat(),
    }

    return summary


def create_customer_topic_profiles(
    df: pl.DataFrame,
    topic_model_artifact: Dict[str, Any],
    customer_params: Dict[str, Any],
) -> pl.DataFrame:
    """Create topic profiles for each customer."""
    logger.info("Creating customer topic profiles...")

    if "customer_id" not in df.columns:
        raise ValueError("customer_id column is required for customer profiling")

    min_reviews = customer_params.get("min_reviews_per_customer", 3)
    topic_model = topic_model_artifact["model"]

    # Get unique topics (excluding -1)
    topic_info = topic_model.get_topic_info()
    # topic_info is pandas
    unique_topics_count = len(topic_info[topic_info["Topic"] != -1])

    # Filter for valid topics directly
    valid_topics_df = df.filter(pl.col("topic") != -1)

    # We need intermediate calculations per customer

    # 1. Total reviews per customer (including outliers if needed, or usually we typically profile based on all reviews)
    cust_total_reviews = (
        df.group_by("customer_id").len().rename({"len": "total_reviews"})
    )

    # 2. Filter out customers with few reviews
    cust_total_reviews = cust_total_reviews.filter(
        pl.col("total_reviews") >= min_reviews
    )

    # Reduce main df to these customers
    df_filtered = df.join(cust_total_reviews.select("customer_id"), on="customer_id")

    # 3. Calculate topic counts per customer (group by customer, topic)
    cust_topic_counts = (
        df_filtered.group_by(["customer_id", "topic", "topic_name"])
        .len()
        .rename({"len": "count"})
    )

    # 4. Dominant topic
    # Sort by count desc, take first
    dominant_topics = cust_topic_counts.sort(
        ["customer_id", "count"], descending=[False, True]
    ).unique(subset=["customer_id"], keep="first")
    dominant_topics = dominant_topics.select(
        pl.col("customer_id"),
        pl.col("topic").alias("dominant_topic"),
        pl.col("topic_name").alias("dominant_topic_name"),
        pl.col("count").alias("dominant_count"),
    )

    # 5. Diversity & Entropy
    # We need to calculate based on non-outlier topics for diversity usually?
    # Original code: unique_customer_topics = len(topic_counts[topic_counts.index != -1])
    # So we filter out -1 for diversity calc

    valid_topic_counts = cust_topic_counts.filter(pl.col("topic") != -1)

    # Unique topics per customer
    entropy_stats = valid_topic_counts.group_by("customer_id").agg(
        [pl.len().alias("unique_topics")]
    )

    # Calculate prop for entropy: p = count / total_reviews
    # Note: total_reviews in original code likely included outlier reviews too.
    # We need to join total_reviews back to valid_topic_counts
    valid_topic_counts = valid_topic_counts.join(cust_total_reviews, on="customer_id")
    valid_topic_counts = valid_topic_counts.with_columns(
        (pl.col("count") / pl.col("total_reviews")).alias("prob")
    )

    # Entropy = -sum(p * log2(p + 1e-10))
    valid_topic_counts = valid_topic_counts.with_columns(
        (pl.col("prob") * (pl.col("prob") + 1e-10).log(2)).alias("entropy_term")
    )

    entropy_agg = valid_topic_counts.group_by("customer_id").agg(
        (-pl.col("entropy_term").sum()).alias("topic_entropy")
    )

    # Join everything
    # Base: cust_total_reviews
    profiles = cust_total_reviews.join(dominant_topics, on="customer_id", how="left")
    profiles = profiles.join(entropy_stats, on="customer_id", how="left")
    profiles = profiles.join(entropy_agg, on="customer_id", how="left")

    # Fill NAs
    profiles = profiles.with_columns(
        [
            pl.col("unique_topics").fill_null(0),
            pl.col("topic_entropy").fill_null(0.0),
            pl.col("dominant_topic").fill_null(-1),
            pl.col("dominant_topic_name").fill_null("Unknown"),
            pl.col("dominant_count").fill_null(0),
        ]
    )

    # Calculate derived metrics
    profiles = profiles.with_columns(
        [
            (pl.col("dominant_count") / pl.col("total_reviews") * 100)
            .round(2)
            .alias("dominant_topic_pct"),
            (pl.col("unique_topics") / max(unique_topics_count, 1))
            .round(3)
            .alias("topic_diversity"),
            pl.col("topic_entropy").round(3),
        ]
    )

    # Cleanup columns
    profiles = profiles.select(
        [
            "customer_id",
            "total_reviews",
            "unique_topics",
            "dominant_topic",
            "dominant_topic_name",
            "dominant_topic_pct",
            "topic_diversity",
            "topic_entropy",
        ]
    )

    logger.info(f"Created topic profiles for {len(profiles)} customers")

    return profiles


def create_customer_topic_summary(
    customer_profiles: pl.DataFrame, customer_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Create summary statistics for customer topic profiles."""
    logger.info("Creating customer topic summary...")

    top_n = customer_params.get("top_customers_count", 20)

    # Overall statistics
    total_customers = len(customer_profiles)
    avg_topics_per_customer = customer_profiles["unique_topics"].mean()
    avg_diversity = customer_profiles["topic_diversity"].mean()

    # Topic distribution across customers
    topic_dist_df = customer_profiles["dominant_topic_name"].value_counts()
    topic_dist = {
        row["dominant_topic_name"]: row["count"]
        for row in topic_dist_df.iter_rows(named=True)
    }

    # Most diverse customers
    most_diverse = (
        customer_profiles.top_k(top_n, by="topic_diversity")
        .select(["customer_id", "topic_diversity", "unique_topics", "total_reviews"])
        .to_dicts()
    )

    # Most focused customers (low diversity, high volume)
    # focused_mask = customer_profiles['total_reviews'] >= customer_profiles['total_reviews'].median()
    median_reviews = customer_profiles["total_reviews"].median()
    focused_profiles = customer_profiles.filter(
        pl.col("total_reviews") >= median_reviews
    )

    # nsmallest is bottom_k
    most_focused = (
        focused_profiles.bottom_k(top_n, by="topic_diversity")
        .select(
            [
                "customer_id",
                "topic_diversity",
                "dominant_topic_name",
                "dominant_topic_pct",
            ]
        )
        .to_dicts()
    )

    summary = {
        "total_customers": total_customers,
        "avg_topics_per_customer": round(avg_topics_per_customer, 2),
        "avg_topic_diversity": round(avg_diversity, 3),
        "dominant_topic_distribution": topic_dist,
        "most_diverse_customers": most_diverse,
        "most_focused_customers": most_focused,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(f"Customer topic summary created for {total_customers} customers")

    return summary
