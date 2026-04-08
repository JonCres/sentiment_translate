import logging
import pandas as pd
from typing import Dict, Any, List
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import os
from utils import get_device

logger = logging.getLogger(__name__)

def fine_tune_embedding_model(
    train_data: pd.DataFrame,
    parameters: Dict[str, Any]
) -> str:
    """Fine-tunes the sentence transformer model on the review data.
    
    Uses the ratings as labels to learn an embedding space where reviews 
    with similar ratings are closer together (using BatchHardTripletLoss 
    or similar Metric Learning approach).

    Args:
        train_data: Data containing 'review_text' and 'rating'.
        parameters: Parameters defined in parameters.yml.

    Returns:
        Path to the fine-tuned model.
    """
    logger.info("Starting embedding model fine-tuning...")
    
    # Extract parameters
    model_name = parameters['model'].get('embedding_model', 'BAAI/bge-small-en-v1.5')
    batch_size = parameters['fine_tuning'].get('batch_size', 16)
    epochs = parameters['fine_tuning'].get('epochs', 4)
    output_path = parameters['fine_tuning'].get('output_path', 'data/06_models/fine_tuned_embeddings')
    warmup_steps = parameters['fine_tuning'].get('warmup_steps', 100)
    num_workers = parameters['fine_tuning'].get('num_workers', 0)
    
    # Check if we have enough data
    df = train_data.dropna(subset=['review_text', 'rating'])
    logger.info(f"Fine-tuning on {len(df)} samples.")
    
    # Prepare training examples
    # We use the rating as a label (int) for Metric Learning
    train_examples = []
    for _, row in df.iterrows():
        # InputExample for Metric Learning (BatchHardTripletLoss) requires text and a label (int)
        train_examples.append(InputExample(texts=[row['review_text']], label=int(row['rating'])))

    # Create DataLoader
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    
    # Determine device
    device = get_device(purpose="rating prediction fine-tuning")

    # Load model
    logger.info(f"Loading base model: {model_name} on device: {device}")
    model = SentenceTransformer(model_name, device=device)
    
    # Define Loss
    # BatchHardTripletLoss is good for ordinal/class-based clustering
    train_loss = losses.BatchHardTripletLoss(model=model)
    
    # Train
    logger.info("Training model...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        output_path=output_path,
        show_progress_bar=True
    )
    
    logger.info(f"Model fine-tuned and saved to {output_path}")
    
    return output_path
