"""
Aspect-Based Sentiment Analysis (ABSA) Pipeline

Extracts aspects from customer reviews and analyzes sentiment per aspect.
Uses DeBERTa-based ABSA models for aspect-level sentiment classification.

Based on Section 1.1 of the Voice of Customer AI Technical Walkthrough.
"""
from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
