"""
Multimodal Emotion Recognition (MER) Pipeline

Placeholder module for future audio/video emotion recognition capabilities.
Based on Section 1.2 of the Voice of Customer AI Technical Walkthrough.

When enabled, this module will:
- Extract acoustic features from audio (MFCC, pitch, energy)
- Extract visual features from video (facial expressions)
- Fuse multimodal features with text for emotion classification

IMPORTANT: This module requires audio/video data sources and additional
dependencies (librosa, opencv, etc.). Currently disabled until data is available.
"""

# Feature flag - set to True when audio/video data and dependencies are available
ENABLED = True

# Required dependencies for MER (not installed by default)
REQUIRED_DEPENDENCIES = [
    "librosa>=0.10.0",      # Audio processing
    "opencv-python>=4.8.0", # Video processing  
    "facenet-pytorch>=2.5.0", # Face detection
    "torchaudio>=2.0.0",    # Audio neural networks
]

__all__ = ["ENABLED", "create_pipeline"]


def create_pipeline(**kwargs):
    """Create the MER pipeline.
    
    Returns an empty pipeline since MER is disabled.
    Enable by setting ENABLED=True after installing dependencies
    and providing audio/video data sources.
    """
    from kedro.pipeline import Pipeline
    
    if not ENABLED:
        # Return empty pipeline when disabled
        return Pipeline([])
    
    # When enabled, this will import and create the actual pipeline
    from .pipeline import create_mer_pipeline
    return create_mer_pipeline(**kwargs)
