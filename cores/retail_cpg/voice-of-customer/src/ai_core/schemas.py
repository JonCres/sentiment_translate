import pandera.polars as pa
import polars as pl
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# --- Pydantic Models for Configuration ---

class MappingParams(BaseModel):
    """Configuration parameters for column mapping."""
    mandatory: Dict[str, Optional[str]] = Field(default_factory=dict, description="Mapping for mandatory skeleton columns")
    optional: Dict[str, Optional[str]] = Field(default_factory=dict, description="Mapping for optional columns")
    defaults: Dict[str, Any] = Field(default_factory=dict, description="Default values for missing columns")

class ValidationParams(BaseModel):
    """Configuration for review validation."""
    required_columns: List[str] = Field(..., description="List of columns that must exist")
    min_review_length: int = Field(10, description="Minimum character length for review text")
    max_review_length: int = Field(10000, description="Maximum character length for review text")
    rating_range: List[float] = Field([1.0, 5.0], description="Allowed range for ratings [min, max]")
    max_null_percentage: float = Field(0.2, description="Maximum allowed percentage of nulls in a column")

class CleaningParams(BaseModel):
    """Configuration for review cleaning."""
    remove_html: bool = True
    remove_urls: bool = True
    lowercase: bool = True
    remove_extra_spaces: bool = True

class TranslationParams(BaseModel):
    """Configuration for translation."""
    target_language: str = Field("en", description="Target language code (ISO 639-1)")
    enabled: bool = Field(True, description="Whether translation is enabled")
    source_language: str = Field("auto", description="Source language code or 'auto'")

class FeatureParams(BaseModel):
    """Configuration for feature engineering."""
    extract_length: bool = True
    extract_word_count: bool = True
    extract_exclamation_count: bool = True
    extract_question_count: bool = True
    extract_caps_ratio: bool = True


# --- Pandera Schemas for Data Validation ---

class VoCSkeletonSchema(pa.DataFrameModel):
    """Schema for Voice of Customer Skeleton Data."""
    Interaction_ID: str
    Interaction_Payload: str
    Customer_ID: str
    Timestamp: pl.Datetime
    Target_Object_ID: str
    Rating: float = pa.Field(ge=0.0, le=5.0, nullable=True) # Rating might be null? Docstring says "Numeric score if available"
    Channel_ID: str
    Language_Code: str

    class Config:
        strict = False # Allow extra columns
        coerce = True

class ReviewFeaturesSchema(pa.DataFrameModel):
    """Schema for engineered features."""
    review_length: Optional[pl.Int64] = pa.Field(ge=0)
    word_count: Optional[pl.Int64] = pa.Field(ge=0)
    caps_ratio: Optional[float] = pa.Field(ge=0.0, le=1.0)
    
    class Config:
        strict = False
        coerce = True