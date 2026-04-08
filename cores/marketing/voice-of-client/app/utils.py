import streamlit as st
import os
import json
import yaml
import polars as pl
from functools import lru_cache

# --- Data Loading Functions ---


@st.cache_data(ttl=3600)
def load_real_data():
    """Load enriched feedback data from Delta table."""
    try:
        path = "data/05_model_input/final_enriched_feedback.delta"
        if os.path.exists(path):
            df = pl.read_delta(path)
            return df.to_pandas()
    except Exception as e:
        st.error(f"Error loading real data: {e}")
    return None


@st.cache_data(ttl=3600)
def load_json_report(filename):
    """Load a JSON report from the reporting directory."""
    try:
        path = f"data/08_reporting/{filename}"
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        else:
            print(f"DEBUG: JSON report file not found: {path}")  # Debugging
            return {}
    except json.JSONDecodeError as e:
        print(f"DEBUG: JSON decoding error for {path}: {e}")  # Debugging
        return {}
    except Exception as e:
        st.warning(f"Could not load {filename}: {e}")
        print(f"DEBUG: Error loading {path}: {e}")  # Debugging
        return {}


@st.cache_data
def load_params():
    """Load configuration parameters."""
    try:
        with open("conf/base/parameters.yml", "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.warning(f"Could not load parameters: {e}")
        return {}


@lru_cache(maxsize=128)
def get_ai_interpretation(section: str, key: str = None, type: str = "slm") -> str:
    """Retrieve pre-generated AI interpretation (SLM or Groq) from the report."""
    interps = load_json_report("ai_interpretations.json")
    if not interps:
        print(
            f"DEBUG: ai_interpretations.json is empty or failed to load."
        )  # Debugging
        return "AI insights are being processed. Check back soon."

    # Check for Groq-specific insights if requested
    if type == "groq" and "groq" in interps:
        # feedback_insights are nested under groq
        if section == "feedback_insights":
            feedback_insights_data = interps["groq"].get("feedback_insights", {})
            result = feedback_insights_data.get(
                str(key), f"No AI insight available for interaction {key}."
            )
            print(
                f"DEBUG (feedback_insights): Requested key='{key}', Result='{result[:100]}...'"
            )  # Debugging
            return result

        section_data = interps["groq"].get(section)
    else:
        section_data = interps.get(section)

    if section_data is None:
        print(
            f"DEBUG: No section_data found for section='{section}' (type='{type}')."
        )  # Debugging
        return f"No AI insight available for {section}."

    if key and isinstance(section_data, dict):
        result = section_data.get(str(key), f"No AI insight available for {key}.")
        if not result:
            return "AI Insight is being generated. Please check back after the next pipeline run."
        print(
            f"DEBUG (general): Requested section='{section}', key='{key}', Result='{result[:100]}...'"
        )  # Debugging
        return result

    if not section_data:
        return "AI Insight is being generated. Please check back after the next pipeline run."

    return (
        section_data
        if isinstance(section_data, str)
        else "AI interpretation format mismatch."
    )
