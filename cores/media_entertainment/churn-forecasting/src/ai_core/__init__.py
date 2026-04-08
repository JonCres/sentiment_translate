__version__ = "0.1"

import os
import warnings

# Suppress Rust/Delta Lake noise from polars/deltalake bindings
os.environ["RUST_LOG"] = "error"

# Suppress Feast/Pydantic deprecation warnings
# Using generic Warning category to catch PydanticDeprecatedSince212
warnings.filterwarnings("ignore", message=".*model_validator.*", category=Warning)
