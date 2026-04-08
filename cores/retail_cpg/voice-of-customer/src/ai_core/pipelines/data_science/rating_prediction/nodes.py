import polars as pl
import pandas as pd
import logging
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, make_scorer, f1_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sentence_transformers import SentenceTransformer, models
import os
from utils import get_device, clear_device_cache

logger = logging.getLogger(__name__)


class SentenceEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        self.model_name = model_name
        self.model = None

    def fit(self, X, y=None):
        clear_device_cache()
        # We don't fit the sentence transformer here, we load it.
        # To enforce mean pooling, we construct the model from modules.
        # Check if model_name is a local path or huggingface ID
        if os.path.exists(self.model_name):
            logger.info(f"Loading local fine-tuned model from: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        else:
            logger.info(f"Loading pretrained model: {self.model_name}")
            word_embedding_model = models.Transformer(self.model_name)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode_mean_tokens=True,
                pooling_mode_cls_token=False,
                pooling_mode_max_tokens=False,
            )
            self.model = SentenceTransformer(
                modules=[word_embedding_model, pooling_model]
            )
        return self

    def transform(self, X):
        if self.model is None:
            self.fit(X)

        # Ensure X is a list of strings
        if isinstance(X, pd.Series):
            X = X.tolist()
        elif isinstance(X, pl.Series):
            X = X.to_list()

        # Determine device
        device = get_device("rating prediction embeddings")
        self.model.to(device)

        embeddings = self.model.encode(
            X, normalize_embeddings=True, show_progress_bar=True
        )
        return embeddings


def split_data(
    df: pl.DataFrame, parameters: Dict[str, Any]
) -> Tuple[Any, Any, Any, Any]:
    """Splits data into features and targets training and test sets.

    Args:
        df: Data containing features and target.
        parameters: Parameters defined in parameters.yml.

    Returns:
        Split data: X_train, X_test, y_train, y_test.
    """
    logger.info("Splitting data for rating prediction model...")

    # Filter out rows with missing ratings or text
    df = df.drop_nulls(subset=["review_text", "rating"])

    # Extract as numpy/pandas or list for sklearn
    X = df["review_text"].to_list()
    y = df["rating"].to_list()

    test_size = parameters["model"]["test_size"]
    random_state = parameters["model"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def train_model(
    X_train: Any, X_test: Any, y_train: Any, y_test: Any, parameters: Dict[str, Any]
) -> Tuple[Any, Dict[str, Any]]:
    """Trains the rating prediction model using sentence embeddings.

    Args:
        X_train: Training data of features (text).
        X_test: Testing data of features (text).
        y_train: Training data of target.
        y_test: Testing data of target.
        parameters: Parameters of the pipeline.

    Returns:
        Trained model and metrics.
    """
    logger.info("Training rating prediction model (Classification)...")

    use_fine_tuned = parameters["model"].get("use_fine_tuned_model", False)

    if use_fine_tuned:
        embedding_model_name = parameters["fine_tuning"].get(
            "output_path", "data/06_models/fine_tuned_embeddings"
        )
        logger.info(f"Using fine-tuned model path: {embedding_model_name}")
    else:
        embedding_model_name = parameters["model"].get(
            "embedding_model", "BAAI/bge-small-en-v1.5"
        )
        logger.info(f"Using pretrained model: {embedding_model_name}")

    random_state = parameters["model"]["random_state"]

    # Define parameter grid for RandomForest hyperparameter tuning
    param_grid = {
        "classifier__n_estimators": parameters["model"]["grid_search"][
            "classifier__n_estimators"
        ],
        "classifier__max_depth": [
            None if x == "None" else x
            for x in parameters["model"]["grid_search"]["classifier__max_depth"]
        ],
        "classifier__min_samples_split": parameters["model"]["grid_search"][
            "classifier__min_samples_split"
        ],
        "classifier__min_samples_leaf": parameters["model"]["grid_search"][
            "classifier__min_samples_leaf"
        ],
        "classifier__max_features": parameters["model"]["grid_search"][
            "classifier__max_features"
        ],
    }

    # Create F1 scorer with macro average (better for imbalanced multiclass ratings)
    f1_scorer = make_scorer(f1_score, average="macro")

    # Create pipeline with custom embedding transformer
    pipeline = Pipeline(
        [
            (
                "embedding",
                SentenceEmbeddingTransformer(model_name=embedding_model_name),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=parameters["model"]["random_state"],
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Perform GridSearchCV with F1 scoring and 5-fold CV
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        scoring=f1_scorer,
        cv=parameters["model"]["grid_search"]["cv"],
        n_jobs=parameters["model"]["grid_search"]["n_jobs"],
        verbose=parameters["model"]["grid_search"]["verbose"],
    )

    # Fit grid search
    grid_search.fit(X_train, y_train)

    # Best model and results
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Evaluate with F1 focus
    f1_macro = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "best_params": grid_search.best_params_,
        "best_f1_score": grid_search.best_score_,
        "test_f1_macro": f1_macro,
        "classification_report": report,
    }

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best cross-validation F1 (macro): {grid_search.best_score_}")
    logger.info(f"Test F1 (macro): {f1_macro}")

    return best_model, metrics
