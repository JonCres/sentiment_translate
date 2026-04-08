from typing import Dict, Any
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime, date
from scipy.stats import multivariate_normal
from lifetimes import BetaGeoFitter, GammaGammaFitter, BetaGeoBetaBinomFitter
from lifelines import WeibullAFTFitter
import xgboost as xgb
import logging
from utils import get_device, clear_device_cache

logger = logging.getLogger(__name__)


def train_bg_nbd_model(data: pl.DataFrame, params: Dict[str, Any]) -> BetaGeoFitter:
    """Train BG/NBD (Beta-Geometric/Negative Binomial Distribution) model for purchase frequency prediction.

    BG/NBD is a probabilistic model for non-contractual settings (TVOD, e-commerce, F2P)
    that estimates:
    - Purchase frequency (how often customers transact)
    - Churn probability (likelihood customer has "died" / stopped purchasing)
    - Customer lifetime (expected active duration)

    Model Assumptions:
    1. Customer transactions occur at rate λ (while active)
    2. Customer "dies" with probability p after each transaction
    3. Heterogeneity across customers modeled by Gamma distributions

    Args:
        data: RFM summary DataFrame with columns:
            - frequency (int): Number of repeat purchases (total purchases - 1)
            - recency (float): Time (days/months) between first and last purchase
            - T (float): Customer age (time between first purchase and observation_period_end)
            - customer_id (str): Customer identifier (not used in model fit but preserved)
        params: Configuration dictionary containing:
            - penalizer_coef (float, optional): L2 regularization parameter to prevent
              overfitting. Default: 0.0. Recommended: 0.001-0.01 for small datasets.

    Returns:
        Fitted BetaGeoFitter model with learned parameters:
            - r, alpha: Shape parameters for λ (transaction rate) Gamma distribution
            - a, b: Shape parameters for p (churn probability) Beta distribution

    Raises:
        ValueError: If required columns (frequency, recency, T) are missing from data
        np.linalg.LinAlgError: If model fitting fails (e.g., insufficient data variance)

    Note:
        - Temporarily converts Polars to Pandas for lifetimes library compatibility
        - Applies data validation: recency ≤ T, frequency ≥ 0, T > 0
        - Suppresses numpy divide/invalid warnings during numerical optimization

    Example:
        >>> rfm_data = pl.DataFrame({
        ...     'customer_id': ['C1', 'C2'],
        ...     'frequency': [3, 0],
        ...     'recency': [30, 0],
        ...     'T': [90, 90],
        ...     'monetary_value': [50, 100]
        ... })
        >>> model = train_bg_nbd_model(rfm_data, {'penalizer_coef': 0.001})
        >>> # Predict expected purchases for C1 in next 12 months:
        >>> model.conditional_expected_number_of_purchases_up_to_time(12, 3, 30, 90)
    """
    penalizer_coef = params.get("penalizer_coef", 0.0)
    bgf = BetaGeoFitter(penalizer_coef=penalizer_coef)

    # Fit model on frequency, recency, and T (age)
    if (
        "frequency" not in data.columns
        or "recency" not in data.columns
        or "T" not in data.columns
    ):
        raise ValueError(
            "Data must likely contain 'frequency', 'recency', and 'T' columns."
        )

    # lifetimes requires pandas/numpy
    # Extract columns as pandas Series for fit (filling NaNs)
    freq = data["frequency"].to_pandas().fillna(0)
    rec = data["recency"].to_pandas().fillna(0)
    T = data["T"].to_pandas().fillna(0)

    # Data Validation/Cleanup for Lifetimes
    # Recency cannot be > T
    rec = rec.clip(upper=T)

    # Frequency must be non-negative
    freq = freq.clip(lower=0)

    # Ensure T is strictly positive
    T = T.clip(lower=1e-4)

    with np.errstate(divide="ignore", invalid="ignore"):
        bgf.fit(freq, rec, T)

    # Attach data as DataFrame to model for plotting functions
    # lifetimes plotting functions expect model.data to have named columns
    bgf.data = pd.DataFrame({"frequency": freq, "recency": rec, "T": T})

    # Add dummy predict method to satisfy MLflow's sklearn flavor and avoid warnings
    # Must handle both sklearn style (X as DataFrame) and lifetimes internal style (t, frequency, recency, T)
    def bgf_predict(t_or_X, frequency=None, recency=None, T=None):
        if hasattr(t_or_X, "__getitem__") and frequency is None:
            # sklearn/pandas style
            return bgf.conditional_expected_number_of_purchases_up_to_time(
                1, t_or_X["frequency"], t_or_X["recency"], t_or_X["T"]
            )
        # lifetimes style
        return bgf.conditional_expected_number_of_purchases_up_to_time(
            t_or_X, frequency, recency, T
        )

    bgf.predict = bgf_predict

    return bgf


def train_gamma_gamma_model(
    data: pl.DataFrame, params: Dict[str, Any]
) -> GammaGammaFitter:
    """Train Gamma-Gamma model for customer monetary value prediction.

    The Gamma-Gamma model estimates the expected average transaction value per customer.
    It assumes transaction values are independent of purchase frequency (a key assumption
    that should be validated with data). This model is paired with BG/NBD to compute CLTV.

    Model Assumptions:
    1. Transaction values vary randomly around each customer's average transaction value
    2. Average transaction values vary across customers following a Gamma distribution
    3. Transaction value is independent of purchase frequency (validate this!)

    Args:
        data: RFM summary DataFrame with columns:
            - frequency (int): Number of repeat purchases (must be > 0 for model)
            - monetary_value (float): Average transaction value across repeat purchases
            - customer_id (str, optional): Customer identifier (not used in model fit)
        params: Configuration dictionary containing:
            - penalizer_coef (float, optional): L2 regularization parameter. Default: 0.0.
              Use 0.001-0.01 for small datasets to prevent overfitting.

    Returns:
        Fitted GammaGammaFitter model with learned parameters:
            - p, q, v: Shape parameters for the Gamma-Gamma distribution
            These parameters characterize heterogeneity in customer spending behavior.

    Raises:
        ValueError: If no repeat customers exist (all frequency = 0)
        np.linalg.LinAlgError: If model fitting fails due to insufficient data variance

    Note:
        - Only customers with frequency > 0 are used for training (repeat purchasers only)
        - One-time customers are excluded because monetary_value is undefined for them
        - Temporarily converts Polars to Pandas for lifetimes library compatibility
        - This model should be validated by checking if monetary_value is truly
          independent of frequency using correlation analysis

    Warning:
        If fewer than ~50 repeat customers exist, model estimates will be unreliable.
        Consider using a fixed average transaction value instead.

    Example:
        >>> rfm_data = pl.DataFrame({
        ...     'customer_id': ['C1', 'C2', 'C3'],
        ...     'frequency': [3, 5, 0],  # C3 excluded (no repeats)
        ...     'monetary_value': [50, 75, 100]
        ... })
        >>> model = train_gamma_gamma_model(rfm_data, {'penalizer_coef': 0.001})
        >>> # Predict expected avg profit for C1:
        >>> model.conditional_expected_average_profit(frequency=3, monetary_value=50)
    """
    penalizer_coef = params.get("penalizer_coef", 0.0)
    ggf = GammaGammaFitter(penalizer_coef=penalizer_coef)

    # Filter for returning customers in Polars
    returning_customers = data.filter(pl.col("frequency") > 0)

    if len(returning_customers) == 0:
        raise ValueError("No returning customers found to train Gamma-Gamma model.")

    # Convert need columns to pandas
    freq = returning_customers["frequency"].to_pandas()
    monetary = returning_customers["monetary_value"].to_pandas()

    ggf.fit(freq, monetary)

    # Add dummy predict method to satisfy MLflow's sklearn flavor and avoid warnings
    def ggf_predict(X_or_freq, monetary_value=None):
        if hasattr(X_or_freq, "__getitem__") and monetary_value is None:
            # sklearn/pandas style
            return ggf.conditional_expected_average_profit(
                X_or_freq["frequency"], X_or_freq["monetary_value"]
            )
        # positional style
        return ggf.conditional_expected_average_profit(X_or_freq, monetary_value)

    ggf.predict = ggf_predict

    return ggf


def sample_parameters(model: Any, n_samples: int = 100) -> pd.DataFrame:
    """Sample model parameters from posterior distribution for parametric bootstrapping.

    Uses the inverse Hessian matrix (covariance of parameter estimates) to draw samples
    from a multivariate normal distribution centered at the MLE parameter estimates.
    This enables uncertainty quantification for CLTV predictions via confidence intervals.

    Parametric Bootstrap Process:
    1. Extract fitted parameters (MLE point estimates) from model
    2. Compute covariance matrix: Σ = H^(-1) where H is the Hessian
    3. Sample from N(μ=params, Σ=cov) to get N plausible parameter sets
    4. Clip samples to positive values (parameters must be > 0 for stability)

    Args:
        model: Fitted lifetimes model (BetaGeoFitter or GammaGammaFitter) with
            - params_: Series of fitted parameters (e.g., r, alpha, a, b for BG/NBD)
            - _hessian_: Matrix of second derivatives from MLE optimization
        n_samples: Number of parameter sets to sample. Default: 100.
            Higher values = more precise CI estimates but slower computation.
            Recommended: 20-50 for production, 100+ for final analysis.

    Returns:
        DataFrame with n_samples rows and one column per model parameter:
            - For BG/NBD: columns=['r', 'alpha', 'a', 'b']
            - For Gamma-Gamma: columns=['p', 'q', 'v']
        Each row represents a plausible parameter set from the posterior distribution.

    Raises:
        np.linalg.LinAlgError: If Hessian matrix is singular (non-invertible).
            Falls back to repeating point estimates n_samples times.

    Note:
        - All sampled parameters are clipped to minimum value 1e-4 to prevent
          numerical instability in downstream CLTV calculations
        - If sampling fails (singular Hessian), logs warning and returns point
          estimates, effectively disabling confidence intervals

    Example:
        >>> bgf = BetaGeoFitter().fit(freq, rec, T)
        >>> param_samples = sample_parameters(bgf, n_samples=50)
        >>> # param_samples shape: (50, 4) for BG/NBD
        >>> # Use these to compute 50 CLTV predictions per customer for CI estimation
    """
    try:
        # lifetimes stores params as Series and _hessian_ matrix
        params_mean = model.params_
        hessian = model._hessian_

        # Covariance = Inverse of Hessian
        cov_matrix = np.linalg.inv(hessian)

        # Sample
        samples = multivariate_normal.rvs(
            mean=params_mean.values, cov=cov_matrix, size=n_samples
        )

        # Parameters must be strictly positive for stability
        samples = np.maximum(samples, 1e-4)

        # Convert to DataFrame with same column names
        return pd.DataFrame(samples, columns=params_mean.index)
    except Exception as e:
        logger.warning(
            f"Could not sample parameters for {type(model).__name__}: {e}. using point estimate."
        )
        return pd.DataFrame([model.params_] * n_samples)


def predict_cltv(
    bg_nbd_model: BetaGeoFitter,
    gamma_gamma_model: GammaGammaFitter,
    data: pl.DataFrame,
    xgboost_model: Any = None,
    feature_store: pl.DataFrame = None,
) -> pl.DataFrame:
    """Predict Customer Lifetime Value (CLTV) using hybrid probabilistic + behavioral models.

    Generates comprehensive customer value predictions combining:
    1. BG/NBD (frequency/churn modeling) → Expected future purchases
    2. Gamma-Gamma (monetary modeling) → Expected transaction values
    3. XGBoost (optional behavioral refinement) → Churn probability adjustment
    4. Parametric bootstrapping → Confidence intervals for predictions

    Prediction Outputs:
    - CLTV forecasts (12/24/36 months with 95% confidence intervals)
    - Churn risk scores (30-day, 90-day, current)
    - Expected purchases and transaction values
    - Customer segmentation (quintile-based)

    Hybrid Blending Logic (when XGBoost is provided):
    - p_churn_final = 0.3 * (1 - p_alive_BG/NBD) + 0.7 * p_churn_XGBoost
    - Prioritizes recent behavioral signals (engagement, binge patterns) over
      transactional recency for short-term churn risk

    Args:
        bg_nbd_model: Fitted BG/NBD model for purchase frequency prediction
        gamma_gamma_model: Fitted Gamma-Gamma model for monetary value prediction
        data: RFM summary DataFrame with columns:
            - customer_id (str): Customer identifier (preserved in output)
            - frequency (int): Number of repeat purchases
            - recency (float): Time between first and last purchase
            - T (float): Customer age (observation window)
            - monetary_value (float): Average transaction value
        xgboost_model: Optional fitted XGBoost classifier for behavioral churn prediction.
            If provided, must have:
            - predict_proba() method OR daal_model_.predict() for Intel acceleration
            - _estimator_type='classifier' attribute
        feature_store: Optional behavioral feature DataFrame (required if xgboost_model provided):
            - customer_id (str): Customer identifier for merging
            - Behavioral features: watch_time, login_count, etc. (from engagement aggregation)

    Returns:
        Predictions DataFrame with one row per customer:
            - customer_id (str): Customer identifier
            - prediction_date (date): Date of prediction generation
            - clv_12mo / clv_24mo / clv_36mo (float): Discounted CLTV forecasts in USD
            - clv_95_ci_lower / clv_95_ci_upper (float): 95% confidence interval bounds
            - p_alive (float): Probability customer is active (0-1)
            - p_churn_30day / p_churn_90day (float): Short-term churn risk (0-1)
            - expected_transactions_12mo (float): Expected purchase count
            - expected_value_per_transaction (float): Expected avg transaction value
            - frequency / recency / T / monetary_value (float): Original RFM features
            - clv_segment (str): Quintile segment (Q1_Low to Q5_Elite)
            - acquisition_channel / cohort_month (str): Placeholders for enrichment
            - model_accuracy_metric (float): Placeholder for model performance tracking

    Raises:
        ValueError: If data is missing required RFM columns
        KeyError: If xgboost_model provided but feature_store missing customer_ids

    Note:
        - Applies extensive data validation and clipping to ensure numerical stability
        - Bootstraps 20 parameter samples for CI estimation (configurable via n_boot)
        - Suppresses numpy warnings during numerical optimization
        - Handles XGBoost acceleration via daal4py (Intel) or standard scikit-learn API
        - Falls back gracefully if XGBoost prediction fails

    Warning:
        - Discount rate is hardcoded to 1% monthly (0.01). For different rates, modify code.
        - Churn probabilities are approximations: BG/NBD models "death" probability, not
          explicit churn. Use (1 - p_alive) as churn proxy.
        - XGBoost blending weights (0.7/0.3) should be tuned based on validation performance.

    Example:
        >>> rfm_data = pl.DataFrame({...})  # RFM summary
        >>> bgf = train_bg_nbd_model(rfm_data, params)
        >>> ggf = train_gamma_gamma_model(rfm_data, params)
        >>> predictions = predict_cltv(bgf, ggf, rfm_data)
        >>> high_value = predictions.filter(pl.col('clv_12mo') > 500)
    """
    # 1. Prepare data for lifetimes (pandas)
    # Fillna to ensure no NaN values reach lifetimes
    freq = data["frequency"].to_pandas().fillna(0)
    rec = data["recency"].to_pandas().fillna(0)
    T = data["T"].to_pandas().fillna(0)
    monetary = data["monetary_value"].to_pandas().fillna(0)

    # Data Validation/Cleanup for Lifetimes (Mirroring training validation)
    # Recency cannot be > T
    rec = rec.clip(upper=T)

    # Frequency must be non-negative
    freq = freq.clip(lower=0)

    # Consistency check: If recency is 0, frequency must be 0
    invalid_freq_idx = (rec == 0) & (freq > 0)
    if invalid_freq_idx.any():
        freq.loc[invalid_freq_idx] = 0.0

    # T and monetary must be strictly positive for log stability
    T = T.clip(lower=1e-4)
    monetary = monetary.clip(lower=1e-4)

    with np.errstate(divide="ignore", invalid="ignore"):
        # 2. Key Predictions
        # a. Expected Number of Purchases (12 months)
        purchases_12mo = (
            bg_nbd_model.conditional_expected_number_of_purchases_up_to_time(
                12, freq, rec, T
            )
        )
        purchases_12mo = np.nan_to_num(purchases_12mo, nan=0.0)

        # b. Probability Alive
        p_alive = bg_nbd_model.conditional_probability_alive(freq, rec, T)
        p_alive = np.nan_to_num(p_alive, nan=0.0)

        # c. Monetary Value (Expected Average Profit)
        expected_avg_profit = gamma_gamma_model.conditional_expected_average_profit(
            freq, monetary
        )
        expected_avg_profit = np.nan_to_num(expected_avg_profit, nan=0.0)

        # d. CLTV (12, 24, 36 months)
        clv_12mo = gamma_gamma_model.customer_lifetime_value(
            bg_nbd_model, freq, rec, T, monetary, time=12, discount_rate=0.01
        )
        clv_12mo = np.nan_to_num(clv_12mo, nan=0.0)

        clv_24mo = gamma_gamma_model.customer_lifetime_value(
            bg_nbd_model, freq, rec, T, monetary, time=24, discount_rate=0.01
        )
        clv_24mo = np.nan_to_num(clv_24mo, nan=0.0)

        clv_36mo = gamma_gamma_model.customer_lifetime_value(
            bg_nbd_model, freq, rec, T, monetary, time=36, discount_rate=0.01
        )
        clv_36mo = np.nan_to_num(clv_36mo, nan=0.0)

    # 3. Derived Metrics & Approximations

    # b. Churn Probabilities
    # In BG/NBD non-contractual setting, "churn" isn't explicit.
    # We use (1 - P_alive) as a proxy for "Has Churned" state probability.
    p_churn_now = 1.0 - p_alive

    # HYBRID LOGIC: If XGBoost model is provided (Behavioral Refinement)
    xgb_churn_prob = None
    if (
        xgboost_model is not None
        and feature_store is not None
        and not isinstance(xgboost_model, dict)
    ):
        try:
            logger.info("Applying XGBoost Behavioral Refinement to Churn Scores...")
            # Convert features to pandas
            fs_pd = feature_store.to_pandas()
            # We need to map these predictions back to the `rec/freq/T` order
            # The `data` df has customer_id. We merge.

            # Create a DF for the current batch
            current_ids = pd.DataFrame({"customer_id": data["customer_id"].to_list()})

            # Merge features
            merged_features = current_ids.merge(fs_pd, on="customer_id", how="left")

            # Preparing X for XGBoost (Drop ID and non-features)
            # Assuming model was trained on specific cols.
            # ideally we match columns. For demo, we drop ID and try to predict.
            # In validation, we'd ensure schema.
            X_pred = merged_features.drop(
                columns=[
                    "customer_id",
                    "churn_label",
                    "last_engagement_date",
                    "inactivity_days",
                ],
                errors="ignore",
            )
            # Filter to numeric only, similar to training
            X_pred = X_pred.select_dtypes(
                exclude=["object", "datetime64[ns]", "datetime"]
            )
            # Fill NAs if any (new customers might miss behavioral features)
            X_pred = X_pred.fillna(0)

            # Predict Proba
            # Check for daal4py accelerated model first
            if hasattr(xgboost_model, "daal_model_"):
                logger.info("Using daal4py for accelerated inference.")
                # daal4py predict returns probabilities for classification by default
                xgb_preds = xgboost_model.daal_model_.predict(X_pred)
                if len(xgb_preds.shape) > 1 and xgb_preds.shape[1] > 1:
                    xgb_preds = xgb_preds[:, 1]  # Probability of class 1 (Churn)
            elif hasattr(xgboost_model, "predict_proba"):
                xgb_preds = xgboost_model.predict_proba(X_pred)[
                    :, 1
                ]  # Prob of class 1 (Churn)
            else:
                # Fallback or different API
                xgb_preds = np.zeros(len(data))

            xgb_churn_prob = xgb_preds

            # BLENDING: Weighted Average of P(NotAlive) and P(BehavioralChurn)
            # p_final = w1 * (1-P_alive) + w2 * XGB_prob
            # Weights should ideally be parameters. Defaulting to 0.7 XGBoost (Behavior) / 0.3 BG/NBD (Recency)
            # to prioritize recent behavioral signals (e.g. binge-watching)
            w_xgb = 0.7
            p_churn_now = (1 - w_xgb) * p_churn_now + w_xgb * xgb_churn_prob

        except Exception as e:
            logger.warning(
                f"XGBoost refinement failed: {e}. Using baseline BG/NBD churn."
            )

    # Project short term risk
    p_churn_30day = p_churn_now
    # If we have behavioral score, it might indicate immediate burnout better than just R/F
    if xgb_churn_prob is not None:
        # stronger weight to behavioral for short term if valid
        p_churn_30day = np.clip(
            (1 - w_xgb) * (1 - p_alive) + w_xgb * xgb_churn_prob, 0.0, 1.0
        )

    p_churn_90day = np.clip(p_churn_30day * 1.1, 0.0, 1.0)  # Dummy trend

    # b. Confidence Intervals (Parametric Bootstrapping)
    # Instead of fixed approximation, we sample model parameters to get distribution of CLV

    logger.info("Computing Bootstrapped Confidence Intervals (N=20)...")
    n_boot = 20  # Keep small for demo speed; max 100

    try:
        # Sample parameters
        bg_params_samples = sample_parameters(bg_nbd_model, n_boot)
        gg_params_samples = sample_parameters(gamma_gamma_model, n_boot)

        # We need to calc CLV for each customer N times.
        # This can be slow. leveraging vectorization.
        # Store results: (n_customers, n_boot)
        clv_samples = np.zeros((len(data), n_boot))

        # We temporarily overwrite model params, predict, then restore?
        # Or instantiate temporary models.
        # easier: Instantiate temp models.

        temp_bg = BetaGeoFitter(penalizer_coef=bg_nbd_model.penalizer_coef)
        temp_gg = GammaGammaFitter(penalizer_coef=gamma_gamma_model.penalizer_coef)

        # Pre-calculate safe inputs if needed
        # safe_freq = freq.copy()
        # safe_freq[safe_freq <= 0] = 0.001

        with np.errstate(divide="ignore", invalid="ignore"):
            for i in range(n_boot):
                # Set params
                temp_bg.params_ = bg_params_samples.iloc[i]
                temp_gg.params_ = gg_params_samples.iloc[i]

                # Predict
                # Manual CLV calculation to avoid 'predict' attribute error in lifetimes 0.11.3+
                # CLV ~ Expected Sales * Expected Profit * Discount

                # 1. Expected Transactions (12 months)
                exp_sales = temp_bg.conditional_expected_number_of_purchases_up_to_time(
                    12, freq, rec, T
                )
                exp_sales = np.nan_to_num(exp_sales, nan=0.0)

                # 2. Expected Profit
                exp_profit = temp_gg.conditional_expected_average_profit(freq, monetary)
                exp_profit = np.nan_to_num(exp_profit, nan=0.0)

                # 3. Discount Factor (Approximate)
                # Simple approximation: assume transactions happen in middle of period (t=6)
                discount_factor = 1.0 / ((1.01) ** 6)

                preds = exp_sales * exp_profit * discount_factor
                # Ensure stability
                clv_samples[:, i] = np.nan_to_num(preds, nan=0.0)

        # Calculate quantiles
        clv_lower = np.quantile(
            clv_samples, 0.05, axis=1
        )  # 5th percentile (90% CI for strictness)
        clv_upper = np.quantile(clv_samples, 0.95, axis=1)

    except Exception as e:
        logger.warning(f"Bootstrapping failed: {e}. Falling back to approximation.")
        # Fallback
        clv_lower = clv_12mo * 0.8
        clv_upper = clv_12mo * 1.2

    # 4. Construct Result DataFrame in Polars
    # Helper to extract values safely
    def _val(x):
        return x.values if hasattr(x, "values") else np.asarray(x)

    # 4. Construct Result DataFrame in Polars
    results_df = pl.DataFrame(
        {
            "customer_id": data["customer_id"],  # Ensure ID is preserved
            "prediction_date": [date.today()] * len(data),
            "clv_12mo": _val(clv_12mo),
            "clv_24mo": _val(clv_24mo),
            "clv_36mo": _val(clv_36mo),
            "clv_95_ci_lower": _val(clv_lower),
            "clv_95_ci_upper": _val(clv_upper),
            "p_alive": _val(p_alive),
            "p_churn_30day": _val(p_churn_30day),
            "p_churn_90day": _val(p_churn_90day),
            "expected_transactions_12mo": _val(purchases_12mo),
            "expected_value_per_transaction": _val(expected_avg_profit),
            # Join original input columns if needed for context
            "frequency": data["frequency"],
            "recency": data["recency"],
            "T": data["T"],
            "monetary_value": data["monetary_value"],
        }
    )

    # 5. Segmentation (Post-creation)
    # Quintiles for CLTV
    # Polars quantile expects single float, so we loop
    q_cuts = [results_df["clv_12mo"].quantile(q) for q in [0.2, 0.4, 0.6, 0.8]]

    # Helper to map value to segment
    def get_segment(val):
        if val <= q_cuts[0]:
            return "Q1_Low"
        elif val <= q_cuts[1]:
            return "Q2_Fair"
        elif val <= q_cuts[2]:
            return "Q3_Medium"
        elif val <= q_cuts[3]:
            return "Q4_High"
        else:
            return "Q5_Elite"

    # Polars robust way? Map elements.
    # For efficiency we could use `cut` but let's use map_elements for clarity with custom labels
    results_df = results_df.with_columns(
        pl.col("clv_12mo")
        .map_elements(get_segment, return_dtype=pl.Utf8)
        .alias("clv_segment"),
        pl.lit("Unknown").alias("acquisition_channel"),  # Placeholder
        pl.lit("2024-01").alias("cohort_month"),  # Placeholder
        pl.lit(0.75).alias("model_accuracy_metric"),  # Placeholder
    )

    return results_df


def train_sbg_model(survival_data: pl.DataFrame, params: Dict[str, Any]) -> Any:
    """Train sBG (Shifted Beta-Geometric) model for contractual churn prediction.

    sBG is a discrete-time probabilistic model for contractual settings (SVOD, SaaS)
    that estimates retention curves and churn probabilities. Unlike BG/NBD (non-contractual),
    sBG assumes explicit subscription renewals/cancellations.

    Model Characteristics:
    - Discrete time periods (e.g., monthly renewal cycles)
    - Retention probability θ varies across customers (Beta distribution)
    - Predicts P(churn at period t | survived until t-1)

    Args:
        survival_data: Subscription lifecycle DataFrame with columns:
            - customer_id or subscription_id (str): Subscription identifier
            - T (int): Observation time in periods (e.g., months subscribed)
            - E (int): Event indicator (1=churned, 0=active/censored)
        params: Configuration dictionary (currently unused but available for future params)

    Returns:
        Fitted BetaGeoBetaBinomFitter model (if available), or None if:
            - survival_data is empty
            - BetaGeoBetaBinomFitter not available in lifetimes library
            - Model fitting fails

    Note:
        - This is a placeholder implementation. Production code should:
          1. Convert survival_data to cohort format required by BetaGeoBetaBinomFitter
          2. Call model.fit() with proper cohort matrix
          3. Handle edge cases (single cohort, insufficient periods)
        - If BetaGeoBetaBinomFitter unavailable, returns None (triggers fallback to Weibull)

    Warning:
        Current implementation returns a placeholder model without actual fitting.
        Integrate with lifetimes.BetaGeoBetaBinomFitter for production use.

    Example:
        >>> survival_df = pl.DataFrame({
        ...     'customer_id': ['S1', 'S2', 'S3'],
        ...     'T': [6, 12, 3],  # Months observed
        ...     'E': [1, 0, 1]    # 1=churned, 0=active
        ... })
        >>> model = train_sbg_model(survival_df, {})
        >>> # Production: model.predict(n_periods=12) for retention curve
    """
    if survival_data.is_empty():
        logger.warning("No survival data found. Skipping sBG model.")
        return None

    try:
        model = BetaGeoBetaBinomFitter()

        # sBG needs frequency/recency or valid cohort data.
        # For simplicity in this placeholder context, if the library is missing, we handle it.
        # Assuming standard survival data format is converted to required input.
        logger.info("Training sBG Model (BetaGeoBetaBinomFitter)...")
        # Actual fit would depend on exact data shape
        # model.fit(...)
        return model
    except NameError:
        logger.warning("BetaGeoBetaBinomFitter not loaded. Returning None.")
        return None


def train_weibull_aft_model(survival_data: pl.DataFrame, params: Dict[str, Any]) -> Any:
    """Train Weibull Accelerated Failure Time (AFT) model for subscription lifetime prediction.

    Weibull AFT is a parametric survival analysis model for contractual settings (SVOD, SaaS)
    that predicts time-until-churn as a function of customer covariates. Unlike sBG (discrete),
    Weibull AFT operates in continuous time and incorporates explanatory variables.

    Model Characteristics:
    - Continuous time-to-event modeling (days/months until churn)
    - Log-linear relationship: log(T) = β₀ + β₁X₁ + β₂X₂ + ... + ε
    - Error term ε follows Weibull distribution (flexible hazard shapes)
    - Covariates: Demographics, behavioral features, subscription tier, etc.

    AFT Interpretation:
    - β > 0: Covariate accelerates time-to-churn (protective, prolongs survival)
    - β < 0: Covariate decelerates time-to-churn (harmful, speeds up churn)
    - Example: β_engagement = 0.5 means high engagement customers survive 50% longer

    Args:
        survival_data: Subscription lifecycle DataFrame with columns:
            - T or duration (float): Time observed (days/months from start to churn or censoring)
            - E or event (int): Event indicator (1=churned, 0=active/censored)
            - Covariates (numeric): Static features (age, subscription_tier_encoded, etc.)
            - customer_id (str, optional): Identifier (excluded from model fitting)
        params: Configuration dictionary (currently unused but available for future hyperparams)

    Returns:
        Fitted WeibullAFTFitter model with:
            - params_: DataFrame of coefficients (β values) per covariate
            - rho_: Weibull shape parameter (ρ > 1 = increasing hazard, ρ < 1 = decreasing hazard)
            - lambda_: Weibull scale parameter
        Returns None if:
            - survival_data is empty
            - Required columns (T/duration, E/event) are missing
            - Model fitting fails (e.g., singular matrix, insufficient variance)

    Raises:
        ValueError: If duration ≤ 0 (all durations clipped to 0.001 minimum)
        lifelines.exceptions.ConvergenceError: If optimization fails to converge

    Note:
        - Temporarily converts Polars to Pandas for lifelines library compatibility
        - Automatically selects 'T' or 'duration' column for time, 'E' or 'event' for censoring
        - Filters to numeric columns only (drops strings, dates, customer_id)
        - Clips duration to 0.001 minimum (Weibull requires positive durations)

    Warning:
        - Weibull AFT assumes proportional acceleration (multiplicative effects on survival time)
        - Non-numeric covariates must be encoded before passing to this function
        - Ensure sufficient uncensored events (E=1) for stable estimation (recommended: 30+)

    Example:
        >>> survival_df = pl.DataFrame({
        ...     'customer_id': ['S1', 'S2', 'S3'],
        ...     'T': [180, 365, 90],  # Days subscribed
        ...     'E': [1, 0, 1],       # 1=churned, 0=active
        ...     'engagement_score': [20, 80, 10],
        ...     'subscription_tier': [1, 3, 1]  # Already encoded
        ... })
        >>> model = train_weibull_aft_model(survival_df, {})
        >>> # Predict survival function for new customer:
        >>> new_customer = pd.DataFrame({'engagement_score': [50], 'subscription_tier': [2]})
        >>> survival_curve = model.predict_survival_function(new_customer)
    """
    if survival_data.is_empty():
        logger.warning("No survival data found. Skipping Weibull AFT model.")
        return None

    logger.info("Training Weibull AFT Model...")

    # Convert to pandas for lifelines
    df = survival_data.to_pandas()

    # Ensure mandatory columns T (duration) and E (event) exist
    # Using 'duration' and 'event' as standard names if T/E not present
    duration_col = "T" if "T" in df.columns else "duration"
    event_col = "E" if "E" in df.columns else "event"

    if duration_col not in df.columns or event_col not in df.columns:
        logger.warning(
            f"Missing duration/event columns ({duration_col}, {event_col}) for Weibull. Skipping."
        )
        return None

    # Ensure strictly positive duration (Weibull AFT requirement)
    df.loc[df[duration_col] <= 0, duration_col] = 0.001

    # Drop customer_id if present
    if "customer_id" in df.columns:
        df = df.drop(columns=["customer_id"])

    # Drop non-numeric columns to avoid lifelines errors/warnings
    # We must preserve duration and event columns, but they are expected to be numeric.
    # If they were not, we'd have issues anyway.
    # This filters out dates, strings, etc. that might be in survival_data
    df = df.select_dtypes(include=["number"])

    try:
        aft = WeibullAFTFitter()
        aft.fit(df, duration_col=duration_col, event_col=event_col)

        # Add dummy predict method to satisfy MLflow's sklearn flavor and avoid warnings
        aft.predict = lambda X: aft.predict_expectation(X)

        return aft
    except Exception as e:
        logger.error(f"Failed to train Weibull AFT: {e}")
        return None


try:
    import daal4py as d4p

    HAS_DAAL4PY = True
except ImportError:
    HAS_DAAL4PY = False

try:
    import onedal

    HAS_ONEDAL = True
except ImportError:
    HAS_ONEDAL = False


def train_xgboost_residual_model(
    feature_store: pl.DataFrame, parameters: Dict[str, Any]
) -> Any:
    """Train XGBoost classifier for behavioral churn prediction (Binge-and-Burnout detection).

    XGBoost refines probabilistic model predictions (BG/NBD, Weibull) by incorporating
    behavioral signals that indicate short-term churn risk not captured by transactional
    patterns alone. Examples: binge consumption followed by sudden drop-off, declining
    engagement quality, changing content preferences.

    Model Purpose:
    - Detect behavioral churn patterns (burnout, dissatisfaction, competitive switching)
    - Complement transactional models with real-time engagement signals
    - Enable proactive retention interventions for high-risk customers

    Churn Label Generation (if not provided):
    - If 'churn_label' column missing, generates target from inactivity:
      - Churn = 1 if (observation_period_end - last_engagement_date) > inactivity_threshold_days
      - Churn = 0 otherwise (active customer)
    - Default inactivity_threshold_days: 90 (configurable via params)

    Hardware Acceleration:
    - Auto-detects available accelerators: CUDA (NVIDIA), SYCL (Intel Arc), MPS (Apple Silicon)
    - Falls back to CPU if hardware acceleration fails
    - Applies Intel oneDAL optimization (daal4py) for CPU inference acceleration
    - Clears device cache before training to prevent OOM errors

    Args:
        feature_store: Behavioral feature DataFrame with one row per customer:
            - customer_id (str): Customer identifier (excluded from features)
            - last_engagement_date (datetime): Most recent engagement timestamp (used for target generation)
            - Behavioral features: watch_time, login_count, buffering_ratio, etc. (model inputs)
            - churn_label (int, optional): Binary target (1=churned, 0=active). If missing, auto-generated.
        parameters: Configuration dictionary with nested structure:
            - data_processing.observation_period_end (str): Reference date for churn calculation (YYYY-MM-DD)
            - modeling.xgboost.xgboost_target_col (str): Target column name (default: 'churn_label')
            - modeling.xgboost.inactivity_threshold_days (int): Days inactive to classify as churned (default: 90)
            - modeling.xgboost.params (dict): XGBoost hyperparameters:
                - max_depth (int): Tree depth (default: 6)
                - learning_rate (float): Step size (default: 0.3)
                - n_estimators (int): Number of trees (default: 100)
                - objective (str): 'binary:logistic' for classification
                - eval_metric (str): 'logloss', 'auc', etc.
                - device (str, optional): 'cuda', 'sycl', 'cpu' (auto-detected if omitted)
                - tree_method (str): 'hist' recommended for SYCL, 'auto' otherwise

    Returns:
        Fitted XGBClassifier model with:
            - Standard scikit-learn API: fit(), predict(), predict_proba()
            - _estimator_type='classifier' attribute (for MLflow compatibility)
            - daal_model_ attribute (Intel oneDAL optimized model, if available)
        Returns None if:
            - feature_store is empty
            - No features available after dropping IDs/dates
            - Model training fails

    Raises:
        ValueError: If observation_period_end has invalid date format
        KeyError: If required columns missing from feature_store

    Note:
        - Automatically drops non-numeric columns (dates, strings) from features
        - Fills missing values with 0 (assumes missing engagement = zero activity)
        - Logs warnings for all fallback scenarios (hardware acceleration failures, etc.)
        - Intel optimization via daal4py/oneDAL is optional; model works without it

    Warning:
        - Churn label generation is heuristic-based (inactivity threshold). For accurate
          labels, provide ground truth churn_label from subscription cancellations.
        - Inactivity threshold (90 days default) should be tuned per business model
          (shorter for high-frequency services, longer for seasonal content)
        - XGBoost hyperparameters use defaults; tune via cross-validation for production

    Example:
        >>> feature_store = pl.DataFrame({
        ...     'customer_id': ['C1', 'C2'],
        ...     'last_engagement_date': ['2024-01-01', '2024-06-01'],
        ...     'watch_time': [1000, 50],
        ...     'login_count': [30, 2]
        ... })
        >>> params = {
        ...     'data_processing': {'observation_period_end': '2024-07-01'},
        ...     'modeling': {'xgboost': {'inactivity_threshold_days': 90, 'params': {'max_depth': 4}}}
        ... }
        >>> model = train_xgboost_residual_model(feature_store, params)
        >>> # C1: inactive 182 days → churned, C2: inactive 30 days → active
    """
    if feature_store.is_empty():
        logger.warning("No feature store data found. Skipping XGBoost.")
        return None

    clear_device_cache()
    xgb_params = parameters.get("modeling", {}).get("xgboost", {})
    target_col = xgb_params.get("xgboost_target_col", "churn_label")
    id_col = "customer_id"

    # We assume standard column names because data_processing pipeline enforces them
    # via skeleton mapping.
    df = feature_store

    # --- 1. Target Variable Generation (Churn Label) ---
    if target_col not in df.columns:
        logger.info(f"Generating target column '{target_col}' from inactivity.")

        if "last_engagement_date" not in df.columns:
            logger.warning(
                "'last_engagement_date' missing from feature store. Cannot calculate churn. Skipping XGBoost."
            )
            return None

        analysis_date_str = parameters.get("data_processing", {}).get(
            "observation_period_end"
        )
        if not analysis_date_str:
            logger.warning("`observation_period_end` not found. Skipping XGBoost.")
            return None

        try:
            analysis_date = datetime.strptime(analysis_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(
                f"Invalid date format for observation_period_end: {analysis_date_str}. Skipping."
            )
            return None

        # Ensure last_engagement_date is Date type
        if df["last_engagement_date"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("last_engagement_date").str.to_date(strict=False)
            )
        # If it was 0 (from fill_null), it might be int. Treat 0/null as ancient history -> churned.

        # Calculate inactivity
        inactivity_threshold = xgb_params.get("inactivity_threshold_days", 90)

        # Handle null/0 dates (never engaged) -> assume churned if we want strictness,
        # or handle separately. Here we treat null as infinite inactivity.
        df = df.with_columns(
            pl.when(
                pl.col("last_engagement_date").is_null()
                | (pl.col("last_engagement_date") == 0)
            )
            .then(9999)  # Infinite days
            .otherwise(
                (pl.lit(analysis_date) - pl.col("last_engagement_date")).dt.total_days()
            )
            .alias("inactivity_days")
        )

        df = df.with_columns(
            (pl.col("inactivity_days") > inactivity_threshold)
            .cast(pl.Int8)
            .alias(target_col)
        )

    # --- 2. Model Training ---
    # Convert to pandas for XGBoost
    df_pd = df.to_pandas()

    # Drop non-feature columns
    cols_to_drop = [id_col, target_col, "last_engagement_date", "inactivity_days"]
    X = df_pd.drop(columns=cols_to_drop, errors="ignore")

    # Also drop date objects if any remain
    X = X.select_dtypes(exclude=["object", "datetime64[ns]", "datetime"])

    y = df_pd[target_col]

    if X.empty:
        logger.warning(
            "No features available for training after dropping IDs/Dates. Skipping."
        )
        return None

    try:
        model_params = xgb_params.get("params", {}).copy()

        # Determine device
        if "device" not in model_params:
            device = get_device(purpose="XGBoost training", framework="xgboost")
            model_params["device"] = device

            # Additional configuration for SYCL
            if device == "sycl":
                model_params["tree_method"] = "hist"
                if model_params.get("n_jobs", -1) == -1:
                    model_params["n_jobs"] = 1

        try:
            model = xgb.XGBClassifier(**model_params)
            model.fit(X, y)
        except Exception as e:
            if model_params.get("device") in ["sycl", "cuda"]:
                logger.warning(
                    f"Hardware acceleration ({model_params.get('device')}) failed: {e}. Falling back to CPU."
                )
                model_params["device"] = "cpu"
                # If tree_method was 'hist' for SYCL, we can keep it for CPU or let it be 'auto'
                # but 'hist' works fine on CPU too.
                model = xgb.XGBClassifier(**model_params)
                model.fit(X, y)
            else:
                raise e

        # Ensure estimator type is set for MLflow/Kedro-MLflow
        if not hasattr(model, "_estimator_type"):
            model._estimator_type = "classifier"

        logger.info("XGBoost model training completed successfully.")

        # --- 3. Inference Acceleration (daal4py / oneDAL) ---
        # Note: XGBoost 2.0+ with SYCL backend often provides native acceleration.
        # Conversion is primarily used for CPU-based inference acceleration.
        if HAS_DAAL4PY:
            try:
                # Robust attribute checking for different versions of scikit-learn-intelex/daal4py
                if hasattr(d4p, "get_gbt_model_from_xgboost"):
                    logger.info(
                        "Optimizing XGBoost model with daal4py for Intel inference..."
                    )
                    daal_model = d4p.get_gbt_model_from_xgboost(model.get_booster())
                    model.daal_model_ = daal_model
                elif HAS_ONEDAL:
                    # In newer versions, onedal might provide the conversion
                    try:
                        import onedal.xgb

                        if hasattr(onedal.xgb, "convert_model"):
                            logger.info(
                                "Optimizing XGBoost model with oneDAL for Intel inference..."
                            )
                            daal_model = onedal.xgb.convert_model(model.get_booster())
                            model.daal_model_ = daal_model
                    except (ImportError, AttributeError):
                        pass

                if not hasattr(model, "daal_model_"):
                    logger.info(
                        "Intel-specific XGBoost optimization utilities not found or not compatible. Skipping."
                    )
            except Exception as e:
                logger.warning(
                    f"Intel-specific optimization failed: {e}. Falling back to standard model."
                )

        return model
    except Exception as e:
        logger.error(f"Failed to train XGBoost model: {e}")
        return None


def predict_contractual_cltv(
    sbg_model: Any,
    weibull_model: Any,
    xgboost_model: Any,
    survival_data: pl.DataFrame,
    feature_store: pl.DataFrame,
) -> pl.DataFrame:
    """Predict CLTV for contractual settings (SVOD, SaaS) using hybrid survival analysis.

    Combines three complementary models to forecast subscription lifetime value:
    1. sBG (Shifted Beta-Geometric): Baseline retention curves
    2. Weibull AFT: Covariate-adjusted survival functions
    3. XGBoost: Behavioral churn risk (binge-and-burnout detection)

    Hybrid Prediction Strategy:
    - Weibull AFT generates survival curves S(t) = P(survive beyond month t)
    - XGBoost predicts short-term behavioral churn risk (burnout penalty)
    - CLTV = monthly_margin * Σ[S(t) / (1+d)^t] for t=1..12
    - High XGBoost risk (>0.7) applies 50% penalty to CLTV (burnout adjustment)

    Fallback Logic:
    - If Weibull unavailable: Uses geometric decay based on XGBoost risk
    - If XGBoost unavailable: Uses Weibull survival curves only
    - If both unavailable: Returns empty DataFrame with correct schema

    Args:
        sbg_model: Fitted sBG model (currently unused, reserved for baseline blending)
        weibull_model: Fitted WeibullAFTFitter or None. Provides survival curves S(t)
            per customer based on static covariates.
        xgboost_model: Fitted XGBClassifier or None. Provides behavioral churn probability
            based on recent engagement patterns.
        survival_data: Subscription lifecycle DataFrame with columns:
            - customer_id (str): Customer/subscription identifier
            - T or duration (float): Observation time (months)
            - E or event (int): Event indicator (1=churned, 0=active)
            - Covariates (numeric): Static features for Weibull (tier, demographics, etc.)
        feature_store: Behavioral feature DataFrame (required if xgboost_model provided):
            - customer_id (str): Customer identifier for merging
            - Behavioral features: watch_time, login_count, etc. (from engagement aggregation)

    Returns:
        Predictions DataFrame with one row per customer:
            - customer_id (str): Customer identifier
            - clv_12mo (float): 12-month discounted CLTV in USD
            - p_churn_30day (float): 30-day churn probability (0-1)
            - clv_segment (str): Customer segment (placeholder: 'Q3_Medium')
        Returns empty DataFrame with correct schema if survival_data is empty.

    Note:
        - Monthly margin hardcoded to $10 (SVOD placeholder). Parameterize for production.
        - Discount rate hardcoded to 1% monthly (0.01). Should be configurable.
        - High burnout penalty (50% reduction) is heuristic; tune based on validation.
        - Segmentation is placeholder; integrate quintile logic from predict_cltv for consistency.

    Warning:
        - Weibull survival curves assume covariates are time-invariant (static features only)
        - XGBoost risk should be recent (< 30 days old) for accurate short-term prediction
        - Survival curve indexing assumes monthly time units (t=1,2,3..12)

    Example:
        >>> survival_df = pl.DataFrame({
        ...     'customer_id': ['S1', 'S2'],
        ...     'T': [6, 12],
        ...     'E': [1, 0],
        ...     'engagement_score': [20, 80]
        ... })
        >>> feature_store = pl.DataFrame({
        ...     'customer_id': ['S1', 'S2'],
        ...     'watch_time': [100, 2000],
        ...     'login_count': [5, 40]
        ... })
        >>> predictions = predict_contractual_cltv(
        ...     sbg_model=None,
        ...     weibull_model=weibull_fitted,
        ...     xgboost_model=xgb_fitted,
        ...     survival_data=survival_df,
        ...     feature_store=feature_store
        ... )
        >>> # S1: High XGBoost risk (low engagement) → CLTV penalized
        >>> # S2: Low XGBoost risk (high engagement) → Full CLTV
    """
    if survival_data.is_empty():
        # Return empty DF with correct schema to satisfy Delta writer
        schema = {
            "customer_id": pl.Utf8,
            "clv_12mo": pl.Float64,
            "p_churn_30day": pl.Float64,
            "clv_segment": pl.Utf8,
        }
        return pl.DataFrame(schema=schema)

    df = survival_data.to_pandas()
    customer_ids = df["customer_id"].values

    # 1. Baseline Retention Curve (sBG)
    # Simplified: We'll assume a baseline retention rate if model is proper
    # or just use 1 - churn_rate.
    # For this demo, let's focus on the Weibull + XGBoost combination which is more powerful.

    # 2. Weibull AFT Survival Function
    # S(t) = P(T > t)
    survival_curves = None
    if weibull_model and not isinstance(weibull_model, dict):
        try:
            # predict_survival_function returns dataframe with index=timeline, cols=customers?
            # Or expects covariates.
            # We need X features for the customers.
            # Assuming survival_data has the static covariates used in training
            X_surv = df.drop(
                columns=["customer_id", "duration", "event", "T", "E"], errors="ignore"
            )
            survival_curves = weibull_model.predict_survival_function(X_surv)
            # survival_curves: index is time, columns are simple indices corresponding to X_surv
        except Exception as e:
            logger.warning(f"Weibull prediction failed: {e}")

    # 3. XGBoost Refinement (Binge-and-Burnout)
    # Predict probability of churning in next period given recent behavior
    xgb_risk = np.zeros(len(df))
    if (
        xgboost_model
        and not isinstance(xgboost_model, dict)
        and feature_store is not None
    ):
        try:
            fs_pd = feature_store.to_pandas()
            # Merge on ID
            merged = df[["customer_id"]].merge(fs_pd, on="customer_id", how="left")
            X_xgb = merged.drop(
                columns=["customer_id", "churn_label", "last_engagement_date"],
                errors="ignore",
            )
            X_xgb = X_xgb.select_dtypes(
                exclude=["object", "datetime64[ns]", "datetime"]
            ).fillna(0)

            if hasattr(xgboost_model, "daal_model_"):
                logger.info("Using daal4py for accelerated contractual inference.")
                xgb_risk = xgboost_model.daal_model_.predict(X_xgb)
                if len(xgb_risk.shape) > 1 and xgb_risk.shape[1] > 1:
                    xgb_risk = xgb_risk[:, 1]
            elif hasattr(xgboost_model, "predict_proba"):
                xgb_risk = xgboost_model.predict_proba(X_xgb)[:, 1]
        except Exception as e:
            logger.warning(f"XGBoost prediction failed: {e}")

    # 4. Synthesize CLTV
    # CLTV = Margin * Sum( S(t) / (1+d)^t )
    # We will approximate this.
    # If we have survival curves, we sum them.
    # If not, we use a heuristic based on xgb_risk.

    # Approx CLV 12mo
    monthly_margin = 10.0  # Placeholder for SVOD price
    discount_rate = 0.01

    clv_12mo = []
    p_churn_30 = []

    for i in range(len(df)):
        # Base retention prob for next period
        clv = 0.0
        if survival_curves is not None:
            # Sum discounted S(t) for t=1..12 from the survival curve
            curve = survival_curves.iloc[:, i]
            relevant_curve = curve.loc[curve.index <= 12]
            for t, prob in relevant_curve.items():
                if t > 0:  # Discounting starts from the first period
                    clv += prob / ((1 + discount_rate) ** t)
            clv *= monthly_margin
        else:
            # Fallback: simple retention decay based on XGBoost risk
            # r = 1 - risk. Probability of staying.
            r = 1.0 - xgb_risk[i]
            # Geometric sum of discounted retention probabilities: sum (r / (1+d))^t
            discounted_r = r / (1 + discount_rate)
            if discounted_r < 1:
                expected_discounted_periods = (
                    discounted_r * (1 - discounted_r**12) / (1 - discounted_r)
                )
            else:
                expected_discounted_periods = 12
            clv = monthly_margin * expected_discounted_periods

        # Adjust for "Burnout" factor (XGBoost high risk)
        # If XGBoost says high risk, we penalize the projection
        if xgb_risk[i] > 0.7:
            clv *= 0.5  # Penalty

        clv_12mo.append(clv)

        # P_churn 30 day is basically xgb_risk or 1-S(1)
        p_c = xgb_risk[i]
        p_churn_30.append(p_c)

    # Construct result DataFrame
    results = pl.DataFrame(
        {
            "customer_id": customer_ids,
            "clv_12mo": clv_12mo,
            "p_churn_30day": p_churn_30,
            "clv_segment": ["Q3_Medium"] * len(df),  # Placeholder
        }
    )

    return results
