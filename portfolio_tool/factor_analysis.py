"""
Fama-French Factor Analysis Module

This module performs factor analysis on portfolio returns using the Fama-French
multi-factor models. It explains portfolio returns through systematic risk factors
and calculates alpha (excess return not explained by factors).

Finance Theory:
- Market Factor (Mkt-RF): Excess return of market over risk-free rate
- Size Factor (SMB): Small Minus Big - returns of small cap stocks minus large cap
- Value Factor (HML): High Minus Low - returns of high book-to-market (value) minus low (growth)
- Profitability Factor (RMW): Robust Minus Weak - profitable firms minus unprofitable
- Investment Factor (CMA): Conservative Minus Aggressive - firms with low investment minus high
- Momentum Factor (MOM): Winners minus Losers - past winners minus past losers

Regression Model:
R_portfolio - R_f = α + β_mkt*(R_market - R_f) + β_smb*SMB + β_hml*HML + ε

Where:
- α (alpha) = excess return NOT explained by factors (skill/luck)
- β coefficients = factor loadings (exposure to each risk factor)
- Higher R² = factors explain more of the returns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging
import warnings

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    warnings.warn("yfinance not available. Install with: pip install yfinance")

try:
    from statsmodels.api import OLS, add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Install with: pip install statsmodels")

from portfolio_tool.fama_french import get_factors

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum observations required for statistical validity
MIN_OBSERVATIONS_MONTHLY = 36  # 3 years minimum
MIN_OBSERVATIONS_WEEKLY = 156  # 3 years minimum
MIN_OBSERVATIONS_DAILY = 252 * 3  # 3 years minimum

# Significance threshold for p-values
DEFAULT_SIGNIFICANCE_THRESHOLD = 0.05


def _fetch_ticker_returns(
    ticker: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    frequency: str = 'monthly'
) -> pd.Series:
    """
    Fetch ticker price data and convert to returns.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (string YYYY-MM-DD or pandas Timestamp)
        end_date: End date (string YYYY-MM-DD or pandas Timestamp)
        frequency: 'daily', 'weekly', or 'monthly'
    
    Returns:
        pd.Series: Returns series with DatetimeIndex
    
    Raises:
        ValueError: If ticker not found or insufficient data
        ImportError: If yfinance not available
    """
    if not YFINANCE_AVAILABLE:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    # Convert dates to strings for yfinance
    if isinstance(start_date, pd.Timestamp):
        start_date = start_date.strftime('%Y-%m-%d')
    if isinstance(end_date, pd.Timestamp):
        end_date = end_date.strftime('%Y-%m-%d')
    
    try:
        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(start=start_date, end=end_date)
        
        if hist.empty:
            raise ValueError(f"No price data available for ticker {ticker} for the specified date range")
        
        # Use adjusted close prices
        prices = hist['Close'] if 'Close' in hist.columns else hist.iloc[:, 0]
        
        # Calculate returns: (P_t - P_t-1) / P_t-1
        returns = prices.pct_change().dropna()
        
        # Resample to requested frequency
        if frequency == 'monthly':
            # Resample to month-end and calculate monthly returns
            # Method: compound returns within each month
            # Use 'ME' (month-end) instead of deprecated 'M'
            returns = (1 + returns).resample('ME').apply(lambda x: x.prod() - 1)
        elif frequency == 'weekly':
            # Resample to week-end (Sunday)
            # Use 'W-SUN' instead of deprecated 'W'
            returns = (1 + returns).resample('W-SUN').apply(lambda x: x.prod() - 1)
        # For daily, keep as is (no resampling needed)
        
        returns = returns.dropna()
        
        if len(returns) == 0:
            raise ValueError(f"No returns data after resampling for frequency {frequency}")
        
        # Strip timezone to ensure compatibility
        if hasattr(returns.index, 'tz') and returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        
        return returns
        
    except Exception as e:
        if "No data found" in str(e) or "not found" in str(e).lower():
            raise ValueError(f"Ticker {ticker} not found or no data available") from e
        raise RuntimeError(f"Error fetching data for {ticker}: {str(e)}") from e


def _calculate_portfolio_returns(
    tickers: List[str],
    weights: List[float],
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    frequency: str = 'monthly'
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Calculate weighted portfolio returns from multiple tickers.

    Args:
        tickers: List of ticker symbols
        weights: List of weights (must sum to 1.0)
        start_date: Start date (YYYY-MM-DD string or pandas Timestamp)
        end_date: End date (YYYY-MM-DD string or pandas Timestamp)
        frequency: 'daily', 'weekly', or 'monthly'

    Returns:
        Tuple of:
            - pd.Series: Portfolio returns indexed by date
            - pd.DataFrame: Individual ticker returns (for attribution analysis)

    Raises:
        ValueError: If tickers/weights mismatch or weights don't sum to 1.0
    """
    # Validate inputs
    if len(tickers) != len(weights):
        raise ValueError(f"Number of tickers ({len(tickers)}) must match number of weights ({len(weights)})")

    weights_sum = sum(weights)
    if not np.isclose(weights_sum, 1.0, atol=0.01):
        raise ValueError(f"Weights must sum to 1.0, got {weights_sum:.4f}")

    # Normalize weights to exactly 1.0
    weights = [w / weights_sum for w in weights]

    # Fetch returns for each ticker
    all_returns = {}
    for ticker in tickers:
        try:
            returns = _fetch_ticker_returns(ticker, start_date, end_date, frequency)
            all_returns[ticker] = returns
            logger.info(f"Fetched {len(returns)} {frequency} returns for {ticker}")
        except Exception as e:
            raise ValueError(f"Failed to fetch returns for {ticker}: {str(e)}") from e

    # Combine into DataFrame
    returns_df = pd.DataFrame(all_returns)

    # Align all returns to common dates (only keep dates where all tickers have data)
    returns_df = returns_df.dropna()

    if len(returns_df) == 0:
        raise ValueError("No overlapping data between tickers after alignment")

    logger.info(f"Portfolio has {len(returns_df)} aligned observations for {len(tickers)} tickers")

    # Calculate weighted portfolio returns
    weights_array = np.array(weights)
    portfolio_returns = (returns_df * weights_array).sum(axis=1)
    portfolio_returns.name = 'returns'

    return portfolio_returns, returns_df


def _calculate_excess_returns(
    returns: pd.Series,
    risk_free_rate: Union[pd.Series, float]
) -> pd.Series:
    """
    Calculate excess returns: portfolio returns minus risk-free rate.
    
    Args:
        returns: Portfolio returns series
        risk_free_rate: Risk-free rate (Series aligned with returns, or scalar)
    
    Returns:
        pd.Series: Excess returns
    """
    if isinstance(risk_free_rate, pd.Series):
        # Align indices and forward fill missing values
        aligned_rf = risk_free_rate.reindex(returns.index).ffill()
        excess = returns - aligned_rf
    else:
        # Scalar risk-free rate
        excess = returns - risk_free_rate
    
    return excess


def _align_data(
    ticker_returns: pd.Series,
    factors: pd.DataFrame,
    risk_free_rate: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Align ticker returns with factor data on date index.
    
    Args:
        ticker_returns: Ticker returns series
        factors: Factor DataFrame with DatetimeIndex
        risk_free_rate: Optional risk-free rate series
    
    Returns:
        pd.DataFrame: Aligned data with columns: returns, excess_returns, and factor columns
    
    Raises:
        ValueError: If insufficient overlapping data
    """
    # Strip timezone from both indices to avoid tz-naive/tz-aware mismatch
    # Make copies to avoid modifying original data
    ticker_returns = ticker_returns.copy()
    factors = factors.copy()
    
    # STEP 1: Initial data
    logger.info(f"=== STEP 1: Initial data ===")
    logger.info(f"Ticker returns shape: {ticker_returns.shape}, columns: {ticker_returns.name if hasattr(ticker_returns, 'name') else 'Series'}")
    logger.info(f"Factors shape: {factors.shape}, columns: {factors.columns.tolist()}")
    logger.info(f"Risk-free rate: {type(risk_free_rate)}, is None: {risk_free_rate is None}")
    
    # Remove timezone from ticker_returns index if present
    if hasattr(ticker_returns.index, 'tz') and ticker_returns.index.tz is not None:
        ticker_returns.index = ticker_returns.index.tz_localize(None)
    
    # Remove timezone from factors index if present
    if hasattr(factors.index, 'tz') and factors.index.tz is not None:
        factors.index = factors.index.tz_localize(None)
    
    # STEP 2: After timezone stripping
    logger.info(f"=== STEP 2: After timezone stripping ===")
    logger.info(f"Ticker returns index has tz: {ticker_returns.index.tz is not None if hasattr(ticker_returns.index, 'tz') else 'No tz attr'}")
    logger.info(f"Factors index has tz: {factors.index.tz is not None if hasattr(factors.index, 'tz') else 'No tz attr'}")
    logger.info(f"Ticker returns index type: {type(ticker_returns.index)}")
    logger.info(f"Ticker returns date range: {ticker_returns.index.min()} to {ticker_returns.index.max()}")
    logger.info(f"Ticker returns first 3 dates: {ticker_returns.index[:3].tolist()}")
    logger.info(f"Factors index type: {type(factors.index)}")
    logger.info(f"Factors date range: {factors.index.min()} to {factors.index.max()}")
    logger.info(f"Factors first 3 dates: {factors.index[:3].tolist()}")
    
    # Normalize dates to month period for alignment
    # Monthly data may have different day-of-month (e.g., 2020-01-01 vs 2020-01-31)
    # Convert to period and back to ensure they align on the same month
    ticker_returns.index = ticker_returns.index.to_period('M').to_timestamp('M')
    factors.index = factors.index.to_period('M').to_timestamp('M')
    
    logger.info(f"After normalization - Ticker returns first 3 dates: {ticker_returns.index[:3].tolist()}")
    logger.info(f"After normalization - Factors first 3 dates: {factors.index[:3].tolist()}")
    
    # Create DataFrame from returns
    aligned = pd.DataFrame({'returns': ticker_returns})
    
    # STEP 3: After creating aligned DataFrame
    logger.info(f"=== STEP 3: After creating aligned DataFrame ===")
    logger.info(f"Aligned shape: {aligned.shape}")
    logger.info(f"Aligned columns: {aligned.columns.tolist()}")
    logger.info(f"Aligned index (first 3): {aligned.index[:3].tolist()}")
    logger.info(f"Aligned has NaN: {aligned.isna().any().any()}")
    logger.info(f"NaN counts per column: {aligned.isna().sum().to_dict()}")
    
    # Merge with factors
    aligned = aligned.join(factors, how='inner')
    
    # STEP 4: After joining with factors
    logger.info(f"=== STEP 4: After joining with factors ===")
    logger.info(f"Aligned shape: {aligned.shape}")
    logger.info(f"Aligned columns: {aligned.columns.tolist()}")
    logger.info(f"Number of rows before dropna: {len(aligned)}")
    logger.info(f"NaN counts per column: {aligned.isna().sum().to_dict()}")
    if len(aligned) > 0:
        logger.info(f"First row:\n{aligned.iloc[0]}")
        logger.info(f"Last row:\n{aligned.iloc[-1]}")
    else:
        logger.error(f"!!! JOIN PRODUCED ZERO ROWS !!!")
        logger.error(f"Ticker index (first 5): {ticker_returns.index[:5].tolist()}")
        logger.error(f"Factors index (first 5): {factors.index[:5].tolist()}")
    
    # Add risk-free rate if provided
    if risk_free_rate is not None:
        # Strip timezone from risk_free_rate index if present
        risk_free_rate = risk_free_rate.copy()
        if hasattr(risk_free_rate.index, 'tz') and risk_free_rate.index.tz is not None:
            risk_free_rate.index = risk_free_rate.index.tz_localize(None)
        
        # Normalize risk_free_rate index to match aligned index (month-end)
        risk_free_rate.index = risk_free_rate.index.to_period('M').to_timestamp('M')
        
        aligned['rf'] = risk_free_rate.reindex(aligned.index).ffill()
        aligned['excess_returns'] = aligned['returns'] - aligned['rf']
    elif 'RF' in aligned.columns:
        # Use RF from factors if available (already aligned)
        aligned['rf'] = aligned['RF']
        aligned['excess_returns'] = aligned['returns'] - aligned['RF']
    else:
        # No risk-free rate available, excess = returns
        aligned['excess_returns'] = aligned['returns']
    
    # STEP 5: After adding risk-free rate
    logger.info(f"=== STEP 5: After adding risk-free rate ===")
    logger.info(f"Aligned shape: {aligned.shape}")
    logger.info(f"Has 'rf' column: {'rf' in aligned.columns}")
    logger.info(f"Has 'excess_returns' column: {'excess_returns' in aligned.columns}")
    logger.info(f"NaN counts: {aligned.isna().sum().to_dict()}")
    
    # Store count before dropna
    rows_before_dropna = len(aligned)
    
    # Drop rows with any NaN values
    aligned = aligned.dropna()
    
    # STEP 6: After dropna
    logger.info(f"=== STEP 6: After dropna ===")
    logger.info(f"Dropna removed {rows_before_dropna - len(aligned)} rows")
    logger.info(f"Final aligned shape: {aligned.shape}")
    if len(aligned) == 0:
        logger.error(f"!!! ALL ROWS DROPPED BY DROPNA !!!")
        logger.error(f"DataFrame before dropna had {rows_before_dropna} rows")
        logger.error(f"This means all rows had at least one NaN value")
        logger.error(f"Check NaN counts from STEP 5 above to see which columns had NaN values")
    else:
        logger.info(f"SUCCESS: {len(aligned)} observations after alignment")
        logger.info(f"Date range: {aligned.index.min()} to {aligned.index.max()}")
    
    if len(aligned) == 0:
        raise ValueError("No overlapping data between ticker returns and factors after alignment")
    
    return aligned


def _annualize_alpha(period_alpha: float, frequency: str) -> float:
    """
    Annualize alpha based on frequency.
    
    Args:
        period_alpha: Alpha for the period (decimal) - can be daily, weekly, or monthly
        frequency: 'daily', 'weekly', or 'monthly'
    
    Returns:
        float: Annualized alpha
    """
    if frequency == 'monthly':
        # Annualize: (1 + monthly_alpha)^12 - 1
        # For small values, approximate: period_alpha * 12
        if abs(period_alpha) < 0.1:
            return period_alpha * 12
        else:
            return (1 + period_alpha) ** 12 - 1
    elif frequency == 'weekly':
        # Annualize: (1 + weekly_alpha)^52 - 1
        if abs(period_alpha) < 0.1:
            return period_alpha * 52
        else:
            return (1 + period_alpha) ** 52 - 1
    else:  # daily
        # Annualize: (1 + daily_alpha)^252 - 1
        if abs(period_alpha) < 0.1:
            return period_alpha * 252
        else:
            return (1 + period_alpha) ** 252 - 1


def _determine_significance(p_value: float, threshold: float = DEFAULT_SIGNIFICANCE_THRESHOLD) -> bool:
    """
    Determine if a coefficient is statistically significant.
    
    Args:
        p_value: P-value from regression
        threshold: Significance threshold (default 0.05)
    
    Returns:
        bool: True if significant (p < threshold)
    """
    if pd.isna(p_value):
        return False
    # Convert numpy bool to Python bool for JSON serialization
    return bool(p_value < threshold)


def _get_factor_columns(factor_model: str) -> List[str]:
    """
    Get list of factor column names for the specified model.
    
    Args:
        factor_model: '3-factor', '5-factor', '4-factor', or 'CAPM'
    
    Returns:
        List[str]: Factor column names
    """
    if factor_model == 'CAPM':
        return ['Mkt-RF']
    elif factor_model == '3-factor':
        return ['Mkt-RF', 'SMB', 'HML']
    elif factor_model == '4-factor':
        return ['Mkt-RF', 'SMB', 'HML', 'MOM']
    elif factor_model == '5-factor':
        return ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
    else:
        raise ValueError(f"Unsupported factor model: {factor_model}")


def _get_model_name(factor_model: str) -> str:
    """Get display name for factor model."""
    model_names = {
        'CAPM': 'Capital Asset Pricing Model (CAPM)',
        '3-factor': 'Fama-French 3-Factor',
        '4-factor': 'Fama-French 4-Factor (with Momentum)',
        '5-factor': 'Fama-French 5-Factor'
    }
    return model_names.get(factor_model, factor_model)


def _get_factor_display_names(factor_model: str) -> Dict[str, str]:
    """Get display names for factors in the model."""
    base_names = {
        'Mkt-RF': 'Market (Mkt-RF)',
        'SMB': 'Size (SMB)',
        'HML': 'Value (HML)',
        'MOM': 'Momentum (MOM)',
        'RMW': 'Profitability (RMW)',
        'CMA': 'Investment (CMA)'
    }
    
    factor_cols = _get_factor_columns(factor_model)
    return {col: base_names.get(col, col) for col in factor_cols}


def analyze_factors(
    ticker: Optional[str] = None,
    tickers: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    factor_model: str = '3-factor',
    frequency: str = 'monthly',
    risk_free_rate: str = '1M_TBILL'
) -> Dict:
    """
    Perform Fama-French factor analysis on a single ticker or portfolio.

    This function:
    1. Fetches ticker/portfolio price data and converts to returns
    2. Fetches Fama-French factor data (3-factor, 5-factor, 4-factor, or CAPM)
    3. Aligns data and calculates excess returns
    4. Runs regression: excess_returns = α + β_factors * factors + ε
    5. Extracts coefficients, t-stats, p-values, R²
    6. Calculates annualized alpha and return attribution

    Args:
        ticker: Single stock ticker symbol (e.g., 'SPY', 'AAPL') - use this OR tickers+weights
        tickers: List of ticker symbols for portfolio analysis
        weights: List of weights for each ticker (must sum to 1.0)
        start_date: Start date (YYYY-MM-DD string or pandas Timestamp)
        end_date: End date (YYYY-MM-DD string or pandas Timestamp)
        factor_model: '3-factor', '5-factor', '4-factor', or 'CAPM'
        frequency: 'daily', 'weekly', or 'monthly' (default: 'monthly')
        risk_free_rate: '1M_TBILL' (use RF from Fama-French), '3M_TBILL', or custom rate

    Returns:
        dict: Comprehensive factor analysis results with structure:
            {
                "model": str,
                "ticker": str (or "Portfolio (N holdings)"),
                "period": {...},
                "coefficients": {...},
                "statistics": {...},
                "regression_stats": {...},
                "factor_premiums": {...},
                "return_attribution": {...},
                "time_series": [...]
            }

    Raises:
        ValueError: Invalid inputs, insufficient data, or unsupported model
        ConnectionError: Network errors fetching data
        ImportError: Missing required dependencies

    Example (single ticker):
        >>> result = analyze_factors(
        ...     ticker="SPY",
        ...     start_date="2020-01-01",
        ...     end_date="2025-01-01",
        ...     factor_model="3-factor"
        ... )

    Example (portfolio):
        >>> result = analyze_factors(
        ...     tickers=["AAPL", "MSFT", "GOOGL"],
        ...     weights=[0.4, 0.35, 0.25],
        ...     start_date="2020-01-01",
        ...     end_date="2025-01-01",
        ...     factor_model="5-factor"
        ... )
    """
    # Validate ticker/portfolio inputs
    is_portfolio = False
    portfolio_tickers = None
    portfolio_weights = None
    ticker_name = None

    if ticker and tickers:
        raise ValueError("Provide either 'ticker' OR 'tickers' with 'weights', not both")

    if ticker:
        # Single ticker mode
        ticker_name = ticker.upper()
        is_portfolio = False
    elif tickers and weights:
        # Portfolio mode
        portfolio_tickers = [t.upper() for t in tickers]
        portfolio_weights = weights
        ticker_name = f"Portfolio ({len(tickers)} holdings)"
        is_portfolio = True
    elif tickers and not weights:
        raise ValueError("Must provide 'weights' when using multiple tickers")
    else:
        raise ValueError("Must provide either 'ticker' or 'tickers' with 'weights'")

    # Validate factor model
    if factor_model not in ['3-factor', '5-factor', '4-factor', 'CAPM']:
        raise ValueError(
            f"Factor model must be one of: '3-factor', '5-factor', '4-factor', 'CAPM'. "
            f"Got: {factor_model}"
        )

    if frequency not in ['daily', 'weekly', 'monthly']:
        raise ValueError(
            f"Frequency must be one of: 'daily', 'weekly', 'monthly'. Got: {frequency}"
        )

    # Validate and convert dates
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required")

    try:
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD. Error: {e}") from e

    if start_date >= end_date:
        raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")

    if not STATSMODELS_AVAILABLE:
        raise ImportError("statsmodels is required. Install with: pip install statsmodels")

    logger.info(f"Starting factor analysis for {ticker_name} using {factor_model} model")

    # Step 1: Fetch returns (single ticker or portfolio)
    individual_returns_df = None
    if is_portfolio:
        logger.info(f"Fetching returns for portfolio: {portfolio_tickers}")
        ticker_returns, individual_returns_df = _calculate_portfolio_returns(
            portfolio_tickers, portfolio_weights, start_date, end_date, frequency
        )
    else:
        logger.info(f"Fetching returns for {ticker_name}")
        ticker_returns = _fetch_ticker_returns(ticker_name, start_date, end_date, frequency)

    # Step 2: Fetch Fama-French factors
    # Determine which factors to fetch based on model
    include_5_factor = factor_model in ['5-factor']
    include_momentum = factor_model in ['4-factor']

    logger.info(f"Fetching Fama-French {factor_model} factors")
    try:
        factors = get_factors(
            start_date, end_date,
            include_5_factor=include_5_factor or True,  # Always fetch 5-factor for flexibility
            include_momentum=include_momentum or True    # Always fetch momentum for flexibility
        )
    except Exception as e:
        raise ConnectionError(f"Failed to fetch Fama-French data: {str(e)}") from e

    # Check if we have required factors
    required_factors = _get_factor_columns(factor_model)
    missing_factors = [f for f in required_factors if f not in factors.columns]

    if missing_factors:
        if factor_model == '4-factor' and 'MOM' not in factors.columns:
            raise ValueError(
                "Momentum (MOM) factor not available. "
                "4-factor model requires momentum data which could not be fetched."
            )
        elif factor_model == '5-factor':
            missing = [f for f in ['RMW', 'CMA'] if f not in factors.columns]
            if missing:
                raise ValueError(
                    f"5-factor model requires RMW and CMA factors. Missing: {missing}. "
                    "These factors could not be fetched."
                )
        else:
            raise ValueError(f"Missing required factors: {missing_factors}")
    
    # Step 3: Align data
    logger.info("Aligning ticker returns with factor data")
    
    # Get risk-free rate
    rf_series = None
    if risk_free_rate == '1M_TBILL' and 'RF' in factors.columns:
        rf_series = factors['RF']
    elif risk_free_rate == '3M_TBILL':
        # Would need to fetch 3-month T-bill separately
        # For now, use RF from factors
        if 'RF' in factors.columns:
            rf_series = factors['RF']
        else:
            logger.warning("3M_TBILL requested but not available, using RF from factors")
            rf_series = None
    # For custom rate, would need to be passed as parameter (future enhancement)
    
    aligned_data = _align_data(ticker_returns, factors, rf_series)
    
    # Debug logging: detailed information about aligned data
    logger.info(f"Date range in aligned data: {aligned_data.index.min()} to {aligned_data.index.max()}")
    logger.info(f"First 3 rows of aligned data:\n{aligned_data.head(3)}")
    logger.info(f"Last 3 rows of aligned data:\n{aligned_data.tail(3)}")
    logger.info(f"Excess returns - Mean: {aligned_data['excess_returns'].mean():.6f}, Std: {aligned_data['excess_returns'].std():.6f}")
    if 'Mkt-RF' in aligned_data.columns:
        logger.info(f"Market factor - Mean: {aligned_data['Mkt-RF'].mean():.6f}, Std: {aligned_data['Mkt-RF'].std():.6f}")
    
    # Check minimum observations
    min_obs = {
        'monthly': MIN_OBSERVATIONS_MONTHLY,
        'weekly': MIN_OBSERVATIONS_WEEKLY,
        'daily': MIN_OBSERVATIONS_DAILY
    }[frequency]
    
    if len(aligned_data) < min_obs:
        raise ValueError(
            f"Insufficient data: need at least {min_obs} {frequency} observations, "
            f"got {len(aligned_data)}"
        )
    
    logger.info(f"Aligned data: {len(aligned_data)} observations")
    
    # Step 4: Prepare regression data
    Y = aligned_data['excess_returns'].values  # Dependent variable
    X_factors = aligned_data[required_factors].values  # Independent variables (factors)
    
    # Add constant for intercept (alpha)
    X = add_constant(X_factors)
    
    # Step 5: Run regression
    logger.info("Running OLS regression")
    model = OLS(Y, X).fit()
    
    # Step 6: Extract results
    coefficients = model.params
    t_stats = model.tvalues
    p_values = model.pvalues
    r_squared = model.rsquared
    adj_r_squared = model.rsquared_adj
    residuals = model.resid
    
    # Alpha is the intercept (first coefficient)
    alpha = coefficients[0]
    alpha_t_stat = t_stats[0]
    alpha_p_value = p_values[0]
    
    # Factor loadings (betas) are the remaining coefficients
    factor_loadings = coefficients[1:]
    factor_t_stats = t_stats[1:]
    factor_p_values = p_values[1:]
    
    # Annualize alpha
    alpha_annualized = _annualize_alpha(alpha, frequency)

    # Step 6b: Calculate additional statistics
    num_observations = len(aligned_data)

    # Regression statistics (F-statistic)
    f_statistic = float(model.fvalue) if model.fvalue is not None else None
    f_statistic_p_value = float(model.f_pvalue) if model.f_pvalue is not None else None

    # Factor premiums (in basis points - multiply by 10000)
    factor_premiums = {}
    if 'Mkt-RF' in aligned_data.columns:
        factor_premiums['market_premium_bps'] = round(float(aligned_data['Mkt-RF'].mean() * 10000), 2)
    if 'SMB' in aligned_data.columns:
        factor_premiums['smb_premium_bps'] = round(float(aligned_data['SMB'].mean() * 10000), 2)
    if 'HML' in aligned_data.columns:
        factor_premiums['hml_premium_bps'] = round(float(aligned_data['HML'].mean() * 10000), 2)
    if 'MOM' in aligned_data.columns:
        factor_premiums['mom_premium_bps'] = round(float(aligned_data['MOM'].mean() * 10000), 2)
    if 'RMW' in aligned_data.columns:
        factor_premiums['rmw_premium_bps'] = round(float(aligned_data['RMW'].mean() * 10000), 2)
    if 'CMA' in aligned_data.columns:
        factor_premiums['cma_premium_bps'] = round(float(aligned_data['CMA'].mean() * 10000), 2)

    # Return attribution calculations
    # Total return = cumulative return over period
    total_return = float((1 + aligned_data['returns']).prod() - 1)

    # Annualized return (assuming monthly frequency for now)
    periods_per_year = {'monthly': 12, 'weekly': 52, 'daily': 252}.get(frequency, 12)
    if num_observations > 0:
        annualized_return = float((1 + total_return) ** (periods_per_year / num_observations) - 1)
    else:
        annualized_return = 0.0

    # Annualized standard deviation
    annualized_std = float(aligned_data['returns'].std() * np.sqrt(periods_per_year))

    # Factor contributions to return - DYNAMIC calculation
    # Contribution = beta * sum of factor returns over period
    # Map factor column names to contribution keys
    factor_contribution_map = {
        'Mkt-RF': 'market',
        'SMB': 'smb',
        'HML': 'hml',
        'MOM': 'mom',
        'RMW': 'rmw',
        'CMA': 'cma'
    }

    # Calculate contributions dynamically for all factors in the model
    # IMPORTANT: Portfolio Visualizer uses TOTAL returns (including RF) for market attribution
    # Standard Fama-French uses EXCESS returns (Mkt-RF)
    # We need to add RF back to get total market returns for proper attribution
    factor_contributions = {}
    for factor_col in required_factors:
        if factor_col in factor_contribution_map:
            idx = required_factors.index(factor_col)
            contribution_key = factor_contribution_map[factor_col]

            # Special handling for market factor: use total returns not excess returns
            if factor_col == 'Mkt-RF' and 'RF' in aligned_data.columns:
                # Market contribution = β_market × Σ(Mkt-RF + RF)
                # This converts excess returns back to total returns
                mkt_rf_sum = float(aligned_data['Mkt-RF'].sum())
                rf_sum = float(aligned_data['RF'].sum())
                total_market_returns = aligned_data['Mkt-RF'] + aligned_data['RF']
                total_market_sum = float(total_market_returns.sum())
                beta_market = float(factor_loadings[idx])
                contribution_value = beta_market * total_market_sum

                # Detailed logging for debugging
                logger.info(f"=== MARKET CONTRIBUTION DEBUG ===")
                logger.info(f"Beta (market): {beta_market:.6f}")
                logger.info(f"Σ(Mkt-RF): {mkt_rf_sum:.6f}")
                logger.info(f"Σ(RF): {rf_sum:.6f}")
                logger.info(f"Σ(Mkt-RF + RF): {total_market_sum:.6f}")
                logger.info(f"Market contribution = {beta_market:.6f} × {total_market_sum:.6f} = {contribution_value:.6f}")
                logger.info(f"=== END DEBUG ===")

                contribution_value = float(contribution_value)
            else:
                # Other factors use excess returns as-is (SMB, HML, etc.)
                contribution_value = float(factor_loadings[idx] * aligned_data[factor_col].sum())
                logger.debug(f"{factor_col} contribution: {contribution_value:.6f}")

            factor_contributions[contribution_key] = contribution_value

    # Alpha contribution = alpha * number of periods
    alpha_contribution = float(alpha * num_observations)
    factor_contributions['alpha'] = alpha_contribution

    # NOTE: We do NOT add a separate RF contribution here!
    # Portfolio Visualizer's methodology includes RF in the market contribution:
    # - Market contribution = β_market × Σ(Mkt-RF + RF) = β_market × Σ(total market returns)
    # - This already includes the risk-free rate component
    # - Adding a separate RF contribution would DOUBLE-COUNT the risk-free rate

    # Log total attribution for verification
    total_contributions = sum(factor_contributions.values())
    logger.info(f"Total factor contributions: {total_contributions:.6f} vs actual total return: {total_return:.6f}")
    residual = total_return - total_contributions
    if abs(residual) > 0.001:
        logger.warning(f"Attribution residual: {residual:.6f} (contributions don't sum to total return)")

    # Risk contribution (variance decomposition using Euler allocation)
    # This properly accounts for covariances between factors
    total_var = aligned_data['returns'].var()
    residual_var = float(np.var(residuals))

    # Initialize risk contributions
    risk_contribution = {}

    if total_var > 0:
        # Get covariance matrix for factors
        factor_data = aligned_data[required_factors]
        factor_cov_matrix = factor_data.cov()

        # Map factor columns to risk keys
        factor_risk_map = {
            'Mkt-RF': 'market_risk',
            'SMB': 'smb_risk',
            'HML': 'hml_risk',
            'MOM': 'mom_risk',
            'RMW': 'rmw_risk',
            'CMA': 'cma_risk'
        }

        # Calculate marginal contribution to variance for each factor (Euler allocation)
        # Risk_i = (β_i × Σ(β_j × Cov(i,j))) / σ²_portfolio
        for i, factor_i in enumerate(required_factors):
            if factor_i not in factor_risk_map:
                continue

            beta_i = factor_loadings[i]

            # Calculate marginal contribution: sum over all factors (including itself)
            marginal_contrib = 0.0
            for j, factor_j in enumerate(required_factors):
                beta_j = factor_loadings[j]
                cov_ij = factor_cov_matrix.loc[factor_i, factor_j]
                marginal_contrib += beta_j * cov_ij

            # Risk contribution as fraction of total variance
            risk_contrib = (beta_i * marginal_contrib) / total_var
            risk_key = factor_risk_map[factor_i]
            risk_contribution[risk_key] = round(float(risk_contrib), 4)
            logger.debug(f"{factor_i} risk contribution: {risk_contrib:.6f}")

        # Alpha (idiosyncratic) risk contribution
        # This is the residual: 1.0 minus sum of all factor risks
        total_factor_risk = sum(risk_contribution.values())
        alpha_risk = 1.0 - total_factor_risk
        risk_contribution['alpha_risk'] = round(float(alpha_risk), 4)

        # Verify risk contributions sum to 1.0 (100%)
        total_risk_check = sum(risk_contribution.values())
        logger.info(f"Risk contributions sum to: {total_risk_check:.6f} (should be ~1.0)")

    else:
        # Handle edge case of zero variance
        risk_contribution = {
            'market_risk': 0.0,
            'smb_risk': 0.0,
            'hml_risk': 0.0,
            'alpha_risk': 0.0
        }

    # Step 7: Build time series data
    dates = aligned_data.index
    actual_returns = aligned_data['excess_returns'].values
    # model.fittedvalues and model.resid are already numpy arrays from statsmodels OLS
    predicted_returns = model.fittedvalues
    residual_values = residuals
    
    time_series = []
    factor_display_names = _get_factor_display_names(factor_model)
    
    for i, date in enumerate(dates):
        # Get factor values for this period
        factor_values = {}
        for j, factor_col in enumerate(required_factors):
            factor_values[factor_col.lower().replace('-', '_')] = float(aligned_data[factor_col].iloc[i])
        
        time_series.append({
            "date": date.strftime('%Y-%m') if frequency == 'monthly' else date.strftime('%Y-%m-%d'),
            "actual_return": float(actual_returns[i]),
            "predicted_return": float(predicted_returns[i]),
            "residual": float(residual_values[i]),
            "factors": factor_values
        })
    
    # Step 8: Build coefficients dictionary
    coefficients_dict = {}
    
    # Market factor
    if 'Mkt-RF' in required_factors:
        idx = required_factors.index('Mkt-RF')
        coefficients_dict['market'] = {
            "name": factor_display_names['Mkt-RF'],
            "loading": float(factor_loadings[idx]),
            "t_stat": float(factor_t_stats[idx]),
            "p_value": float(factor_p_values[idx]),
            "significant": _determine_significance(factor_p_values[idx])
        }
    
    # Size factor
    if 'SMB' in required_factors:
        idx = required_factors.index('SMB')
        coefficients_dict['size'] = {
            "name": factor_display_names['SMB'],
            "loading": float(factor_loadings[idx]),
            "t_stat": float(factor_t_stats[idx]),
            "p_value": float(factor_p_values[idx]),
            "significant": _determine_significance(factor_p_values[idx])
        }
    
    # Value factor
    if 'HML' in required_factors:
        idx = required_factors.index('HML')
        coefficients_dict['value'] = {
            "name": factor_display_names['HML'],
            "loading": float(factor_loadings[idx]),
            "t_stat": float(factor_t_stats[idx]),
            "p_value": float(factor_p_values[idx]),
            "significant": _determine_significance(factor_p_values[idx])
        }
    
    # Momentum factor
    if 'MOM' in required_factors:
        idx = required_factors.index('MOM')
        coefficients_dict['momentum'] = {
            "name": factor_display_names['MOM'],
            "loading": float(factor_loadings[idx]),
            "t_stat": float(factor_t_stats[idx]),
            "p_value": float(factor_p_values[idx]),
            "significant": _determine_significance(factor_p_values[idx])
        }
    
    # Profitability factor
    if 'RMW' in required_factors:
        idx = required_factors.index('RMW')
        coefficients_dict['profitability'] = {
            "name": factor_display_names['RMW'],
            "loading": float(factor_loadings[idx]),
            "t_stat": float(factor_t_stats[idx]),
            "p_value": float(factor_p_values[idx]),
            "significant": _determine_significance(factor_p_values[idx])
        }
    
    # Investment factor
    if 'CMA' in required_factors:
        idx = required_factors.index('CMA')
        coefficients_dict['investment'] = {
            "name": factor_display_names['CMA'],
            "loading": float(factor_loadings[idx]),
            "t_stat": float(factor_t_stats[idx]),
            "p_value": float(factor_p_values[idx]),
            "significant": _determine_significance(factor_p_values[idx])
        }
    
    # Step 9: Build final result dictionary
    result = {
        "model": _get_model_name(factor_model),
        "ticker": ticker_name,
        "is_portfolio": is_portfolio,
        "period": {
            "start": start_date.strftime('%Y-%m-%d'),
            "end": end_date.strftime('%Y-%m-%d'),
            "observations": num_observations,
            "frequency": frequency
        },
        "coefficients": coefficients_dict,
        "statistics": {
            "r_squared": float(r_squared),
            "adjusted_r_squared": float(adj_r_squared),
            "alpha_monthly": float(alpha) if frequency == 'monthly' else None,
            "alpha_annualized": float(alpha_annualized),
            "alpha_t_stat": float(alpha_t_stat),
            "alpha_p_value": float(alpha_p_value),
            "alpha_significant": _determine_significance(alpha_p_value),
            "residual_std": float(np.std(residuals))
        },
        "regression_stats": {
            "f_statistic": round(f_statistic, 4) if f_statistic is not None else None,
            "f_statistic_p_value": round(f_statistic_p_value, 6) if f_statistic_p_value is not None else None
        },
        "factor_premiums": factor_premiums,
        "return_attribution": {
            "total_return": round(total_return, 4),
            "annualized_return": round(annualized_return, 4),
            "annualized_std": round(annualized_std, 4),
            "risk_contribution": risk_contribution
        },
        "time_series": time_series
    }

    # Add factor contributions dynamically with proper suffixes
    # Extract alpha separately (it's already in factor_contributions)
    for key, value in factor_contributions.items():
        if key == 'alpha':
            result["return_attribution"]["alpha_contribution"] = round(value, 4)
        else:
            # Add with _contribution suffix (e.g., market -> market_contribution)
            result["return_attribution"][f"{key}_contribution"] = round(value, 4)
    
    # Add frequency-specific alpha if not monthly
    if frequency == 'daily':
        result["statistics"]["alpha_daily"] = float(alpha)
    elif frequency == 'weekly':
        result["statistics"]["alpha_weekly"] = float(alpha)

    # Add portfolio composition if portfolio analysis
    if is_portfolio and portfolio_tickers and portfolio_weights:
        result["portfolio_composition"] = [
            {"ticker": t, "weight": round(w, 4)}
            for t, w in zip(portfolio_tickers, portfolio_weights)
        ]

    logger.info(f"Factor analysis complete. R² = {r_squared:.3f}, Alpha (annualized) = {alpha_annualized:.4f}")

    return result

