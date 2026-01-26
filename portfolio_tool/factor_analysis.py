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
    ticker: str,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    factor_model: str = '3-factor',
    frequency: str = 'monthly',
    risk_free_rate: str = '1M_TBILL'
) -> Dict:
    """
    Perform Fama-French factor analysis on a ticker.
    
    This function:
    1. Fetches ticker price data and converts to returns
    2. Fetches Fama-French factor data
    3. Aligns data and calculates excess returns
    4. Runs regression: excess_returns = α + β_factors * factors + ε
    5. Extracts coefficients, t-stats, p-values, R²
    6. Calculates annualized alpha
    
    Args:
        ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL')
        start_date: Start date (YYYY-MM-DD string or pandas Timestamp)
        end_date: End date (YYYY-MM-DD string or pandas Timestamp)
        factor_model: '3-factor', '5-factor', '4-factor', or 'CAPM'
        frequency: 'daily', 'weekly', or 'monthly' (default: 'monthly')
        risk_free_rate: '1M_TBILL' (use RF from Fama-French), '3M_TBILL', or custom rate
    
    Returns:
        dict: Comprehensive factor analysis results with structure:
            {
                "model": str,
                "ticker": str,
                "period": {...},
                "coefficients": {...},
                "statistics": {...},
                "time_series": [...]
            }
    
    Raises:
        ValueError: Invalid inputs, insufficient data, or unsupported model
        ConnectionError: Network errors fetching data
        ImportError: Missing required dependencies
    
    Example:
        >>> result = analyze_factors(
        ...     ticker="SPY",
        ...     start_date="2020-01-01",
        ...     end_date="2025-01-01",
        ...     factor_model="3-factor",
        ...     frequency="monthly"
        ... )
        >>> print(result['statistics']['alpha_annualized'])
        0.0123
    """
    # Validate inputs
    if factor_model not in ['3-factor', '5-factor', '4-factor', 'CAPM']:
        raise ValueError(
            f"Factor model must be one of: '3-factor', '5-factor', '4-factor', 'CAPM'. "
            f"Got: {factor_model}"
        )
    
    if frequency not in ['daily', 'weekly', 'monthly']:
        raise ValueError(
            f"Frequency must be one of: 'daily', 'weekly', 'monthly'. Got: {frequency}"
        )
    
    # Convert dates
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
    
    logger.info(f"Starting factor analysis for {ticker} using {factor_model} model")
    
    # Step 1: Fetch ticker returns
    logger.info(f"Fetching returns for {ticker}")
    ticker_returns = _fetch_ticker_returns(ticker, start_date, end_date, frequency)
    
    # Step 2: Fetch Fama-French factors
    logger.info(f"Fetching Fama-French {factor_model} factors")
    try:
        factors = get_factors(start_date, end_date)
    except Exception as e:
        raise ConnectionError(f"Failed to fetch Fama-French data: {str(e)}") from e
    
    # Check if we have required factors
    required_factors = _get_factor_columns(factor_model)
    missing_factors = [f for f in required_factors if f not in factors.columns]
    
    if missing_factors:
        # For 4-factor and 5-factor, we may need to fetch additional data
        # For now, raise error if factors are missing
        if factor_model == '4-factor' and 'MOM' not in factors.columns:
            raise ValueError(
                "Momentum (MOM) factor not available in Fama-French 3-factor data. "
                "4-factor model requires separate momentum data."
            )
        elif factor_model == '5-factor':
            missing = [f for f in ['RMW', 'CMA'] if f not in factors.columns]
            if missing:
                raise ValueError(
                    f"5-factor model requires RMW and CMA factors. Missing: {missing}. "
                    "These are not available in the 3-factor dataset."
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
        "ticker": ticker.upper(),
        "period": {
            "start": start_date.strftime('%Y-%m-%d'),
            "end": end_date.strftime('%Y-%m-%d'),
            "observations": len(aligned_data),
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
        "time_series": time_series
    }
    
    # Add frequency-specific alpha if not monthly
    if frequency == 'daily':
        result["statistics"]["alpha_daily"] = float(alpha)
    elif frequency == 'weekly':
        result["statistics"]["alpha_weekly"] = float(alpha)
    
    logger.info(f"Factor analysis complete. R² = {r_squared:.3f}, Alpha (annualized) = {alpha_annualized:.4f}")
    
    return result

