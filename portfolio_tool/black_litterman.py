"""
Black-Litterman Portfolio Optimization Module

The Black-Litterman model combines market equilibrium returns with investor views
to produce a set of expected returns that can be used for portfolio optimization.

Key Concepts:
- Prior Returns (π): Market-implied equilibrium returns derived from market cap weights
- Views: Investor's beliefs about expected returns (absolute or relative)
- Posterior Returns: Blended returns after incorporating investor views
- Confidence (τ, Ω): Uncertainty parameters that control how much views affect the result

Formula:
E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]

Where:
- π = market equilibrium returns (from reverse optimization)
- Σ = covariance matrix
- τ = uncertainty scalar (typically 0.025 to 0.05)
- P = view matrix (which assets the views are about)
- Q = view vector (expected returns from views)
- Ω = uncertainty matrix for views (based on confidence)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from pypfopt import BlackLittermanModel, EfficientFrontier
from pypfopt import risk_models, expected_returns
from pypfopt.black_litterman import market_implied_risk_aversion, market_implied_prior_returns

from portfolio_tool.market_data import get_price_history

logger = logging.getLogger("portfolio-api.black-litterman")


def fetch_price_data(
    tickers: List[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch historical price data for the given tickers.

    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        DataFrame with adjusted close prices, indexed by date

    Raises:
        ValueError: If no data available for any ticker
    """
    logger.info(f"Fetching price data for {tickers} from {start_date} to {end_date}")

    prices = get_price_history(tickers, start_date, end_date)

    if prices.empty:
        raise ValueError(f"No price data available for tickers: {tickers}")

    # Check for missing tickers
    missing = set(tickers) - set(prices.columns)
    if missing:
        raise ValueError(f"No data available for tickers: {list(missing)}")

    # Check date range
    data_days = (prices.index.max() - prices.index.min()).days
    if data_days < 252:  # Minimum 1 year of data
        raise ValueError(
            f"Insufficient data: {data_days} days available, minimum 252 days (1 year) required. "
            f"Extend your date range or use tickers with longer history."
        )

    logger.info(f"Fetched {len(prices)} days of price data from {prices.index.min()} to {prices.index.max()}")

    return prices


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from price data.

    Args:
        prices: DataFrame of daily prices

    Returns:
        DataFrame of daily returns
    """
    returns = prices.pct_change().dropna()
    logger.info(f"Calculated {len(returns)} daily returns")
    return returns


def calculate_covariance_matrix(
    returns: pd.DataFrame,
    method: str = "sample"
) -> np.ndarray:
    """
    Calculate the covariance matrix of returns.

    Args:
        returns: DataFrame of daily returns
        method: Covariance estimation method ('sample', 'ledoit_wolf', 'shrunk')

    Returns:
        Annualized covariance matrix as numpy array
    """
    if method == "ledoit_wolf":
        from pypfopt import risk_models
        cov = risk_models.CovarianceShrinkage(returns).ledoit_wolf()
    elif method == "shrunk":
        from pypfopt import risk_models
        cov = risk_models.CovarianceShrinkage(returns).shrunk_covariance()
    else:
        # Sample covariance, annualized
        cov = returns.cov() * 252

    logger.info(f"Calculated {method} covariance matrix, shape: {cov.shape}")

    return cov


def calculate_market_cap_weights(
    market_caps: Dict[str, float]
) -> Dict[str, float]:
    """
    Calculate market-cap weights from market capitalizations.

    Args:
        market_caps: Dictionary mapping ticker to market cap in dollars

    Returns:
        Dictionary mapping ticker to weight (sums to 1.0)
    """
    total_cap = sum(market_caps.values())
    if total_cap <= 0:
        raise ValueError("Total market capitalization must be positive")

    weights = {ticker: cap / total_cap for ticker, cap in market_caps.items()}

    logger.info(f"Market cap weights: {weights}")

    return weights


def calculate_prior_returns(
    market_caps: Dict[str, float],
    cov_matrix: pd.DataFrame,
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.04
) -> pd.Series:
    """
    Calculate market-implied prior returns using reverse optimization.

    The prior returns (π) represent the market's equilibrium expected returns,
    derived from the assumption that the market portfolio is optimal.

    Formula: π = δ * Σ * w_mkt

    Where:
    - δ = risk aversion coefficient
    - Σ = covariance matrix
    - w_mkt = market capitalization weights

    Args:
        market_caps: Market capitalizations for each asset
        cov_matrix: Covariance matrix (as DataFrame with ticker index/columns)
        risk_aversion: Risk aversion coefficient (default 2.5)
        risk_free_rate: Annual risk-free rate (default 0.04)

    Returns:
        Series of prior expected returns for each asset
    """
    # Calculate market cap weights
    mcap_weights = calculate_market_cap_weights(market_caps)

    # Ensure tickers are aligned
    tickers = list(cov_matrix.columns)
    weights_array = np.array([mcap_weights[t] for t in tickers])

    # Prior returns: π = δ * Σ * w_mkt
    prior_returns = risk_aversion * cov_matrix.values @ weights_array

    prior_series = pd.Series(prior_returns, index=tickers)

    logger.info(f"Prior returns (market equilibrium):")
    for ticker, ret in prior_series.items():
        logger.info(f"  {ticker}: {ret:.4f} ({ret*100:.2f}%)")

    return prior_series


def build_view_matrices(
    views: Dict[str, List[Dict]],
    tickers: List[str],
    cov_matrix: pd.DataFrame,
    tau: float = 0.05
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build the view matrices P, Q, and Ω from the views specification.

    Args:
        views: Dictionary with 'absolute' and 'relative' view lists
        tickers: List of ticker symbols
        cov_matrix: Covariance matrix
        tau: Uncertainty scalar

    Returns:
        Tuple of (P, Q, omega) matrices, or (None, None, None) if no views
    """
    absolute_views = views.get('absolute', [])
    relative_views = views.get('relative', [])

    n_views = len(absolute_views) + len(relative_views)
    n_assets = len(tickers)

    if n_views == 0:
        logger.info("No views provided, returning None matrices")
        return None, None, None

    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    P = np.zeros((n_views, n_assets))
    Q = np.zeros(n_views)
    confidences = []

    view_idx = 0

    # Process absolute views
    for view in absolute_views:
        asset = view['asset']
        expected_return = view['return']
        confidence = view['confidence']

        if asset not in ticker_to_idx:
            raise ValueError(f"Asset '{asset}' in absolute view not found in tickers")

        P[view_idx, ticker_to_idx[asset]] = 1.0
        Q[view_idx] = expected_return
        confidences.append(confidence)

        logger.info(f"Absolute view {view_idx}: {asset} will return {expected_return:.2%} (confidence: {confidence:.2f})")
        view_idx += 1

    # Process relative views
    for view in relative_views:
        asset1 = view['asset1']
        asset2 = view['asset2']
        outperformance = view['outperformance']
        confidence = view['confidence']

        if asset1 not in ticker_to_idx:
            raise ValueError(f"Asset '{asset1}' in relative view not found in tickers")
        if asset2 not in ticker_to_idx:
            raise ValueError(f"Asset '{asset2}' in relative view not found in tickers")

        P[view_idx, ticker_to_idx[asset1]] = 1.0
        P[view_idx, ticker_to_idx[asset2]] = -1.0
        Q[view_idx] = outperformance
        confidences.append(confidence)

        logger.info(f"Relative view {view_idx}: {asset1} will outperform {asset2} by {outperformance:.2%} (confidence: {confidence:.2f})")
        view_idx += 1

    # Build omega matrix based on confidence
    # Omega represents uncertainty in views
    # Lower confidence = higher variance = larger omega diagonal
    # Formula: Ω_ii = (1/confidence - 1) * τ * (P * Σ * P')_ii
    omega_diag = []
    for i, conf in enumerate(confidences):
        # Compute view variance: P_i * Σ * P_i'
        view_variance = P[i] @ cov_matrix.values @ P[i]
        # Scale by uncertainty: higher confidence = lower omega
        # Using (1/conf - 1) to map confidence 0-1 to uncertainty
        uncertainty_scale = (1.0 / max(conf, 0.01)) - 1.0
        omega_ii = uncertainty_scale * tau * view_variance
        omega_diag.append(max(omega_ii, 1e-8))  # Ensure positive

    omega = np.diag(omega_diag)

    logger.info(f"Built view matrices: P shape {P.shape}, Q shape {Q.shape}, Ω shape {omega.shape}")

    return P, Q, omega


def calculate_posterior_returns(
    prior_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    P: Optional[np.ndarray],
    Q: Optional[np.ndarray],
    omega: Optional[np.ndarray],
    tau: float = 0.05
) -> pd.Series:
    """
    Calculate posterior expected returns by blending prior with views.

    Black-Litterman formula:
    E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 π + P'Ω^-1 Q]

    Args:
        prior_returns: Market equilibrium returns (π)
        cov_matrix: Covariance matrix (Σ)
        P: View matrix
        Q: View vector
        omega: View uncertainty matrix (Ω)
        tau: Uncertainty scalar

    Returns:
        Series of posterior expected returns
    """
    tickers = list(prior_returns.index)
    Sigma = cov_matrix.values
    pi = prior_returns.values

    # If no views, posterior equals prior
    if P is None or Q is None or omega is None:
        logger.info("No views provided, posterior returns equal prior returns")
        return prior_returns

    # Precision matrix of prior: (τΣ)^-1
    tau_sigma_inv = np.linalg.inv(tau * Sigma)

    # Precision matrix of views: Ω^-1
    omega_inv = np.linalg.inv(omega)

    # Posterior precision: (τΣ)^-1 + P'Ω^-1P
    posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P

    # Posterior covariance
    posterior_cov = np.linalg.inv(posterior_precision)

    # Posterior mean: posterior_cov * [(τΣ)^-1 π + P'Ω^-1 Q]
    posterior_mean = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    posterior_series = pd.Series(posterior_mean, index=tickers)

    logger.info(f"Posterior returns (after views):")
    for ticker, ret in posterior_series.items():
        prior_ret = prior_returns[ticker]
        logger.info(f"  {ticker}: {ret:.4f} ({ret*100:.2f}%) [prior: {prior_ret:.4f}]")

    return posterior_series


def optimize_portfolio(
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.04
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Optimize portfolio weights using mean-variance optimization.

    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        risk_free_rate: Annual risk-free rate

    Returns:
        Tuple of (weights dict, optimization stats dict)
    """
    try:
        # Use PyPortfolioOpt's EfficientFrontier
        ef = EfficientFrontier(expected_returns, cov_matrix)

        # Add constraints: no shorting, full investment
        ef.add_constraint(lambda w: w >= 0)
        ef.add_constraint(lambda w: sum(w) == 1)

        # Optimize for maximum Sharpe ratio
        try:
            weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        except Exception as e:
            logger.warning(f"Max Sharpe optimization failed: {e}, trying min volatility")
            # Fall back to minimum volatility
            ef = EfficientFrontier(expected_returns, cov_matrix)
            ef.add_constraint(lambda w: w >= 0)
            ef.add_constraint(lambda w: sum(w) == 1)
            weights = ef.min_volatility()

        # Clean weights (remove very small allocations)
        cleaned_weights = ef.clean_weights(cutoff=0.001)

        # Get performance metrics
        performance = ef.portfolio_performance(
            verbose=False,
            risk_free_rate=risk_free_rate
        )

        stats = {
            'expected_return': performance[0],
            'volatility': performance[1],
            'sharpe_ratio': performance[2]
        }

        logger.info(f"Optimal weights: {cleaned_weights}")
        logger.info(f"Portfolio stats: return={stats['expected_return']:.4f}, vol={stats['volatility']:.4f}, sharpe={stats['sharpe_ratio']:.4f}")

        return cleaned_weights, stats

    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        # Return equal weights as fallback
        n_assets = len(expected_returns)
        tickers = list(expected_returns.index)
        equal_weights = {t: 1.0 / n_assets for t in tickers}

        # Calculate stats for equal weights
        w = np.array([1.0 / n_assets] * n_assets)
        port_return = expected_returns.values @ w
        port_vol = np.sqrt(w @ cov_matrix.values @ w)
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0

        stats = {
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe
        }

        logger.warning(f"Using equal weights fallback due to optimization failure")

        return equal_weights, stats


def calculate_portfolio_metrics(
    weights: Dict[str, float],
    expected_returns: pd.Series,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float = 0.04
) -> Dict[str, float]:
    """
    Calculate portfolio performance metrics for given weights.

    Args:
        weights: Portfolio weights
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix
        risk_free_rate: Annual risk-free rate

    Returns:
        Dictionary with expected_return, expected_risk, sharpe_ratio
    """
    tickers = list(expected_returns.index)
    w = np.array([weights.get(t, 0.0) for t in tickers])

    # Expected return
    port_return = expected_returns.values @ w

    # Expected volatility (risk)
    port_vol = np.sqrt(w @ cov_matrix.values @ w)

    # Sharpe ratio
    sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0.0

    return {
        'expected_return': float(port_return),
        'expected_risk': float(port_vol),
        'sharpe_ratio': float(sharpe)
    }


def run_black_litterman_optimization(
    tickers: List[str],
    start_date: str,
    end_date: str,
    market_caps: Dict[str, float],
    views: Dict[str, List[Dict]],
    risk_aversion: float = 2.5,
    risk_free_rate: float = 0.04,
    tau: float = 0.05
) -> Dict[str, Any]:
    """
    Run full Black-Litterman portfolio optimization.

    This is the main entry point that orchestrates the entire BL process:
    1. Fetch price data
    2. Calculate returns and covariance
    3. Calculate prior (equilibrium) returns
    4. Apply investor views to get posterior returns
    5. Optimize portfolio weights
    6. Compare with market-cap weighted portfolio

    Args:
        tickers: List of ticker symbols
        start_date: Start date for historical data
        end_date: End date for historical data
        market_caps: Market capitalizations for each ticker
        views: Investor views (absolute and relative)
        risk_aversion: Risk aversion coefficient (higher = more conservative)
        risk_free_rate: Annual risk-free rate
        tau: Uncertainty scalar for Black-Litterman (typically 0.025-0.05)

    Returns:
        Dictionary with all optimization results
    """
    logger.info("=" * 60)
    logger.info("Starting Black-Litterman Optimization")
    logger.info(f"Tickers: {tickers}")
    logger.info(f"Date range: {start_date} to {end_date}")
    logger.info(f"Risk aversion: {risk_aversion}")
    logger.info(f"Risk-free rate: {risk_free_rate}")
    logger.info(f"Tau: {tau}")
    logger.info("=" * 60)

    # 1. Fetch and validate data
    prices = fetch_price_data(tickers, start_date, end_date)

    # 2. Calculate returns and covariance
    returns = calculate_returns(prices)
    cov_matrix = calculate_covariance_matrix(returns, method="ledoit_wolf")

    # Convert to DataFrame for easier handling
    cov_df = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)

    # 3. Calculate prior returns (market equilibrium)
    prior_returns = calculate_prior_returns(
        market_caps=market_caps,
        cov_matrix=cov_df,
        risk_aversion=risk_aversion,
        risk_free_rate=risk_free_rate
    )

    # 4. Build view matrices and calculate posterior returns
    P, Q, omega = build_view_matrices(views, tickers, cov_df, tau)

    posterior_returns = calculate_posterior_returns(
        prior_returns=prior_returns,
        cov_matrix=cov_df,
        P=P,
        Q=Q,
        omega=omega,
        tau=tau
    )

    # 5. Optimize portfolio using posterior returns
    optimal_weights, optimal_stats = optimize_portfolio(
        expected_returns=posterior_returns,
        cov_matrix=cov_df,
        risk_free_rate=risk_free_rate
    )

    # 6. Calculate market-cap weighted portfolio metrics for comparison
    mcap_weights = calculate_market_cap_weights(market_caps)
    mcap_metrics = calculate_portfolio_metrics(
        weights=mcap_weights,
        expected_returns=prior_returns,  # Use prior returns for fair comparison
        cov_matrix=cov_df,
        risk_free_rate=risk_free_rate
    )

    # 7. Calculate BL portfolio metrics
    bl_metrics = calculate_portfolio_metrics(
        weights=optimal_weights,
        expected_returns=posterior_returns,
        cov_matrix=cov_df,
        risk_free_rate=risk_free_rate
    )

    # Validate outputs
    weight_sum = sum(optimal_weights.values())
    if abs(weight_sum - 1.0) > 0.01:
        logger.warning(f"Optimal weights sum to {weight_sum}, normalizing...")
        optimal_weights = {k: v / weight_sum for k, v in optimal_weights.items()}

    # Check for reasonable expected returns
    for ticker, ret in posterior_returns.items():
        if ret < -0.50 or ret > 0.50:
            logger.warning(f"Unusual expected return for {ticker}: {ret:.2%}")

    result = {
        'optimal_weights': {k: round(v, 6) for k, v in optimal_weights.items()},
        'expected_return': round(bl_metrics['expected_return'], 6),
        'expected_risk': round(bl_metrics['expected_risk'], 6),
        'sharpe_ratio': round(bl_metrics['sharpe_ratio'], 4),
        'prior_returns': {k: round(v, 6) for k, v in prior_returns.items()},
        'posterior_returns': {k: round(v, 6) for k, v in posterior_returns.items()},
        'market_cap_weights': {k: round(v, 6) for k, v in mcap_weights.items()},
        'comparison': {
            'market_weighted': {
                'return': round(mcap_metrics['expected_return'], 6),
                'risk': round(mcap_metrics['expected_risk'], 6),
                'sharpe': round(mcap_metrics['sharpe_ratio'], 4)
            },
            'black_litterman': {
                'return': round(bl_metrics['expected_return'], 6),
                'risk': round(bl_metrics['expected_risk'], 6),
                'sharpe': round(bl_metrics['sharpe_ratio'], 4)
            }
        }
    }

    logger.info("=" * 60)
    logger.info("Black-Litterman Optimization Complete")
    logger.info(f"Optimal Sharpe Ratio: {result['sharpe_ratio']:.4f}")
    logger.info("=" * 60)

    return result
