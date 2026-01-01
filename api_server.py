"""FastAPI server wrapper for Portfolio Analysis backend."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import io
from datetime import datetime, timedelta

def safe_float(value, default=0.0):
    """
    Safely convert a value to float, handling NaN and None.
    Returns default (0.0) if value is NaN, None, or cannot be converted.
    """
    if value is None:
        return default
    if pd.isna(value):
        return default
    try:
        result = float(value)
        if pd.isna(result) or np.isnan(result):
            return default
        return result
    except (ValueError, TypeError):
        return default

def slice_to_effective_window(series_or_df, start_date, end_date):
    """
    Slice a Series or DataFrame to the effective window (start_date to end_date inclusive).
    
    This is a pure filtering helper - no math, just date alignment.
    Ensures all metrics use the same date window for consistency.
    
    Args:
        series_or_df: pandas Series or DataFrame with DatetimeIndex
        start_date: start date (inclusive)
        end_date: end date (inclusive)
    
    Returns:
        Sliced Series or DataFrame
    """
    if series_or_df.empty:
        return series_or_df
    
    if start_date is None or end_date is None:
        return series_or_df
    
    # Slice to the effective window
    mask = (series_or_df.index >= start_date) & (series_or_df.index <= end_date)
    return series_or_df.loc[mask]


def compute_volatility_from_window(daily_returns):
    """
    Compute annualized volatility using the entire windowed daily returns series.
    No internal lookback - uses all data provided.
    
    Formula: std(daily_returns) * sqrt(252)
    """
    if len(daily_returns) == 0:
        return None
    
    if len(daily_returns) < 20:  # Need at least ~20 trading days
        return None
    
    # Annualized volatility: std * sqrt(252 trading days per year)
    volatility = daily_returns.std() * np.sqrt(252)
    return float(volatility)


def compute_sharpe_from_window(daily_returns, period_years, risk_free_rate=0.0):
    """
    Compute annualized Sharpe ratio using the entire windowed daily returns series.
    Uses fractional years for proper annualization.
    
    Formula: (mean daily excess return / daily return volatility) * sqrt(252)
    """
    if len(daily_returns) == 0:
        return None
    
    if len(daily_returns) < MIN_OBS_SHARPE:
        return None
    
    # Convert annual risk-free rate to daily using compound interest
    daily_rf = (1 + risk_free_rate) ** (1 / 252.0) - 1
    
    # Compute mean daily excess return
    excess_returns = daily_returns - daily_rf
    mean_daily_excess_return = excess_returns.mean()
    
    # Compute standard deviation of daily returns
    daily_volatility = daily_returns.std()
    
    if daily_volatility == 0 or pd.isna(daily_volatility) or pd.isna(mean_daily_excess_return):
        return None
    
    # Sharpe = (mean daily excess return / daily return volatility) * sqrt(252)
    sharpe = (mean_daily_excess_return / daily_volatility) * np.sqrt(252)
    return float(sharpe)


def compute_sortino_from_window(daily_returns, period_years, risk_free_rate=0.0):
    """
    Compute annualized Sortino ratio using the entire windowed daily returns series.
    Uses fractional years for proper annualization.
    """
    if len(daily_returns) == 0:
        return None
    
    if len(daily_returns) < MIN_OBS_SORTINO:
        return None
    
    # Convert annual risk-free rate to daily
    daily_rf = (1 + risk_free_rate) ** (1 / 252.0) - 1
    
    # Compute mean daily excess return
    excess_returns = daily_returns - daily_rf
    mean_daily_excess_return = excess_returns.mean()
    
    # Compute downside deviation (std of negative returns only)
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) == 0:
        # No negative returns - downside deviation is 0, Sortino undefined
        return None
    
    downside_deviation = negative_returns.std()
    if downside_deviation == 0 or pd.isna(downside_deviation) or pd.isna(mean_daily_excess_return):
        return None
    
    # Sortino = (mean daily excess return / downside deviation) * sqrt(252)
    sortino = (mean_daily_excess_return / downside_deviation) * np.sqrt(252)
    return float(sortino)


def compute_beta_from_window(portfolio_daily_returns, benchmark_daily_returns):
    """
    Compute beta using the entire windowed daily returns series.
    No internal lookback - uses all data provided.
    
    Formula: Covariance(portfolio, benchmark) / Variance(benchmark)
    """
    if len(portfolio_daily_returns) == 0 or len(benchmark_daily_returns) == 0:
        return None
    
    # Align on common dates
    common_dates = portfolio_daily_returns.index.intersection(benchmark_daily_returns.index)
    if len(common_dates) < 20:
        return None
    
    portfolio_aligned = portfolio_daily_returns.loc[common_dates]
    benchmark_aligned = benchmark_daily_returns.loc[common_dates]
    
    # Beta = Covariance(portfolio, benchmark) / Variance(benchmark)
    covariance = portfolio_aligned.cov(benchmark_aligned)
    benchmark_variance = benchmark_aligned.var()
    
    if benchmark_variance == 0:
        return None
    
    beta = covariance / benchmark_variance
    return float(beta)


def compute_max_drawdown_from_window(cumulative_index):
    """
    Compute maximum drawdown using the entire windowed cumulative index series.
    No internal lookback - uses all data provided.
    """
    if len(cumulative_index) == 0:
        return None
    
    if len(cumulative_index) < 20:
        return None
    
    # Calculate running maximum (peak)
    running_max = cumulative_index.expanding().max()
    
    # Calculate drawdown from peak
    drawdown = (cumulative_index - running_max) / running_max
    
    # Maximum drawdown (most negative)
    max_dd = drawdown.min()
    return float(max_dd)


def compute_annualized_return_and_volatility_from_window(daily_returns, period_years):
    """
    Compute annualized return and volatility using the entire windowed daily returns series.
    Uses fractional years for proper annualization.
    
    Returns: tuple (annualized_return, annualized_volatility) as decimals
    """
    if len(daily_returns) == 0:
        return None, None
    
    if len(daily_returns) < 20:
        return None, None
    
    if period_years is None or period_years <= 0:
        return None, None
    
    # Total return = (1 + daily_returns).prod() - 1
    total_return = (1 + daily_returns).prod() - 1
    
    # Annualized return (CAGR) = (1 + total_return) ** (1 / period_years) - 1
    annualized_return = (1 + total_return) ** (1 / period_years) - 1 if period_years > 0 else None
    
    # Annualized volatility = std(daily_returns) * sqrt(252)
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    
    if annualized_return is None or pd.isna(annualized_return) or pd.isna(annualized_volatility):
        return None, None
    
    return float(annualized_return), float(annualized_volatility)

from portfolio_tool.data_io import load_portfolio
from portfolio_tool.market_data import get_price_history, get_sector_info, get_risk_free_rate
from portfolio_tool.analytics import (
    compute_returns,
    compute_period_returns,
    compute_period_returns_vs_benchmark,
    compute_cumulative_index,
    compute_monthly_portfolio_returns,
    compute_risk_metrics,
    compute_correlation_matrix,
    compute_annualized_return_and_volatility,
    compute_efficient_frontier_analysis,
    compute_daily_returns,
    compute_volatility,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_beta,
    compute_max_drawdown,
    compute_ulcer_index,
    compute_drawdown_series,
    compute_rolling_sharpe_ratio,
    compute_ytd_contribution,
    compute_rolling_volatility,
    compute_rolling_beta,
    get_effective_start_date,
    get_as_of_date,
    compute_ytd_risk_contribution,
    compute_window_cagr,
    compute_asset_breakdown,
    MIN_OBS_CALMAR,
    MIN_OBS_SHARPE,
    MIN_OBS_SORTINO,
)

app = FastAPI(
    title="Portfolio Analysis API",
    description="Backend API for portfolio analytics using real market data",
    version="1.0.0"
)

# CORS middleware - allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:5173",  # Vite default port
        "http://localhost:5174",  # Vite alternate port
        "http://localhost:8080",  # Current frontend port
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PortfolioRequest(BaseModel):
    portfolioText: str
    requested_start_date: Optional[str] = None  # ISO format date string (YYYY-MM-DD)


def parse_portfolio_text(portfolio_text: str) -> pd.DataFrame:
    """Parse portfolio text (CSV format) into DataFrame matching load_portfolio format."""
    lines = portfolio_text.strip().split('\n')
    holdings = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and header row
        if not line or line.lower().startswith('ticker'):
            continue
        
        parts = [p.strip() for p in line.split(',')]
        if len(parts) >= 2:
            ticker = parts[0].upper()
            weight_str = parts[1].replace('%', '').strip()
            try:
                weight = float(weight_str)
                # Convert to decimal if > 1 (assume percentage)
                weight_decimal = weight / 100.0 if weight > 1.0 else weight
                holdings.append({'ticker': ticker, 'weight': weight_decimal})
            except ValueError:
                continue
    
    if not holdings:
        raise ValueError("No valid portfolio entries found")
    
    df = pd.DataFrame(holdings)
    
    # Normalize weights to sum to 1
    total_weight = df['weight'].sum()
    if total_weight > 0:
        df['weight'] = df['weight'] / total_weight
    
    return df


@app.post("/analyze")
async def analyze_portfolio(request: PortfolioRequest):
    """Analyze portfolio and return formatted results matching frontend expectations."""
    try:
        # Parse portfolio from text
        portfolio_df = parse_portfolio_text(request.portfolioText)
        
        tickers = portfolio_df['ticker'].tolist()
        weights = portfolio_df.set_index('ticker')['weight'].to_dict()
        
        # Filter out cash tickers before fetching prices (cash has no price data)
        from portfolio_tool.analytics import is_cash_ticker
        invested_tickers = [ticker for ticker in tickers if not is_cash_ticker(ticker)]
        
        benchmark_ticker = "SPY"
        today = datetime.today()
        
        # Parse user-selected start date if provided
        user_selected_start_date = None
        if request.requested_start_date:
            try:
                user_selected_start_date = pd.Timestamp(request.requested_start_date)
                if user_selected_start_date > pd.Timestamp(today):
                    user_selected_start_date = None  # Invalid future date
            except (ValueError, TypeError):
                user_selected_start_date = None
        
        # Determine fetch range: need to fetch enough data to find common_start_date
        # Fetch from max(user_selected_start - buffer, 10 years back) to ensure we have data
        if user_selected_start_date:
            fetch_start_date = (user_selected_start_date - timedelta(days=30)).strftime('%Y-%m-%d')  # 30 day buffer
        else:
            fetch_start_date = (today - timedelta(days=365 * 10)).strftime('%Y-%m-%d')  # Default: 10 years back
        
        # Use tomorrow as end_date to ensure we get data through today (yfinance end parameter is exclusive)
        fetch_end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch real market data (only for invested tickers, not cash)
        # If no invested tickers, we still need benchmark data
        if invested_tickers:
            all_tickers = invested_tickers + [benchmark_ticker]
            prices_all = get_price_history(all_tickers, fetch_start_date, fetch_end_date)
            prices_portfolio = prices_all[invested_tickers]
            prices_benchmark = prices_all[[benchmark_ticker]]
        else:
            # Portfolio is 100% cash - only fetch benchmark
            prices_all = get_price_history([benchmark_ticker], fetch_start_date, fetch_end_date)
            prices_portfolio = pd.DataFrame()  # Empty DataFrame for 100% cash portfolio
            prices_benchmark = prices_all[[benchmark_ticker]]
        
        # Get as_of_date: last common available trading date across portfolio and benchmark
        # Find minimum of last available dates to ensure we have data for all holdings
        last_dates = []
        if not prices_portfolio.empty:
            for ticker in invested_tickers:
                if ticker in prices_portfolio.columns:
                    ticker_prices = prices_portfolio[ticker].dropna()
                    if not ticker_prices.empty:
                        last_dates.append(ticker_prices.index.max())
        if not prices_benchmark.empty:
            bench_prices = prices_benchmark[benchmark_ticker].dropna()
            if not bench_prices.empty:
                last_dates.append(bench_prices.index.max())
        
        if last_dates:
            as_of_date = min(last_dates)  # Use minimum to ensure all holdings have data
        else:
            as_of_date = pd.Timestamp(today)
        
        # Find common_start_date: latest first-valid date across all non-cash holdings
        common_start_date = None
        limiting_ticker = None
        limiting_start_date = None
        
        if not prices_portfolio.empty and invested_tickers:
            ticker_first_dates = {}
            for ticker in invested_tickers:
                if ticker in prices_portfolio.columns:
                    first_valid = prices_portfolio[ticker].first_valid_index()
                    if first_valid is not None:
                        ticker_first_dates[ticker] = first_valid
            
            if ticker_first_dates:
                common_start_date = max(ticker_first_dates.values())
                # Find limiting ticker
                for ticker in sorted(ticker_first_dates.keys()):
                    if ticker_first_dates[ticker] == common_start_date:
                        limiting_ticker = ticker
                        limiting_start_date = ticker_first_dates[ticker]
                        break
        
        # Fallback if no non-cash holdings
        if common_start_date is None:
            if not prices_benchmark.empty:
                benchmark_first = prices_benchmark[benchmark_ticker].first_valid_index()
                if benchmark_first is not None:
                    common_start_date = benchmark_first
            if common_start_date is None:
                common_start_date = pd.Timestamp(fetch_start_date)
        
        # Compute effective_start_date = max(user_selected_start_date, common_start_date)
        if user_selected_start_date is not None:
            effective_start_date = max(user_selected_start_date, common_start_date)
        else:
            effective_start_date = common_start_date
        
        # Window prices to [effective_start_date, as_of_date]
        prices_portfolio_windowed = slice_to_effective_window(prices_portfolio, effective_start_date, as_of_date)
        prices_benchmark_windowed = slice_to_effective_window(prices_benchmark, effective_start_date, as_of_date)
        
        # Calculate period length in fractional years for annualization
        period_days = (as_of_date - effective_start_date).days
        period_years = period_days / 365.25 if period_days > 0 else None
        
        # Get risk-free rate
        risk_free_rate = get_risk_free_rate('^TNX')
        if risk_free_rate is None:
            risk_free_rate = 0.02  # Default 2%
        
        # Compute daily returns and cumulative index ONCE using the windowed prices
        # These will be used for ALL metrics to ensure consistency
        portfolio_daily_windowed = compute_daily_returns(prices_portfolio_windowed, weights=weights)
        benchmark_daily_windowed = compute_daily_returns(prices_benchmark_windowed, weights=None)
        portfolio_cum_windowed = compute_cumulative_index(prices_portfolio_windowed, weights=weights)
        benchmark_cum_windowed = compute_cumulative_index(prices_benchmark_windowed, weights=None)
        
        # Compute yearly returns for returns table using windowed prices
        ticker_returns, portfolio_returns = compute_returns(
            prices_portfolio_windowed, weights=weights, years_back=6
        )
        bench_ticker_returns, _ = compute_returns(
            prices_benchmark_windowed, weights=None, years_back=6
        )
        benchmark_returns = bench_ticker_returns[benchmark_ticker]
        
        # Build returns table (yearly + YTD)
        returns_table = []
        # Use as_of_date.year for consistency (matches the actual data range)
        current_year = as_of_date.year if isinstance(as_of_date, pd.Timestamp) else pd.Timestamp(as_of_date).year
        
        # Get years from portfolio_returns index (excluding YTD)
        years = [idx for idx in portfolio_returns.index if isinstance(idx, int)]
        years.sort()
        # Include YTD if present
        if "YTD" in portfolio_returns.index:
            years.append("YTD")
        
        for period in years:
            port_ret = safe_float(portfolio_returns[period] if period in portfolio_returns.index else 0.0)
            bench_ret = safe_float(benchmark_returns[period] if period in benchmark_returns.index else 0.0)
            returns_table.append({
                "period": str(period),
                "portfolioReturn": port_ret,
                "benchmarkReturn": bench_ret
            })
        
        # Compute Period Returns vs Benchmark (1M, 3M, YTD, 1Y, 3Y, 5Y)
        # IMPORTANT: This is SEPARATE from dashboard logic. Uses FULL prices (not windowed) and
        # fixed-horizon buckets independent of the user-selected analysis window.
        # Dashboard stat cards should NOT source from this table - they use windowed data.
        # Handle empty portfolio case (100% cash)
        if prices_portfolio.empty:
            # For 100% cash portfolio, use empty DataFrame (will return empty dict)
            portfolio_period = compute_period_returns_vs_benchmark(pd.DataFrame(), weights=weights)
        else:
            portfolio_period = compute_period_returns_vs_benchmark(prices_portfolio, weights=weights)
        
        benchmark_period = compute_period_returns_vs_benchmark(prices_benchmark, weights=None)
        
        # Map period returns - preserve None values for periods with insufficient history
        # None values indicate insufficient data for that period (frontend shows "—" with tooltip)
        def safe_float_or_none(value):
            """Convert to float if valid, preserve None if value is None or NaN."""
            if value is None:
                return None
            result = safe_float(value, default=None)
            # If safe_float returns default (None) for NaN/Invalid, preserve None
            return result if result is not None else None
        
        period_returns_portfolio = {
            '1M': safe_float_or_none(portfolio_period.get('1M')),
            '3M': safe_float_or_none(portfolio_period.get('3M')),
            'YTD': safe_float_or_none(portfolio_period.get('YTD')),
            '1Y': safe_float_or_none(portfolio_period.get('1Y')),
            '3Y': safe_float_or_none(portfolio_period.get('3Y')),
            '5Y': safe_float_or_none(portfolio_period.get('5Y')),
        }
        
        period_returns_benchmark = {
            '1M': safe_float_or_none(benchmark_period.get('1M')),
            '3M': safe_float_or_none(benchmark_period.get('3M')),
            'YTD': safe_float_or_none(benchmark_period.get('YTD')),
            '1Y': safe_float_or_none(benchmark_period.get('1Y')),
            '3Y': safe_float_or_none(benchmark_period.get('3Y')),
            '5Y': safe_float_or_none(benchmark_period.get('5Y')),
        }
        
        # Compute risk metrics using the FULL effective window (no internal lookbacks)
        # All risk metrics use the exact window: [effective_start_date, as_of_date]
        risk_metrics = {
            'annualVolatility': None,
            'sharpeRatio': None,
            'sortinoRatio': None,
            'calmarRatio': None,
            'beta': None,
            'maxDrawdown': None,
        }
        
        # Portfolio risk metrics using windowed data only (no internal lookbacks)
        # Return as PERCENTAGES (46.0 for 46%) - frontend expects percentages
        if len(portfolio_daily_windowed) >= 20:
            vol = compute_volatility_from_window(portfolio_daily_windowed)
            risk_metrics['annualVolatility'] = safe_float_or_none(vol * 100) if vol is not None else None
        
        if len(portfolio_daily_windowed) >= MIN_OBS_SHARPE:
            sharpe = compute_sharpe_from_window(portfolio_daily_windowed, period_years, risk_free_rate)
            risk_metrics['sharpeRatio'] = safe_float_or_none(sharpe)
        
        if len(portfolio_daily_windowed) >= MIN_OBS_SORTINO:
            sortino = compute_sortino_from_window(portfolio_daily_windowed, period_years, risk_free_rate)
            risk_metrics['sortinoRatio'] = safe_float_or_none(sortino)
        
        if len(portfolio_daily_windowed) >= 20 and len(benchmark_daily_windowed) >= 20:
            beta = compute_beta_from_window(portfolio_daily_windowed, benchmark_daily_windowed)
            risk_metrics['beta'] = safe_float_or_none(beta)
        
        if len(portfolio_cum_windowed) >= 20:
            max_dd = compute_max_drawdown_from_window(portfolio_cum_windowed)
            risk_metrics['maxDrawdown'] = safe_float_or_none(max_dd * 100) if max_dd is not None else None
        
        # Calculate Calmar Ratio: CAGR / |Max Drawdown| using windowed data
        # Compute CAGR from the full window using fractional years
        # Both CAGR and Max Drawdown must be in the same units (percentages) for the ratio
        calmar_ratio = None
        if period_years and period_years >= 1.0 and len(portfolio_daily_windowed) >= MIN_OBS_CALMAR:
            # Compute total return from windowed data
            total_return = (1 + portfolio_daily_windowed).prod() - 1
            # CAGR = (1 + total_return) ** (1 / period_years) - 1 (as decimal, e.g., 0.15 for 15%)
            cagr_value_decimal = (1 + total_return) ** (1 / period_years) - 1 if period_years > 0 else None
            
            if cagr_value_decimal is not None and risk_metrics['maxDrawdown'] is not None:
                # Convert CAGR to percentage to match maxDrawdown (which is already a percentage)
                cagr_value_pct = cagr_value_decimal * 100
                max_dd_abs = abs(risk_metrics['maxDrawdown'])  # Already a percentage
                if max_dd_abs > 0 and not pd.isna(max_dd_abs):
                    calmar_ratio = safe_float(cagr_value_pct / max_dd_abs)
        risk_metrics['calmarRatio'] = calmar_ratio
        
        # Benchmark risk metrics using windowed data only (no internal lookbacks)
        benchmark_risk_metrics = {
            'annualVolatility': None,
            'sharpeRatio': None,
            'sortinoRatio': None,
            'calmarRatio': None,
            'maxDrawdown': None,
            'beta': 1.0,  # Benchmark beta is always 1.0 (beta against itself)
        }
        
        if len(benchmark_daily_windowed) >= 20:
            bench_vol = compute_volatility_from_window(benchmark_daily_windowed)
            benchmark_risk_metrics['annualVolatility'] = safe_float_or_none(bench_vol * 100) if bench_vol is not None else None
        
        if len(benchmark_daily_windowed) >= MIN_OBS_SHARPE:
            bench_sharpe = compute_sharpe_from_window(benchmark_daily_windowed, period_years, risk_free_rate)
            benchmark_risk_metrics['sharpeRatio'] = safe_float_or_none(bench_sharpe)
        
        if len(benchmark_daily_windowed) >= MIN_OBS_SORTINO:
            bench_sortino = compute_sortino_from_window(benchmark_daily_windowed, period_years, risk_free_rate)
            benchmark_risk_metrics['sortinoRatio'] = safe_float_or_none(bench_sortino)
        
        if len(benchmark_cum_windowed) >= 20:
            bench_max_dd = compute_max_drawdown_from_window(benchmark_cum_windowed)
            benchmark_risk_metrics['maxDrawdown'] = safe_float_or_none(bench_max_dd * 100) if bench_max_dd is not None else None
        
        # Benchmark Calmar Ratio using windowed data
        # Both CAGR and Max Drawdown must be in the same units (percentages) for the ratio
        bench_calmar_ratio = None
        if period_years and period_years >= 1.0 and len(benchmark_daily_windowed) >= MIN_OBS_CALMAR:
            # Compute CAGR from the full window using fractional years
            bench_total_return = (1 + benchmark_daily_windowed).prod() - 1
            bench_cagr_value_decimal = (1 + bench_total_return) ** (1 / period_years) - 1 if period_years > 0 else None
            
            if bench_cagr_value_decimal is not None and benchmark_risk_metrics['maxDrawdown'] is not None:
                # Convert CAGR to percentage to match maxDrawdown (which is already a percentage)
                bench_cagr_value_pct = bench_cagr_value_decimal * 100
                bench_max_dd_abs = abs(benchmark_risk_metrics['maxDrawdown'])  # Already a percentage
                if bench_max_dd_abs > 0 and not pd.isna(bench_max_dd_abs):
                    bench_calmar_ratio = safe_float(bench_cagr_value_pct / bench_max_dd_abs)
        benchmark_risk_metrics['calmarRatio'] = bench_calmar_ratio
        
        # Cumulative returns using windowed data [effective_start_date, as_of_date]
        # Return as PERCENTAGES (46.0 for 46%) - frontend expects percentages
        cumulative_return_portfolio = None
        cumulative_return_benchmark = None
        
        if len(portfolio_daily_windowed) > 0:
            # Total return = (1 + daily_returns).prod() - 1
            total_return = (1 + portfolio_daily_windowed).prod() - 1
            cumulative_return_portfolio = safe_float(total_return * 100)  # Convert to percentage
        
        if len(benchmark_daily_windowed) > 0:
            bench_total_return = (1 + benchmark_daily_windowed).prod() - 1
            cumulative_return_benchmark = safe_float(bench_total_return * 100)  # Convert to percentage
        
        # Compute window-based CAGR using fractional years
        # CAGR = (1 + total_return) ** (1 / period_years) - 1
        # Return as PERCENTAGES (46.0 for 46%) - frontend expects percentages
        summary_cagr_portfolio = None
        summary_cagr_benchmark = None
        
        if period_years and period_years > 0:
            if len(portfolio_daily_windowed) > 0:
                total_return = (1 + portfolio_daily_windowed).prod() - 1
                summary_cagr_portfolio = safe_float(((1 + total_return) ** (1 / period_years) - 1) * 100)
            
            if len(benchmark_daily_windowed) > 0:
                bench_total_return = (1 + benchmark_daily_windowed).prod() - 1
                summary_cagr_benchmark = safe_float(((1 + bench_total_return) ** (1 / period_years) - 1) * 100)
        
        # Performance metrics using windowed data
        # Note: "5Y" naming is kept for backward compatibility but uses the actual window
        performance_metrics_5y = {
            'cumulativeReturn5Y': cumulative_return_portfolio,
            'cumulativeReturn5YBenchmark': cumulative_return_benchmark,
            'cagr5Y': safe_float_or_none(summary_cagr_portfolio),
            'cagr5YBenchmark': safe_float_or_none(summary_cagr_benchmark),
            'maxDrawdown5Y': risk_metrics['maxDrawdown'],
            'maxDrawdown5YBenchmark': benchmark_risk_metrics['maxDrawdown'],
            'sharpeRatio5Y': risk_metrics['sharpeRatio'],
            'sharpeRatio5YBenchmark': benchmark_risk_metrics['sharpeRatio'],
        }
        
        # Align indices and create growth of $1,000 data using windowed series
        growth_of_100 = []
        common_dates = portfolio_cum_windowed.index.intersection(benchmark_cum_windowed.index)
        
        for date in common_dates:
            growth_of_100.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio': safe_float(portfolio_cum_windowed[date]),
                'benchmark': safe_float(benchmark_cum_windowed[date])
            })
        
        # Drawdown series (portfolio vs benchmark) using windowed series
        portfolio_drawdown = compute_drawdown_series(portfolio_cum_windowed)
        benchmark_drawdown = compute_drawdown_series(benchmark_cum_windowed)
        
        # Align drawdown series on common dates
        drawdown_common_dates = portfolio_drawdown.index.intersection(benchmark_drawdown.index)
        drawdown_data = []
        for date in drawdown_common_dates:
            drawdown_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'portfolio': safe_float(portfolio_drawdown[date] * 100),  # Convert to percentage
                'benchmark': safe_float(benchmark_drawdown[date] * 100)  # Convert to percentage
            })
        
        # Rolling Sharpe ratio (6-month window) using windowed series
        # portfolio_daily_windowed and benchmark_daily_windowed already computed above
        portfolio_rolling_sharpe = compute_rolling_sharpe_ratio(portfolio_daily_windowed, window_months=6, risk_free_rate=risk_free_rate)
        benchmark_rolling_sharpe = compute_rolling_sharpe_ratio(benchmark_daily_windowed, window_months=6, risk_free_rate=risk_free_rate)
        
        # Align rolling Sharpe series on common dates
        # Filter out NaN values (periods before full window is available)
        rolling_sharpe_common_dates = portfolio_rolling_sharpe.index.intersection(benchmark_rolling_sharpe.index)
        rolling_sharpe_data = []
        for date in rolling_sharpe_common_dates:
            port_sharpe = portfolio_rolling_sharpe[date]
            bench_sharpe = benchmark_rolling_sharpe[date]
            # Only include points where both values are valid (not NaN)
            if (port_sharpe is not None and not pd.isna(port_sharpe) and 
                bench_sharpe is not None and not pd.isna(bench_sharpe)):
                try:
                    rolling_sharpe_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'portfolio': safe_float(port_sharpe),
                        'benchmark': safe_float(bench_sharpe)
                    })
                except (ValueError, TypeError):
                    # Skip invalid values
                    continue
        
        # Rolling Volatility (6-month window, 126 trading days) using windowed prices
        # Use the windowed prices which already have the effective window applied
        portfolio_rolling_vol = compute_rolling_volatility(prices_portfolio_windowed, weights=weights) if not prices_portfolio_windowed.empty else pd.Series(dtype=float)
        benchmark_rolling_vol = compute_rolling_volatility(prices_benchmark_windowed, weights=None) if not prices_benchmark_windowed.empty else pd.Series(dtype=float)
        
        # Verify rolling volatility series end dates match common_end_date (if sufficient data exists)
        # Note: Rolling series may end earlier if insufficient data for full window
        # Both portfolio and benchmark now use the same common_end_date, ensuring consistent start dates
        if len(portfolio_rolling_vol) > 0 and len(benchmark_rolling_vol) > 0:
            portfolio_vol_end = portfolio_rolling_vol.index.max()
            benchmark_vol_end = benchmark_rolling_vol.index.max()
            # Both should end at the same date (or earlier if insufficient data for full window)
            # This is expected if there's insufficient data for the rolling window
            # The series ends earlier because dropna() removes periods before full window
            pass  # Log if needed: rolling series may end earlier due to min_periods requirement
        
        # Align rolling volatility series on common dates (only non-null values)
        rolling_vol_common_dates = portfolio_rolling_vol.index.intersection(benchmark_rolling_vol.index)
        rolling_volatility_data = []
        for date in rolling_vol_common_dates:
            port_vol = portfolio_rolling_vol[date]
            bench_vol = benchmark_rolling_vol[date]
            # Only include points where both values are valid (not NaN)
            if (port_vol is not None and not pd.isna(port_vol) and 
                bench_vol is not None and not pd.isna(bench_vol)):
                try:
                    rolling_volatility_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'portfolio': safe_float(port_vol * 100),  # Convert to percentage
                        'benchmark': safe_float(bench_vol * 100)  # Convert to percentage
                    })
                except (ValueError, TypeError):
                    # Skip invalid values
                    continue
        
        # Rolling Beta (6-month window, 126 trading days) vs SPY using windowed prices
        rolling_beta_series = compute_rolling_beta(prices_portfolio_windowed, prices_benchmark_windowed, portfolio_weights=weights, window_days=126)
        
        # Verify rolling beta series end date matches as_of_date (if sufficient data exists)
        # Note: Rolling series may end earlier if insufficient data for full window
        if len(rolling_beta_series) > 0:
            rolling_beta_end = rolling_beta_series.index.max()
            if rolling_beta_end != as_of_date:
                # This is expected if there's insufficient data for the rolling window
                # The series ends earlier because dropna() removes periods before full window
                pass  # Log if needed: rolling series may end earlier due to min_periods requirement
        
        # Format rolling beta data for frontend (only non-null values)
        rolling_beta_data = []
        for date in rolling_beta_series.index:
            beta_value = rolling_beta_series[date]
            if beta_value is not None and not pd.isna(beta_value):
                try:
                    rolling_beta_data.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'beta': safe_float(beta_value)
                    })
                except (ValueError, TypeError):
                    # Skip invalid values
                    continue
        
        # YTD contributions to return using windowed prices
        ytd_contributions_list = []
        try:
            from portfolio_tool.analytics import is_cash_ticker
            ytd_contributions = compute_ytd_contribution(prices_portfolio_windowed, weights=weights)
            # Filter out cash tickers from visualization
            for ticker, contribution in ytd_contributions.items():
                if not pd.isna(contribution) and not is_cash_ticker(ticker):
                    ytd_contributions_list.append({
                        'ticker': ticker,
                        'contribution': safe_float(contribution * 100)  # Convert to percentage
                    })
        except (ValueError, Exception) as e:
            # If no YTD data is available, return empty list
            # This can happen if the current year just started or if there's insufficient data
            ytd_contributions_list = []
        
        # YTD risk contributions (percentage of total portfolio risk) using windowed prices
        ytd_risk_contributions_list = []
        try:
            from portfolio_tool.analytics import is_cash_ticker
            ytd_risk_contributions = compute_ytd_risk_contribution(prices_portfolio_windowed, weights=weights)
            # Filter out cash tickers from visualization
            for ticker, contribution in ytd_risk_contributions.items():
                if not pd.isna(contribution) and not is_cash_ticker(ticker):
                    ytd_risk_contributions_list.append({
                        'ticker': ticker,
                        'contribution': safe_float(contribution * 100)  # Convert to percentage
                    })
            # Renormalize to sum to 100% across non-cash holdings
            if ytd_risk_contributions_list:
                total = sum(c['contribution'] for c in ytd_risk_contributions_list)
                if total > 0:
                    for contrib in ytd_risk_contributions_list:
                        contrib['contribution'] = contrib['contribution'] / total * 100  # Convert to percentage
        except (ValueError, Exception) as e:
            # If no YTD data is available, return empty list
            ytd_risk_contributions_list = []
        
        # Monthly returns heatmap using windowed prices
        # years_back parameter is used for display filtering, but data comes from windowed prices
        monthly_portfolio = compute_monthly_portfolio_returns(
            prices_portfolio_windowed, weights=weights, years_back=5
        )
        
        # Format monthly returns heatmap
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        years_list = sorted(monthly_portfolio.index.tolist(), reverse=True)
        
        heatmap_values = []
        for year in years_list:
            for month_idx, month_name in enumerate(months, start=1):
                if month_name in monthly_portfolio.columns:
                    return_val = monthly_portfolio.loc[year, month_name]
                    if pd.notna(return_val):
                        heatmap_values.append({
                            'year': int(year),
                            'month': month_idx,
                            'return': safe_float(return_val)
                        })
        
        # Correlation matrix using windowed prices
        # years parameter is ignored since we're using windowed prices, but kept for function signature
        correlation_matrix = compute_correlation_matrix(prices_portfolio_windowed, years=None)
        
        # Risk-return scatter
        benchmark_tickers = ['SPY', 'QQQ', 'AGG', 'ACWI']
        all_benchmark_tickers = [benchmark_ticker] + [t for t in benchmark_tickers if t != benchmark_ticker]
        # Fetch benchmark data using the same fetch range as portfolio
        benchmark_prices_all = get_price_history(all_benchmark_tickers, fetch_start_date, fetch_end_date)
        
        risk_return_scatter = []
        
        # Portfolio using windowed daily returns
        # Return as PERCENTAGES (46.0 for 46%) - frontend expects percentages
        port_ret, port_vol = compute_annualized_return_and_volatility_from_window(
            portfolio_daily_windowed, period_years
        )
        if port_ret is not None and port_vol is not None:
            risk_return_scatter.append({
                'label': 'Portfolio',
                'return': safe_float(port_ret * 100),  # Convert to percentage
                'risk': safe_float(port_vol * 100)  # Convert to percentage
            })
        
        # SPY using windowed daily returns
        spy_ret, spy_vol = compute_annualized_return_and_volatility_from_window(
            benchmark_daily_windowed, period_years
        )
        if spy_ret is not None and spy_vol is not None:
            risk_return_scatter.append({
                'label': 'SPY',
                'return': safe_float(spy_ret * 100),  # Convert to percentage
                'risk': safe_float(spy_vol * 100)  # Convert to percentage
            })
        
        # Other benchmarks (exclude SPY since it's already added above)
        for ticker in benchmark_tickers:
            if ticker == benchmark_ticker:  # Skip SPY - already added above
                continue
            if ticker in benchmark_prices_all.columns:
                ticker_prices = benchmark_prices_all[[ticker]]
                # Slice to effective window for consistency with portfolio and SPY
                ticker_prices_windowed = slice_to_effective_window(ticker_prices, effective_start_date, as_of_date)
                # Compute daily returns for this ticker
                ticker_daily = compute_daily_returns(ticker_prices_windowed, weights=None)
                ticker_ret, ticker_vol = compute_annualized_return_and_volatility_from_window(
                    ticker_daily, period_years
                )
                if ticker_ret is not None and ticker_vol is not None:
                    risk_return_scatter.append({
                        'label': ticker,
                        'return': safe_float(ticker_ret * 100),  # Convert to percentage
                        'risk': safe_float(ticker_vol * 100)  # Convert to percentage
                    })
        
        # Efficient frontier using windowed prices
        # Note: compute_efficient_frontier_analysis may do internal lookbacks
        # For now, pass period_years as an integer for compatibility, but the function should use windowed data
        # TODO: Update compute_efficient_frontier_analysis to not do internal lookbacks if needed
        benchmark_prices_dict = {benchmark_ticker: prices_benchmark_windowed}
        for ticker in benchmark_tickers:
            if ticker in benchmark_prices_all.columns:
                ticker_prices = benchmark_prices_all[[ticker]]
                ticker_prices_windowed = slice_to_effective_window(ticker_prices, effective_start_date, as_of_date)
                benchmark_prices_dict[ticker] = ticker_prices_windowed
        
        # Use integer years for compatibility, but the function should work with windowed data
        ef_years_int = int(np.ceil(period_years)) if period_years and period_years > 0 else 5
        ef_data = compute_efficient_frontier_analysis(
            portfolio_prices=prices_portfolio_windowed,
            portfolio_weights=weights,
            benchmark_prices_dict=benchmark_prices_dict,
            years=ef_years_int
        )
        
        # Format efficient frontier
        efficient_frontier = {
            'points': [],
            'current': {'risk': 0.0, 'return': 0.0},
            'maxSharpe': {'risk': 0.0, 'return': 0.0},
            'minVariance': {'risk': 0.0, 'return': 0.0}
        }
        
        if ef_data and ef_data.get('frontier') is not None:
            frontier_df = ef_data['frontier']
            for _, row in frontier_df.iterrows():
                efficient_frontier['points'].append({
                    'risk': safe_float(row['vol']),
                    'return': safe_float(row['ret'])
                })
            
            # Current portfolio point - use computed portfolio metrics
            # Return as PERCENTAGES (46.0 for 46%) - frontend expects percentages
            if port_ret is not None and port_vol is not None:
                efficient_frontier['current'] = {
                    'risk': safe_float(port_vol * 100),  # Convert to percentage
                    'return': safe_float(port_ret * 100)  # Convert to percentage
                }
            elif ef_data.get('portfolio_point'):
                portfolio_point = ef_data['portfolio_point']
                if isinstance(portfolio_point, dict):
                    efficient_frontier['current'] = {
                        'risk': safe_float(portfolio_point.get('vol', 0.0)),
                        'return': safe_float(portfolio_point.get('ret', 0.0))
                    }
            
            # Max Sharpe (tangency portfolio)
            if ef_data.get('tangency'):
                tangency = ef_data['tangency']
                efficient_frontier['maxSharpe'] = {
                    'risk': safe_float(tangency.get('vol', 0.0)),
                    'return': safe_float(tangency.get('ret', 0.0))
                }
            
            # Min variance - find point with lowest risk in frontier
            if efficient_frontier['points']:
                min_var_point = min(efficient_frontier['points'], key=lambda x: x['risk'])
                efficient_frontier['minVariance'] = min_var_point
        
        # Asset Breakdown (replaces simple holdings)
        # Compute asset-level metrics using windowed prices
        asset_breakdown = compute_asset_breakdown(prices_portfolio_windowed, weights)
        
        # Format for API response - return as PERCENTAGES (46.0 for 46%) - frontend expects percentages
        holdings = []
        for asset in asset_breakdown:
            holdings.append({
                'ticker': asset['ticker'],
                'weight': safe_float(asset['weight'] * 100),  # Convert to percentage
                'cagr': safe_float(asset['cagr'] * 100) if asset['cagr'] is not None else None,  # Convert to percentage
                'volatility': safe_float(asset['volatility'] * 100) if asset['volatility'] is not None else None,  # Convert to percentage
                'bestDay': safe_float(asset['bestDay'] * 100) if asset['bestDay'] is not None else None,  # Convert to percentage
                'worstDay': safe_float(asset['worstDay'] * 100) if asset['worstDay'] is not None else None  # Convert to percentage
            })
        
        # Warnings
        warnings = []
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.05:  # 5% tolerance
            warnings.append(f'Portfolio weights sum to {total_weight*100:.1f}%, expected 100%')
        
        # ===== AUDIT LOG =====
        # Count number of daily return rows (non-NaN)
        n_obs_portfolio = len(portfolio_daily_windowed.dropna()) if not portfolio_daily_windowed.empty else 0
        n_obs_benchmark = len(benchmark_daily_windowed.dropna()) if not benchmark_daily_windowed.empty else 0
        n_obs = max(n_obs_portfolio, n_obs_benchmark)
        
        audit_log = {
            'start_date_used': effective_start_date.strftime('%Y-%m-%d') if effective_start_date is not None else None,
            'end_date_used': as_of_date.strftime('%Y-%m-%d') if as_of_date is not None else None,
            'period_days': period_days,
            'period_years': safe_float(period_years) if period_years else None,
            'number_of_daily_return_rows': n_obs,
            'user_selected_start_date': request.requested_start_date if request.requested_start_date else None,
            'common_start_date': common_start_date.strftime('%Y-%m-%d') if common_start_date is not None else None,
            'limiting_ticker': limiting_ticker,
            'limiting_start_date': limiting_start_date.strftime('%Y-%m-%d') if limiting_start_date is not None else None,
        }
        
        # Initialize consistency warnings list for checks
        consistency_warnings = []
        
        # Check: cumulative return from Summary should equal (last value of growth chart / 1000 - 1)
        if len(portfolio_cum_windowed) > 0 and len(growth_of_100) > 0:
            growth_chart_final = growth_of_100[-1]['portfolio'] if growth_of_100 else None
            if growth_chart_final is not None:
                growth_chart_return = ((growth_chart_final / 1000.0) - 1) * 100  # Convert to percentage
                if abs(growth_chart_return - cumulative_return_portfolio) > 0.01:  # 0.01% tolerance
                    consistency_warnings.append(
                        f"INCONSISTENCY: Cumulative return ({cumulative_return_portfolio:.2f}%) != "
                        f"Growth chart final value ({growth_chart_return:.2f}%)"
                    )
        
        # Check: max drawdown stat should equal min drawdown from drawdown chart
        if len(portfolio_drawdown) > 0 and risk_metrics['maxDrawdown'] is not None:
            drawdown_chart_min = min([d['portfolio'] for d in drawdown_data]) if drawdown_data else None
            if drawdown_chart_min is not None:
                # Both values are percentages, compare directly
                max_dd_stat = risk_metrics['maxDrawdown']  # Already percentage (46.0 for 46%)
                # Both values are percentages, compare directly
                if abs(drawdown_chart_min - max_dd_stat) > 0.01:  # 0.01% tolerance
                    consistency_warnings.append(
                        f"INCONSISTENCY: Max drawdown stat ({max_dd_stat:.2f}%) != "
                        f"Min drawdown from chart ({drawdown_chart_min:.2f}%)"
                    )
        
        # Check: YTD contribution bars should sum to YTD portfolio return
        if len(ytd_contributions_list) > 0 and period_returns_portfolio.get('YTD') is not None:
            ytd_contrib_sum = sum([c['contribution'] for c in ytd_contributions_list])
            ytd_return_pct = period_returns_portfolio['YTD'] * 100  # Convert to percentage
            # Both values are percentages, compare directly
            if abs(ytd_contrib_sum - ytd_return_pct) > 0.1:  # 0.1% tolerance
                    consistency_warnings.append(
                        f"INCONSISTENCY: YTD contributions sum ({ytd_contrib_sum:.2f}%) != "
                        f"YTD portfolio return ({ytd_return_pct:.2f}%)"
                    )
        
        # Check: Asset breakdown uses same window as portfolio
        # Verify that asset breakdown is computed from the same windowed prices
        if len(asset_breakdown) > 0 and len(prices_portfolio_windowed) > 0:
            breakdown_window_start = prices_portfolio_windowed.index[0].strftime('%Y-%m-%d')
            breakdown_window_end = prices_portfolio_windowed.index[-1].strftime('%Y-%m-%d')
            expected_start = effective_start_date.strftime('%Y-%m-%d') if effective_start_date else None
            expected_end = as_of_date.strftime('%Y-%m-%d') if isinstance(as_of_date, pd.Timestamp) else pd.Timestamp(as_of_date).strftime('%Y-%m-%d')
            
            if expected_start and breakdown_window_start != expected_start:
                consistency_warnings.append(
                    f"INCONSISTENCY: Asset breakdown window start ({breakdown_window_start}) != "
                    f"Effective start date ({expected_start})"
                )
            if expected_end and breakdown_window_end != expected_end:
                consistency_warnings.append(
                    f"INCONSISTENCY: Asset breakdown window end ({breakdown_window_end}) != "
                    f"As-of date ({expected_end})"
                )
        
        # Add consistency warnings to audit log
        if consistency_warnings:
            audit_log['consistency_warnings'] = consistency_warnings
            # Also add to main warnings for visibility
            warnings.extend(consistency_warnings)
        
        # ===== END CONSISTENCY CHECKS =====
        
        # Calculate window length
        window_length_days = period_days
        window_length_years = safe_float(period_years) if period_years else None
        
        # Build response matching PortfolioDashboardResponse
        response_data = {
            'meta': {
                'analysisDate': as_of_date.isoformat() if isinstance(as_of_date, pd.Timestamp) else pd.Timestamp(as_of_date).isoformat(),
                'benchmarkTicker': benchmark_ticker,
                'riskFreeRate': safe_float(risk_free_rate),
                'lookbackYears': 5,  # Kept for backward compatibility
                'effectiveStartDate': effective_start_date.strftime('%Y-%m-%d') if effective_start_date is not None else None
            },
            'analysisWindow': {
                'requestedStartDate': request.requested_start_date if request.requested_start_date else None,
                'effectiveStartDate': effective_start_date.strftime('%Y-%m-%d') if effective_start_date is not None else None,
                'asOfDate': as_of_date.strftime('%Y-%m-%d') if as_of_date is not None else None,
                'windowLengthDays': window_length_days,
                'windowLengthYears': window_length_years,
                'limitingTicker': limiting_ticker,
                'limitingTickerStartDate': limiting_start_date.strftime('%Y-%m-%d') if limiting_start_date is not None else None,
            },
            'periodReturns': {
                'portfolio': period_returns_portfolio,
                'benchmark': period_returns_benchmark,
                'annualized': True
            },
            'returnsTable': returns_table,
            'riskMetrics': risk_metrics,
            'benchmarkRiskMetrics': benchmark_risk_metrics,
            'performanceMetrics5Y': performance_metrics_5y,
            'charts': {
                'growthOf100': growth_of_100,
                'drawdown': drawdown_data,
                'rollingSharpe': rolling_sharpe_data,
                'rollingVolatility': rolling_volatility_data,
                'rollingBeta': rolling_beta_data,
                'ytdContributions': ytd_contributions_list,
                'ytdRiskContributions': ytd_risk_contributions_list,
                'monthlyReturnsHeatmap': {
                    'years': years_list,
                    'months': months,
                    'values': heatmap_values
                },
                'correlationMatrix': {
                    'tickers': correlation_matrix.columns.tolist(),
                    'matrix': [[safe_float(val) for val in row] for row in correlation_matrix.values.tolist()]
                },
                'riskReturnScatter': risk_return_scatter,
                'efficientFrontier': efficient_frontier
            },
            'holdings': holdings,
            'warnings': warnings,
            'auditLog': audit_log  # Temporary audit log for debugging consistency
        }
        
        return {
            'success': True,
            'data': response_data
        }
        
    except Exception as e:
        import traceback
        error_detail = str(e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Portfolio analysis error: {error_detail}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "portfolio-analysis-api"}


if __name__ == "__main__":
    import uvicorn
    # Note: For auto-reload during development, use from command line:
    # uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
    # Running directly with reload=False avoids the warning
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=False)

