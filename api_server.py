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

from portfolio_tool.data_io import load_portfolio
from portfolio_tool.market_data import get_price_history, get_sector_info, get_risk_free_rate
from portfolio_tool.analytics import (
    compute_returns,
    compute_period_returns,
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
        start_date = (today - timedelta(days=365 * 6)).strftime('%Y-%m-%d')  # 6 years back
        # Use tomorrow as end_date to ensure we get data through today (yfinance end parameter is exclusive)
        end_date = (today + timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Fetch real market data (only for invested tickers, not cash)
        # If no invested tickers, we still need benchmark data
        if invested_tickers:
            all_tickers = invested_tickers + [benchmark_ticker]
            prices_all = get_price_history(all_tickers, start_date, end_date)
            # Create prices_portfolio with only invested tickers (cash will be handled separately in analytics)
            prices_portfolio = prices_all[invested_tickers]
            prices_benchmark = prices_all[[benchmark_ticker]]
        else:
            # Portfolio is 100% cash - only fetch benchmark
            prices_all = get_price_history([benchmark_ticker], start_date, end_date)
            prices_portfolio = pd.DataFrame()  # Empty DataFrame for 100% cash portfolio
            prices_benchmark = prices_all[[benchmark_ticker]]
        
        # Get the actual last date from the data (accounts for weekends, holidays, and yfinance delays)
        # This ensures the displayed end date matches the actual last date in the dataset
        if not prices_portfolio.empty:
            actual_end_date = get_as_of_date(prices_portfolio.index)
        elif not prices_benchmark.empty:
            actual_end_date = get_as_of_date(prices_benchmark.index)
        else:
            actual_end_date = pd.Timestamp(today)
        
        # Determine effective start date: latest first available date across all securities
        # This makes the "latest common start date" behavior explicit rather than relying
        # on implicit dropna() behavior in return calculations
        if not prices_portfolio.empty:
            # Combine portfolio and benchmark prices to find effective start date
            # (benchmark should be included to ensure alignment)
            combined_prices = pd.concat([prices_portfolio, prices_benchmark], axis=1)
            effective_start_date = get_effective_start_date(combined_prices, default_months_back=60)
        elif not prices_benchmark.empty:
            # Portfolio is 100% cash, only use benchmark for effective start date
            effective_start_date = get_effective_start_date(prices_benchmark, default_months_back=60)
        else:
            effective_start_date = pd.Timestamp(start_date)
        
        # Filter prices to effective_start_date:end_date for consistent calculations
        # This ensures all metrics use the same date range
        if not prices_portfolio.empty and effective_start_date is not None:
            prices_portfolio = prices_portfolio[prices_portfolio.index >= effective_start_date]
        if not prices_benchmark.empty and effective_start_date is not None:
            prices_benchmark = prices_benchmark[prices_benchmark.index >= effective_start_date]
        
        # Slice prices to effective window (effective_start_date to actual_end_date)
        # This is the SINGLE source of truth for all metrics
        prices_portfolio_windowed = slice_to_effective_window(prices_portfolio, effective_start_date, actual_end_date)
        prices_benchmark_windowed = slice_to_effective_window(prices_benchmark, effective_start_date, actual_end_date)
        
        # Calculate actual period length in years for proper annualization
        if not prices_portfolio_windowed.empty:
            actual_period_years = (actual_end_date - effective_start_date).days / 365.25
        elif not prices_benchmark_windowed.empty:
            actual_period_years = (actual_end_date - effective_start_date).days / 365.25
        else:
            actual_period_years = None
        
        # Round up to nearest integer for DateOffset (which requires integers)
        # This ensures all windowed data is included when functions do lookback
        # The actual float value is still used for precise annualization calculations where possible
        actual_period_years_int = int(np.ceil(actual_period_years)) if actual_period_years else None
        
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
        # Use actual_end_date.year for consistency (matches the actual data range)
        current_year = actual_end_date.year if isinstance(actual_end_date, pd.Timestamp) else pd.Timestamp(actual_end_date).year
        
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
        
        # Compute period returns (1M, 3M, YTD, 1Y, 3Y, 5Y) using windowed prices
        portfolio_period = compute_period_returns(prices_portfolio_windowed, weights=weights)
        benchmark_period = compute_period_returns(prices_benchmark_windowed, weights=None)
        
        # Map period returns - preserve None values for periods with insufficient history
        # None values indicate that the portfolio doesn't have enough history for that period
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
        
        # Compute risk metrics using the FULL effective window (not lookback periods)
        # All risk metrics now use the same window: effective_start_date to actual_end_date
        # Calculate actual period length for proper annualization
        risk_metrics = {
            'annualVolatility': 0.0,
            'sharpeRatio': None,
            'sortinoRatio': None,
            'calmarRatio': None,
            'beta': 1.0,
            'maxDrawdown': 0.0,
        }
        
        # Portfolio risk metrics using full effective window
        # Use integer years for DateOffset compatibility, but functions will use all windowed data
        if actual_period_years_int and actual_period_years_int > 0 and len(portfolio_daily_windowed) >= 20:
            # Use windowed series - functions do lookback from end, which will include all windowed data
            vol = compute_volatility(portfolio_daily_windowed, years=actual_period_years_int)
            risk_metrics['annualVolatility'] = safe_float(vol) if vol is not None else 0.0
        
        if actual_period_years_int and actual_period_years_int > 0 and len(portfolio_daily_windowed) >= MIN_OBS_SHARPE:
            sharpe = compute_sharpe_ratio(portfolio_daily_windowed, years=actual_period_years_int, risk_free_rate=risk_free_rate)
            risk_metrics['sharpeRatio'] = safe_float_or_none(sharpe)
        
        if actual_period_years_int and actual_period_years_int > 0 and len(portfolio_daily_windowed) >= MIN_OBS_SORTINO:
            sortino = compute_sortino_ratio(portfolio_daily_windowed, years=actual_period_years_int, risk_free_rate=risk_free_rate)
            risk_metrics['sortinoRatio'] = safe_float_or_none(sortino)
        
        if len(portfolio_daily_windowed) >= 20 and len(benchmark_daily_windowed) >= 20:
            beta = compute_beta(portfolio_daily_windowed, benchmark_daily_windowed, years=actual_period_years_int if actual_period_years_int else 1)
            risk_metrics['beta'] = safe_float(beta, default=1.0) if beta is not None else 1.0
        
        if len(portfolio_cum_windowed) >= 20:
            max_dd = compute_max_drawdown(portfolio_cum_windowed, years=actual_period_years_int if actual_period_years_int else 1)
            risk_metrics['maxDrawdown'] = safe_float(max_dd) if max_dd is not None else 0.0
        
        # Calculate Calmar Ratio: CAGR / |Max Drawdown| using full effective window
        # Find the longest available CAGR period that matches our window
        calmar_ratio = None
        if actual_period_years and actual_period_years >= 1.0:
            # Determine which CAGR period to use (prefer longest available that matches our window)
            cagr_value = None
            if period_returns_portfolio.get('5Y') is not None and actual_period_years >= 4.75:
                cagr_value = period_returns_portfolio['5Y']
            elif period_returns_portfolio.get('3Y') is not None and actual_period_years >= 2.85:
                cagr_value = period_returns_portfolio['3Y']
            elif period_returns_portfolio.get('1Y') is not None and actual_period_years >= 0.95:
                cagr_value = period_returns_portfolio['1Y']
            
            if cagr_value is not None and len(portfolio_daily_windowed) >= MIN_OBS_CALMAR:
                max_dd_abs = abs(risk_metrics['maxDrawdown']) if risk_metrics['maxDrawdown'] is not None else 0
                if max_dd_abs > 0 and not pd.isna(max_dd_abs):
                    calmar_ratio = safe_float(cagr_value / max_dd_abs)
        risk_metrics['calmarRatio'] = calmar_ratio
        
        # Benchmark risk metrics using full effective window
        benchmark_risk_metrics = {
            'annualVolatility': 0.0,
            'sharpeRatio': None,
            'sortinoRatio': None,
            'calmarRatio': None,
            'maxDrawdown': 0.0,
            'beta': 1.0,  # Benchmark beta is always 1.0 (beta against itself)
        }
        
        if actual_period_years_int and actual_period_years_int > 0 and len(benchmark_daily_windowed) >= 20:
            bench_vol = compute_volatility(benchmark_daily_windowed, years=actual_period_years_int)
            benchmark_risk_metrics['annualVolatility'] = safe_float(bench_vol) if bench_vol is not None else 0.0
        
        if actual_period_years_int and actual_period_years_int > 0 and len(benchmark_daily_windowed) >= MIN_OBS_SHARPE:
            bench_sharpe = compute_sharpe_ratio(benchmark_daily_windowed, years=actual_period_years_int, risk_free_rate=risk_free_rate)
            benchmark_risk_metrics['sharpeRatio'] = safe_float_or_none(bench_sharpe)
        
        if actual_period_years_int and actual_period_years_int > 0 and len(benchmark_daily_windowed) >= MIN_OBS_SORTINO:
            bench_sortino = compute_sortino_ratio(benchmark_daily_windowed, years=actual_period_years_int, risk_free_rate=risk_free_rate)
            benchmark_risk_metrics['sortinoRatio'] = safe_float_or_none(bench_sortino)
        
        if len(benchmark_cum_windowed) >= 20:
            bench_max_dd = compute_max_drawdown(benchmark_cum_windowed, years=actual_period_years_int if actual_period_years_int else 1)
            benchmark_risk_metrics['maxDrawdown'] = safe_float(bench_max_dd) if bench_max_dd is not None else 0.0
        
        # Benchmark Calmar Ratio using full effective window
        bench_calmar_ratio = None
        if actual_period_years and actual_period_years >= 1.0:
            # Determine which CAGR period to use (prefer longest available that matches our window)
            bench_cagr_value = None
            if period_returns_benchmark.get('5Y') is not None and actual_period_years >= 4.75:
                bench_cagr_value = period_returns_benchmark['5Y']
            elif period_returns_benchmark.get('3Y') is not None and actual_period_years >= 2.85:
                bench_cagr_value = period_returns_benchmark['3Y']
            elif period_returns_benchmark.get('1Y') is not None and actual_period_years >= 0.95:
                bench_cagr_value = period_returns_benchmark['1Y']
            
            if bench_cagr_value is not None and len(benchmark_daily_windowed) >= MIN_OBS_CALMAR:
                bench_max_dd_abs = abs(benchmark_risk_metrics['maxDrawdown']) if benchmark_risk_metrics['maxDrawdown'] is not None else 0
                if bench_max_dd_abs > 0 and not pd.isna(bench_max_dd_abs):
                    bench_calmar_ratio = safe_float(bench_cagr_value / bench_max_dd_abs)
        benchmark_risk_metrics['calmarRatio'] = bench_calmar_ratio
        
        # Cumulative returns (growth of $1,000) - using windowed series
        # portfolio_cum_windowed and benchmark_cum_windowed already computed above
        
        # Calculate cumulative return using the full effective window
        # This matches the window used for all other metrics
        cumulative_return_portfolio = 0.0
        cumulative_return_benchmark = 0.0
        
        if len(portfolio_cum_windowed) > 0 and len(benchmark_cum_windowed) > 0:
            # Get starting and ending values for the effective window
            portfolio_start = portfolio_cum_windowed.iloc[0]
            portfolio_end = portfolio_cum_windowed.iloc[-1]
            benchmark_start = benchmark_cum_windowed.iloc[0]
            benchmark_end = benchmark_cum_windowed.iloc[-1]
            
            # Calculate cumulative return: (end / start - 1) * 100
            if portfolio_start > 0:
                cumulative_return_portfolio = safe_float(((portfolio_end / portfolio_start) - 1) * 100)
            if benchmark_start > 0:
                cumulative_return_benchmark = safe_float(((benchmark_end / benchmark_start) - 1) * 100)
        
        # For backward compatibility, also calculate 5Y cumulative return if we have 5+ years
        # But use the windowed series, not a separate lookback
        cumulative_return_5y_portfolio = cumulative_return_portfolio
        cumulative_return_5y_benchmark = cumulative_return_benchmark
        if actual_period_years and actual_period_years >= 4.75:
            # If we have 5+ years, the cumulative return already represents the full period
            # For display purposes, we can still call it "5Y" if the period is close to 5 years
            pass
        
        # Extract 5Y metrics for Performance Metrics table (using windowed series)
        # Note: These use the full effective window, not a fixed 5Y lookback
        performance_metrics_5y = {
            'cumulativeReturn5Y': cumulative_return_5y_portfolio,
            'cumulativeReturn5YBenchmark': cumulative_return_5y_benchmark,
            'cagr5Y': period_returns_portfolio.get('5Y', 0.0),
            'cagr5YBenchmark': period_returns_benchmark.get('5Y', 0.0),
            'maxDrawdown5Y': risk_metrics['maxDrawdown'],  # Use the windowed max drawdown
            'maxDrawdown5YBenchmark': benchmark_risk_metrics['maxDrawdown'],  # Use the windowed max drawdown
            'sharpeRatio5Y': risk_metrics['sharpeRatio'] if risk_metrics['sharpeRatio'] is not None else 0.0,
            'sharpeRatio5YBenchmark': benchmark_risk_metrics['sharpeRatio'] if benchmark_risk_metrics['sharpeRatio'] is not None else 0.0,
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
        
        # Verify rolling beta series end date matches actual_end_date (if sufficient data exists)
        # Note: Rolling series may end earlier if insufficient data for full window
        if len(rolling_beta_series) > 0:
            rolling_beta_end = rolling_beta_series.index.max()
            if rolling_beta_end != actual_end_date:
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
                        'contribution': safe_float(contribution * 100)  # Convert to percentage points
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
                        'contribution': safe_float(contribution * 100)  # Convert to percentage (already normalized)
                    })
            # Renormalize to sum to 100% across non-cash holdings
            if ytd_risk_contributions_list:
                total = sum(c['contribution'] for c in ytd_risk_contributions_list)
                if total > 0:
                    for contrib in ytd_risk_contributions_list:
                        contrib['contribution'] = contrib['contribution'] / total * 100
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
        benchmark_prices_all = get_price_history(all_benchmark_tickers, start_date, end_date)
        
        risk_return_scatter = []
        
        # Portfolio using windowed prices
        # years parameter uses integer for DateOffset compatibility
        port_ret, port_vol = compute_annualized_return_and_volatility(
            prices_portfolio_windowed, weights=weights, years=actual_period_years_int if actual_period_years_int else 5
        )
        if port_ret is not None and port_vol is not None:
            risk_return_scatter.append({
                'label': 'Portfolio',
                'return': safe_float(port_ret),
                'risk': safe_float(port_vol)
            })
        
        # SPY using windowed prices
        spy_ret, spy_vol = compute_annualized_return_and_volatility(
            prices_benchmark_windowed, weights=None, years=actual_period_years_int if actual_period_years_int else 5
        )
        if spy_ret is not None and spy_vol is not None:
            risk_return_scatter.append({
                'label': 'SPY',
                'return': safe_float(spy_ret),
                'risk': safe_float(spy_vol)
            })
        
        # Other benchmarks (exclude SPY since it's already added above)
        for ticker in benchmark_tickers:
            if ticker == benchmark_ticker:  # Skip SPY - already added above
                continue
            if ticker in benchmark_prices_all.columns:
                ticker_prices = benchmark_prices_all[[ticker]]
                # Slice to effective window for consistency with portfolio and SPY
                ticker_prices_windowed = slice_to_effective_window(ticker_prices, effective_start_date, actual_end_date)
                ticker_ret, ticker_vol = compute_annualized_return_and_volatility(
                    ticker_prices_windowed, weights=None, years=actual_period_years_int if actual_period_years_int else 5
                )
                if ticker_ret is not None and ticker_vol is not None:
                    risk_return_scatter.append({
                        'label': ticker,
                        'return': safe_float(ticker_ret),
                        'risk': safe_float(ticker_vol)
                    })
        
        # Efficient frontier using windowed prices
        benchmark_prices_dict = {benchmark_ticker: prices_benchmark_windowed}
        for ticker in benchmark_tickers:
            if ticker in benchmark_prices_all.columns:
                ticker_prices = benchmark_prices_all[[ticker]]
                ticker_prices_windowed = slice_to_effective_window(ticker_prices, effective_start_date, actual_end_date)
                benchmark_prices_dict[ticker] = ticker_prices_windowed
        
        ef_data = compute_efficient_frontier_analysis(
            portfolio_prices=prices_portfolio_windowed,
            portfolio_weights=weights,
            benchmark_prices_dict=benchmark_prices_dict,
            years=actual_period_years_int if actual_period_years_int else 5
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
            if port_ret is not None and port_vol is not None:
                efficient_frontier['current'] = {
                    'risk': safe_float(port_vol),
                    'return': safe_float(port_ret)
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
        
        # Holdings
        holdings = [
            {'ticker': ticker, 'weight': safe_float(weight)}
            for ticker, weight in weights.items()
        ]
        
        # Warnings
        warnings = []
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.05:  # 5% tolerance
            warnings.append(f'Portfolio weights sum to {total_weight*100:.1f}%, expected 100%')
        
        # ===== CONSISTENCY CHECKS AND AUDIT LOG =====
        audit_log = {
            'effective_start_date': effective_start_date.strftime('%Y-%m-%d') if effective_start_date is not None else None,
            'as_of_date': actual_end_date.strftime('%Y-%m-%d') if isinstance(actual_end_date, pd.Timestamp) else pd.Timestamp(actual_end_date).strftime('%Y-%m-%d'),
            'actual_period_years': safe_float(actual_period_years) if actual_period_years else None,
            'portfolio_daily_returns_rows': len(portfolio_daily_windowed),
            'benchmark_daily_returns_rows': len(benchmark_daily_windowed),
            'portfolio_cumulative_index_rows': len(portfolio_cum_windowed),
            'benchmark_cumulative_index_rows': len(benchmark_cum_windowed),
            'metrics': {}
        }
        
        # Track date ranges used by each metric/chart
        if len(portfolio_daily_windowed) > 0:
            audit_log['metrics']['portfolio_daily_returns'] = {
                'start': portfolio_daily_windowed.index[0].strftime('%Y-%m-%d'),
                'end': portfolio_daily_windowed.index[-1].strftime('%Y-%m-%d'),
                'rows': len(portfolio_daily_windowed)
            }
        
        if len(benchmark_daily_windowed) > 0:
            audit_log['metrics']['benchmark_daily_returns'] = {
                'start': benchmark_daily_windowed.index[0].strftime('%Y-%m-%d'),
                'end': benchmark_daily_windowed.index[-1].strftime('%Y-%m-%d'),
                'rows': len(benchmark_daily_windowed)
            }
        
        if len(portfolio_cum_windowed) > 0:
            audit_log['metrics']['growth_of_100_chart'] = {
                'start': portfolio_cum_windowed.index[0].strftime('%Y-%m-%d'),
                'end': portfolio_cum_windowed.index[-1].strftime('%Y-%m-%d'),
                'rows': len(portfolio_cum_windowed)
            }
        
        if len(portfolio_drawdown) > 0:
            audit_log['metrics']['drawdown_chart'] = {
                'start': portfolio_drawdown.index[0].strftime('%Y-%m-%d'),
                'end': portfolio_drawdown.index[-1].strftime('%Y-%m-%d'),
                'rows': len(portfolio_drawdown)
            }
        
        # Consistency assertions (warnings, not errors)
        consistency_warnings = []
        
        # Check: cumulative return from Summary should equal (last value of growth chart / 1000 - 1)
        if len(portfolio_cum_windowed) > 0 and len(growth_of_100) > 0:
            growth_chart_final = growth_of_100[-1]['portfolio'] if growth_of_100 else None
            if growth_chart_final is not None:
                growth_chart_return = ((growth_chart_final / 1000.0) - 1) * 100
                if abs(growth_chart_return - cumulative_return_portfolio) > 0.01:  # 0.01% tolerance
                    consistency_warnings.append(
                        f"INCONSISTENCY: Cumulative return ({cumulative_return_portfolio:.2f}%) != "
                        f"Growth chart final value ({growth_chart_return:.2f}%)"
                    )
        
        # Check: max drawdown stat should equal min drawdown from drawdown chart
        if len(portfolio_drawdown) > 0 and risk_metrics['maxDrawdown'] is not None:
            drawdown_chart_min = min([d['portfolio'] for d in drawdown_data]) if drawdown_data else None
            if drawdown_chart_min is not None:
                # drawdown_chart_min is already in percentage (multiplied by 100)
                max_dd_stat = risk_metrics['maxDrawdown'] * 100  # Convert to percentage
                if abs(drawdown_chart_min - max_dd_stat) > 0.01:  # 0.01% tolerance
                    consistency_warnings.append(
                        f"INCONSISTENCY: Max drawdown stat ({max_dd_stat:.2f}%) != "
                        f"Min drawdown from chart ({drawdown_chart_min:.2f}%)"
                    )
        
        # Check: YTD contribution bars should sum to YTD portfolio return
        if len(ytd_contributions_list) > 0 and period_returns_portfolio.get('YTD') is not None:
            ytd_contrib_sum = sum([c['contribution'] for c in ytd_contributions_list])
            ytd_return_pct = period_returns_portfolio['YTD'] * 100  # Convert to percentage
            if abs(ytd_contrib_sum - ytd_return_pct) > 0.1:  # 0.1% tolerance (accounting for rounding)
                consistency_warnings.append(
                    f"INCONSISTENCY: YTD contributions sum ({ytd_contrib_sum:.2f}%) != "
                    f"YTD portfolio return ({ytd_return_pct:.2f}%)"
                )
        
        # Add consistency warnings to audit log
        if consistency_warnings:
            audit_log['consistency_warnings'] = consistency_warnings
            # Also add to main warnings for visibility
            warnings.extend(consistency_warnings)
        
        # ===== END CONSISTENCY CHECKS =====
        
        # Build response matching PortfolioDashboardResponse
        # Use actual_end_date (last date in data) instead of today for consistency with displayed end date
        response_data = {
            'meta': {
                'analysisDate': actual_end_date.isoformat() if isinstance(actual_end_date, pd.Timestamp) else pd.Timestamp(actual_end_date).isoformat(),
                'benchmarkTicker': benchmark_ticker,
                'riskFreeRate': safe_float(risk_free_rate),
                'lookbackYears': 5,
                'effectiveStartDate': effective_start_date.strftime('%Y-%m-%d') if effective_start_date is not None else None
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

