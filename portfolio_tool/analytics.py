from datetime import datetime
import pandas as pd
import numpy as np
from scipy.optimize import minimize

# Cash detection helper - centralized for consistency
CASH_TICKERS = {'CASH', 'USD', 'CASH_USD', 'UNINVESTED', 'SWEEP', 'MONEY'}

# Minimum observations for risk metrics (in trading days)
TRADING_DAYS_PER_YEAR = 252
MIN_OBS_SHARPE = int(0.5 * TRADING_DAYS_PER_YEAR)  # 6 months ≈ 126 trading days
MIN_OBS_SORTINO = int(0.5 * TRADING_DAYS_PER_YEAR)  # 6 months ≈ 126 trading days
MIN_OBS_CALMAR = int(1.0 * TRADING_DAYS_PER_YEAR)  # 1 year ≈ 252 trading days

def is_cash_ticker(ticker):
    """
    Check if a ticker represents cash/uninvested funds.
    
    Args:
        ticker: ticker symbol (string)
    
    Returns:
        bool: True if ticker is cash, False otherwise
    """
    if ticker is None:
        return False
    return str(ticker).upper() in CASH_TICKERS

def get_cash_weight(weights):
    """
    Extract cash weight from portfolio weights.
    
    Args:
        weights: dict/Series of ticker -> weight
    
    Returns:
        float: total cash weight (0.0 to 1.0)
    """
    if weights is None:
        return 0.0
    
    w = pd.Series(weights) if isinstance(weights, dict) else pd.Series(weights)
    cash_weight = sum(w.get(ticker, 0.0) for ticker in w.index if is_cash_ticker(ticker))
    return float(cash_weight)

def filter_cash_from_weights(weights):
    """
    Filter out cash tickers from weights, returning only invested assets.
    
    Args:
        weights: dict/Series of ticker -> weight
    
    Returns:
        dict: weights without cash tickers (not normalized)
    """
    if weights is None:
        return {}
    
    w = pd.Series(weights) if isinstance(weights, dict) else pd.Series(weights)
    invested = {ticker: weight for ticker, weight in w.items() if not is_cash_ticker(ticker)}
    return invested

def get_as_of_date(index):
    """
    Get the as-of date from a DatetimeIndex (last available date in the dataset).
    
    Args:
        index: pandas DatetimeIndex or Index with datetime values
    
    Returns:
        pandas.Timestamp: maximum (last) date in the index
    """
    return index.max()

def get_effective_start_date(prices, default_months_back=60):
    """
    Determine the effective portfolio start date as the maximum of:
    1. The latest first available date across all ticker columns
    2. The default start date (last_date - default_months_back months)
    
    When securities have different start dates (e.g., IPOs), the portfolio can only
    be constructed from the date when all securities have data available. This function
    makes that behavior explicit rather than relying on implicit dropna() behavior.
    
    Args:
        prices: DataFrame with DatetimeIndex (dates) and columns (tickers)
        default_months_back: Number of months to look back from the last date (default 60)
    
    Returns:
        pandas.Timestamp: Effective start date for portfolio calculations
    """
    if prices.empty:
        return None
    
    # Get the last date in the dataset
    last_date = prices.index.max()
    default_start = last_date - pd.DateOffset(months=default_months_back)
    
    # Find first valid (non-NaN) date for each ticker column
    first_valid_dates = []
    for col in prices.columns:
        first_valid = prices[col].first_valid_index()
        if first_valid is not None:
            first_valid_dates.append(first_valid)
    
    if not first_valid_dates:
        # If no valid dates found, use default
        return default_start
    
    # Latest first date across all securities
    latest_first_date = max(first_valid_dates)
    
    # Effective start is the maximum of latest_first_date and default_start
    # This ensures we use at least the default lookback period if all securities
    # have sufficient history, but use the latest_first_date if any security
    # has shorter history
    effective_start = max(latest_first_date, default_start)
    
    return effective_start

def compute_returns(prices, weights=None, years_back=5):
    """
    Compute calendar-year & YTD returns.

    Includes cash as a zero-return asset if present in weights.

    prices: DataFrame of daily prices (index = dates, columns = tickers)
    weights: optional dict/Series of ticker -> weight for portfolio.
    years_back: number of calendar years to include (including current year as YTD).
    """
    prices = prices.sort_index()

    # Group directly by year using the index
    yearly_first = prices.groupby(prices.index.year).first()
    yearly_last  = prices.groupby(prices.index.year).last()

    # Yearly returns per ticker: (last / first) - 1
    ticker_yearly_returns = (yearly_last / yearly_first) - 1

    # Keep only the last N years (including current year)
    all_years = ticker_yearly_returns.index.to_list()
    if len(all_years) > years_back:
        years_to_keep = all_years[-years_back:]
        ticker_yearly_returns = ticker_yearly_returns.loc[years_to_keep]

    # Portfolio-level yearly returns if weights provided
    portfolio_yearly_returns = None
    if weights is not None:
        if isinstance(weights, dict):
            w = pd.Series(weights)
        else:
            w = pd.Series(weights)

        # Use raw weights directly - cash will contribute 0% automatically since cash has no price data
        # Portfolio return = sum(weight × return) for all assets
        # Cash contributes 0% since it's not in prices DataFrame
        w_aligned = w.reindex(ticker_yearly_returns.columns).fillna(0)
        portfolio_yearly_returns = ticker_yearly_returns.dot(w_aligned)

    # Rename current year (year of last date in dataset) to "YTD"
    as_of_date = get_as_of_date(prices.index)
    current_year = as_of_date.year
    if current_year in ticker_yearly_returns.index:
        ticker_yearly_returns = ticker_yearly_returns.rename(index={current_year: "YTD"})
        if portfolio_yearly_returns is not None:
            portfolio_yearly_returns = portfolio_yearly_returns.rename(index={current_year: "YTD"})

    return ticker_yearly_returns, portfolio_yearly_returns


def compute_period_returns(prices, weights=None):
    """
    Compute returns over various periods:
    1M, 3M, 6M, 1Y, YTD, 3Y (CAGR), 5Y (CAGR)
    
    All periods use daily rebalanced portfolio returns (constant weights) to ensure
    consistency across the platform. This matches the methodology used for CAGR,
    volatility, Sharpe ratio, beta, and drawdown calculations.
    
    For multi-year periods (3Y, 5Y), returns are annualized (CAGR).
    Multi-year periods are only computed if the portfolio has sufficient history
    (at least that many years from effective_start_date). If insufficient history,
    returns None for that period.
    
    For shorter periods (1M, 3M, 6M, YTD, 1Y), returns are total period returns.
    
    Note: Prices should already be filtered to effective_start_date before calling
    this function, so the available history length can be determined from the price
    data itself.
    """
    prices = prices.sort_index()
    
    # Compute daily returns once for all periods
    # Note: Prices should already be filtered to effective_start_date before calling this function,
    # so all securities should have data from that date forward. dropna() removes the first row
    # (NaN from pct_change()) and any rows with missing data (should be rare after filtering).
    daily_returns = prices.pct_change().dropna()
    
    if len(prices) == 0:
        return {}
    
    # Calculate available history length in years
    # Use the first and last dates in the prices DataFrame (which is already filtered to effective_start_date)
    # This gives us the actual date range available, accounting for the fact that dropna() removes the first row
    start_date = prices.index[0]
    end_date = prices.index[-1]
    history_years = (end_date - start_date).days / 365.25  # Account for leap years
    
    # Compute daily returns (needed for period calculations)
    daily_returns = prices.pct_change().dropna()
    
    if len(daily_returns) == 0:
        return {}
    
    periods = {
        '1M': 1,
        '3M': 3,
        '6M': 6,
        '1Y': 12,
        'YTD': None,
        '3Y': 36,
        '5Y': 60
    }

    results = {}

    for label, months in periods.items():
        # For multi-year periods (1Y, 3Y, 5Y), check if sufficient history exists
        # Use 95% threshold to account for weekends, holidays, and edge cases in date calculations
        if label in ['1Y', '3Y', '5Y']:
            if label == '1Y':
                required_years = 1.0
            else:
                required_years = int(label[0])
            if history_years < (required_years * 0.95):
                # Insufficient history - return None (do NOT compute/shown misleading returns)
                results[label] = None
                continue
        
        # Filter to the appropriate period
        if label == 'YTD':
            # YTD: filter from January 1 to as-of date (last date in dataset)
            as_of = get_as_of_date(daily_returns.index)
            start = pd.Timestamp(as_of.year, 1, 1)
            period_returns = daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= as_of)]
        elif label in ['3Y', '5Y']:
            # Multi-year periods: use years-based filtering
            years = int(label[0])
            cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)
            period_returns = daily_returns[daily_returns.index >= cutoff_date]
        else:
            # Shorter periods: use months-based filtering
            cutoff_date = daily_returns.index[-1] - pd.DateOffset(months=months)
            period_returns = daily_returns[daily_returns.index >= cutoff_date]
        
        # For 1Y specifically, require at least ~240 trading days (95% of 252 trading days per year)
        # This ensures we have sufficient data quality even if date range passes the initial check
        if label == '1Y' and len(period_returns) < 240:
            results[label] = None
            continue
        
        # For other periods, require at least 20 trading days as a minimum sanity check
        if len(period_returns) < 20:
            results[label] = None
            continue
        
        # Build portfolio returns if weights provided
        if weights is not None:
            w = pd.Series(weights).reindex(period_returns.columns).fillna(0)
            portfolio_daily = (period_returns * w).sum(axis=1)
            returns_series = portfolio_daily
        else:
            # Assume single-column DF for benchmark
            if period_returns.shape[1] == 1:
                returns_series = period_returns.iloc[:, 0]
            else:
                w = pd.Series(1 / period_returns.shape[1], index=period_returns.columns)
                returns_series = (period_returns * w).sum(axis=1)
        
        # Compound daily returns to get total return
        total_return = (1 + returns_series).prod() - 1
        
        # For multi-year periods (3Y, 5Y), compute annualized return (CAGR)
        if label in ['3Y', '5Y']:
            years = int(label[0])
            cagr = (1 + total_return) ** (1 / years) - 1
            results[label] = float(cagr)
        else:
            # For shorter periods, return total period return (not annualized)
            results[label] = float(total_return)

    return results


def compute_period_returns_vs_benchmark(prices, weights=None):
    """
    Compute fixed-horizon period returns (1M, 3M, YTD, 1Y, 3Y, 5Y) for Period Returns vs Benchmark table.
    
    This function is SEPARATE from the main dashboard logic. It uses the FULL available data
    (not windowed to user-selected analysis window) and computes fixed-horizon buckets.
    
    Key differences from compute_period_returns():
    - Uses FULL prices (not windowed to effective_start_date)
    - Independent of user-selected analysis window
    - Fixed-horizon buckets (1M, 3M, YTD, 1Y, 3Y, 5Y)
    - More strict insufficient data logic with calendar coverage checks
    
    Insufficient data requirements:
    - 1M/3M/YTD: require at least 20 trading days inside the bucket
    - 1Y: require at least ~240 trading days AND at least 0.95 years of calendar coverage
    - 3Y: require at least 2.85 years (95% of 3Y) AND sufficient trading days (>= 252*2.85)
    - 5Y: require at least 4.75 years (95% of 5Y) of calendar coverage
    
    Calendar coverage check: For each bucket [bucket_start, bucket_end], we check if we have
    sufficient data. The actual coverage is from max(bucket_start, bucket_start_available) to bucket_end.
    actual_years = (bucket_end - max(bucket_start, bucket_start_available)).days / 365.25
    This checks coverage within the bucket window itself, not the user-selected analysis window.
    
    Args:
        prices: DataFrame of daily prices (index = dates, columns = tickers)
                Should be FULL prices (not windowed to effective_start_date)
        weights: optional dict/Series of ticker -> weight for portfolio
    
    Returns:
        dict: {period: return_value or None} where None indicates insufficient data
              Returns are as decimals (0.05 = 5%)
              For 3Y and 5Y, returns are annualized (CAGR)
              For 1M, 3M, YTD, 1Y, returns are total period returns (not annualized)
    """
    prices = prices.sort_index()
    
    if len(prices) == 0:
        return {}
    
    # Compute daily returns
    daily_returns = prices.pct_change().dropna()
    
    if len(daily_returns) == 0:
        return {}
    
    # Get as_of_date (bucket_end) from the full dataset
    bucket_end = get_as_of_date(daily_returns.index)
    
    # Get the first available date in the dataset (bucket_start_available)
    bucket_start_available = daily_returns.index[0]
    
    periods = {
        '1M': 1,
        '3M': 3,
        'YTD': None,
        '1Y': 12,
        '3Y': 36,
        '5Y': 60
    }
    
    results = {}
    
    for label, months in periods.items():
        # Determine the bucket window [bucket_start, bucket_end]
        if label == 'YTD':
            bucket_start = pd.Timestamp(bucket_end.year, 1, 1)
        elif label in ['3Y', '5Y']:
            years = int(label[0])
            bucket_start = bucket_end - pd.DateOffset(years=years)
        else:
            bucket_start = bucket_end - pd.DateOffset(months=months)
        
        # Check calendar coverage: actual_years = (bucket_end - bucket_start_available).days / 365.25
        # We need bucket_start_available <= bucket_start for full coverage
        # The actual coverage is from max(bucket_start, bucket_start_available) to bucket_end
        actual_bucket_start = max(bucket_start, bucket_start_available)
        actual_years = (bucket_end - actual_bucket_start).days / 365.25
        
        # Filter returns to the bucket window
        period_returns = daily_returns.loc[
            (daily_returns.index >= actual_bucket_start) & 
            (daily_returns.index <= bucket_end)
        ]
        
        n_trading_days = len(period_returns)
        
        # Insufficient data checks
        if label in ['1M', '3M', 'YTD']:
            # Require at least 20 trading days
            if n_trading_days < 20:
                results[label] = None
                continue
        
        elif label == '1Y':
            # Require at least ~240 trading days AND at least 0.95 years of calendar coverage
            required_years = 0.95
            if n_trading_days < 240 or actual_years < required_years:
                results[label] = None
                continue
        
        elif label == '3Y':
            # Require at least 2.85 years (95% of 3Y) AND sufficient trading days (>= 252*2.85)
            required_years = 2.85
            required_trading_days = int(252 * 2.85)  # ~718 trading days
            if actual_years < required_years or n_trading_days < required_trading_days:
                results[label] = None
                continue
        
        elif label == '5Y':
            # Require at least 4.75 years (95% of 5Y) of calendar coverage
            required_years = 4.75
            if actual_years < required_years:
                results[label] = None
                continue
        
        # Build portfolio returns if weights provided
        if weights is not None:
            w = pd.Series(weights).reindex(period_returns.columns).fillna(0)
            portfolio_daily = (period_returns * w).sum(axis=1)
            returns_series = portfolio_daily
        else:
            # Assume single-column DF for benchmark
            if period_returns.shape[1] == 1:
                returns_series = period_returns.iloc[:, 0]
            else:
                w = pd.Series(1 / period_returns.shape[1], index=period_returns.columns)
                returns_series = (period_returns * w).sum(axis=1)
        
        # Compound daily returns to get total return
        total_return = (1 + returns_series).prod() - 1
        
        # For multi-year periods (3Y, 5Y), compute annualized return (CAGR)
        if label in ['3Y', '5Y']:
            years = int(label[0])
            cagr = (1 + total_return) ** (1 / years) - 1
            results[label] = float(cagr)
        else:
            # For shorter periods, return total period return (not annualized)
            results[label] = float(total_return)
    
    return results

# Adding cumulative returns plot
def compute_cumulative_index(prices, weights=None):
    """
    Compute cumulative growth of $1,000 invested over time.
    
    Includes cash as a zero-return asset if present in weights.
    """
    # Use compute_daily_returns which handles cash properly
    series_ret = compute_daily_returns(prices, weights=weights)

    # cumulative wealth index
    cumulative_index = (1 + series_ret).cumprod()
    cumulative_index = cumulative_index/cumulative_index.iloc[0] * 1000  # scale to $1,000 start

    return cumulative_index

# Adding main contributor to overall returns
from datetime import datetime

def compute_ytd_contribution(prices, weights):
    """
    Compute each asset's contribution to the portfolio's YTD return using daily rebalanced methodology.
    
    This matches the methodology used in compute_period_returns() for consistency.
    
    YTD is calculated from January 1 to the last available date in the dataset.
    
    Uses daily rebalanced portfolio returns: r_p,t = Σ(w_i × r_i,t), then compounds to get total return.
    Each asset's contribution accounts for the compounding effect of the portfolio.
    
    Includes cash as a zero-return asset (cash contribution = 0.0).
    Cash tickers will have 0.0 contribution but are included in the result.
    Filter cash from visualization in plotting functions.
    
    Returns: Series indexed by ticker, values = contribution to total YTD return as decimal
             (e.g., 0.05 = 5% contribution to portfolio return)
             Contributions sum to the total YTD portfolio return
    """
    prices = prices.sort_index()
    
    # Compute daily returns FIRST (matches compute_period_returns() exactly)
    # Note: Prices should already be filtered to effective_start_date before calling this function,
    # so all securities should have data from that date forward. dropna() removes the first row
    # (NaN from pct_change()) and any rows with missing data (should be rare after filtering).
    daily_returns = prices.pct_change().dropna()
    
    if len(daily_returns) == 0:
        raise ValueError("No daily returns data available.")
    
    # Filter for YTD data using daily_returns index (matches compute_period_returns() exactly)
    # Use get_as_of_date(daily_returns.index) to ensure consistency with compute_period_returns()
    as_of = get_as_of_date(daily_returns.index)
    start = pd.Timestamp(as_of.year, 1, 1)
    period_returns = daily_returns.loc[(daily_returns.index >= start) & (daily_returns.index <= as_of)]
    
    if len(period_returns) == 0:
        raise ValueError(f"No price data available for YTD period (year {as_of.year}).")
    
    if len(period_returns) < 20:
        raise ValueError("Insufficient YTD data for contribution calculation.")
    
    # Build portfolio daily returns using daily rebalanced methodology
    # This matches compute_period_returns() exactly
    w = pd.Series(weights)
    w_invested = w.reindex(period_returns.columns).fillna(0)
    portfolio_daily = (period_returns * w_invested).sum(axis=1)
    
    # Compute cumulative portfolio value (starting from 1.0)
    # This tracks how $1 invested at the start grows over time
    portfolio_cumulative = (1 + portfolio_daily).cumprod()
    portfolio_total_return = portfolio_cumulative.iloc[-1] - 1.0
    
    # Initialize contributions Series with all weights (including cash)
    contributions = pd.Series(0.0, index=w.index)
    
    # Compute each asset's contribution
    # For daily rebalanced, contribution accounts for compounding:
    # Contribution_i = Σ_t [w_i × r_i,t × (V_T / V_t)]
    # where V_t is cumulative portfolio value at END of day t, and V_T is final value
    # This ensures contributions sum exactly to the total return
    portfolio_cumulative_final = portfolio_cumulative.iloc[-1]
    
    # For each invested asset, compute its contribution
    for ticker in period_returns.columns:
        if ticker in w.index and w[ticker] != 0:
            # Asset's weighted daily returns
            asset_weighted_returns = w[ticker] * period_returns[ticker]
            
            # Scale by (V_T / V_t) to account for compounding
            # An asset's contribution on day t compounds by the portfolio return from day t+1 to T
            # V_t is cumulative at END of day t, so (V_T / V_t) is the compounding factor
            contribution = (asset_weighted_returns * (portfolio_cumulative_final / portfolio_cumulative)).sum()
            contributions[ticker] = contribution
    
    # Cash tickers contribute 0.0 (already initialized to 0.0)
    # Cash weight is included in the Series but with 0.0 contribution value
    
    # Verify contributions sum to total return (within floating point precision)
    contribution_sum = contributions.sum()
    if abs(contribution_sum - portfolio_total_return) > 1e-6:
        # Log diagnostic info if contributions don't sum correctly
        import warnings
        warnings.warn(
            f"YTD contributions do not sum to total return. "
            f"Sum: {contribution_sum:.10f}, Total: {portfolio_total_return:.10f}, "
            f"Diff: {abs(contribution_sum - portfolio_total_return):.10f}. "
            f"As_of: {as_of}, YTD start: {start}, Rows: {len(period_returns)}, "
            f"Tickers: {list(period_returns.columns)}"
        )

    return contributions

def compute_ytd_risk_contribution(prices, weights):
    """
    Compute each asset's percentage contribution to the portfolio's YTD total risk.
    
    YTD is calculated from January 1 to the last available date in the dataset.
    
    Uses variance decomposition: each asset's contribution to portfolio variance is
    w_i × (Σ × w)_i, where Σ is the covariance matrix and w are portfolio weights.
    Normalized by portfolio variance to get percentages that sum to 100%.
    
    Includes cash as a zero-risk asset (cash contribution = 0.0).
    Cash tickers will have 0.0 contribution but are included in the result.
    Filter cash from visualization in plotting functions.
    
    Returns: Series indexed by ticker, values = percentage contribution as decimal
             (e.g., 0.35 = 35% of total portfolio risk)
             These percentages sum to 100% (cash contributes 0%)
    
    prices: DataFrame of daily prices (index = dates, columns = tickers)
    weights: dict/Series of ticker -> weight for portfolio
    """
    prices = prices.sort_index()
    as_of = get_as_of_date(prices.index)
    start = pd.Timestamp(as_of.year, 1, 1)
    
    # Filter for YTD data (from January 1 to as-of date)
    year_data = prices.loc[(prices.index >= start) & (prices.index <= as_of)]
    
    if year_data.empty:
        raise ValueError(f"No price data available for YTD period (year {as_of.year}).")
    
    # Compute daily returns for YTD period (invested assets only)
    daily_returns = year_data.pct_change().dropna()
    
    if len(daily_returns) < 20:  # Need sufficient data for meaningful covariance
        raise ValueError("Insufficient YTD data for risk calculation.")
    
    # Compute annualized covariance matrix (multiply by 252 trading days)
    cov_matrix = daily_returns.cov() * 252
    
    # Separate cash from invested weights
    w = pd.Series(weights)
    cash_weight = get_cash_weight(w)
    invested_weights = {ticker: weight for ticker, weight in w.items() 
                        if not is_cash_ticker(ticker)}
    
    # Initialize risk contributions with all weights (including cash)
    risk_contrib_series = pd.Series(0.0, index=w.index)
    
    # Compute risk contributions for invested assets only
    if invested_weights:
        w_invested = pd.Series(invested_weights).reindex(prices.columns).fillna(0)
        # Normalize invested weights to sum to (1 - cash_weight)
        total_invested = w_invested.sum()
        if total_invested > 0:
            w_invested = w_invested / total_invested * (1 - cash_weight)
        
        w_invested = w_invested.reindex(cov_matrix.columns).fillna(0)
        
        # Compute portfolio variance: w^T × Σ × w (only invested portion)
        w_arr = w_invested.values
        cov_arr = cov_matrix.values
        portfolio_variance = w_arr @ cov_arr @ w_arr
        
        if portfolio_variance > 0 and not pd.isna(portfolio_variance):
            # Compute variance contributions: w_i × (Σ × w)_i
            cov_times_w = cov_arr @ w_arr  # (Σ * w)
            variance_contributions = w_arr * cov_times_w  # w_i × (Σ × w)_i
            
            # Normalize by portfolio variance to get percentages (sum to 100% of invested risk)
            risk_percentages = variance_contributions / portfolio_variance
            
            # Add invested asset contributions to result
            for ticker, contrib in zip(w_invested.index, risk_percentages):
                if ticker in risk_contrib_series.index:
                    risk_contrib_series[ticker] = contrib
    
    # Cash tickers contribute 0.0 to risk (already initialized to 0.0)
    # Note: The percentages sum to 100% of the invested portion's risk
    # Cash reduces total portfolio risk but contributes 0% to the risk decomposition
    
    return risk_contrib_series

def compute_monthly_portfolio_returns(prices, weights, years_back: int = 5) -> pd.DataFrame:
    """
    Compute monthly portfolio returns over the past N years.
    
    Years are calculated relative to the dataset's last available date, not the current date.

    Includes cash as a zero-return asset if present in weights.

    prices: DataFrame of daily prices (index = dates, columns = tickers)
    weights: dict/Series of ticker -> weight for portfolio.
    years_back: number of years to include.
    """
    prices = prices.sort_index()

    # Use compute_daily_returns which handles cash properly
    port_daily = compute_daily_returns(prices, weights=weights)

    # monthly returns from daily (compounded within each month)
    monthly = (1 + port_daily).resample('ME').prod() - 1

    # restrict to last "years_back" calendar years (based on dataset's last date)
    as_of_date = get_as_of_date(prices.index)
    current_year = as_of_date.year
    earliest_year = current_year - years_back + 1
    monthly = monthly[monthly.index.year >= earliest_year]

    # build year x month table
    df = monthly.to_frame(name='Monthly Return')
    df["Year"] = df.index.year
    df["Month"] = df.index.month

    table = df.pivot(index='Year', columns='Month', values='Monthly Return')

    # order years descending
    table = table.sort_index(ascending=False)

    # rename month columns to names
    month_names = {i: name for i, name in enumerate(
        ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
    table = table.rename(columns=month_names)

    # ensure all 12 months exist as columns
    all_months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    table = table.reindex(columns=all_months)

    return table

def compute_sector_weights(tickers, weights, sector_info):
    """
    Compute portfolio weights aggregated by sector.
    
    Includes cash as "Cash / Uninvested" sector if present in weights.
    
    tickers: list of ticker symbols
    weights: dict/Series of ticker -> weight
    sector_info: DataFrame with columns: ticker, sector, industry
    
    Returns:
        Series indexed by sector, values = aggregated weights
    """
    # Convert weights to Series if dict
    if isinstance(weights, dict):
        w = pd.Series(weights)
    else:
        w = pd.Series(weights)
    
    # Separate cash from invested assets
    cash_weight = get_cash_weight(w)
    invested_tickers = [ticker for ticker in tickers if not is_cash_ticker(ticker)]
    
    # Create a mapping from ticker to sector
    sector_map = sector_info.set_index('ticker')['sector'].to_dict()
    
    # Map each invested ticker to its sector
    ticker_sectors = pd.Series([sector_map.get(ticker, 'Unknown') for ticker in invested_tickers], 
                              index=invested_tickers)
    
    # Create DataFrame with ticker, weight, and sector (invested assets only)
    df = pd.DataFrame({
        'ticker': invested_tickers,
        'weight': [w.get(ticker, 0) for ticker in invested_tickers],
        'sector': ticker_sectors.values
    })
    
    # Aggregate weights by sector
    sector_weights = df.groupby('sector')['weight'].sum()
    
    # Add cash as "Cash / Uninvested" if present
    if cash_weight > 0:
        sector_weights['Cash / Uninvested'] = cash_weight
    
    # Sort by weight descending
    sector_weights = sector_weights.sort_values(ascending=False)
    
    return sector_weights

def compute_window_cagr(prices, weights=None):
    """
    Compute CAGR over the actual analysis window (effective_start_date → as_of).
    
    This function is designed for Summary metrics and uses the actual period length
    from the windowed data, not hardcoded horizons like 1Y/3Y/5Y.
    
    IMPORTANT: This function assumes prices are already windowed to the desired period.
    It does NOT apply any additional lookback - uses all provided data.
    
    prices: DataFrame of daily prices (index = dates, columns = tickers)
           Should already be filtered to the effective window (no additional lookback applied)
    weights: optional dict/Series of ticker -> weight for portfolio
    
    Returns: tuple (cagr, audit_info) where:
            - cagr: float or None if insufficient data
            - audit_info: dict with 'start_date', 'end_date', 'n_obs', 'actual_years' or None if insufficient data
    """
    if prices.empty:
        return None, None
    
    prices = prices.sort_index()
    
    # Compute daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Require at least 20 trading days for reliable CAGR calculation
    if len(daily_returns) < 20:
        return None, None
    
    # Build portfolio returns if weights provided
    if weights is not None:
        w = pd.Series(weights).reindex(daily_returns.columns).fillna(0)
        portfolio_daily = (daily_returns * w).sum(axis=1)
        returns_series = portfolio_daily
    else:
        # Assume single-column DF for benchmark
        if daily_returns.shape[1] == 1:
            returns_series = daily_returns.iloc[:, 0]
        else:
            w = pd.Series(1 / daily_returns.shape[1], index=daily_returns.columns)
            returns_series = (daily_returns * w).sum(axis=1)
    
    # Calculate actual period length from return index endpoints
    start_date = returns_series.index[0]
    end_date = returns_series.index[-1]
    actual_years = (end_date - start_date).days / 365.25
    
    # Require positive period length
    if actual_years <= 0:
        return None, None
    
    # Compute total return: compound all daily returns
    total_return = (1 + returns_series).prod() - 1
    
    # Compute CAGR using ACTUAL period length (not hardcoded)
    cagr = (1 + total_return) ** (1 / actual_years) - 1
    
    # Audit info
    audit_info = {
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'n_obs': len(returns_series),
        'actual_years': float(actual_years)
    }
    
    return float(cagr), audit_info

def compute_asset_breakdown(prices, weights):
    """
    Compute asset-level metrics for the Asset Breakdown table.
    
    Uses the same analysis window as all other metrics (effective_start_date → actual_end_date).
    No additional lookback is applied - uses all provided windowed prices.
    
    prices: DataFrame of daily prices (index = dates, columns = tickers)
           Should already be filtered to the effective window (no additional lookback applied)
    weights: dict/Series of ticker -> weight for portfolio
    
    Returns: list of dicts, one per asset, with keys:
            - ticker: str
            - weight: float (as decimal, e.g., 0.25 for 25%)
            - cagr: float (as decimal, e.g., 0.12 for 12%) or None if insufficient data
            - volatility: float (as decimal, e.g., 0.15 for 15%) or None if insufficient data
            - bestDay: float (as decimal, e.g., 0.05 for 5%) or None if insufficient data
            - worstDay: float (as decimal, e.g., -0.03 for -3%) or None if insufficient data
    """
    if prices.empty:
        return []
    
    prices = prices.sort_index()
    
    # Compute daily returns for all assets
    daily_returns = prices.pct_change().dropna()
    
    if len(daily_returns) < 20:
        # Insufficient data - return basic structure with None values
        return [
            {
                'ticker': ticker,
                'weight': float(weights.get(ticker, 0.0)),
                'cagr': None,
                'volatility': None,
                'bestDay': None,
                'worstDay': None
            }
            for ticker in weights.keys()
        ]
    
    # Calculate window length once from daily returns index
    start_date = daily_returns.index[0]
    end_date = daily_returns.index[-1]
    window_length_years = (end_date - start_date).days / 365.25
    
    asset_breakdown = []
    
    for ticker in weights.keys():
        weight = float(weights.get(ticker, 0.0))
        
        # Handle cash tickers
        if is_cash_ticker(ticker):
            asset_breakdown.append({
                'ticker': ticker,
                'weight': weight,
                'cagr': 0.0,
                'volatility': 0.0,
                'bestDay': 0.0,
                'worstDay': 0.0
            })
            continue
        
        # Get asset daily returns
        if ticker not in daily_returns.columns:
            # Ticker not in prices (shouldn't happen if data is properly aligned)
            asset_breakdown.append({
                'ticker': ticker,
                'weight': weight,
                'cagr': None,
                'volatility': None,
                'bestDay': None,
                'worstDay': None
            })
            continue
        
        asset_daily = daily_returns[ticker].dropna()
        
        if len(asset_daily) < 20:
            # Insufficient data for this asset
            asset_breakdown.append({
                'ticker': ticker,
                'weight': weight,
                'cagr': None,
                'volatility': None,
                'bestDay': None,
                'worstDay': None
            })
            continue
        
        # Compute total return over window
        total_return = (1 + asset_daily).prod() - 1
        
        # Compute CAGR using actual window length
        if window_length_years > 0:
            cagr = (1 + total_return) ** (1 / window_length_years) - 1
        else:
            cagr = None
        
        # Compute annualized volatility (pandas std uses ddof=1 by default)
        volatility = asset_daily.std() * (252 ** 0.5)
        
        # Best and worst day
        best_day = float(asset_daily.max())
        worst_day = float(asset_daily.min())
        
        asset_breakdown.append({
            'ticker': ticker,
            'weight': weight,
            'cagr': float(cagr) if cagr is not None else None,
            'volatility': float(volatility) if not pd.isna(volatility) else None,
            'bestDay': best_day,
            'worstDay': worst_day
        })
    
    return asset_breakdown

def compute_daily_returns(prices, weights=None):
    """
    Compute daily returns for portfolio or benchmark.
    
    Includes cash as a zero-return asset if present in weights.
    Cash tickers (CASH, USD, CASH_USD, etc.) contribute 0.0 daily return.
    
    Note: Prices should already be filtered to the effective start date (where all
    securities have data available) before calling this function. This function only
    drops rows where ALL securities have missing data (non-trading days).
    
    prices: DataFrame of daily prices (index = dates, columns = tickers)
    weights: optional dict/Series of ticker -> weight for portfolio
    
    Returns: Series of daily returns
    """
    prices = prices.sort_index()
    # Only drop rows where ALL columns are NaN (non-trading days)
    # Individual security NaN values should not occur if prices are properly filtered
    # to the effective start date, but if they do, they will result in NaN portfolio returns
    daily_returns = prices.pct_change().dropna(how='all')
    
    if weights is not None:
        w = pd.Series(weights)
        
        # Use raw weights directly - cash will contribute 0% automatically since cash has no price data
        # Portfolio return = sum(weight × return) for all assets
        # Cash contributes 0% since it's not in daily_returns DataFrame
        w_aligned = w.reindex(daily_returns.columns).fillna(0)
        portfolio_daily = (daily_returns * w_aligned).sum(axis=1)
        
        return portfolio_daily
    else:
        # assume single-column DF for benchmark
        if daily_returns.shape[1] == 1:
            return daily_returns.iloc[:, 0]
        else:
            w = pd.Series(1 / daily_returns.shape[1], index=daily_returns.columns)
            return (daily_returns * w).sum(axis=1)

def compute_volatility(daily_returns, years):
    """
    Compute annualized volatility (standard deviation) for a given period.
    
    daily_returns: Series of daily returns
    years: number of years to look back
    
    Returns: annualized volatility (float)
    """
    if len(daily_returns) == 0:
        return None
    
    # Get the last N years of data
    cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)
    period_returns = daily_returns[daily_returns.index >= cutoff_date]
    
    if len(period_returns) < 20:  # Need at least ~20 trading days
        return None
    
    # Annualized volatility: std * sqrt(252 trading days per year)
    volatility = period_returns.std() * (252 ** 0.5)
    return float(volatility)

def compute_rolling_volatility(prices, weights=None):
    """
    Compute 6-month rolling annualized volatility using the most recent 5 years of data.
    
    Data preparation:
    - Filter prices to last 5 years (60 months) ending today
    - Compute daily percentage returns
    - Sort by date ascending
    
    Rolling calculation:
    - 6-month window = 126 trading days
    - Rolling volatility = std(returns) * sqrt(252) for annualization
    - min_periods=126 (no values before full window)
    
    Returns: Series of rolling volatility (annualized), with NaN for periods before full window
    Only non-null values are included in the result.
    
    prices: DataFrame of daily prices (index = dates, columns = tickers)
    weights: optional dict/Series of ticker -> weight for portfolio
    
    Returns: Series with date index and annualized volatility values (only non-null)
    """
    # Filter to last 5 years (60 months) ending at actual data end date
    # Use actual end date from data to ensure consistency with other metrics
    actual_end_date = prices.index.max()
    cutoff_date = actual_end_date - pd.DateOffset(months=60)
    prices_filtered = prices[prices.index >= cutoff_date].sort_index()
    
    if len(prices_filtered) == 0:
        return pd.Series(dtype=float)
    
    # Compute daily percentage returns
    daily_returns = prices_filtered.pct_change().dropna()
    
    if len(daily_returns) == 0:
        return pd.Series(dtype=float)
    
    # Build portfolio returns if weights provided
    if weights is not None:
        w = pd.Series(weights).reindex(daily_returns.columns).fillna(0)
        returns_series = (daily_returns * w).sum(axis=1)
    else:
        # Assume single-column DF for benchmark
        if daily_returns.shape[1] == 1:
            returns_series = daily_returns.iloc[:, 0]
        else:
            w = pd.Series(1 / daily_returns.shape[1], index=daily_returns.columns)
            returns_series = (daily_returns * w).sum(axis=1)
    
    # Define 6-month rolling window as 126 trading days
    window_days = 126
    
    # Compute rolling standard deviation with min_periods=126
    rolling_std = returns_series.rolling(window=window_days, min_periods=window_days).std()
    
    # Annualize by multiplying by sqrt(252)
    rolling_volatility = rolling_std * np.sqrt(252)
    
    # Drop NaN values (periods before full window is complete)
    rolling_volatility = rolling_volatility.dropna()
    
    return rolling_volatility

def compute_sharpe_ratio(daily_returns, years, risk_free_rate=0.0):
    """
    Compute annualized Sharpe ratio for a given period.
    
    Uses mean daily excess returns methodology (matches compute_rolling_sharpe_ratio):
    - Converts annual risk-free rate to daily
    - Computes mean daily excess return (daily return - daily risk-free rate)
    - Computes standard deviation of daily returns
    - Sharpe = (mean daily excess return / daily return volatility) × sqrt(252)
    
    daily_returns: Series of daily returns
    years: number of years to look back
    risk_free_rate: annual risk-free rate (default 0.0, typically from ^TNX 10-year Treasury)
    
    Returns: Sharpe ratio (float) or None if insufficient data
    """
    if len(daily_returns) == 0:
        return None
    
    # Get the last N years of data
    cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)
    period_returns = daily_returns[daily_returns.index >= cutoff_date]
    
    # Require minimum observations for reliable Sharpe ratio estimation (6 months ≈ 126 trading days)
    if len(period_returns) < MIN_OBS_SHARPE:
        return None
    
    # Convert annual risk-free rate to daily using compound interest
    # Formula: (1 + r_annual)^(1/252) - 1
    daily_rf = (1 + risk_free_rate) ** (1 / 252.0) - 1
    
    # Compute mean daily excess return (daily return - daily risk-free rate)
    excess_returns = period_returns - daily_rf
    mean_daily_excess_return = excess_returns.mean()
    
    # Compute standard deviation of daily returns (not excess returns)
    daily_volatility = period_returns.std()
    
    # Handle divide-by-zero: return None if volatility is zero
    if daily_volatility == 0 or pd.isna(daily_volatility) or pd.isna(mean_daily_excess_return):
        return None
    
    # Compute Sharpe ratio
    # Sharpe = (mean daily excess return / daily return volatility) × sqrt(252)
    sharpe = (mean_daily_excess_return / daily_volatility) * np.sqrt(252)
    return float(sharpe)

def compute_sortino_ratio(daily_returns, years, risk_free_rate=0.0):
    """
    Compute annualized Sortino ratio for a given period.
    
    Sortino ratio is like Sharpe ratio but only penalizes downside volatility
    (standard deviation of negative returns only).
    
    Uses mean daily excess returns methodology (consistent with compute_sharpe_ratio):
    - Converts annual risk-free rate to daily
    - Computes mean daily excess return (daily return - daily risk-free rate)
    - Computes downside deviation of daily returns (std of negative returns only)
    - Sortino = (mean daily excess return / downside deviation) × sqrt(252)
    
    daily_returns: Series of daily returns
    years: number of years to look back
    risk_free_rate: annual risk-free rate (default 0.0, typically from ^TNX 10-year Treasury)
    
    Returns: Sortino ratio (float) or None if insufficient data
    """
    if len(daily_returns) == 0:
        return None
    
    # Get the last N years of data
    cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)
    period_returns = daily_returns[daily_returns.index >= cutoff_date]
    
    # Require minimum observations for reliable Sortino ratio estimation
    if len(period_returns) < MIN_OBS_SORTINO:
        return None
    
    # Convert annual risk-free rate to daily using compound interest
    # Formula: (1 + r_annual)^(1/252) - 1
    daily_rf = (1 + risk_free_rate) ** (1 / 252.0) - 1
    
    # Compute mean daily excess return (daily return - daily risk-free rate)
    excess_returns = period_returns - daily_rf
    mean_daily_excess_return = excess_returns.mean()
    
    # Downside deviation: std of only negative returns (not excess returns)
    negative_returns = period_returns[period_returns < 0]
    
    if len(negative_returns) == 0:
        # If no negative returns, downside deviation is 0, ratio would be infinity
        # Return None to indicate it's not meaningful (no downside volatility to measure)
        return None
    
    # Daily downside deviation: std of only negative returns
    daily_downside_std = negative_returns.std()
    
    if daily_downside_std == 0 or pd.isna(daily_downside_std) or pd.isna(mean_daily_excess_return):
        # Downside deviation is zero or NaN - cannot compute meaningful ratio
        return None
    
    # Sortino ratio: (mean daily excess return / daily downside deviation) × sqrt(252)
    sortino = (mean_daily_excess_return / daily_downside_std) * np.sqrt(252)
    return float(sortino)

def compute_rolling_sharpe_ratio(daily_returns, window_months=6, risk_free_rate=0.0):
    """
    Compute rolling Sharpe ratio over time using a specified window.
    
    Implements the correct 6-month rolling Sharpe calculation:
    - Uses 126 trading days (21 trading days × 6 months) as window size
    - Computes mean daily excess return (daily return - daily risk-free rate)
    - Computes standard deviation of daily returns
    - Sharpe = (mean daily excess return / daily return volatility) × sqrt(252)
    
    daily_returns: Series of daily returns (already computed from prices)
    window_months: number of months for rolling window (default 6)
    risk_free_rate: annual risk-free rate (default 0.0)
    
    Returns: Series of rolling Sharpe ratios with same index as daily_returns,
             with NaN values for periods before the full window is available
    """
    if len(daily_returns) == 0:
        return pd.Series(dtype=float, index=daily_returns.index)
    
    # Define 6-month rolling window as 126 trading days (21 trading days × 6 months)
    window_days = 126
    
    # Convert annual risk-free rate to daily using compound interest
    # Formula: (1 + r_annual)^(1/252) - 1
    daily_rf = (1 + risk_free_rate) ** (1 / 252.0) - 1
    
    # Initialize result series with NaN
    rolling_sharpe = pd.Series(index=daily_returns.index, dtype=float)
    rolling_sharpe[:] = np.nan
    
    # Calculate rolling Sharpe for each point
    for i in range(len(daily_returns)):
        # Get window of returns up to and including current point
        start_idx = max(0, i - window_days + 1)
        window_returns = daily_returns.iloc[start_idx:i+1]
        
        # Use minimum window size of 126 observations
        if len(window_returns) < window_days:
            rolling_sharpe.iloc[i] = np.nan
            continue
        
        # Compute mean daily excess return (daily return - daily risk-free rate)
        excess_returns = window_returns - daily_rf
        mean_daily_excess_return = excess_returns.mean()
        
        # Compute standard deviation of daily returns (not excess returns)
        daily_volatility = window_returns.std()
        
        # Handle divide-by-zero: return NaN if volatility is zero
        if daily_volatility == 0 or pd.isna(daily_volatility) or pd.isna(mean_daily_excess_return):
            rolling_sharpe.iloc[i] = np.nan
            continue
        
        # Compute rolling Sharpe ratio
        # Sharpe = (mean daily excess return / daily return volatility) × sqrt(252)
        sharpe = (mean_daily_excess_return / daily_volatility) * np.sqrt(252)
        
        # Store result (will be NaN if calculation failed)
        rolling_sharpe.iloc[i] = float(sharpe) if not pd.isna(sharpe) else np.nan
    
    return rolling_sharpe

def compute_beta(portfolio_daily_returns, benchmark_daily_returns, years):
    """
    Compute beta of portfolio relative to benchmark for a given period.
    
    portfolio_daily_returns: Series of portfolio daily returns
    benchmark_daily_returns: Series of benchmark daily returns
    years: number of years to look back
    
    Returns: beta (float)
    """
    if len(portfolio_daily_returns) == 0 or len(benchmark_daily_returns) == 0:
        return None
    
    # Get the last N years of data
    cutoff_date = portfolio_daily_returns.index[-1] - pd.DateOffset(years=years)
    
    # Align on common dates
    portfolio_period = portfolio_daily_returns[portfolio_daily_returns.index >= cutoff_date]
    benchmark_period = benchmark_daily_returns[benchmark_daily_returns.index >= cutoff_date]
    
    # Align indices
    common_dates = portfolio_period.index.intersection(benchmark_period.index)
    if len(common_dates) < 20:
        return None
    
    portfolio_aligned = portfolio_period.loc[common_dates]
    benchmark_aligned = benchmark_period.loc[common_dates]
    
    # Beta = Covariance(portfolio, benchmark) / Variance(benchmark)
    covariance = portfolio_aligned.cov(benchmark_aligned)
    benchmark_variance = benchmark_aligned.var()
    
    if benchmark_variance == 0:
        return None
    
    beta = covariance / benchmark_variance
    return float(beta)

def compute_rolling_beta(portfolio_prices, benchmark_prices, portfolio_weights, window_days=126):
    """
    Compute 6-month rolling beta vs benchmark using daily returns and a 126-trading-day window.
    
    Rolling beta assumes constant weights (daily rebalanced) unless holdings history is available.
    
    portfolio_prices: DataFrame (dates index, columns are tickers like AAPL/MSFT/GOOGL)
    benchmark_prices: DataFrame with exactly 1 column (e.g., "SPY")
    portfolio_weights: dict mapping ticker->weight (sums to 1.0)
    window_days: number of trading days for rolling window (default 126 for 6 months)
    
    Returns: Series with date index and rolling beta values (only non-null, after min_periods)
    """
    # Filter both price DataFrames to the last 5 years ending at actual data end date
    # Use the minimum of the two end dates to ensure alignment between portfolio and benchmark
    # This ensures consistency with other metrics and the API analysisDate
    actual_end_date = min(portfolio_prices.index.max(), benchmark_prices.index.max())
    cutoff_date = actual_end_date - pd.DateOffset(months=60)
    
    portfolio_filtered = portfolio_prices[portfolio_prices.index >= cutoff_date].sort_index()
    benchmark_filtered = benchmark_prices[benchmark_prices.index >= cutoff_date].sort_index()
    
    if len(portfolio_filtered) == 0 or len(benchmark_filtered) == 0:
        return pd.Series(dtype=float, name='beta')
    
    # Compute daily returns
    portfolio_ret_df = portfolio_filtered.pct_change()
    bench_ret = benchmark_filtered.iloc[:, 0].pct_change()
    
    # Construct portfolio daily returns using the weights dict
    w = pd.Series(portfolio_weights).reindex(portfolio_ret_df.columns).fillna(0.0)
    
    # Normalize weights if sum != 1 (handle zero-sum safely)
    if w.sum() != 0 and abs(w.sum() - 1.0) > 1e-10:
        w = w / w.sum()
    elif w.sum() == 0:
        # If all weights are zero, return empty series
        return pd.Series(dtype=float, name='beta')
    
    # Compute portfolio returns: weighted sum of individual returns
    portfolio_ret = portfolio_ret_df.mul(w, axis=1).sum(axis=1, min_count=1)
    
    # Align portfolio_ret and bench_ret on common dates using pd.concat(..., join="inner")
    aligned_df = pd.concat([portfolio_ret, bench_ret], axis=1, join='inner')
    aligned_df.columns = ['portfolio', 'benchmark']
    aligned_df = aligned_df.dropna()
    
    if len(aligned_df) < window_days:
        return pd.Series(dtype=float, name='beta')
    
    portfolio_ret_aligned = aligned_df['portfolio']
    bench_ret_aligned = aligned_df['benchmark']
    
    # Compute rolling beta using vectorized pandas operations (NO for-loops)
    WINDOW = window_days
    min_periods = window_days
    
    # Compute rolling covariance using vectorized operations
    # Formula: cov(x,y) = mean((x - mean(x)) * (y - mean(y)))
    # For sample covariance, we use: sum((x-mean(x))*(y-mean(y))) / (n-1)
    portfolio_mean = portfolio_ret_aligned.rolling(WINDOW, min_periods=min_periods).mean()
    bench_mean = bench_ret_aligned.rolling(WINDOW, min_periods=min_periods).mean()
    
    # Compute product of deviations from their respective rolling means
    deviations_product = (portfolio_ret_aligned - portfolio_mean) * (bench_ret_aligned - bench_mean)
    
    # Compute rolling mean of deviations product (population covariance)
    cov_pop = deviations_product.rolling(WINDOW, min_periods=min_periods).mean()
    
    # Convert to sample covariance: multiply by n/(n-1)
    n_effective = portfolio_ret_aligned.rolling(WINDOW, min_periods=min_periods).count()
    cov = cov_pop * (n_effective / (n_effective - 1)).fillna(1.0)
    
    # Compute rolling variance of benchmark (sample variance, ddof=1)
    var = bench_ret_aligned.rolling(WINDOW, min_periods=min_periods).var(ddof=1)
    
    # Compute beta = cov / var with epsilon guard (var.abs() > 1e-12) to avoid divide-by-zero
    epsilon = 1e-12
    beta = cov / var.where(var.abs() > epsilon, np.nan)
    
    # Return beta.dropna() as a Series indexed by date (no backfill/fill missing values)
    return beta.dropna()

def compute_drawdown_series(cumulative_index):
    """
    Compute drawdown series over time from cumulative index.
    
    Drawdown is calculated as the percentage decline from the most recent peak.
    
    cumulative_index: Series of cumulative index values (e.g., from compute_cumulative_index)
    
    Returns: Series of drawdown percentages (negative values, 0 at peaks)
    """
    if len(cumulative_index) == 0:
        return pd.Series(dtype=float)
    
    # Calculate running maximum (peak)
    running_max = cumulative_index.expanding().max()
    
    # Calculate drawdown from peak: (current - peak) / peak
    drawdown = (cumulative_index - running_max) / running_max
    
    return drawdown

def compute_max_drawdown(cumulative_index, years):
    """
    Compute maximum drawdown for a given period.
    
    cumulative_index: Series of cumulative index values (e.g., from compute_cumulative_index)
    years: number of years to look back
    
    Returns: max drawdown as a percentage (float, negative value)
    """
    if len(cumulative_index) == 0:
        return None
    
    # Get the last N years of data
    cutoff_date = cumulative_index.index[-1] - pd.DateOffset(years=years)
    period_index = cumulative_index[cumulative_index.index >= cutoff_date]
    
    if len(period_index) < 20:
        return None
    
    # Calculate running maximum (peak)
    running_max = period_index.expanding().max()
    
    # Calculate drawdown from peak
    drawdown = (period_index - running_max) / running_max
    
    # Maximum drawdown (most negative)
    max_dd = drawdown.min()
    return float(max_dd)

def compute_ulcer_index(cumulative_index, years):
    """
    Compute Ulcer Index for a given period.
    
    Ulcer Index measures the depth and duration of drawdowns.
    Formula: sqrt(mean(squared_drawdowns)) * 100
    where drawdowns are calculated from the most recent peak.
    
    cumulative_index: Series of cumulative index values (e.g., from compute_cumulative_index)
    years: number of years to look back
    
    Returns: Ulcer Index as a percentage (float, non-negative value)
    """
    if len(cumulative_index) == 0:
        return None
    
    # Get the last N years of data
    cutoff_date = cumulative_index.index[-1] - pd.DateOffset(years=years)
    period_index = cumulative_index[cumulative_index.index >= cutoff_date]
    
    if len(period_index) < 20:
        return None
    
    # Calculate running maximum (peak)
    running_max = period_index.expanding().max()
    
    # Calculate drawdown from peak (negative values)
    drawdown = (period_index - running_max) / running_max
    
    # Square the drawdowns (only negative values contribute)
    squared_drawdowns = drawdown ** 2
    
    # Average the squared drawdowns
    mean_squared_drawdown = squared_drawdowns.mean()
    
    # Take square root and convert to percentage
    ulcer_index = np.sqrt(mean_squared_drawdown) * 100
    
    return float(ulcer_index)

def compute_risk_metrics(portfolio_prices, benchmark_prices, portfolio_weights, periods=[1, 3, 5], risk_free_rate=0.0):
    """
    Compute comprehensive risk metrics for multiple time periods.
    
    portfolio_prices: DataFrame of portfolio prices
    benchmark_prices: DataFrame of benchmark prices
    portfolio_weights: dict/Series of portfolio weights
    periods: list of years to compute metrics for (default [1, 3, 5])
    risk_free_rate: annual risk-free rate (default 0.0)
    
    Returns: DataFrame with risk metrics (rows: metrics, columns: periods)
    """
    # Compute daily returns
    portfolio_daily = compute_daily_returns(portfolio_prices, weights=portfolio_weights)
    benchmark_daily = compute_daily_returns(benchmark_prices, weights=None)
    
    # Compute cumulative index for drawdown calculation
    portfolio_cum = compute_cumulative_index(portfolio_prices, weights=portfolio_weights)
    
    results = {}
    
    for years in periods:
        period_key = f"{years}Y"
        metrics = {}
        
        # Volatility
        vol = compute_volatility(portfolio_daily, years)
        metrics['Volatility'] = vol
        
        # Sharpe ratio
        sharpe = compute_sharpe_ratio(portfolio_daily, years, risk_free_rate)
        metrics['Sharpe Ratio'] = sharpe
        
        # Sortino ratio
        sortino = compute_sortino_ratio(portfolio_daily, years, risk_free_rate)
        metrics['Sortino Ratio'] = sortino
        
        # Beta
        beta = compute_beta(portfolio_daily, benchmark_daily, years)
        metrics['Beta'] = beta
        
        # Max drawdown
        max_dd = compute_max_drawdown(portfolio_cum, years)
        metrics['Max Drawdown'] = max_dd
        
        # Ulcer Index
        ulcer_idx = compute_ulcer_index(portfolio_cum, years)
        metrics['Ulcer Index'] = ulcer_idx
        
        results[period_key] = metrics
    
    # Convert to DataFrame (metrics as rows, periods as columns)
    df = pd.DataFrame(results)
    
    return df

def compute_correlation_matrix(prices, years=None):
    """
    Compute correlation matrix between all assets in the portfolio.
    
    prices: DataFrame of daily prices (index = dates, columns = tickers)
    years: optional number of years to look back (if None, uses all available data)
    
    Returns: DataFrame with correlation matrix (tickers x tickers)
    """
    prices = prices.sort_index()
    
    # Compute daily returns
    daily_returns = prices.pct_change().dropna()
    
    # If years specified, filter to last N years
    if years is not None:
        cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)
        daily_returns = daily_returns[daily_returns.index >= cutoff_date]
    
    # Compute correlation matrix
    correlation_matrix = daily_returns.corr()
    
    return correlation_matrix

def compute_annualized_return_and_volatility(prices, weights=None, years=5):
    """
    Compute annualized return and volatility for a given period.
    
    prices: DataFrame of daily prices (index = dates, columns = tickers)
    weights: optional dict/Series of ticker -> weight for portfolio
    years: number of years to look back
    
    Returns: tuple (annualized_return, annualized_volatility) as decimals
    """
    prices = prices.sort_index()
    
    # Compute daily returns
    daily_returns = prices.pct_change().dropna()
    
    # Get the last N years of data
    cutoff_date = daily_returns.index[-1] - pd.DateOffset(years=years)
    period_returns = daily_returns[daily_returns.index >= cutoff_date]
    
    if len(period_returns) < 20:
        return None, None
    
    # Build portfolio returns if weights provided
    if weights is not None:
        w = pd.Series(weights).reindex(period_returns.columns).fillna(0)
        portfolio_daily = (period_returns * w).sum(axis=1)
        returns_series = portfolio_daily
    else:
        # Assume single-column DF for benchmark
        if period_returns.shape[1] == 1:
            returns_series = period_returns.iloc[:, 0]
        else:
            w = pd.Series(1 / period_returns.shape[1], index=period_returns.columns)
            returns_series = (period_returns * w).sum(axis=1)
    
    # Annualized return: (1 + total_return)^(1/years) - 1
    total_return = (1 + returns_series).prod() - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1
    
    # Annualized volatility: std * sqrt(252)
    annualized_volatility = returns_series.std() * (252 ** 0.5)
    
    return float(annualized_return), float(annualized_volatility)

def portfolio_stats(w, mu, cov):
    """
    Compute portfolio return and volatility from weights.
    
    w: 1D numpy array of weights
    mu: Series or array of expected returns
    cov: DataFrame or 2D array of covariance matrix
    
    Returns: tuple (return, volatility) as floats
    """
    # Convert to numpy arrays if needed
    if isinstance(mu, pd.Series):
        mu_arr = mu.values
    else:
        mu_arr = mu
    
    if isinstance(cov, pd.DataFrame):
        cov_arr = cov.values
    else:
        cov_arr = cov
    
    ret = float(w @ mu_arr)
    vol = float(np.sqrt(w @ cov_arr @ w))
    return ret, vol

def compute_efficient_frontier_inputs(daily_returns):
    """
    Compute annualized expected returns vector and covariance matrix from daily returns.
    
    daily_returns: DataFrame of daily returns (index = dates, columns = tickers)
    
    Returns: tuple (mu, cov) where:
        mu: Series of annualized expected returns (mean * 252)
        cov: DataFrame of annualized covariance matrix (cov * 252)
    """
    # Annualized expected returns: mean of daily returns * 252
    mu = daily_returns.mean() * 252
    
    # Annualized covariance matrix: covariance of daily returns * 252
    cov = daily_returns.cov() * 252
    
    return mu, cov

def optimize_min_variance(mu, cov):
    """
    Compute the minimum variance portfolio.
    
    Objective: Minimize portfolio variance w.T @ cov @ w
    Constraints: sum(w) = 1, w >= 0
    
    mu: Series of expected returns
    cov: DataFrame of covariance matrix
    
    Returns: tuple (weights_array, return, volatility)
    """
    n = len(mu)
    w0 = np.ones(n) / n
    
    # Convert to numpy arrays
    if isinstance(cov, pd.DataFrame):
        cov_arr = cov.values
    else:
        cov_arr = cov
    
    def objective(w):
        return w @ cov_arr @ w
    
    bounds = [(0.0, 1.0)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    
    res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
    
    if not res.success:
        return None, None, None
    
    w_opt = res.x
    ret, vol = portfolio_stats(w_opt, mu, cov)
    return w_opt, ret, vol

def optimize_max_sharpe(mu, cov, rf):
    """
    Compute the tangency (max Sharpe) portfolio.
    
    Objective: Maximize Sharpe ratio = (w.T @ mu - rf) / sqrt(w.T @ cov @ w)
    Implemented by minimizing negative Sharpe ratio.
    Constraints: sum(w) = 1, w >= 0
    
    mu: Series of expected returns
    cov: DataFrame of covariance matrix
    rf: risk-free rate (annual, as decimal)
    
    Returns: tuple (weights_array, return, volatility, sharpe)
    """
    n = len(mu)
    w0 = np.ones(n) / n
    
    def neg_sharpe(w):
        ret, vol = portfolio_stats(w, mu, cov)
        if vol < 1e-8:
            return 1e6  # Penalty for zero volatility
        return -(ret - rf) / vol
    
    bounds = [(0.0, 1.0)] * n
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    
    res = minimize(neg_sharpe, w0, method='SLSQP', bounds=bounds, constraints=cons)
    
    if not res.success:
        return None, None, None, None
    
    w_opt = res.x
    ret, vol = portfolio_stats(w_opt, mu, cov)
    sharpe = (ret - rf) / vol if vol > 0 else None
    return w_opt, ret, vol, sharpe

def compute_efficient_frontier(mu, cov, num_points=100):
    """
    Compute the efficient frontier curve.
    
    For a grid of target returns between min(mu) and max(mu), minimize variance
    subject to: sum(w) = 1, w >= 0, w.T @ mu = target_mu
    
    mu: Series of expected returns
    cov: DataFrame of covariance matrix
    num_points: number of points on the frontier
    
    Returns: DataFrame with columns ['vol', 'ret'] sorted by volatility
    """
    n = len(mu)
    weights_list = []
    rets = []
    vols = []
    
    ret_min = float(mu.min())
    ret_max = float(mu.max())
    target_rets = np.linspace(ret_min, ret_max, num_points)
    
    # Convert to numpy arrays
    if isinstance(mu, pd.Series):
        mu_arr = mu.values
    else:
        mu_arr = mu
    
    if isinstance(cov, pd.DataFrame):
        cov_arr = cov.values
    else:
        cov_arr = cov
    
    for target in target_rets:
        w0 = np.ones(n) / n
        
        def objective(w):
            return w @ cov_arr @ w
        
        # Create constraint function that captures target value
        def make_return_constraint(target_ret):
            return lambda w: w @ mu_arr - target_ret
        
        cons = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'eq', 'fun': make_return_constraint(target)},
        )
        bounds = [(0.0, 1.0)] * n
        
        res = minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
        if not res.success:
            continue
        
        w_opt = res.x
        ret, vol = portfolio_stats(w_opt, mu, cov)
        weights_list.append(w_opt)
        rets.append(ret)
        vols.append(vol)
    
    frontier = pd.DataFrame({'vol': vols, 'ret': rets})
    
    if frontier.empty:
        return pd.DataFrame(columns=['vol', 'ret'])
    
    # Sort by volatility
    frontier = frontier.sort_values('vol').reset_index(drop=True)
    
    # Filter to keep only efficient points (upper envelope)
    # Keep only points where return is non-decreasing as volatility increases
    if len(frontier) > 1:
        filtered = [frontier.iloc[0]]
        for i in range(1, len(frontier)):
            current_vol = frontier.iloc[i]['vol']
            current_ret = frontier.iloc[i]['ret']
            prev_ret = filtered[-1]['ret']
            
            # Only add if return is greater than or equal to previous (efficient frontier)
            # Also ensure volatility is increasing
            if current_ret >= prev_ret and current_vol > filtered[-1]['vol']:
                filtered.append(frontier.iloc[i])
            elif current_ret > prev_ret:
                # If return is higher but vol is same or lower, replace previous point
                filtered[-1] = frontier.iloc[i]
        
        frontier = pd.DataFrame(filtered).reset_index(drop=True)
        frontier = frontier.sort_values('vol').reset_index(drop=True)
    
    return frontier

def generate_random_portfolios(mu, cov, n_portfolios=3000):
    """
    Generate random long-only portfolios and compute their risk-return characteristics.
    
    mu: Series of expected returns
    cov: DataFrame of covariance matrix
    n_portfolios: number of random portfolios to generate
    
    Returns: DataFrame with columns ['vol', 'ret']
    """
    n = len(mu)
    random_portfolios = []
    
    for _ in range(n_portfolios):
        # Generate random positive weights (Dirichlet-like: uniform then normalize)
        weights = np.random.uniform(0, 1, n)
        weights = weights / weights.sum()
        
        # Compute portfolio return and volatility using helper
        ret, vol = portfolio_stats(weights, mu, cov)
        random_portfolios.append({'vol': vol, 'ret': ret})
    
    return pd.DataFrame(random_portfolios)

def compute_efficient_frontier_analysis(portfolio_prices, portfolio_weights, benchmark_prices_dict, years=5):
    """
    High-level function to compute efficient frontier, tangency portfolio, random portfolios,
    and risk-return points for portfolio, assets, and benchmarks.
    
    portfolio_prices: DataFrame of portfolio asset prices
    portfolio_weights: dict/Series of portfolio weights
    benchmark_prices_dict: dict of {ticker: DataFrame} for benchmarks (SPY, QQQ, AGG, etc.)
    years: number of years of data to use
    
    Returns: dict with all computed data for plotting
    """
    # Get risk-free rate
    from portfolio_tool.market_data import get_risk_free_rate
    rf = get_risk_free_rate('^TNX')
    if rf is None:
        rf = 0.04
        print("Warning: Could not fetch risk-free rate. Using 4% as fallback.")
    
    # Compute daily returns for portfolio assets
    portfolio_daily = portfolio_prices.pct_change().dropna()
    
    # Filter to last N years
    cutoff_date = portfolio_daily.index[-1] - pd.DateOffset(years=years)
    portfolio_daily = portfolio_daily[portfolio_daily.index >= cutoff_date]
    
    if len(portfolio_daily) < 20:
        return None
    
    # Compute annualized inputs
    mu, cov = compute_efficient_frontier_inputs(portfolio_daily)
    
    # Compute efficient frontier
    frontier_df = compute_efficient_frontier(mu, cov)
    
    # Compute tangency portfolio
    tangency_weights, tangency_ret, tangency_vol, tangency_sharpe = optimize_max_sharpe(mu, cov, rf)
    
    # Generate random portfolios
    random_portfolios = generate_random_portfolios(mu, cov, n_portfolios=3000)
    
    # Compute individual asset points using consistent annualization
    asset_points = []
    for ticker in portfolio_daily.columns:
        # Use same annualization: mean * 252 for return, std * sqrt(252) for vol
        asset_ret = mu[ticker]  # Already annualized (mean * 252)
        asset_vol = np.sqrt(cov.loc[ticker, ticker])  # Already annualized (cov * 252, so sqrt gives vol)
        asset_points.append({'ticker': ticker, 'vol': float(asset_vol), 'ret': float(asset_ret)})
    asset_points_df = pd.DataFrame(asset_points)
    
    # Compute user portfolio point
    # Separate cash from invested weights
    w = pd.Series(portfolio_weights)
    cash_weight = get_cash_weight(w)
    invested_weights = {ticker: weight for ticker, weight in w.items() 
                       if not is_cash_ticker(ticker)}
    
    # Compute invested portion risk/return
    w_invested = pd.Series(invested_weights).reindex(portfolio_daily.columns).fillna(0)
    # Normalize invested weights to sum to (1 - cash_weight)
    total_invested = w_invested.sum()
    if total_invested > 0:
        w_invested = w_invested / total_invested * (1 - cash_weight)
        w_arr = w_invested.values
        invested_ret, invested_vol = portfolio_stats(w_arr, mu, cov)
    else:
        invested_ret, invested_vol = 0.0, 0.0
    
    # Portfolio return including cash: (1 - cash_weight) * invested_ret + cash_weight * 0.0
    port_ret = invested_ret * (1 - cash_weight)
    # Portfolio volatility including cash: (1 - cash_weight) * invested_vol (cash has 0 vol, 0 correlation)
    port_vol = invested_vol * (1 - cash_weight)
    
    portfolio_point = {'vol': port_vol, 'ret': port_ret}
    
    # Compute benchmark points using consistent annualization
    benchmark_points = []
    for ticker, prices_df in benchmark_prices_dict.items():
        bench_daily = prices_df.pct_change().dropna()
        bench_daily = bench_daily[bench_daily.index >= cutoff_date]
        if len(bench_daily) >= 20:
            # Consistent annualization: mean * 252, std * sqrt(252)
            bench_ret = bench_daily.mean().iloc[0] * 252
            bench_vol = bench_daily.std().iloc[0] * np.sqrt(252)
            benchmark_points.append({'ticker': ticker, 'vol': float(bench_vol), 'ret': float(bench_ret)})
    benchmark_points_df = pd.DataFrame(benchmark_points)
    
    # Compute 60/40 SPY/AGG portfolio if both available
    if 'SPY' in benchmark_prices_dict and 'AGG' in benchmark_prices_dict:
        spy_prices = benchmark_prices_dict['SPY']
        agg_prices = benchmark_prices_dict['AGG']
        
        # Align dates
        common_dates = spy_prices.index.intersection(agg_prices.index)
        spy_aligned = spy_prices.loc[common_dates]
        agg_aligned = agg_prices.loc[common_dates]
        
        # Compute 60/40 portfolio returns
        spy_returns = spy_aligned.pct_change().dropna()
        agg_returns = agg_aligned.pct_change().dropna()
        common_returns = spy_returns.index.intersection(agg_returns.index)
        
        if len(common_returns) >= 20:
            spy_ret_series = spy_returns.loc[common_returns].iloc[:, 0]
            agg_ret_series = agg_returns.loc[common_returns].iloc[:, 0]
            mix_returns = 0.6 * spy_ret_series + 0.4 * agg_ret_series
            mix_returns = mix_returns[mix_returns.index >= cutoff_date]
            
            if len(mix_returns) >= 20:
                # Consistent annualization: mean * 252, std * sqrt(252)
                mix_ret = mix_returns.mean() * 252
                mix_vol = mix_returns.std() * np.sqrt(252)
                benchmark_points_df = pd.concat([
                    benchmark_points_df,
                    pd.DataFrame([{'ticker': '60/40 SPY/AGG', 'vol': float(mix_vol), 'ret': float(mix_ret)}])
                ], ignore_index=True)
    
    # Compute tangency portfolio point
    tangency_point = None
    if tangency_ret is not None and tangency_vol is not None:
        tangency_point = {'vol': tangency_vol, 'ret': tangency_ret}
    
    return {
        'frontier': frontier_df,
        'tangency': tangency_point,
        'tangency_weights': tangency_weights,
        'random_portfolios': random_portfolios,
        'asset_points': asset_points_df,
        'portfolio_point': portfolio_point,
        'benchmark_points': benchmark_points_df,
        'risk_free_rate': rf
    }