"""FastAPI server wrapper for Portfolio Analysis backend."""

import os
import re
import traceback

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Dict, Literal, Optional, Tuple
import pandas as pd
import numpy as np
import io
import logging
import time
import uuid
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("portfolio-api")


# --- Rate Limiter Setup ---
def get_real_client_ip(request: Request) -> str:
    """Extract real client IP from X-Forwarded-For header (Railway runs behind a proxy)."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For can contain multiple IPs: client, proxy1, proxy2
        # The first one is the real client IP
        return forwarded_for.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=get_real_client_ip)

# Ticker validation pattern: 1-10 alphanumeric chars plus dots and hyphens
_TICKER_PATTERN = re.compile(r'^[A-Za-z0-9.\-]{1,10}$')


# Minimum observations required for statistical validity
MIN_OBSERVATIONS = {
    'monthly': 36,   # 3 years
    'weekly': 156,   # 3 years
    'daily': 756     # 3 years
}


def calculate_adjusted_dates(
    start_date: str,
    end_date: str,
    frequency: str
) -> Tuple[str, str, bool, str]:
    """
    Calculate adjusted start date to ensure minimum observations for factor analysis.

    For monthly frequency, we need at least 36 observations (3 years).
    If the requested period is shorter, we extend the start date.

    Args:
        start_date: Requested start date (YYYY-MM-DD)
        end_date: Requested end date (YYYY-MM-DD)
        frequency: 'daily', 'weekly', or 'monthly'

    Returns:
        Tuple of (adjusted_start_date, end_date, was_adjusted, adjustment_message)
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    # Calculate requested period in years
    requested_days = (end_dt - start_dt).days
    requested_years = requested_days / 365.25

    # Minimum years needed based on frequency
    min_years_needed = {
        'monthly': 3.0,   # 36 months minimum
        'weekly': 3.0,    # 156 weeks minimum
        'daily': 3.0      # 756 days minimum
    }

    min_years = min_years_needed.get(frequency, 3.0)
    was_adjusted = False
    adjustment_message = None

    if requested_years < min_years:
        # Need to extend the start date
        was_adjusted = True

        # Add buffer for alignment issues
        buffer_months = 3 if frequency == 'monthly' else 6

        # Calculate new start date
        adjusted_start_dt = end_dt - relativedelta(years=int(min_years), months=buffer_months)

        # Ensure we don't go before a reasonable date (1990)
        earliest_allowed = pd.Timestamp('1990-01-01')
        if adjusted_start_dt < earliest_allowed:
            adjusted_start_dt = earliest_allowed

        adjusted_start_date = adjusted_start_dt.strftime('%Y-%m-%d')
        adjustment_message = (
            f"Requested period ({requested_years:.1f} years) extended to {min_years:.0f} years "
            f"to meet minimum {MIN_OBSERVATIONS[frequency]} {frequency} observations requirement"
        )

        logger.info(f"Date adjustment: {start_date} -> {adjusted_start_date} ({adjustment_message})")

        return adjusted_start_date, end_date, was_adjusted, adjustment_message

    return start_date, end_date, was_adjusted, adjustment_message


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
from portfolio_tool.factor_analysis import analyze_factors
from portfolio_tool.ai_insights import generate_insight
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

# --- Environment Variables ---
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
APP_VERSION = os.getenv("APP_VERSION", "1.1.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",") if os.getenv("ALLOWED_ORIGINS") else [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:8080",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
    "http://127.0.0.1:8080",
]
# Strip whitespace from origins
ALLOWED_ORIGINS = [o.strip() for o in ALLOWED_ORIGINS if o.strip()]

ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",") if os.getenv("ALLOWED_HOSTS") else ["*"]
ALLOWED_HOSTS = [h.strip() for h in ALLOWED_HOSTS if h.strip()]

# --- App Creation ---
app = FastAPI(
    title="Portfolio Analysis API",
    description="Backend API for portfolio analytics using real market data",
    version=APP_VERSION,
)

# Attach rate limiter to app
app.state.limiter = limiter


# --- Custom Rate Limit Exception Handler ---
@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    retry_after = exc.detail if hasattr(exc, "detail") else "unknown"
    return JSONResponse(
        status_code=429,
        content={
            "error": "Rate limit exceeded",
            "retry_after": str(retry_after),
            "message": "You've made too many requests. Please try again later.",
        },
        headers={"Retry-After": str(retry_after)},
    )


# --- Global Exception Handler ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Let FastAPI handle HTTPExceptions with their proper status codes
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {type(exc).__name__}")
    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
        },
    )


# --- Middleware Stack (last added = first executed) ---

# 1. TrustedHostMiddleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# 2. CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-RateLimit-Limit",
        "X-RateLimit-Remaining",
        "X-RateLimit-Reset",
        "Retry-After",
    ],
)


# 3. Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = "default-src 'self'"
    if ENVIRONMENT == "production":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# 4. Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    client_ip = get_real_client_ip(request)
    logger.info(f"Incoming {request.method} {request.url.path} from {client_ip}")
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(f"Completed {request.method} {request.url.path} -> {response.status_code} in {elapsed:.2f}s")
    return response


# --- Startup Event ---
@app.on_event("startup")
async def startup_event():
    openai_status = "configured" if os.getenv("OPENAI_API_KEY") else "NOT SET"
    logger.info(f"Starting Portfolio Analysis API v{APP_VERSION}")
    logger.info(f"Environment: {ENVIRONMENT}")
    logger.info(f"CORS origins: {len(ALLOWED_ORIGINS)} configured")
    logger.info(f"Trusted hosts: {ALLOWED_HOSTS}")
    logger.info(f"OpenAI API key: {openai_status}")
    if openai_status == "NOT SET":
        logger.warning("OPENAI_API_KEY not set - AI insights will be disabled")


class PortfolioRequest(BaseModel):
    portfolioText: str
    requested_start_date: Optional[str] = None  # ISO format date string (YYYY-MM-DD)


class FactorAnalysisRequest(BaseModel):
    # Single ticker (backward compatible)
    ticker: Optional[str] = None
    # Multiple tickers for portfolio analysis
    tickers: Optional[List[str]] = None
    weights: Optional[List[float]] = None  # Must sum to 1.0
    # Date range
    start_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    end_date: Optional[str] = Field(None, pattern=r'^\d{4}-\d{2}-\d{2}$')
    # Analysis parameters
    factor_model: Literal["CAPM", "3-factor", "4-factor", "5-factor"] = "3-factor"
    frequency: Literal["daily", "weekly", "monthly"] = "monthly"
    risk_free_rate: Literal["1M_TBILL", "3M_TBILL"] = "1M_TBILL"

    @field_validator('ticker')
    @classmethod
    def validate_ticker(cls, v):
        if v is not None:
            v = v.strip().upper()
            if not _TICKER_PATTERN.match(v):
                raise ValueError(f"Invalid ticker format: '{v}'. Must be 1-10 alphanumeric characters.")
        return v

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        if v is not None:
            if len(v) > 20:
                raise ValueError("Maximum 20 tickers allowed")
            validated = []
            for t in v:
                t = t.strip().upper()
                if not _TICKER_PATTERN.match(t):
                    raise ValueError(f"Invalid ticker format: '{t}'. Must be 1-10 alphanumeric characters.")
                validated.append(t)
            return validated
        return v

    @model_validator(mode='after')
    def validate_weights_match_tickers(self):
        if self.tickers and self.weights:
            if len(self.tickers) != len(self.weights):
                raise ValueError(f"Number of tickers ({len(self.tickers)}) must match number of weights ({len(self.weights)})")
            weights_sum = sum(self.weights)
            if not (0.99 <= weights_sum <= 1.01):
                raise ValueError(f"Weights must sum to ~1.0, got {weights_sum:.4f}")
        return self


class AbsoluteView(BaseModel):
    """Absolute view: belief about a single asset's expected return."""
    model_config = {"populate_by_name": True}

    asset: str  # Ticker symbol
    return_: float = Field(alias="return")  # Expected annual return (e.g., 0.15 for 15%)
    confidence: float = Field(gt=0.0, le=1.0)  # Confidence in view (0.0 to 1.0)

    @field_validator('asset')
    @classmethod
    def validate_asset(cls, v):
        v = v.strip().upper()
        if not _TICKER_PATTERN.match(v):
            raise ValueError(f"Invalid asset ticker: '{v}'")
        return v


class RelativeView(BaseModel):
    """Relative view: belief that one asset will outperform another."""
    asset1: str  # Ticker that will outperform
    asset2: str  # Ticker that will underperform
    outperformance: float  # Expected outperformance (e.g., 0.03 for 3%)
    confidence: float = Field(gt=0.0, le=1.0)  # Confidence in view (0.0 to 1.0)

    @field_validator('asset1', 'asset2')
    @classmethod
    def validate_assets(cls, v):
        v = v.strip().upper()
        if not _TICKER_PATTERN.match(v):
            raise ValueError(f"Invalid asset ticker: '{v}'")
        return v


class ViewsInput(BaseModel):
    """Container for investor views."""
    absolute: Optional[List[AbsoluteView]] = []
    relative: Optional[List[RelativeView]] = []


class BlackLittermanRequest(BaseModel):
    """
    Request body for Black-Litterman portfolio optimization.

    The Black-Litterman model combines market equilibrium with investor views
    to produce optimal portfolio weights.

    Attributes:
        tickers: List of ticker symbols to include in the portfolio
        start_date: Start date for historical data (YYYY-MM-DD)
        end_date: End date for historical data (YYYY-MM-DD)
        market_caps: Market capitalizations for each ticker (in dollars)
        views: Investor views (absolute and/or relative)
        risk_aversion: Risk aversion coefficient (higher = more conservative, default 2.5)
        risk_free_rate: Annual risk-free rate (default 0.04 for 4%)
    """
    tickers: List[str] = Field(min_length=2, max_length=20)
    start_date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')
    end_date: str = Field(pattern=r'^\d{4}-\d{2}-\d{2}$')
    market_caps: Dict[str, float]
    views: Optional[ViewsInput] = None
    risk_aversion: float = Field(default=2.5, ge=0.1, le=100)
    risk_free_rate: float = Field(default=0.04, ge=0.0, le=1.0)

    @field_validator('tickers')
    @classmethod
    def validate_tickers(cls, v):
        validated = []
        for t in v:
            t = t.strip().upper()
            if not _TICKER_PATTERN.match(t):
                raise ValueError(f"Invalid ticker format: '{t}'. Must be 1-10 alphanumeric characters.")
            validated.append(t)
        return validated

    @field_validator('market_caps')
    @classmethod
    def validate_market_caps(cls, v):
        for ticker, cap in v.items():
            if cap <= 0:
                raise ValueError(f"Market cap for {ticker} must be positive, got {cap}")
        return v

    @model_validator(mode='after')
    def validate_dates_and_caps(self):
        # Validate end_date > start_date
        try:
            start_dt = pd.Timestamp(self.start_date)
            end_dt = pd.Timestamp(self.end_date)
            if end_dt <= start_dt:
                raise ValueError(f"end_date ({self.end_date}) must be after start_date ({self.start_date})")
            date_range_days = (end_dt - start_dt).days
            if date_range_days < 365:
                raise ValueError(f"Date range too short ({date_range_days} days). Minimum 1 year required.")
            if date_range_days > 3652:
                raise ValueError(f"Date range too long ({date_range_days} days). Maximum ~10 years allowed.")
        except ValueError:
            raise
        # Validate market_caps keys match tickers
        caps_upper = {k.strip().upper() for k in self.market_caps.keys()}
        tickers_set = set(self.tickers)
        missing = tickers_set - caps_upper
        if missing:
            raise ValueError(f"Missing market_caps for tickers: {list(missing)}")
        return self


class AIExplainRequest(BaseModel):
    analysis_type: Literal["performance", "factor_analysis", "black_litterman"]
    data: dict
    section: str = Field(..., min_length=1, max_length=50)

    @model_validator(mode='after')
    def validate_data_size(self):
        import json
        # Limit serialized data size to prevent abuse (100KB max)
        data_str = json.dumps(self.data, default=str)
        if len(data_str) > 102400:
            raise ValueError(f"Data payload too large ({len(data_str)} bytes). Maximum 100KB allowed.")
        return self


class AIExplainResponse(BaseModel):
    insight: str
    requests_remaining: int
    cached: bool
    tokens_used: int


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
@limiter.limit("100/hour")
async def analyze_portfolio(request: Request, body: PortfolioRequest):
    """Analyze portfolio and return formatted results matching frontend expectations."""
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    # Log incoming request
    logger.info(f"[{request_id}] === NEW /analyze REQUEST ===")
    logger.info(f"[{request_id}] Received request at {datetime.now().isoformat()}")
    logger.info(f"[{request_id}] Portfolio text length: {len(body.portfolioText) if body.portfolioText else 0} chars")
    logger.info(f"[{request_id}] Requested start date: {body.requested_start_date}")

    # Log first 500 chars of portfolio text for debugging (avoid logging huge payloads)
    portfolio_preview = body.portfolioText[:500] if body.portfolioText else "(empty)"
    logger.info(f"[{request_id}] Portfolio preview: {portfolio_preview}")

    try:
        # Parse portfolio from text
        logger.info(f"[{request_id}] Parsing portfolio text...")
        portfolio_df = parse_portfolio_text(body.portfolioText)
        
        tickers = portfolio_df['ticker'].tolist()
        weights = portfolio_df.set_index('ticker')['weight'].to_dict()

        logger.info(f"[{request_id}] Parsed {len(tickers)} tickers: {tickers}")
        logger.info(f"[{request_id}] Weights: {weights}")

        # Filter out cash tickers before fetching prices (cash has no price data)
        from portfolio_tool.analytics import is_cash_ticker
        invested_tickers = [ticker for ticker in tickers if not is_cash_ticker(ticker)]
        logger.info(f"[{request_id}] Invested (non-cash) tickers: {invested_tickers}")
        
        benchmark_ticker = "SPY"
        today = datetime.today()
        
        # Parse user-selected start date if provided
        user_selected_start_date = None
        if body.requested_start_date:
            try:
                user_selected_start_date = pd.Timestamp(body.requested_start_date)
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
        logger.info(f"[{request_id}] Fetching market data from {fetch_start_date} to {fetch_end_date}")
        if invested_tickers:
            all_tickers = invested_tickers + [benchmark_ticker]
            prices_all = get_price_history(all_tickers, fetch_start_date, fetch_end_date)
            prices_portfolio = prices_all[invested_tickers]
            prices_benchmark = prices_all[[benchmark_ticker]]
            logger.info(f"[{request_id}] Fetched {len(prices_all)} days of price data for {len(all_tickers)} tickers")
        else:
            # Portfolio is 100% cash - only fetch benchmark
            prices_all = get_price_history([benchmark_ticker], fetch_start_date, fetch_end_date)
            prices_portfolio = pd.DataFrame()  # Empty DataFrame for 100% cash portfolio
            prices_benchmark = prices_all[[benchmark_ticker]]
            logger.info(f"[{request_id}] 100% cash portfolio - fetched benchmark only")
        
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
        # Track if we had to adjust the user's selected date due to insufficient data
        start_date_adjusted = False
        if user_selected_start_date is not None:
            if user_selected_start_date < common_start_date:
                # User's date is too old - use common_start_date instead
                effective_start_date = common_start_date
                start_date_adjusted = True
            else:
                effective_start_date = user_selected_start_date
        else:
            effective_start_date = common_start_date
        
        # Window prices to [effective_start_date, as_of_date]
        prices_portfolio_windowed = slice_to_effective_window(prices_portfolio, effective_start_date, as_of_date)
        prices_benchmark_windowed = slice_to_effective_window(prices_benchmark, effective_start_date, as_of_date)

        # Log analysis window details
        logger.info(f"[{request_id}] Analysis window computed:")
        logger.info(f"[{request_id}]   - Common start date: {common_start_date}")
        logger.info(f"[{request_id}]   - Effective start date: {effective_start_date}")
        logger.info(f"[{request_id}]   - As-of date: {as_of_date}")
        logger.info(f"[{request_id}]   - Start date adjusted: {start_date_adjusted}")
        if limiting_ticker:
            logger.info(f"[{request_id}]   - Limiting ticker: {limiting_ticker} (started {limiting_start_date})")

        # Calculate period length in fractional years for annualization
        period_days = (as_of_date - effective_start_date).days
        period_years = period_days / 365.25 if period_days > 0 else None
        logger.info(f"[{request_id}] Analysis period: {period_days} days ({period_years:.2f} years)" if period_years else f"[{request_id}] Analysis period: {period_days} days")

        # Get risk-free rate
        risk_free_rate = get_risk_free_rate('^TNX')
        if risk_free_rate is None:
            risk_free_rate = 0.02  # Default 2%
            logger.warning(f"[{request_id}] Could not fetch risk-free rate, using default 2%")
        else:
            logger.info(f"[{request_id}] Risk-free rate: {risk_free_rate:.4f}")
        
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
        # Require at least 60 trading days for reliable CAGR calculation
        summary_cagr_portfolio = None
        summary_cagr_benchmark = None
        
        if period_years and period_years > 0:
            if len(portfolio_daily_windowed) >= 60:
                total_return = (1 + portfolio_daily_windowed).prod() - 1
                summary_cagr_portfolio = safe_float(((1 + total_return) ** (1 / period_years) - 1) * 100)
            
            if len(benchmark_daily_windowed) >= 60:
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
                
                # Add optimal allocation weights if available
                tangency_weights = ef_data.get('tangency_weights')
                if tangency_weights is not None and len(tangency_weights) > 0:
                    # Get asset tickers from the portfolio
                    portfolio_tickers = list(weights.keys())
                    if len(tangency_weights) == len(portfolio_tickers):
                        target_allocation = []
                        for i, ticker in enumerate(portfolio_tickers):
                            target_allocation.append({
                                'ticker': ticker,
                                'currentWeight': safe_float(weights.get(ticker, 0) * 100),  # Current weight as percentage
                                'targetWeight': safe_float(tangency_weights[i] * 100)  # Optimal weight as percentage
                            })
                        efficient_frontier['targetAllocation'] = target_allocation
            
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
        if start_date_adjusted:
            limiting_info = f'Limited by {limiting_ticker}' if limiting_ticker else 'Limited by benchmark'
            warnings.append(
                f'Start date automatically adjusted from {user_selected_start_date.strftime("%Y-%m-%d")} '
                f'to {common_start_date.strftime("%Y-%m-%d")} due to insufficient data availability for all holdings. '
                f'{limiting_info} (earliest available: {common_start_date.strftime("%Y-%m-%d")}).'
            )
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
            'user_selected_start_date': body.requested_start_date if body.requested_start_date else None,
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
                'requestedStartDate': body.requested_start_date if body.requested_start_date else None,
                'effectiveStartDate': effective_start_date.strftime('%Y-%m-%d') if effective_start_date is not None else None,
                'startDateAdjusted': start_date_adjusted,
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

        # Log successful response
        elapsed_time = time.time() - start_time
        logger.info(f"[{request_id}] === RESPONSE READY ===")
        logger.info(f"[{request_id}] Processing time: {elapsed_time:.2f} seconds")
        logger.info(f"[{request_id}] Response contains {len(holdings)} holdings")
        logger.info(f"[{request_id}] Warnings: {warnings if warnings else 'None'}")
        logger.info(f"[{request_id}] Analysis window: {response_data['analysisWindow']['effectiveStartDate']} to {response_data['analysisWindow']['asOfDate']}")
        logger.info(f"[{request_id}] Risk metrics - Sharpe: {risk_metrics.get('sharpeRatio')}, Volatility: {risk_metrics.get('annualVolatility')}")
        logger.info(f"[{request_id}] === END REQUEST [{request_id}] ===")

        return {
            'success': True,
            'data': response_data
        }

    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        error_detail = str(e)

        # Log error details
        logger.error(f"[{request_id}] === ERROR ===")
        logger.error(f"[{request_id}] Error after {elapsed_time:.2f} seconds: {error_detail}")
        logger.error(f"[{request_id}] Error type: {type(e).__name__}")
        logger.error(f"[{request_id}] Stack trace:\n{traceback.format_exc()}")
        logger.error(f"[{request_id}] === END ERROR [{request_id}] ===")

        raise HTTPException(status_code=500, detail=f"Portfolio analysis error: {error_detail}")


@app.post("/factors/analyze")
@limiter.limit("20/hour")
async def analyze_factors_endpoint(request: Request, body: FactorAnalysisRequest):
    """
    Perform Fama-French factor analysis on a single ticker.
    
    This endpoint analyzes how a ticker's or portfolio's returns are explained by systematic
    risk factors (market, size, value, etc.) and calculates alpha (excess return not explained by factors).

    Request body (Single Ticker):
    - ticker: Stock ticker symbol (e.g., "SPY", "AAPL")
    - start_date: Start date in YYYY-MM-DD format
    - end_date: End date in YYYY-MM-DD format
    - factor_model: "3-factor", "5-factor", "4-factor", or "CAPM" (default: "3-factor")
    - frequency: "daily", "weekly", or "monthly" (default: "monthly")
    - risk_free_rate: "1M_TBILL" or "3M_TBILL" (default: "1M_TBILL")

    Request body (Portfolio):
    - tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
    - weights: List of weights for each ticker (must sum to 1.0)
    - start_date, end_date, factor_model, frequency, risk_free_rate: Same as above

    Returns:
    - Complete factor analysis results including coefficients, statistics, and time series data
    """
    try:
        # Validate ticker/portfolio inputs
        has_single_ticker = body.ticker and body.ticker.strip()
        has_portfolio = body.tickers and len(body.tickers) > 0

        if not has_single_ticker and not has_portfolio:
            raise HTTPException(
                status_code=400,
                detail="Must provide either 'ticker' for single ticker analysis or 'tickers' with 'weights' for portfolio analysis"
            )

        if has_single_ticker and has_portfolio:
            raise HTTPException(
                status_code=400,
                detail="Provide either 'ticker' OR 'tickers' with 'weights', not both"
            )

        if has_portfolio:
            if not body.weights or len(body.weights) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Must provide 'weights' when using multiple tickers"
                )
            if len(body.tickers) != len(body.weights):
                raise HTTPException(
                    status_code=400,
                    detail=f"Number of tickers ({len(body.tickers)}) must match number of weights ({len(body.weights)})"
                )
            weights_sum = sum(body.weights)
            if not (0.99 <= weights_sum <= 1.01):
                raise HTTPException(
                    status_code=400,
                    detail=f"Weights must sum to 1.0, got {weights_sum:.4f}"
                )

        # Validate dates
        if not body.start_date or not body.end_date:
            raise HTTPException(
                status_code=400,
                detail="start_date and end_date are required"
            )

        try:
            start_dt = pd.Timestamp(body.start_date)
            end_dt = pd.Timestamp(body.end_date)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}"
            )

        if start_dt >= end_dt:
            raise HTTPException(
                status_code=400,
                detail=f"start_date ({body.start_date}) must be before end_date ({body.end_date})"
            )

        # Validate factor_model
        valid_models = ["3-factor", "5-factor", "4-factor", "CAPM"]
        if body.factor_model not in valid_models:
            raise HTTPException(
                status_code=400,
                detail=f"factor_model must be one of {valid_models}. Got: {body.factor_model}"
            )

        # Validate frequency
        valid_frequencies = ["daily", "weekly", "monthly"]
        if body.frequency not in valid_frequencies:
            raise HTTPException(
                status_code=400,
                detail=f"frequency must be one of {valid_frequencies}. Got: {body.frequency}"
            )

        # Validate risk_free_rate
        valid_rf_rates = ["1M_TBILL", "3M_TBILL"]
        if body.risk_free_rate not in valid_rf_rates:
            raise HTTPException(
                status_code=400,
                detail=f"risk_free_rate must be one of {valid_rf_rates}. Got: {body.risk_free_rate}"
            )

        # Adjust dates to ensure minimum observations
        adjusted_start_date, adjusted_end_date, was_adjusted, adjustment_message = calculate_adjusted_dates(
            body.start_date,
            body.end_date,
            body.frequency
        )

        # Call the factor analysis function with adjusted dates
        if has_single_ticker:
            result = analyze_factors(
                ticker=body.ticker.strip().upper(),
                start_date=adjusted_start_date,
                end_date=adjusted_end_date,
                factor_model=body.factor_model,
                frequency=body.frequency,
                risk_free_rate=body.risk_free_rate
            )
        else:
            result = analyze_factors(
                tickers=[t.strip().upper() for t in body.tickers],
                weights=body.weights,
                start_date=adjusted_start_date,
                end_date=adjusted_end_date,
                factor_model=body.factor_model,
                frequency=body.frequency,
                risk_free_rate=body.risk_free_rate
            )

        # Add date adjustment info to response
        if was_adjusted:
            result['period']['requested_start'] = body.start_date
            result['period']['requested_end'] = body.end_date
            result['period']['was_adjusted'] = True
            result['period']['adjustment_reason'] = adjustment_message
        else:
            result['period']['was_adjusted'] = False

        return {
            "success": True,
            "data": result
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (these are intentional)
        raise
    except ValueError as e:
        # Invalid inputs, insufficient data, or unsupported models
        error_msg = str(e)
        if "ticker" in error_msg.lower() and ("not found" in error_msg.lower() or "no data" in error_msg.lower()):
            raise HTTPException(
                status_code=400,
                detail=f"Ticker '{body.ticker}' not found or no data available. "
                       f"Please verify the ticker symbol is correct and has data for the specified date range."
            )
        elif "insufficient" in error_msg.lower() or "need at least" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {error_msg}. "
                       f"Please use a longer date range or a different frequency."
            )
        elif "date" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Date error: {error_msg}. "
                       f"Please check that dates are in YYYY-MM-DD format and start_date < end_date."
            )
        elif "model" in error_msg.lower() or "factor" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Factor model error: {error_msg}. "
                       f"Note: 4-factor and 5-factor models require additional data sources."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request: {error_msg}. "
                       f"Please check your inputs and try again."
            )
    except ConnectionError as e:
        # Network errors fetching Fama-French data
        error_msg = str(e)
        raise HTTPException(
            status_code=503,
            detail=f"Service temporarily unavailable: Failed to fetch Fama-French factor data. "
                  f"{error_msg}. "
                  f"Please check your internet connection and try again later."
        )
    except ImportError as e:
        # Missing dependencies
        error_msg = str(e)
        raise HTTPException(
            status_code=500,
            detail=f"Server configuration error: {error_msg}. "
                  f"Please contact the administrator."
        )
    except Exception as e:
        # Unexpected errors
        import traceback
        error_detail = str(e)
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Factor analysis error: {error_detail}. "
                  f"Please check your inputs and try again. If the problem persists, contact support."
        )


@app.post("/black-litterman/optimize")
@limiter.limit("10/hour")
async def black_litterman_optimize(request: Request, body: BlackLittermanRequest):
    """
    Perform Black-Litterman portfolio optimization.

    The Black-Litterman model combines market equilibrium returns with investor views
    to produce optimal portfolio weights. This is a powerful tool for portfolio construction
    that allows investors to express their beliefs about expected returns while maintaining
    a well-diversified portfolio.

    Key Concepts:
    - Prior Returns: Market-implied equilibrium returns derived from market cap weights
    - Views: Your beliefs about expected returns (absolute or relative)
    - Posterior Returns: Blended returns after incorporating your views
    - Optimal Weights: Portfolio weights that maximize Sharpe ratio given posterior returns

    Request body:
    - tickers: List of ticker symbols (e.g., ["AAPL", "MSFT", "GOOGL"])
    - start_date: Start date for historical data (YYYY-MM-DD)
    - end_date: End date for historical data (YYYY-MM-DD)
    - market_caps: Market capitalizations for each ticker
    - views: Your investment views (optional)
        - absolute: Views about single assets (e.g., "AAPL will return 15%")
        - relative: Views about relative performance (e.g., "GOOGL will outperform MSFT by 3%")
    - risk_aversion: Risk aversion coefficient (default 2.5, higher = more conservative)
    - risk_free_rate: Annual risk-free rate (default 0.04 for 4%)

    Returns:
    - optimal_weights: Recommended portfolio weights
    - expected_return: Expected annual return of optimal portfolio
    - expected_risk: Expected annual volatility of optimal portfolio
    - sharpe_ratio: Sharpe ratio of optimal portfolio
    - prior_returns: Market equilibrium returns before views
    - posterior_returns: Expected returns after incorporating views
    - market_cap_weights: Starting weights based on market capitalization
    - comparison: Side-by-side comparison of market-weighted vs Black-Litterman portfolio
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()

    logger.info(f"[{request_id}] === NEW /black-litterman/optimize REQUEST ===")
    logger.info(f"[{request_id}] Received request at {datetime.now().isoformat()}")
    logger.info(f"[{request_id}] Tickers: {body.tickers}")
    logger.info(f"[{request_id}] Date range: {body.start_date} to {body.end_date}")
    logger.info(f"[{request_id}] Risk aversion: {body.risk_aversion}")
    logger.info(f"[{request_id}] Risk-free rate: {body.risk_free_rate}")

    try:
        # Import the black-litterman module
        from portfolio_tool.black_litterman import run_black_litterman_optimization

        # Validate tickers
        if not body.tickers or len(body.tickers) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 tickers are required for portfolio optimization"
            )

        # Normalize tickers
        tickers = [t.strip().upper() for t in body.tickers]
        logger.info(f"[{request_id}] Normalized tickers: {tickers}")

        # Validate dates
        try:
            start_dt = pd.Timestamp(body.start_date)
            end_dt = pd.Timestamp(body.end_date)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}"
            )

        if start_dt >= end_dt:
            raise HTTPException(
                status_code=400,
                detail=f"start_date ({body.start_date}) must be before end_date ({body.end_date})"
            )

        # Check minimum date range (1 year)
        date_range_days = (end_dt - start_dt).days
        if date_range_days < 365:
            raise HTTPException(
                status_code=400,
                detail=f"Date range too short ({date_range_days} days). Minimum 1 year (365 days) required for reliable optimization."
            )

        # Validate market caps
        if not body.market_caps:
            raise HTTPException(
                status_code=400,
                detail="market_caps dictionary is required"
            )

        # Normalize market_caps keys to uppercase
        market_caps = {k.strip().upper(): v for k, v in body.market_caps.items()}

        # Check that all tickers have market caps
        missing_caps = set(tickers) - set(market_caps.keys())
        if missing_caps:
            logger.warning(f"[{request_id}] Missing market caps for: {missing_caps}")
            raise HTTPException(
                status_code=400,
                detail=f"Missing market_caps for tickers: {list(missing_caps)}"
            )

        # Validate market cap values
        for ticker, cap in market_caps.items():
            if cap <= 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Market cap for {ticker} must be positive, got {cap}"
                )

        # Only keep market caps for requested tickers
        market_caps = {t: market_caps[t] for t in tickers}
        logger.info(f"[{request_id}] Market caps: {market_caps}")

        # Process views
        views = {'absolute': [], 'relative': []}
        if body.views:
            # Process absolute views
            if body.views.absolute:
                for view in body.views.absolute:
                    asset = view.asset.strip().upper()
                    if asset not in tickers:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Asset '{asset}' in absolute view not found in tickers list"
                        )
                    if not (0.0 <= view.confidence <= 1.0):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Confidence for {asset} must be between 0 and 1, got {view.confidence}"
                        )
                    if not (-0.50 <= view.return_ <= 0.50):
                        logger.warning(f"[{request_id}] Unusual expected return for {asset}: {view.return_}")

                    views['absolute'].append({
                        'asset': asset,
                        'return': view.return_,
                        'confidence': view.confidence
                    })

            # Process relative views
            if body.views.relative:
                for view in body.views.relative:
                    asset1 = view.asset1.strip().upper()
                    asset2 = view.asset2.strip().upper()

                    if asset1 not in tickers:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Asset '{asset1}' in relative view not found in tickers list"
                        )
                    if asset2 not in tickers:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Asset '{asset2}' in relative view not found in tickers list"
                        )
                    if asset1 == asset2:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Relative view assets must be different, got {asset1} vs {asset2}"
                        )
                    if not (0.0 <= view.confidence <= 1.0):
                        raise HTTPException(
                            status_code=400,
                            detail=f"Confidence for relative view must be between 0 and 1, got {view.confidence}"
                        )

                    views['relative'].append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'outperformance': view.outperformance,
                        'confidence': view.confidence
                    })

        logger.info(f"[{request_id}] Views: {len(views['absolute'])} absolute, {len(views['relative'])} relative")

        # Validate risk_aversion
        if body.risk_aversion <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"risk_aversion must be positive, got {body.risk_aversion}"
            )

        # Validate risk_free_rate
        if not (0.0 <= body.risk_free_rate <= 0.20):
            logger.warning(f"[{request_id}] Unusual risk-free rate: {body.risk_free_rate}")

        # Run Black-Litterman optimization
        logger.info(f"[{request_id}] Starting Black-Litterman optimization...")

        result = run_black_litterman_optimization(
            tickers=tickers,
            start_date=body.start_date,
            end_date=body.end_date,
            market_caps=market_caps,
            views=views,
            risk_aversion=body.risk_aversion,
            risk_free_rate=body.risk_free_rate,
            tau=0.05  # Standard uncertainty scalar
        )

        # Log successful response
        elapsed_time = time.time() - start_time
        logger.info(f"[{request_id}] === RESPONSE READY ===")
        logger.info(f"[{request_id}] Processing time: {elapsed_time:.2f} seconds")
        logger.info(f"[{request_id}] Optimal Sharpe ratio: {result['sharpe_ratio']:.4f}")
        logger.info(f"[{request_id}] === END REQUEST [{request_id}] ===")

        return {
            "success": True,
            "data": result
        }

    except HTTPException:
        raise
    except ValueError as e:
        error_msg = str(e)
        elapsed_time = time.time() - start_time
        logger.error(f"[{request_id}] ValueError after {elapsed_time:.2f}s: {error_msg}")

        if "no data" in error_msg.lower() or "not found" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Data error: {error_msg}. Please verify ticker symbols and date range."
            )
        elif "insufficient" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient data: {error_msg}. Please use a longer date range."
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid request: {error_msg}"
            )
    except np.linalg.LinAlgError as e:
        elapsed_time = time.time() - start_time
        logger.error(f"[{request_id}] Matrix error after {elapsed_time:.2f}s: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail="Matrix computation error. This can happen with highly correlated assets. "
                   "Try using fewer assets or a longer date range."
        )
    except Exception as e:
        import traceback
        elapsed_time = time.time() - start_time
        error_detail = str(e)

        logger.error(f"[{request_id}] === ERROR ===")
        logger.error(f"[{request_id}] Error after {elapsed_time:.2f} seconds: {error_detail}")
        logger.error(f"[{request_id}] Error type: {type(e).__name__}")
        logger.error(f"[{request_id}] Stack trace:\n{traceback.format_exc()}")
        logger.error(f"[{request_id}] === END ERROR [{request_id}] ===")

        raise HTTPException(
            status_code=500,
            detail=f"Black-Litterman optimization error: {error_detail}"
        )


@app.post("/ai/explain", response_model=AIExplainResponse)
@limiter.limit("20/hour")
async def explain_analysis(request: Request, body: AIExplainRequest):
    """
    Generate AI-powered insights for portfolio analysis results.

    Accepts analysis data (performance, factor analysis, or Black-Litterman)
    and returns a natural-language explanation powered by GPT-4o-mini.
    Results are cached for 5 minutes to reduce API costs.
    """
    # Placeholder user_id until auth is integrated
    user_id = "test_user"

    result = await generate_insight(
        analysis_type=body.analysis_type,
        data=body.data,
        section=body.section,
        user_id=user_id,
    )

    return result


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "portfolio-analysis-api",
        "version": APP_VERSION,
        "environment": ENVIRONMENT,
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
    }


if __name__ == "__main__":
    import uvicorn
    # Note: For auto-reload during development, use from command line:
    # uvicorn api_server:app --host 0.0.0.0 --port 8001 --reload
    # Running directly with reload=False avoids the warning
    uvicorn.run("api_server:app", host="0.0.0.0", port=8001, reload=False)

