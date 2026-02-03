# Fama-French Factor Analysis Backend

## Project Overview

Backend API for portfolio/fund analysis using Fama-French factor models.
Users can input any ticker (stocks, ETFs, mutual funds) to understand what drives their returns.

**This is the BACKEND only. The frontend is built separately in Lovable.**

## Full Stack Architecture

```
LOVABLE FRONTEND (React/Vite)
    ↓
  Supabase Edge Function (proxy)
    ↓
  THIS BACKEND (FastAPI on Railway)
    ↓
  Yahoo Finance + Fama-French Data
```

### Frontend (Lovable - SEPARATE PROJECT)

- Built with Lovable (React/Vite)
- Uses Supabase for auth/database
- Supabase Project ID: xiiysrejbzrvhsdmgjds
- Supabase URL: https://xiiysrejbzrvhsdmgjds.supabase.co
- Calls this backend via: https://portfolio-backend-production-701d.up.railway.app

### Backend (This Repository - Railway)

- GitHub: https://github.com/elis340/portfolio-backend
- Railway URL: https://portfolio-backend-production-701d.up.railway.app
- Main Endpoint: POST /analyze
- Docs: https://portfolio-backend-production-701d.up.railway.app/docs

### Important: Frontend-Backend Communication

1. Frontend sends requests to Supabase Edge Function
2. Supabase Edge Function proxies to Railway backend with retry logic
3. Backend responds with factor analysis data
4. Frontend displays results to user

**DO NOT modify frontend code in this repo - it's in a separate Lovable project.**

## Architecture (Backend Flow)

```
User Request (from Lovable frontend)
    ↓
Supabase Edge Function (proxy with retry)
    ↓
Railway Backend (FastAPI) - THIS REPO
    ↓
1. Fetch ticker returns (Yahoo Finance)
2. Fetch Fama-French factors (Kenneth French Library)
3. Align data (normalize to month-end)
4. Run OLS regression
5. Return results
    ↓
Back to Lovable frontend for display
```

## Key Files

- `api_server.py` - Main FastAPI server with all endpoints
- `portfolio_tool/factor_analysis.py` - Core factor regression analysis logic
- `portfolio_tool/fama_french.py` - Fama-French factor data fetcher
- `portfolio_tool/black_litterman.py` - Black-Litterman portfolio optimization
- `requirements.txt` - Python dependencies

## API Endpoints

### POST /analyze

Expected request from frontend:

```json
{
  "portfolioText": "ticker data...",
  "requested_start_date": "2020-01-01"  // optional
}
```

Response format:

```json
{
  "success": true,
  "data": {
    "model": "Fama-French 3-Factor",
    "coefficients": {
      "market": {"loading": 1.0, "significant": true},
      "size": {"loading": 0.0, "significant": false},
      "value": {"loading": 0.0, "significant": false}
    },
    "statistics": {
      "r_squared": 0.995,
      "alpha_annualized": 0.0015,
      "alpha_significant": false
    }
  }
}
```

### POST /factors/analyze

Perform Fama-French factor analysis on a single ticker or portfolio.

**Single Ticker Analysis:**

```json
{
  "ticker": "SPY",
  "start_date": "2020-01-01",
  "end_date": "2025-01-01",
  "factor_model": "3-factor",
  "frequency": "monthly",
  "risk_free_rate": "1M_TBILL"
}
```

**Portfolio Analysis (Multiple Tickers):**

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL"],
  "weights": [0.4, 0.35, 0.25],
  "start_date": "2020-01-01",
  "end_date": "2025-01-01",
  "factor_model": "5-factor",
  "frequency": "monthly",
  "risk_free_rate": "1M_TBILL"
}
```

**Factor Models:**
- `"CAPM"` - Capital Asset Pricing Model (Market only)
- `"3-factor"` - Fama-French 3-Factor (Market, Size, Value)
- `"4-factor"` - Carhart 4-Factor (Market, Size, Value, Momentum)
- `"5-factor"` - Fama-French 5-Factor (Market, Size, Value, Profitability, Investment)

**Frequencies:**
- `"daily"` - Daily returns
- `"weekly"` - Weekly returns
- `"monthly"` - Monthly returns (recommended)

**Risk-Free Rates:**
- `"1M_TBILL"` - 1-Month Treasury Bill (from Fama-French data)
- `"3M_TBILL"` - 3-Month Treasury Bill

### POST /black-litterman/optimize

Perform Black-Litterman portfolio optimization. Combines market equilibrium returns
with investor views to produce optimal portfolio weights.

**Request:**

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN"],
  "start_date": "2020-01-01",
  "end_date": "2025-01-27",
  "market_caps": {
    "AAPL": 2800000000000,
    "MSFT": 2500000000000,
    "GOOGL": 1700000000000,
    "AMZN": 1500000000000
  },
  "views": {
    "absolute": [
      {"asset": "AAPL", "return": 0.15, "confidence": 0.7}
    ],
    "relative": [
      {"asset1": "GOOGL", "asset2": "MSFT", "outperformance": 0.03, "confidence": 0.5}
    ]
  },
  "risk_aversion": 2.5,
  "risk_free_rate": 0.04
}
```

**Parameters:**
- `tickers`: List of ticker symbols (minimum 2)
- `start_date`, `end_date`: Date range for historical data (minimum 1 year)
- `market_caps`: Market capitalizations in dollars for each ticker
- `views`: Investor views (optional)
  - `absolute`: Belief about a single asset's return (e.g., "AAPL will return 15%")
  - `relative`: Belief about relative performance (e.g., "GOOGL will beat MSFT by 3%")
- `risk_aversion`: Risk aversion coefficient (default 2.5, higher = more conservative)
- `risk_free_rate`: Annual risk-free rate (default 0.04)

**Response:**

```json
{
  "success": true,
  "data": {
    "optimal_weights": {"AAPL": 0.35, "MSFT": 0.25, "GOOGL": 0.20, "AMZN": 0.20},
    "expected_return": 0.108,
    "expected_risk": 0.165,
    "sharpe_ratio": 0.41,
    "prior_returns": {"AAPL": 0.085, "MSFT": 0.082, "GOOGL": 0.091, "AMZN": 0.098},
    "posterior_returns": {"AAPL": 0.12, "MSFT": 0.085, "GOOGL": 0.105, "AMZN": 0.098},
    "market_cap_weights": {"AAPL": 0.34, "MSFT": 0.30, "GOOGL": 0.21, "AMZN": 0.18},
    "comparison": {
      "market_weighted": {"return": 0.091, "risk": 0.152, "sharpe": 0.36},
      "black_litterman": {"return": 0.108, "risk": 0.165, "sharpe": 0.41}
    }
  }
}
```

**Key Concepts:**
- **Prior Returns**: Market-implied equilibrium returns (what the market "expects")
- **Posterior Returns**: Adjusted returns after incorporating your views
- **Confidence**: How strongly you believe in your view (0 to 1)

**Response includes:**
- `coefficients` - Factor loadings with t-stats and p-values
- `statistics` - R², adjusted R², alpha (annualized), significance
- `regression_stats` - F-statistic and p-value
- `factor_premiums` - Average factor premiums in basis points
- `return_attribution` - Total return, factor contributions, risk decomposition
- `time_series` - Monthly actual vs predicted returns
- `portfolio_composition` - (Portfolio only) List of tickers and weights

### Error Handling

- Frontend has retry logic (3 attempts, exponential backoff)
- Returns 400 for invalid requests
- Returns 500 for server errors
- Frontend handles errors gracefully and shows user-friendly messages

## Critical Technical Details (MUST REMEMBER)

### Data Processing

- **Fama-French Data Format:** Data comes as percentages → divide by 100
- **Date Alignment:** Monthly data uses PeriodIndex → convert with `.to_timestamp()` and normalize to month-end before joining
- **Risk-Free Rate:** Must normalize RF index separately to avoid NaN values
- **Pandas Frequency:** Use 'ME' instead of deprecated 'M' for month-end

### API Design

- CORS is enabled for Supabase domain
- Requests come through Supabase Edge Function (not directly from browser)
- Frontend handles authentication (Supabase), backend handles computation
- No authentication required on backend (Supabase handles this)

### Issues Already Resolved ✅

- `distutils` missing → Added setuptools for Python 3.12+
- pandas 3.0 incompatibility → Pinned to pandas 2.x
- Daily vs monthly data → Now fetching monthly data correctly
- Date misalignment → Normalize both ticker returns and FF factors to month-end
- Risk-free rate NaN → Normalize RF index before reindexing
- Pandas frequency deprecation → Use 'ME' instead of 'M'

## Deployment

### Railway (Backend)

- **Platform:** Railway
- **URL:** https://portfolio-backend-production-701d.up.railway.app
- **Auto-deploy:** Pushes to main branch trigger automatic deployment
- **Build:** Automatically runs `pip install -r requirements.txt`
- **Start:** Runs FastAPI server
- **Environment Variables:** Set in Railway dashboard (if any)

### Lovable (Frontend)

- Separate deployment - not managed in this repo
- Environment Variables:
  - `VITE_SUPABASE_PROJECT_ID`
  - `VITE_SUPABASE_PUBLISHABLE_KEY`
  - `VITE_SUPABASE_URL`
  - `VITE_PORTFOLIO_ANALYSIS_API_URL` (points to Railway backend)

### Important Deployment Notes

- Changes to THIS repo only affect the backend
- Frontend changes are made in Lovable (separate project)
- If you change the Railway URL, must update `VITE_PORTFOLIO_ANALYSIS_API_URL` in Lovable
- Test backend independently using `/docs` endpoint before testing with frontend

## Testing

### Backend Testing (This Repo)

Test with these verified tickers:

- **SPY:** S&P 500 ETF (R²=99.5%, Beta=1.0, Alpha≈0%)
- **FMAGX:** Active fund (R²=98.9%, Beta=1.1, Alpha≈0%)
- **BRK-B:** Berkshire (R²=95.5%, Beta=0.97, Alpha=2.4%)

### Testing Methods

1. **Direct API Test:** Use Railway's `/docs` endpoint (FastAPI Swagger UI)
2. **Curl Test:**
   ```bash
   curl -X POST https://portfolio-backend-production-701d.up.railway.app/analyze \
     -H "Content-Type: application/json" \
     -d '{"portfolioText": "SPY", "requested_start_date": "2020-01-01"}'
   ```
3. **Frontend Test:** Use Lovable frontend to test end-to-end flow

### When to Test What

- Backend code changes → Test with `/docs` or curl first
- API contract changes → Update frontend accordingly
- New endpoints → Document in Railway `/docs` (FastAPI auto-generates)

## Common Commands

### Local Development

```bash
# Run locally
uvicorn api_server:app --reload

# Run tests
pytest

# Install dependencies
pip install -r requirements.txt
```

### Deployment

```bash
# Deploy to Railway (automatic on push to main)
git push origin main

# Create feature branch
git checkout -b feature-name

# Push feature branch (won't deploy to Railway)
git push origin feature-name
```

### Claude Code Workflow

```bash
# Start Claude Code
claude

# Common prompts:
"Create a branch for this feature"
"Test the /analyze endpoint with SPY"
"Commit and push to main"
"Show me the Railway deployment logs"
```

## Important Constraints

### What to Change in This Repo

✅ Backend logic (factor analysis, data processing)
✅ API endpoints and responses
✅ Error handling
✅ Dependencies (requirements.txt)
✅ Documentation

### What NOT to Change in This Repo

❌ Frontend code (it's in Lovable)
❌ Supabase configuration (managed in Lovable)
❌ Frontend environment variables (managed in Lovable)
❌ UI/UX (that's frontend)

### When Making API Changes

If you change the API contract (request/response format):

1. Update this backend first
2. Test with `/docs` or curl
3. Deploy to Railway
4. Update Lovable frontend to match new contract
5. Test end-to-end

## Troubleshooting

### Backend Issues

- Check Railway logs for errors
- Test endpoint at `/docs`
- Verify pandas/numpy versions in requirements.txt

### Frontend Can't Reach Backend

- Check Railway is deployed and running
- Verify URL in Lovable: `VITE_PORTFOLIO_ANALYSIS_API_URL`
- Check CORS headers in backend
- Check Supabase Edge Function logs

### Data/Calculation Issues

- Test with known tickers (SPY, FMAGX, BRK-B)
- Check date alignment (month-end normalization)
- Verify Fama-French data download
- Check for NaN values in regression

## Links & Resources

### This Project

- GitHub: https://github.com/elis340/portfolio-backend
- Railway: https://portfolio-backend-production-701d.up.railway.app
- API Docs: https://portfolio-backend-production-701d.up.railway.app/docs

### Related Projects

- Frontend: Built in Lovable (separate project)
- Supabase Project: xiiysrejbzrvhsdmgjds

### External Resources

- Fama-French Data: [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html)
- Yahoo Finance: [yfinance Python package](https://github.com/ranaroussi/yfinance)
