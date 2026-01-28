"""Fama-French factor data fetcher from Kenneth French's data library."""

import pandas as pd
import numpy as np
from functools import lru_cache
from datetime import datetime
import warnings

try:
    import pandas_datareader.data as web
    PANDAS_DATAREADER_AVAILABLE = True
except ImportError as e:
    PANDAS_DATAREADER_AVAILABLE = False
    import_error_msg = str(e)
    warnings.warn(
        f"pandas-datareader not available. Fama-French factor analysis will not work. "
        f"Import error: {import_error_msg}. "
        f"Install with: pip install pandas-datareader>=0.10.0"
    )
except Exception as e:
    # Catch other potential errors (e.g., missing dependencies)
    PANDAS_DATAREADER_AVAILABLE = False
    import_error_msg = str(e)
    warnings.warn(
        f"pandas-datareader import failed. Fama-French factor analysis will not work. "
        f"Error: {import_error_msg}. "
        f"Install with: pip install pandas-datareader>=0.10.0"
    )


@lru_cache(maxsize=32)
def get_factors(start_date, end_date, _version=2, include_5_factor=True, include_momentum=True, use_3month_tbill=True):
    """
    Fetch Fama-French factor data from Kenneth French's data library.

    The Fama-French factors include:
    - Mkt-RF: Market excess return (market return minus risk-free rate)
    - SMB: Small Minus Big (size factor - small cap outperformance)
    - HML: High Minus Low (value factor - value stock outperformance)
    - RF: Risk-free rate (1-month T-Bill from Kenneth French, or 3-month T-Bill from FRED)
    - RMW: Robust Minus Weak (profitability factor) [5-factor]
    - CMA: Conservative Minus Aggressive (investment factor) [5-factor]
    - MOM: Momentum factor [Carhart 4-factor]

    IMPORTANT: Risk-Free Rate Source
    - Kenneth French's original RF column uses 1-month Treasury Bills
    - Portfolio Visualizer uses 3-month Treasury Bills from FRED
    - By default (use_3month_tbill=True), we fetch 3-month T-Bills from FRED to match
      Portfolio Visualizer's methodology
    - Set use_3month_tbill=False to use Kenneth French's original 1-month T-Bill rate

    Source: Kenneth French Data Library
    https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

    Args:
        start_date: Start date as string (YYYY-MM-DD) or pandas Timestamp
        end_date: End date as string (YYYY-MM-DD) or pandas Timestamp
        _version: Internal version parameter for cache invalidation (don't pass this)
        include_5_factor: If True, fetch RMW and CMA factors (default: True)
        include_momentum: If True, fetch MOM factor (default: True)

    Returns:
        pandas.DataFrame: DataFrame with DatetimeIndex and columns:
            - 'Mkt-RF': Market excess return (decimal, e.g., 0.01 for 1%)
            - 'SMB': Small Minus Big factor (decimal)
            - 'HML': High Minus Low factor (decimal)
            - 'RF': Risk-free rate (decimal)
            - 'RMW': Robust Minus Weak factor (decimal) [if include_5_factor]
            - 'CMA': Conservative Minus Aggressive factor (decimal) [if include_5_factor]
            - 'MOM': Momentum factor (decimal) [if include_momentum]

        Returns empty DataFrame if data cannot be fetched.

    Raises:
        ValueError: If dates are invalid or start_date > end_date
        ConnectionError: If network request fails (wrapped in error handling)

    Example:
        >>> factors = get_factors('2020-01-01', '2023-12-31')
        >>> print(factors.head())
                        Mkt-RF     SMB     HML      RF     RMW     CMA     MOM
        Date
        2020-01-02     0.0045   0.0021  -0.0012  0.0001  0.0010 -0.0005  0.0030
        ...
    """
    if not PANDAS_DATAREADER_AVAILABLE:
        raise ImportError(
            "pandas-datareader is required for Fama-French factor analysis. "
            "Install with: pip install pandas-datareader>=0.10.0"
        )
    
    # Convert string dates to pandas Timestamps
    try:
        if isinstance(start_date, str):
            start_date = pd.Timestamp(start_date)
        elif not isinstance(start_date, pd.Timestamp):
            start_date = pd.Timestamp(start_date)
        
        if isinstance(end_date, str):
            end_date = pd.Timestamp(end_date)
        elif not isinstance(end_date, pd.Timestamp):
            end_date = pd.Timestamp(end_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid date format. Use YYYY-MM-DD string or pandas Timestamp. Error: {e}")
    
    # Validate date range
    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")
    
    # Check if dates are too far in the future
    today = pd.Timestamp.today()
    if end_date > today:
        warnings.warn(
            f"end_date ({end_date}) is in the future. Clamping to today ({today})."
        )
        end_date = today
    
    # Kenneth French data library identifiers
    # 'F-F_Research_Data_Factors' is the 3-factor monthly frequency dataset
    # 'F-F_Research_Data_5_Factors_2x3' is the 5-factor monthly dataset
    # 'F-F_Momentum_Factor' is the momentum factor dataset

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress pandas_datareader warnings

            # Decide which base dataset to fetch
            if include_5_factor:
                # Fetch 5-factor data (includes Mkt-RF, SMB, HML, RMW, CMA, RF)
                factor_name = 'F-F_Research_Data_5_Factors_2x3'
            else:
                # Fetch 3-factor data (includes Mkt-RF, SMB, HML, RF)
                factor_name = 'F-F_Research_Data_Factors'

            # Fetch the main factor data
            factors_raw = web.DataReader(
                factor_name,
                'famafrench',
                start=start_date,
                end=end_date
            )
        
        # pandas_datareader returns a dict with different frequencies
        # For monthly data, key 0 typically has the monthly data
        if isinstance(factors_raw, dict):
            # Get monthly data (first dataframe in the dict, typically index 0)
            if 0 in factors_raw:
                factors_df = factors_raw[0]
            elif len(factors_raw) > 0:
                # Use first available dataframe
                factors_df = list(factors_raw.values())[0]
            else:
                raise ValueError("No data found in factors_raw dict")
        else:
            factors_df = factors_raw
        
        # Ensure we have a DataFrame
        if not isinstance(factors_df, pd.DataFrame):
            raise ValueError(f"Unexpected data format from pandas_datareader: {type(factors_df)}")

        # CRITICAL: Convert PeriodIndex to DatetimeIndex immediately
        # Fama-French data comes with PeriodIndex (e.g., Period('2021-02', 'M'))
        # We need DatetimeIndex for proper alignment with ticker returns
        if isinstance(factors_df.index, pd.PeriodIndex):
            factors_df.index = factors_df.index.to_timestamp()

        # Verify we got monthly data (not daily)
        # Monthly data should have fewer than 200 rows for a 5-year period
        # Daily data would have 1000+ rows
        if len(factors_df) > 200:
            # This looks like daily data, which means the library structure changed
            # Try to resample to monthly
            warnings.warn(
                f"Got {len(factors_df)} rows, appears to be daily data. Resampling to monthly.",
                UserWarning
            )
            # Ensure index is DatetimeIndex before resampling
            if not isinstance(factors_df.index, pd.DatetimeIndex):
                # Convert PeriodIndex to DatetimeIndex for monthly data
                if isinstance(factors_df.index, pd.PeriodIndex):
                    factors_df.index = factors_df.index.to_timestamp()
                else:
                    factors_df.index = pd.to_datetime(factors_df.index)
            # Resample to month-end
            factors_df = factors_df.resample('ME').last()
        
        # Kenneth French data is in percentages, convert to decimals
        # Base columns: Mkt-RF, SMB, HML, RF (always required)
        # 5-factor adds: RMW, CMA
        # Momentum adds: MOM (fetched separately)
        base_columns = ['Mkt-RF', 'SMB', 'HML', 'RF']
        if include_5_factor:
            base_columns.extend(['RMW', 'CMA'])

        # Check which columns are available
        available_columns = [col for col in factors_df.columns if col in base_columns]

        # Case-insensitive column matching
        column_map = {}
        for req_col in base_columns:
            if req_col not in factors_df.columns:
                for df_col in factors_df.columns:
                    if req_col.lower() == df_col.lower():
                        column_map[df_col] = req_col
                        break

        if column_map:
            factors_df = factors_df.rename(columns=column_map)

        # Select only the columns that exist
        available_base_columns = [col for col in base_columns if col in factors_df.columns]
        factors_df = factors_df[available_base_columns].copy()

        # Convert from percentages to decimals
        # Fama-French data is in percentage points (1.5 = 1.5%, not 0.015)
        # Divide by 100 to convert to decimal form (0.015 = 1.5%) to match yfinance returns
        factors_df = factors_df / 100.0

        # Fetch momentum factor if requested
        if include_momentum:
            try:
                mom_raw = web.DataReader(
                    'F-F_Momentum_Factor',
                    'famafrench',
                    start=start_date,
                    end=end_date
                )

                # Extract momentum data
                if isinstance(mom_raw, dict):
                    mom_df = mom_raw[0] if 0 in mom_raw else list(mom_raw.values())[0]
                else:
                    mom_df = mom_raw

                # Convert PeriodIndex to DatetimeIndex if needed
                if isinstance(mom_df.index, pd.PeriodIndex):
                    mom_df.index = mom_df.index.to_timestamp()

                # Rename column to 'MOM' if needed
                if 'Mom' in mom_df.columns:
                    mom_df = mom_df.rename(columns={'Mom': 'MOM'})
                elif 'MOM' not in mom_df.columns and len(mom_df.columns) > 0:
                    # Use first column as momentum
                    mom_df = mom_df.rename(columns={mom_df.columns[0]: 'MOM'})

                # Convert to decimal
                mom_df = mom_df[['MOM']] / 100.0

                # Merge with main factors
                factors_df = factors_df.join(mom_df['MOM'], how='left')

            except Exception as e:
                warnings.warn(f"Could not fetch momentum factor: {e}. Proceeding without MOM.")

        # Fetch 3-month T-Bill from FRED if requested
        # Kenneth French's RF column uses 1-month T-Bills
        # Portfolio Visualizer uses 3-month T-Bills (DTB3 from FRED)
        if use_3month_tbill and 'RF' in factors_df.columns:
            try:
                # Fetch 3-month T-Bill rate from FRED
                # DTB3 = 3-Month Treasury Bill: Secondary Market Rate
                # Data is daily, in percent (e.g., 4.5 for 4.5%)
                tbill_3m = web.DataReader('DTB3', 'fred', start=start_date, end=end_date)

                # Convert PeriodIndex to DatetimeIndex if needed
                if isinstance(tbill_3m.index, pd.PeriodIndex):
                    tbill_3m.index = tbill_3m.index.to_timestamp()

                # DTB3 is daily, resample to monthly (take last value of each month)
                # Use 'ME' for month-end
                tbill_3m_monthly = tbill_3m.resample('ME').last()

                # Convert from annual percentage to monthly decimal
                # DTB3 is annualized rate (e.g., 4.5%), convert to monthly decimal
                # Formula: (1 + annual_rate/100)^(1/12) - 1
                tbill_3m_monthly_decimal = ((1 + tbill_3m_monthly / 100) ** (1/12)) - 1

                # Rename column to RF
                tbill_3m_monthly_decimal.columns = ['RF_3M']

                # Replace Kenneth French's 1-month RF with 3-month RF
                # Join and fill missing values with original RF if needed
                factors_df = factors_df.join(tbill_3m_monthly_decimal['RF_3M'], how='left')

                # Replace RF with RF_3M where available, keep original RF as fallback
                factors_df['RF_1M'] = factors_df['RF']  # Save original 1-month rate
                factors_df['RF'] = factors_df['RF_3M'].fillna(factors_df['RF_1M'])
                factors_df = factors_df.drop(columns=['RF_3M'])  # Clean up

                warnings.warn(
                    "Using 3-month T-Bill from FRED (DTB3) as risk-free rate to match Portfolio Visualizer. "
                    "Original 1-month T-Bill from Kenneth French saved as RF_1M.",
                    UserWarning
                )

            except Exception as e:
                warnings.warn(
                    f"Could not fetch 3-month T-Bill from FRED: {e}. "
                    f"Using Kenneth French's 1-month T-Bill instead.",
                    UserWarning
                )

        # Ensure index is DatetimeIndex
        if not isinstance(factors_df.index, pd.DatetimeIndex):
            try:
                # Convert PeriodIndex to DatetimeIndex for monthly data
                if isinstance(factors_df.index, pd.PeriodIndex):
                    factors_df.index = factors_df.index.to_timestamp()
                else:
                    factors_df.index = pd.to_datetime(factors_df.index)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Could not convert index to DatetimeIndex: {e}") from e
        
        # Filter to requested date range
        factors_df = factors_df.loc[start_date:end_date].copy()
        
        # Remove any rows with all NaN values
        factors_df = factors_df.dropna(how='all')
        
        # Sort by date
        factors_df = factors_df.sort_index()
        
        return factors_df
        
    except Exception as e:
        error_msg = str(e)
        
        # Provide helpful error messages for common issues
        if "Connection" in error_msg or "timeout" in error_msg.lower() or "network" in error_msg.lower():
            raise ConnectionError(
                f"Failed to fetch Fama-French data due to network error: {error_msg}. "
                "Please check your internet connection and try again."
            ) from e
        elif "404" in error_msg or "not found" in error_msg.lower():
            raise ValueError(
                f"Fama-French data not found for the requested date range. "
                f"Error: {error_msg}. "
                f"Kenneth French's data library may not have data for dates {start_date} to {end_date}."
            ) from e
        else:
            raise RuntimeError(
                f"Error fetching Fama-French data: {error_msg}. "
                f"Date range: {start_date} to {end_date}"
            ) from e

