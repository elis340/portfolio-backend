"""Fama-French factor data fetcher from Kenneth French's data library."""

import pandas as pd
import numpy as np
from functools import lru_cache
from datetime import datetime
import warnings

try:
    import pandas_datareader.data as web
    PANDAS_DATAREADER_AVAILABLE = True
except ImportError:
    PANDAS_DATAREADER_AVAILABLE = False
    warnings.warn(
        "pandas-datareader not available. Fama-French factor analysis will not work. "
        "Install with: pip install pandas-datareader>=0.10.0"
    )


@lru_cache(maxsize=32)
def get_factors(start_date, end_date):
    """
    Fetch Fama-French 3-factor data from Kenneth French's data library.
    
    The Fama-French 3-factor model includes:
    - Mkt-RF: Market excess return (market return minus risk-free rate)
    - SMB: Small Minus Big (size factor - small cap outperformance)
    - HML: High Minus Low (value factor - value stock outperformance)
    - RF: Risk-free rate
    
    Args:
        start_date: Start date as string (YYYY-MM-DD) or pandas Timestamp
        end_date: End date as string (YYYY-MM-DD) or pandas Timestamp
    
    Returns:
        pandas.DataFrame: DataFrame with DatetimeIndex and columns:
            - 'Mkt-RF': Market excess return (decimal, e.g., 0.01 for 1%)
            - 'SMB': Small Minus Big factor (decimal)
            - 'HML': High Minus Low factor (decimal)
            - 'RF': Risk-free rate (decimal)
        
        Returns empty DataFrame if data cannot be fetched.
    
    Raises:
        ValueError: If dates are invalid or start_date > end_date
        ConnectionError: If network request fails (wrapped in error handling)
    
    Example:
        >>> factors = get_factors('2020-01-01', '2023-12-31')
        >>> print(factors.head())
                        Mkt-RF     SMB     HML      RF
        Date
        2020-01-02     0.0045   0.0021  -0.0012  0.0001
        2020-01-03    -0.0023  -0.0005   0.0008  0.0001
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
    
    # Kenneth French data library identifier for Fama-French 3 factors (daily)
    # 'F-F_Research_Data_Factors_daily' is the daily frequency dataset
    factor_name = 'F-F_Research_Data_Factors_daily'
    
    try:
        # Fetch Fama-French 3-factor data
        # Note: Kenneth French's data is provided as percentages, so we divide by 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress pandas_datareader warnings
            
            # Fetch the data
            factors_raw = web.DataReader(
                factor_name,
                'famafrench',
                start=start_date,
                end=end_date
            )
        
        # pandas_datareader returns a dict with different frequencies
        # For daily data, the key is typically 0 or 'daily'
        if isinstance(factors_raw, dict):
            # Try to find the daily data
            if 0 in factors_raw:
                factors_df = factors_raw[0]
            elif 'daily' in factors_raw:
                factors_df = factors_raw['daily']
            elif len(factors_raw) == 1:
                # Only one key, use it
                factors_df = list(factors_raw.values())[0]
            else:
                # Multiple keys, try to find daily
                for key in factors_raw.keys():
                    if 'daily' in str(key).lower() or key == 0:
                        factors_df = factors_raw[key]
                        break
                else:
                    # Use first available
                    factors_df = list(factors_raw.values())[0]
        else:
            factors_df = factors_raw
        
        # Ensure we have a DataFrame
        if not isinstance(factors_df, pd.DataFrame):
            raise ValueError(f"Unexpected data format from pandas_datareader: {type(factors_df)}")
        
        # Kenneth French data is in percentages, convert to decimals
        # Expected columns: Mkt-RF, SMB, HML, RF
        required_columns = ['Mkt-RF', 'SMB', 'HML', 'RF']
        
        # Check if columns exist (case-insensitive)
        available_columns = [col for col in factors_df.columns if col in required_columns]
        if not available_columns:
            # Try case-insensitive matching
            column_map = {}
            for req_col in required_columns:
                for df_col in factors_df.columns:
                    if req_col.lower() == df_col.lower():
                        column_map[req_col] = df_col
                        break
            
            if column_map:
                factors_df = factors_df.rename(columns=column_map)
                available_columns = list(column_map.keys())
        
        # Select only the required columns
        factors_df = factors_df[required_columns].copy()
        
        # Convert from percentages to decimals (divide by 100)
        for col in required_columns:
            if col in factors_df.columns:
                factors_df[col] = factors_df[col] / 100.0
        
        # Ensure index is DatetimeIndex
        if not isinstance(factors_df.index, pd.DatetimeIndex):
            try:
                factors_df.index = pd.to_datetime(factors_df.index)
            except (ValueError, TypeError):
                raise ValueError("Could not convert index to DatetimeIndex")
        
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

