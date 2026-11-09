import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

logging.basicConfig(level=logging.INFO, format='%(message)s')

def load_and_prep_retail_data(filepath: str) -> pd.DataFrame:
    """Loads dataset and creates a robust datetime index."""
    file_path = Path(filepath)
    if not file_path.exists():
        logging.error(f"Dataset not found at {filepath}. Please verify path.")
        raise FileNotFoundError(f"Missing dataset: {filepath}")

    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df.dropna(subset=['Date'], inplace=True)
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)
    return df

def analyze_time_series(df: pd.DataFrame) -> None:
    """Executes seasonal decomposition and forecasting on aggregated monthly sales."""
    # Resample to monthly total revenue
    monthly_revenue = df['Weekly_Sales'].resample('ME').sum()
    
    # 1. Revenue breakdown by Top 3 Stores (Regions) over time
    store_revenue = df.groupby(['Store', pd.Grouper(freq='ME')])['Weekly_Sales'].sum().unstack(level=0)
    top_stores = store_revenue.sum().nlargest(3).index
    
    # 2. Forecasting using Holt-Winters (captures both trend and seasonality)
    # Using a 12-month seasonal period for retail data
    model = ExponentialSmoothing(
        monthly_revenue, 
        trend='add', 
        seasonal='add', 
        seasonal_periods=12,
        initialization_method="estimated"
    ).fit()
    forecast = model.forecast(6) # Forecast next 6 months

    # 3. Presentation-Ready Visualization
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Top Plot: Aggregate Trends, Moving Averages, and Forecast
    ax1 = axes[0]
    ax1.plot(monthly_revenue.index, monthly_revenue.values, label='Actual Monthly Revenue', color='#2c3e50', linewidth=2)
    ax1.plot(monthly_revenue.rolling(window=3).mean(), label='3-Month Moving Avg (Trend)', color='#e74c3c', linestyle='-.')
    ax1.plot(forecast.index, forecast.values, label='Holt-Winters 6-Month Forecast', color='#27ae60', linewidth=2, linestyle='--')
    
    ax1.set_title("Retail Revenue: Historical Trends & Predictive Forecast", fontsize=16, weight='bold', pad=15)
    ax1.set_ylabel("Total Revenue ($)", weight='bold')
    ax1.legend(frameon=True, facecolor='white', shadow=True)
    ax1.grid(True, alpha=0.3)

    # Bottom Plot: Breakdown by Top Regions
    ax2 = axes[1]
    store_revenue[top_stores].plot(ax=ax2, colormap='Set2', linewidth=1.5)
    ax2.set_title("Revenue Breakdown by Top Performing Regions", fontsize=14, weight='bold')
    ax2.set_ylabel("Revenue ($)")
    ax2.set_xlabel("Date", weight='bold')
    ax2.legend(title="Store ID")
    
    # Format x-axis for readability
    for ax in axes:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    sns.despine(fig)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Wrap execution in try-except for graceful failing if the user lacks the CSV
    try:
        logging.info("Initializing retail time series analysis...")
        retail_df = load_and_prep_retail_data('walmart_sales.csv')
        analyze_time_series(retail_df)
        logging.info("Analysis complete.")
    except FileNotFoundError:
        # Fallback dummy data generation for demonstration if CSV is missing
        logging.info("Generating sample time series data to demonstrate functionality...")
        import numpy as np
        dates = pd.date_range(start='2010-01-01', end='2012-12-31', freq='W')
        dummy_data = pd.DataFrame({
            'Date': dates,
            'Store': np.random.choice([1, 2, 3, 4], size=len(dates)),
            'Dept': np.random.choice([10, 20, 30], size=len(dates)),
            'Weekly_Sales': np.random.normal(15000, 5000, size=len(dates)) + np.sin(np.arange(len(dates)) / 5) * 5000
        })
        dummy_data.to_csv('walmart_sales.csv', index=False)
        
        retail_df = load_and_prep_retail_data('walmart_sales.csv')
        analyze_time_series(retail_df)