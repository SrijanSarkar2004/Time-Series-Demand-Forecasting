## TIME SERIES DEMAND FORECASTING
# -------------------------------------------------------------------
# PART 1 – PREPARATION & LIBRARY IMPORTS
# -------------------------------------------------------------------

# Data manipulation libraries
import pandas as pd
import numpy as np

# Statistical modeling library (for time series models)
import statsmodels.api as sm
import statsmodels as sms

# Visualization library
import matplotlib.pyplot as plt

# Utility libraries
import warnings
import itertools

# Plot configuration utilities
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Ignore warning messages for clean output
warnings.filterwarnings("ignore")

# Plot styling
plt.style.use('fivethirtyeight')
rcParams['figure.figsize'] = 16, 8


# -------------------------------------------------------------------
# PART 1 – DATA LOADING & CLEANING
# -------------------------------------------------------------------

# Load dataset from GitHub (raw CSV link)
sales = pd.read_csv(
    'https://raw.githubusercontent.com/dwoo-work/time-series-demand-forecasting/main/src/sales_data_sample_utf8.csv'
)

# Remove duplicate records
sales = sales.drop_duplicates()

# Create a clean copy of the dataset
sales_clean = sales.copy()
sales_clean.info()

# Convert ORDERDATE column to datetime format
sales_clean['ORDERDATE'] = pd.to_datetime(sales_clean['ORDERDATE'])

# Create a clean date column (YYYY-MM-DD)
sales_clean['date'] = sales_clean['ORDERDATE'].dt.strftime("%Y-%m-%d")
sales_clean['date'] = pd.to_datetime(sales_clean['date'])

# Extract time-based features
sales_clean['month'] = sales_clean.date.dt.month
sales_clean['year'] = sales_clean.date.dt.year
sales_clean['week'] = sales_clean.date.dt.week

# Filter motorcycle sales quantity
sales_clean['motorcycles_QUANTITYORDERED'] = sales_clean.loc[
    sales_clean['PRODUCTLINE'] == 'Motorcycles',
    'QUANTITYORDERED'
]

# Aggregate weekly motorcycle sales
time_series = (
    sales_clean
    .groupby(['week', 'month', 'year'])
    .agg(
        date=('date', 'first'),
        motorcycles_total_qty_ordered=('motorcycles_QUANTITYORDERED', np.sum)
    )
    .reset_index()
    .sort_values('date')
)

# Set date as index for time series operations
time_series['date'] = pd.to_datetime(time_series['date'])
time_series = time_series.set_index('date')

# Convert weekly data to monthly time series
monthly_series = time_series.motorcycles_total_qty_ordered.resample('M').sum()

# Plot monthly motorcycle sales
monthly_series.plot(label='actual')
plt.title('Total Qty. of Motorcycles Sold from Feb 2003 - May 2005')
plt.legend(loc='upper left')
plt.show()


# -------------------------------------------------------------------
# PART 2 – SEASONAL DECOMPOSITION
# -------------------------------------------------------------------

# Decompose time series into trend, seasonality, and residuals
components = sm.tsa.seasonal_decompose(monthly_series)
components.plot()
plt.show()

# Store individual components
seasonality = components.seasonal
trend = components.trend
remainder = components.resid


# -------------------------------------------------------------------
# PART 3 – STATIONARITY TEST (ADF TEST)
# -------------------------------------------------------------------

# Plot rolling mean and standard deviation
monthly_series.plot(label='actual')
monthly_series.rolling(window=12).mean().plot(label='mean')
monthly_series.rolling(window=12).std().plot(label='s.d')
plt.legend(loc='upper left')
plt.title('Monthly Sales with Rolling Statistics')
plt.show()

# Perform Augmented Dickey-Fuller test
ad_fuller_test = sm.tsa.stattools.adfuller(monthly_series, autolag='AIC')
ad_fuller_test   # p-value < 0.05 implies stationarity


# -------------------------------------------------------------------
# PART 4 – ARIMA MODEL IDENTIFICATION
# -------------------------------------------------------------------

# Plot Autocorrelation and Partial Autocorrelation
plot_acf(monthly_series)
plt.show()

plot_pacf(monthly_series, lags=13)
plt.show()

# Define different ARIMA family models
model_MA = sm.tsa.statespace.SARIMAX(monthly_series, order=(0, 0, 1))
model_AR = sm.tsa.statespace.SARIMAX(monthly_series, order=(1, 0, 0))
model_ARMA = sm.tsa.statespace.SARIMAX(monthly_series, order=(1, 0, 1))
model_ARIMA = sm.tsa.statespace.SARIMAX(monthly_series, order=(1, 1, 1))

# Fit all models
result_MA = model_MA.fit()
result_AR = model_AR.fit()
result_ARMA = model_ARMA.fit()
result_ARIMA = model_ARIMA.fit()

# Compare models using AIC
result_MA.aic
result_AR.aic
result_ARMA.aic
result_ARIMA.aic

# Diagnostic plots for ARIMA model
result_ARIMA.plot_diagnostics(figsize=[20, 16])
plt.show()


# -------------------------------------------------------------------
# PART 5 – GRID SEARCH FOR BEST ARIMA PARAMETERS
# -------------------------------------------------------------------

# Define parameter ranges
p = d = q = P = D = Q = range(0, 3)
S = 12  # Seasonal period (12 months)

# Generate all parameter combinations
combinations = list(itertools.product(p, d, q, P, D, Q))

# Separate non-seasonal and seasonal parameters
arima_orders = [(x[0], x[1], x[2]) for x in combinations]
seasonal_orders = [(x[3], x[4], x[5], S) for x in combinations]

# Store grid search results
results_data = pd.DataFrame(columns=['p', 'd', 'q', 'P', 'D', 'Q', 'AIC'])

# Perform grid search using AIC
for i in range(len(combinations)):
    try:
        model = sm.tsa.statespace.SARIMAX(
            monthly_series,
            order=arima_orders[i],
            seasonal_order=seasonal_orders[i]
        )
        result = model.fit()
        results_data.loc[i] = [
            arima_orders[i][0],
            arima_orders[i][1],
            arima_orders[i][2],
            seasonal_orders[i][0],
            seasonal_orders[i][1],
            seasonal_orders[i][2],
            result.aic
        ]
    except:
        continue

# Display best ARIMA configuration
results_data[results_data.AIC == min(results_data.AIC)]


# -------------------------------------------------------------------
# PART 6 – FINAL ARIMA FORECAST
# -------------------------------------------------------------------

# Fit best ARIMA model
best_model = sm.tsa.statespace.SARIMAX(
    monthly_series,
    order=(2, 1, 0),
    seasonal_order=(0, 2, 0, 12)
)

results = best_model.fit()

# In-sample predictions
fitting = results.get_prediction(start='2003-01-31')
fitting_mean = fitting.predicted_mean

# Forecast next 12 months
forecast = results.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean

# Plot ARIMA forecast
fitting_mean.plot(label='fitting')
forecast_mean.plot(label='forecast')
monthly_series.plot(label='actual')
plt.title('ARIMA Forecast')
plt.legend(loc='upper left')
plt.show()

# Calculate MAE
mean_absolute_error = abs(monthly_series - fitting_mean).mean()


# -------------------------------------------------------------------
# PART 7 – EXPONENTIAL SMOOTHING (HOLT-WINTERS)
# -------------------------------------------------------------------

# Define different Holt-Winters models
model_expo1 = sms.tsa.holtwinters.ExponentialSmoothing(
    monthly_series, trend='add', seasonal='add', seasonal_periods=12
)
model_expo2 = sms.tsa.holtwinters.ExponentialSmoothing(
    monthly_series, trend='mul', seasonal='add', seasonal_periods=12
)
model_expo3 = sms.tsa.holtwinters.ExponentialSmoothing(
    monthly_series, trend='add', seasonal='mul', seasonal_periods=12
)
model_expo4 = sms.tsa.holtwinters.ExponentialSmoothing(
    monthly_series, trend='mul', seasonal='mul', seasonal_periods=12
)

# Fit models
results_1 = model_expo1.fit()
results_2 = model_expo2.fit()
results_3 = model_expo3.fit()
results_4 = model_expo4.fit()

# Fitted values
fit1 = results_1.predict(0, len(monthly_series))
fit2 = results_2.predict(0, len(monthly_series))
fit3 = results_3.predict(0, len(monthly_series))
fit4 = results_4.predict(0, len(monthly_series))

# Calculate MAE for each model
mae1 = abs(monthly_series - fit1).mean()
mae2 = abs(monthly_series - fit2).mean()
mae3 = abs(monthly_series - fit3).mean()
mae4 = abs(monthly_series - fit4).mean()

# Forecast using best exponential smoothing model
forecast = results_1.predict(0, len(monthly_series) + 12)

# Plot Exponential Smoothing forecast
monthly_series.plot(label='actual')
forecast.plot(label='forecast')
plt.title('Exponential Smoothing Forecast')
plt.legend(loc='upper left')
plt.show()
