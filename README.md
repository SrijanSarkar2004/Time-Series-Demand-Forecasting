# Time-Series-Demand-Forecasting
This project focuses on forecasting future demand using historical time series data of motorcycle sales. The goal is to analyze past sales patterns, identify trends and seasonality, and predict demand for the next twelve months using statistical forecasting techniques. The project is purely analytical and does not include any deployment or full-stack components.

The data is cleaned and aggregated into a monthly time series to enable effective analysis. Exploratory analysis and seasonal decomposition are used to understand the underlying structure of the data. Stationarity is tested using the Augmented Dickey-Fuller (ADF) test, and autocorrelation plots are analyzed to guide model selection.

Two forecasting approaches are implemented: ARIMA and Exponential Smoothing (Holt-Winters). ARIMA models are evaluated using AIC and optimized through grid search, while Exponential Smoothing models are compared using error metrics. Model performance is assessed using MAE, MSE, and RMSE.

The final output includes a twelve-month demand forecast visualized alongside historical data. This project demonstrates practical skills in time series analysis, model selection, and forecasting, and is applicable to real-world problems such as inventory planning and demand forecasting.
