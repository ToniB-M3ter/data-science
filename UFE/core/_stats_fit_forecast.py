
from statsforecast import StatsForecast
from statsforecast.models import (
    SeasonalNaive, # model using the previous season's data as the forecast
    Naive, # Simple naive model using the last observed value as the forecast
    HistoricAverage, # Average of all historical data
    AutoETS, # Automatically selects best ETS model based on AIC
    AutoARIMA, # ARIMA model that automatically select the parameters for given time series with AIC and cross validation
    HoltWinters, #HoltWinters ETS model
    AutoCES, # Auto Complex Exponential Smoothing
    MSTL,
    OptimizedTheta,
    AutoTheta
    )



class FitForecast():
    df
    h =
    season =
    freq =
    ts_models =

    def fit(df: pd.DataFrame, h, season, freq, ts_models):
        # create the model object, for each model and let user know time required for each fit
        model = StatsForecast(df=df,
                              models=ts_models,
                              freq=freq,
                              n_jobs=1,
                              verbose=True)

        model.fit(df)
        return model


