import logging
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive

module_logger = logging.getLogger('UFE.stats_fit_forecast')
logger = logging.getLogger('UFE.stats_fit_forecast')

class FitForecast():
    def __init__(self,
                    datadf: pd.DataFrame,
                    h: int, # forecast horizon
                    season: int, # seasonality
                    freq: str, # frequency
                    ts_models: list,
                    n_jobs: int,
                    confidence: int ):
        self.datadf = datadf
        self.h = h
        self.freq = freq
        self.ts_models = ts_models
        self.n_jobs = n_jobs
        self.confidence = confidence


    def fit(data: pd.DataFrame, h, season, freq, ts_models, n_jobs, confidence):
        # create the model object, for each model and let user know time required for each fit
        model = StatsForecast(df=data,
                              models=ts_models,
                              freq=freq,
                              n_jobs=n_jobs,
                              verbose=True)

        model.fit(data)
        return model

    def make_prediction(model, datadf, h, confidence):
        # predict future h periods, with level of confidence
        prediction = model.predict(h=h, level=[confidence])
        return prediction

    def both(datadf: pd.DataFrame, h, season, freq, ts_models, n_jobs, confidence):
        # create the model object, for each model and let user know time required for each fit
        model = StatsForecast(
            df=datadf,
            models=ts_models,
            freq=freq,
            n_jobs=n_jobs,
            fallback_model=Naive()
        )

        forecast = model.forecast(df=datadf, h=h, level=[confidence])
        return forecast, model

