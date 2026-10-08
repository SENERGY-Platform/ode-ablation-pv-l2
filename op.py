"""The operator: a day-ahead PV forecast from archived weather forecasts.

Two inputs reach infer():
  * "pv": the PV series itself. It is an input so the model can train on it and
    so the evaluation can score against it; infer() answers nothing for it.
  * "forecast": the Open-Meteo previous-runs archive. Every message is one
    forecast hour at one lead time, published at its issue time.

The model predicts hour B from the forecasts for B and B + 1h at the trained lead
time. The forecast for B + 1h is issued an hour after the one for B, so infer()
keeps the forecasts it has seen by hour and answers for B when B + 1h arrives:
still about 23 h ahead at lead 1. The result is stamped with B, the hour it is
about, not the hour it was made.
"""

import datetime
import typing

import pandas as pd
from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import FORECAST_ARGS, FORECAST_FIELDS, LEAD_DAYS, PV_ARG, train_model

HOUR = datetime.timedelta(hours=1)


class CustomConfig(Config):
    """Deployment configuration, typed."""

    # Retrain at most this often, in seconds.
    retrain_after_s = 86400


def _lead_of(model: PyFuncModel) -> int:
    try:
        return int(model.unwrap_python_model().lead_days)
    except Exception:
        return LEAD_DAYS


def _utc(value) -> typing.Optional[datetime.datetime]:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if parsed is None or pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


class Operator(MLOperator):
    configType = CustomConfig

    selectors = [
        Selector({"name": "pv", "args": [PV_ARG]}),
        Selector({"name": "forecast", "args": FORECAST_ARGS}),
    ]

    def init(self, *args, **kwargs):
        # State first: under a data split, super().init() trains and replays the
        # test window before it returns.
        self.trained_at: typing.Optional[datetime.datetime] = None
        self.pending: typing.Dict[datetime.datetime, typing.Dict[str, typing.Any]] = {}
        super().init(*args, **kwargs)

    def infer(
        self,
        model: typing.Optional[PyFuncModel],
        data: typing.Dict[str, typing.Any],
        selector: str,
        device_id: str,
        timestamp: datetime.datetime,
    ) -> typing.Tuple[
        typing.Optional[datetime.datetime], typing.Optional[typing.Any], typing.Optional[PythonModel]
    ]:
        if selector != "forecast" or model is None:
            return None, None, None
        try:
            if int(data.get("lead_days")) != _lead_of(model):
                return None, None, None
        except (TypeError, ValueError):
            return None, None, None
        hour = _utc(data.get("forecasted_for"))
        if hour is None:
            return None, None, None

        self.pending[hour] = {field: data.get(field) for field in FORECAST_FIELDS}
        # Two hours of forecasts are all a prediction needs.
        for stale in [h for h in self.pending if h < hour - 2 * HOUR]:
            del self.pending[stale]

        bucket = hour - HOUR
        current = self.pending.get(bucket)
        if current is None:
            return None, None, None

        payload = {"bucket": bucket.isoformat(), **current}
        for field in FORECAST_FIELDS:
            payload[f"{field}_next"] = data.get(field)
        out = model.predict(payload)
        if not out:
            return None, None, None
        about = datetime.datetime.fromisoformat(out["time"])
        if about.tzinfo is None:
            about = about.replace(tzinfo=datetime.timezone.utc)
        return about, {"prediction": out["prediction"]}, None

    def train(
        self, model: typing.Optional[PyFuncModel], logger: TrainMlflowLogger
    ) -> typing.Optional[PythonModel]:
        self.trained_at = datetime.datetime.now(datetime.timezone.utc)
        return train_model(logger)

    def need_retraining(self, model: typing.Optional[PyFuncModel]) -> bool:
        if model is None:
            return True
        if self.trained_at is None:
            return False
        age = datetime.datetime.now(datetime.timezone.utc) - self.trained_at
        return age.total_seconds() >= self.config.retrain_after_s
