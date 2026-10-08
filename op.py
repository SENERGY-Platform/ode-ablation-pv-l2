"""The operator: a day-ahead PV forecast, one prediction per archived weather forecast.

Two inputs reach infer():
  * "pv": the PV series itself. It is an input so the model can train on it and
    so the evaluation can score against it; infer() answers nothing for it.
  * "forecast": the Open-Meteo previous-runs archive. Every message is one
    forecast hour at one lead time, published at its issue time. For the lead
    time the model was trained on, infer() predicts the PV hourly mean for the
    hour that forecast describes and stamps the result with that hour, so the
    output carries the time it is about rather than the time it was made.
"""

import datetime
import typing

from mlflow.pyfunc import PyFuncModel, PythonModel

from operator_lib.util import Config, MLOperator, Selector
from operator_lib.util.helpers import TrainMlflowLogger

from training import FORECAST_ARGS, PV_ARG, train_model


class CustomConfig(Config):
    """Deployment configuration, typed."""

    # Retrain at most this often, in seconds.
    retrain_after_s = 86400


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

        out = model.predict(data)
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
