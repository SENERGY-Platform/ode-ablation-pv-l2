"""Training, on Ray.

A day-ahead PV forecast from archived numerical weather forecasts.

Inputs (one dataset per input topic, in topic order):
  * the PV series, mapped to "pv" (W, instantaneous)
  * the Open-Meteo previous-runs forecast archive, one row per forecast hour and
    lead day, mapped to FORECAST_FIELDS + "forecasted_for" + "lead_days"

The target is the PV series' hourly mean, bucketed exactly as Operator Lib's
evaluation buckets it: the arithmetic mean of the samples whose time falls in
[hour, hour + 1h). Each training row joins one such bucket with the forecast that
was issued LEAD_DAYS before it, so the model only ever learns from what would have
been known a day ahead.

Open-Meteo labels an hourly radiation value with the end of the hour it averages
over, so the forecast stamped F most likely describes the bucket starting F - 1h.
That is a convention, not something verified here, so training tries both
alignments on a held-out tail and keeps the better one. The choice is logged.
"""

import datetime
import math
import os
import typing

import numpy as np
import pandas as pd
import ray
from mlflow.pyfunc import PythonModel

from operator_lib.util.helpers import provide_historic_data, TrainMlflowLogger


# How much history one training pass reads. Kept under 365 days: Operator Lib
# refuses longer reads from Kafka, and an import topic read without its export
# would fall back to Kafka.
TRAINING_WINDOW = datetime.timedelta(days=int(os.environ.get("PV_TRAINING_DAYS", "360")))

# The forecast horizon the operator answers with: 1 is "issued 24 h before".
LEAD_DAYS = int(os.environ.get("PV_LEAD_DAYS", "1"))

# The held-out tail used to choose the alignment and report a validation error.
HOLDOUT = datetime.timedelta(days=int(os.environ.get("PV_HOLDOUT_DAYS", "30")))

# Site of the forecast archive (the import is configured for 51.7 / 10).
LAT = float(os.environ.get("PV_LAT", "51.7"))
LON = float(os.environ.get("PV_LON", "10.0"))

FORECAST_FIELDS = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "temperature_2m",
]
FORECAST_ARGS = FORECAST_FIELDS + ["forecasted_for", "lead_days"]
PV_ARG = "pv"

HOUR = pd.Timedelta(hours=1)


# --------------------------------------------------------------------------- #
# Features. Shared by training and by the registered model, so the two are one
# contract.
# --------------------------------------------------------------------------- #

def _solar_geometry(bucket_start: pd.DatetimeIndex, lat: float, lon: float) -> pd.DataFrame:
    """Sun position at the middle of each hourly bucket (NOAA approximation, UTC)."""
    mid = bucket_start + pd.Timedelta(minutes=30)
    doy = mid.dayofyear.to_numpy(dtype=float)
    hour = mid.hour.to_numpy(dtype=float) + mid.minute.to_numpy(dtype=float) / 60.0
    gamma = 2.0 * math.pi / 365.0 * (doy - 1.0 + (hour - 12.0) / 24.0)
    eqtime = 229.18 * (
        0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
    )
    decl = (
        0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
    )
    tst = hour * 60.0 + eqtime + 4.0 * lon
    ha = np.radians(tst / 4.0 - 180.0)
    phi = math.radians(lat)
    cos_zen = np.sin(phi) * np.sin(decl) + np.cos(phi) * np.cos(decl) * np.cos(ha)
    cos_zen = np.clip(cos_zen, -1.0, 1.0)
    elevation = 90.0 - np.degrees(np.arccos(cos_zen))
    azimuth = np.arctan2(np.sin(ha), np.cos(ha) * np.sin(phi) - np.tan(decl) * np.cos(phi))
    return pd.DataFrame(
        {
            "cos_zenith": np.clip(cos_zen, 0.0, None),
            "elevation": elevation,
            "azimuth_sin": np.sin(azimuth),
            "azimuth_cos": np.cos(azimuth),
            "doy_sin": np.sin(2 * math.pi * doy / 365.25),
            "doy_cos": np.cos(2 * math.pi * doy / 365.25),
        },
        index=bucket_start,
    )


def build_features(forecast: pd.DataFrame, shift_h: int, lat: float, lon: float) -> pd.DataFrame:
    """One feature row per target bucket.

    `forecast` carries FORECAST_FIELDS and a UTC "forecasted_for" column. The
    bucket a row describes is forecasted_for - shift_h hours.
    """
    bucket = pd.DatetimeIndex(forecast["forecasted_for"]) - pd.Timedelta(hours=shift_h)
    values = forecast[FORECAST_FIELDS].astype(float).set_axis(bucket)
    geometry = _solar_geometry(bucket, lat, lon)
    features = pd.concat([values, geometry], axis=1)
    # Irradiance projected on the sun's position: a cheap clear-sky-like term
    # trees otherwise have to assemble from two splits.
    features["beam_on_horizontal"] = features["direct_normal_irradiance"] * features["cos_zenith"]
    return features


FEATURE_COLUMNS = FORECAST_FIELDS + [
    "cos_zenith", "elevation", "azimuth_sin", "azimuth_cos", "doy_sin", "doy_cos",
    "beam_on_horizontal",
]


def _parse_utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True, errors="coerce")


# --------------------------------------------------------------------------- #
# The registered model.
# --------------------------------------------------------------------------- #

class OdeAblationPvL2Model(PythonModel):
    """Maps one archived forecast message to a PV prediction for the hour it covers.

    predict() receives exactly the dict infer() gets for the forecast selector and
    returns None, for a message at another lead time or with missing values, or a
    dict with the bucket start ("time", ISO 8601 UTC) and the prediction in W.
    """

    def __init__(self, regressor, shift_h: int, lead_days: int, lat: float, lon: float,
                 upper: float) -> None:
        self.regressor = regressor
        self.shift_h = shift_h
        self.lead_days = lead_days
        self.lat = lat
        self.lon = lon
        self.upper = upper

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        try:
            if int(payload.get("lead_days")) != self.lead_days:
                return None
        except (TypeError, ValueError):
            return None
        forecasted_for = _parse_utc([payload.get("forecasted_for")]).iloc[0]
        if pd.isna(forecasted_for):
            return None
        row = {"forecasted_for": [forecasted_for]}
        for field in FORECAST_FIELDS:
            value = payload.get(field)
            row[field] = [np.nan if value is None else float(value)]
        features = build_features(pd.DataFrame(row), self.shift_h, self.lat, self.lon)
        value = float(self.regressor.predict(features[FEATURE_COLUMNS].to_numpy())[0])
        value = min(max(value, 0.0), self.upper)
        return {"time": features.index[0].isoformat(), "prediction": value}


# --------------------------------------------------------------------------- #
# Training.
# --------------------------------------------------------------------------- #

def _to_pandas(dataset) -> pd.DataFrame:
    if isinstance(dataset, ray.ObjectRef):
        dataset = ray.get(dataset)
    if isinstance(dataset, pd.DataFrame):
        return dataset
    return dataset.to_pandas()


def _split_inputs(frames: typing.List[pd.DataFrame]) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
    pv, forecast = None, None
    for frame in frames:
        if "shortwave_radiation" in frame.columns:
            forecast = frame
        elif PV_ARG in frame.columns:
            pv = frame
    if pv is None or forecast is None:
        raise RuntimeError(
            f"expected one input mapped to '{PV_ARG}' and one carrying the forecast "
            f"fields; got columns {[list(f.columns) for f in frames]}")
    return pv, forecast


def _hourly_target(pv: pd.DataFrame) -> pd.Series:
    """The evaluation's own bucket mean: numeric samples only, [h, h+1h)."""
    times = _parse_utc(pv["time"])
    values = pd.to_numeric(pv[PV_ARG], errors="coerce")
    frame = pd.DataFrame({"bucket": times.dt.floor("1h"), "pv": values.to_numpy()})
    frame = frame.dropna()
    return frame.groupby("bucket")["pv"].mean()


def _forecast_rows(forecast: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    frame = forecast.copy()
    frame["lead_days"] = pd.to_numeric(frame["lead_days"], errors="coerce")
    frame = frame[frame["lead_days"] == lead_days]
    frame["forecasted_for"] = _parse_utc(frame["forecasted_for"]).to_numpy()
    frame = frame.dropna(subset=["forecasted_for"])
    if "time" in frame.columns:
        frame = frame.sort_values("time", kind="stable")
    # One forecast per hour: the latest issue of this lead time wins.
    frame = frame.drop_duplicates(subset=["forecasted_for"], keep="last")
    return frame


def _dataset(target: pd.Series, forecast: pd.DataFrame, shift_h: int) -> pd.DataFrame:
    features = build_features(forecast, shift_h, LAT, LON)
    features = features[~features.index.duplicated(keep="last")]
    joined = features.join(target.rename("target"), how="inner")
    joined = joined.dropna(subset=["target"])
    return joined.sort_index()


def _regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=float(os.environ.get("PV_LR", "0.05")),
        max_iter=int(os.environ.get("PV_MAX_ITER", "600")),
        max_leaf_nodes=int(os.environ.get("PV_MAX_LEAF_NODES", "31")),
        min_samples_leaf=int(os.environ.get("PV_MIN_SAMPLES_LEAF", "20")),
        l2_regularization=float(os.environ.get("PV_L2", "0.0")),
        random_state=0,
    )


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


@ray.remote
def _fit(datasets: typing.List[typing.Any]) -> typing.Dict[str, typing.Any]:
    """The distributed part: read, align, choose the alignment, fit."""
    frames = [_to_pandas(d) for d in datasets]
    pv, forecast = _split_inputs(frames)
    target = _hourly_target(pv)
    rows = _forecast_rows(forecast, LEAD_DAYS)

    report: typing.Dict[str, typing.Any] = {
        "pv_samples": int(len(pv)),
        "target_buckets": int(len(target)),
        "forecast_rows_at_lead": int(len(rows)),
    }

    best = None
    for shift_h in (1, 0):
        data = _dataset(target, rows, shift_h)
        if len(data) < 200:
            report[f"shift{shift_h}_rows"] = int(len(data))
            continue
        cut = data.index.max() - HOLDOUT
        train, hold = data[data.index <= cut], data[data.index > cut]
        model = _regressor().fit(train[FEATURE_COLUMNS].to_numpy(), train["target"].to_numpy())
        pred = np.clip(model.predict(hold[FEATURE_COLUMNS].to_numpy()), 0.0, None)
        mae = _mae(pred, hold["target"].to_numpy())
        report[f"shift{shift_h}_rows"] = int(len(data))
        report[f"shift{shift_h}_holdout_mae"] = mae
        if best is None or mae < best[1]:
            best = (shift_h, mae, data, hold, pred)

    if best is None:
        raise RuntimeError(f"too few aligned rows to train on: {report}")

    shift_h, holdout_mae, data, hold, pred = best
    report["shift_h"] = shift_h
    report["holdout_mae"] = holdout_mae
    report["holdout_buckets"] = int(len(hold))
    daylight = hold["cos_zenith"].to_numpy() > 0
    report["holdout_mae_daylight"] = _mae(pred[daylight], hold["target"].to_numpy()[daylight])
    report["holdout_target_mean"] = float(hold["target"].mean())

    final = _regressor().fit(data[FEATURE_COLUMNS].to_numpy(), data["target"].to_numpy())
    upper = float(np.nanpercentile(data["target"].to_numpy(), 99.9)) * 1.2
    report["train_rows"] = int(len(data))
    report["train_from"] = data.index.min().isoformat()
    report["train_to"] = data.index.max().isoformat()
    return {"regressor": final, "shift_h": shift_h, "upper": upper, "report": report}


def train_model(logger: TrainMlflowLogger) -> typing.Optional[PythonModel]:
    """Read the history, fit, and hand back a model for MLflow to register."""
    with logger.trace("read history"):
        datasets = provide_historic_data(TRAINING_WINDOW)
    if not datasets:
        return None

    with logger.trace("fit"):
        fitted = ray.get(_fit.remote(datasets))

    report = fitted["report"]
    logger.log_params({
        "training_window_days": TRAINING_WINDOW.days,
        "lead_days": LEAD_DAYS,
        "holdout_days": HOLDOUT.days,
        "shift_h": report["shift_h"],
        "train_from": report["train_from"],
        "train_to": report["train_to"],
        "lat": LAT,
        "lon": LON,
    })
    logger.log_metrics({
        key: float(value) for key, value in report.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    })
    return OdeAblationPvL2Model(
        regressor=fitted["regressor"],
        shift_h=fitted["shift_h"],
        lead_days=LEAD_DAYS,
        lat=LAT,
        lon=LON,
        upper=fitted["upper"],
    )
