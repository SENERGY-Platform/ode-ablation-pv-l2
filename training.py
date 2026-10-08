"""Training, on Ray.

A day-ahead PV forecast from archived numerical weather forecasts and the
inverter's own output a day earlier.

Inputs (one dataset per input topic, in topic order):
  * the PV series, mapped to "pv" (W, instantaneous)
  * the Open-Meteo previous-runs forecast archive, one row per forecast hour and
    lead day, mapped to FORECAST_FIELDS + "forecasted_for" + "lead_days"
  * optional, diagnostic only: the Open-Meteo weather history, mapped to
    OBS_ARGS. It never reaches the registered model; it feeds
    holdout_mae_observed_weather, the error a weather-only model reaches on the
    weather as it turned out.

The target is the PV series' hourly mean, bucketed exactly as Operator Lib's
evaluation buckets it: the arithmetic mean of the samples whose time falls in
[hour, hour + 1h). Each training row joins one such bucket B with
  * the forecasts issued LEAD_DAYS before it for B and for B + 1h, and
  * pv_lag24h, the PV hourly mean of B - 24h. The operator predicts B when the
    forecast for B + 1h arrives, at about B - 23h, when that bucket is complete.

Run 5 (experiment ees26ce6srvjx5zd3wkeomgy54) put a weather-only model on the
weather as it turned out at 37.6 W holdout MAE against 45.0 W on the day-ahead
forecast: most of the error is not forecast error. pv_lag24h carries what the
weather cannot: an outage, a frozen reading, seasonal shading.
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

# The held-out tail used to report a validation error before the final fit.
HOLDOUT = datetime.timedelta(days=int(os.environ.get("PV_HOLDOUT_DAYS", "30")))

# Site of the forecast archive (the import is configured for 51.7 / 10).
LAT = float(os.environ.get("PV_LAT", "51.7"))
LON = float(os.environ.get("PV_LON", "10.0"))

# Regressor settings. Defaults are run 3's (experiment 4kwhois2darvdhqvfexhj3qnie),
# which matched or beat the looser settings of run 2 and fit three times faster.
HYPERPARAMS = {
    "learning_rate": float(os.environ.get("PV_LR", "0.03")),
    "max_iter": int(os.environ.get("PV_MAX_ITER", "800")),
    "max_leaf_nodes": int(os.environ.get("PV_MAX_LEAF_NODES", "15")),
    "min_samples_leaf": int(os.environ.get("PV_MIN_SAMPLES_LEAF", "60")),
    "l2_regularization": float(os.environ.get("PV_L2", "0.0")),
}

FORECAST_FIELDS = [
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "cloud_cover",
    "temperature_2m",
]
NEXT_FIELDS = [f"{field}_next" for field in FORECAST_FIELDS]
FORECAST_ARGS = FORECAST_FIELDS + ["forecasted_for", "lead_days"]
PV_ARG = "pv"
LAG_FIELD = "pv_lag24h"
LAG = pd.Timedelta(hours=24)

# The diagnostic weather-history input. Prefixed so the names never collide with
# the forecast's own.
OBS_FIELDS = [
    "obs_shortwave_radiation",
    "obs_direct_radiation",
    "obs_diffuse_radiation",
    "obs_cloudcover",
    "obs_temperature_2m",
]
OBS_TIME = "obs_weather_time"
OBS_ARGS = OBS_FIELDS + [OBS_TIME]

# Run 4 read weather_time as UTC and found the best alignment at the edge of the
# search; run 5 confirmed it is Europe/Berlin local time with hours labelled by
# their start (shift 0: 37.6 W, shift 1: 38.2 W).
OBS_TIMEZONE = os.environ.get("PV_OBS_TIMEZONE", "Europe/Berlin")
OBS_SHIFTS_H = (0, 1)

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


GEOMETRY_COLUMNS = ["cos_zenith", "elevation", "azimuth_sin", "azimuth_cos", "doy_sin", "doy_cos"]


def build_features(frame: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """One feature row per target bucket.

    `frame` carries a UTC "bucket" column (the hour the row predicts, which is
    also the forecasted_for of the current-hour forecast), FORECAST_FIELDS for
    that hour, NEXT_FIELDS for the hour after and, where known, LAG_FIELD. A
    missing value is NaN, which the regressor handles natively.
    """
    bucket = pd.DatetimeIndex(frame["bucket"])
    values = frame[FORECAST_FIELDS + NEXT_FIELDS].astype(float).set_axis(bucket)
    geometry = _solar_geometry(bucket, lat, lon)
    features = pd.concat([values, geometry], axis=1)
    features["beam_on_horizontal"] = features["direct_normal_irradiance"] * features["cos_zenith"]
    features["shortwave_mean2"] = (
        features[["shortwave_radiation", "shortwave_radiation_next"]].mean(axis=1)
    )
    if LAG_FIELD in frame.columns:
        features[LAG_FIELD] = pd.to_numeric(frame[LAG_FIELD], errors="coerce").to_numpy(dtype=float)
    else:
        features[LAG_FIELD] = np.nan
    return features


FEATURE_COLUMNS = FORECAST_FIELDS + NEXT_FIELDS + GEOMETRY_COLUMNS + [
    "beam_on_horizontal", "shortwave_mean2", LAG_FIELD,
]


def _parse_utc(values) -> pd.Series:
    return pd.to_datetime(pd.Series(values), utc=True, errors="coerce")


def _parse_local_to_utc(values, zone: str) -> typing.Tuple[pd.Series, bool]:
    """Timestamps without an offset read as local time in `zone`, then UTC.

    Returns the parsed series and whether localisation was applied. A timestamp
    that already carries an offset is converted as it stands. The hour that
    repeats at the end of summer time and the hour skipped at its start cannot be
    placed and become NaT rather than being guessed.
    """
    raw = pd.to_datetime(pd.Series(values), errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(raw):
        return raw.dt.tz_convert("UTC"), False
    if not pd.api.types.is_datetime64_any_dtype(raw):
        return _parse_utc(values), False
    local = raw.dt.tz_localize(zone, ambiguous="NaT", nonexistent="NaT")
    return local.dt.tz_convert("UTC"), True


# --------------------------------------------------------------------------- #
# The registered model.
# --------------------------------------------------------------------------- #

class OdeAblationPvL2Model(PythonModel):
    """Maps the forecasts for hour B and B + 1h, and PV at B - 24h, to a prediction for B.

    predict() takes a dict with "bucket" (the hour predicted, ISO 8601 or a
    datetime), FORECAST_FIELDS for that hour, NEXT_FIELDS for the hour after and
    LAG_FIELD (None when unknown). It returns {"time": ISO bucket start,
    "prediction": W}, or None when the bucket cannot be parsed.
    """

    def __init__(self, regressor, lead_days: int, lat: float, lon: float, upper: float) -> None:
        self.regressor = regressor
        self.lead_days = lead_days
        self.lat = lat
        self.lon = lon
        self.upper = upper

    def predict(self, context, model_input=None, params=None):
        payload = model_input if model_input is not None else context
        bucket = _parse_utc([payload.get("bucket")]).iloc[0]
        if pd.isna(bucket):
            return None
        row = {"bucket": [bucket]}
        for field in FORECAST_FIELDS + NEXT_FIELDS + [LAG_FIELD]:
            value = payload.get(field)
            row[field] = [np.nan if value is None else float(value)]
        features = build_features(pd.DataFrame(row), self.lat, self.lon)
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


def _split_inputs(frames: typing.List[pd.DataFrame]):
    pv, forecast, observed = None, None, None
    for frame in frames:
        if OBS_TIME in frame.columns:
            observed = frame
        elif "shortwave_radiation" in frame.columns:
            forecast = frame
        elif PV_ARG in frame.columns:
            pv = frame
    if pv is None or forecast is None:
        raise RuntimeError(
            f"expected one input mapped to '{PV_ARG}' and one carrying the forecast "
            f"fields; got columns {[list(f.columns) for f in frames]}")
    return pv, forecast, observed


def _hourly_target(pv: pd.DataFrame) -> pd.Series:
    """The evaluation's own bucket mean: numeric samples only, [h, h+1h)."""
    times = _parse_utc(pv["time"])
    values = pd.to_numeric(pv[PV_ARG], errors="coerce")
    frame = pd.DataFrame({"bucket": times.dt.floor("1h"), "pv": values.to_numpy()})
    frame = frame.dropna()
    return frame.groupby("bucket")["pv"].mean()


def _forecast_by_hour(forecast: pd.DataFrame, lead_days: int) -> pd.DataFrame:
    """FORECAST_FIELDS at one lead time, indexed by forecasted_for, one row per hour."""
    frame = forecast.copy()
    frame["lead_days"] = pd.to_numeric(frame["lead_days"], errors="coerce")
    frame = frame[frame["lead_days"] == lead_days]
    frame["forecasted_for"] = _parse_utc(frame["forecasted_for"]).to_numpy()
    frame = frame.dropna(subset=["forecasted_for"])
    if "time" in frame.columns:
        frame = frame.sort_values("time", kind="stable")
    frame = frame.drop_duplicates(subset=["forecasted_for"], keep="last")
    return frame.set_index("forecasted_for")[FORECAST_FIELDS].astype(float).sort_index()


def _with_next(by_hour: pd.DataFrame) -> pd.DataFrame:
    """Each hour with its own columns and the columns of the hour after (suffix _next)."""
    nxt = by_hour.copy()
    nxt.index = nxt.index - HOUR
    nxt.columns = [f"{c}_next" for c in by_hour.columns]
    joined = by_hour.join(nxt, how="left")
    joined.index.name = "bucket"
    return joined


def _dataset(target: pd.Series, by_hour: pd.DataFrame) -> pd.DataFrame:
    paired = _with_next(by_hour)
    lag = target.copy()
    lag.index = lag.index + LAG
    paired[LAG_FIELD] = lag.reindex(paired.index).to_numpy()
    features = build_features(paired.reset_index(), LAT, LON)
    features = features[~features.index.duplicated(keep="last")]
    joined = features.join(target.rename("target"), how="inner")
    return joined.dropna(subset=["target"]).sort_index()


def _regressor():
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(loss="absolute_error", random_state=0, **HYPERPARAMS)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b))) if len(a) else float("nan")


def _holdout_mae(data: pd.DataFrame, columns: typing.List[str], cut) -> typing.Tuple[float, float, int]:
    train, hold = data[data.index <= cut], data[data.index > cut]
    if len(train) < 200 or len(hold) == 0:
        return float("nan"), float("nan"), int(len(hold))
    model = _regressor().fit(train[columns].to_numpy(), train["target"].to_numpy())
    pred = np.clip(model.predict(hold[columns].to_numpy()), 0.0, None)
    actual = hold["target"].to_numpy()
    daylight = hold["cos_zenith"].to_numpy() > 0
    return _mae(pred, actual), _mae(pred[daylight], actual[daylight]), int(len(hold))


def _observed_ceiling(observed: pd.DataFrame, target: pd.Series, holdout_index: pd.DatetimeIndex,
                      cut) -> typing.Dict[str, float]:
    """A weather-only model on the weather as it turned out, on the same holdout hours.

    Deliberately without pv_lag24h, so it stays comparable with runs 4 and 5.
    """
    frame = observed.copy()
    hours, localized = _parse_local_to_utc(frame[OBS_TIME], OBS_TIMEZONE)
    frame["hour"] = hours.to_numpy()
    unparsed = float(frame["hour"].isna().mean()) if len(frame) else float("nan")
    frame = frame.dropna(subset=["hour"])
    if "time" in frame.columns:
        frame = frame.sort_values("time", kind="stable")
    frame = frame.drop_duplicates(subset=["hour"], keep="last")
    by_hour = frame.set_index("hour")[OBS_FIELDS].apply(pd.to_numeric, errors="coerce").sort_index()

    report: typing.Dict[str, float] = {
        "observed_hours": float(len(by_hour)),
        "observed_time_localized": 1.0 if localized else 0.0,
        "observed_time_unparsed_ratio": unparsed,
    }
    best = None
    for shift in OBS_SHIFTS_H:
        shifted = by_hour.copy()
        shifted.index = shifted.index - pd.Timedelta(hours=shift)
        paired = _with_next(shifted)
        geometry = _solar_geometry(pd.DatetimeIndex(paired.index), LAT, LON)
        features = pd.concat([paired, geometry], axis=1)
        data = features.join(target.rename("target"), how="inner").dropna(subset=["target"])
        columns = list(paired.columns) + GEOMETRY_COLUMNS
        hold_part = data[data.index.isin(holdout_index)]
        train_part = data[data.index <= cut]
        data = pd.concat([train_part, hold_part])
        mae, mae_day, n = _holdout_mae(data, columns, cut)
        report[f"observed_shift{shift}_holdout_mae"] = mae
        if not math.isnan(mae) and (best is None or mae < best[1]):
            best = (shift, mae, mae_day, n)
    if best is not None:
        report["observed_shift_h"] = float(best[0])
        report["holdout_mae_observed_weather"] = best[1]
        report["holdout_mae_observed_weather_daylight"] = best[2]
        report["observed_holdout_buckets"] = float(best[3])
    return report


@ray.remote
def _fit(datasets: typing.List[typing.Any]) -> typing.Dict[str, typing.Any]:
    """The distributed part: read, align, validate on the tail, fit on everything."""
    frames = [_to_pandas(d) for d in datasets]
    pv, forecast, observed = _split_inputs(frames)
    target = _hourly_target(pv)
    by_hour = _forecast_by_hour(forecast, LEAD_DAYS)
    data = _dataset(target, by_hour)

    report: typing.Dict[str, typing.Any] = {
        "pv_samples": int(len(pv)),
        "target_buckets": int(len(target)),
        "forecast_rows_at_lead": int(len(by_hour)),
        "train_rows": int(len(data)),
        "next_hour_missing_ratio": float(data["shortwave_radiation_next"].isna().mean())
        if len(data) else float("nan"),
        "pv_lag24h_missing_ratio": float(data[LAG_FIELD].isna().mean())
        if len(data) else float("nan"),
    }
    if len(data) < 200:
        raise RuntimeError(f"too few aligned rows to train on: {report}")

    cut = data.index.max() - HOLDOUT
    mae, mae_day, n = _holdout_mae(data, FEATURE_COLUMNS, cut)
    report["holdout_mae"] = mae
    report["holdout_mae_daylight"] = mae_day
    report["holdout_buckets"] = n
    report["holdout_target_mean"] = float(data[data.index > cut]["target"].mean())

    # The same model without the lag, on the same split: what pv_lag24h added.
    no_lag = [c for c in FEATURE_COLUMNS if c != LAG_FIELD]
    report["holdout_mae_without_lag"], _, _ = _holdout_mae(data, no_lag, cut)

    if observed is not None:
        holdout_index = data.index[data.index > cut]
        report.update(_observed_ceiling(observed, target, holdout_index, cut))

    final = _regressor().fit(data[FEATURE_COLUMNS].to_numpy(), data["target"].to_numpy())
    upper = float(np.nanpercentile(data["target"].to_numpy(), 99.9)) * 1.2
    report["train_from"] = data.index.min().isoformat()
    report["train_to"] = data.index.max().isoformat()
    return {"regressor": final, "upper": upper, "report": report}


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
        "features": "current+next hour forecast, pv_lag24h",
        "train_from": report["train_from"],
        "train_to": report["train_to"],
        "lat": LAT,
        "lon": LON,
        "observed_timezone": OBS_TIMEZONE,
        **{f"hp.{key}": value for key, value in HYPERPARAMS.items()},
    })
    logger.log_metrics({
        key: float(value) for key, value in report.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and not (isinstance(value, float) and math.isnan(value))
    })
    return OdeAblationPvL2Model(
        regressor=fitted["regressor"],
        lead_days=LEAD_DAYS,
        lat=LAT,
        lon=LON,
        upper=fitted["upper"],
    )
