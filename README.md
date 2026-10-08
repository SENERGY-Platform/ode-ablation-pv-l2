# ode-ablation-pv-l2

A day-ahead forecast of photovoltaic generation for one site, as hourly mean
power in watts, built as an analytics operator for the SENERGY platform.

## Summary: the problem and how it was approached

**The task.** Forecast PV generation 24 hours ahead as hourly mean power (W).
The criterion, fixed in `evaluation.yaml` before any code existed, is the MAE
over a held-out test month, with a threshold of 30 W. The session's data split
trains on everything before 2026-09-01 and evaluates on September 2026.

**Outcome in one line.** The operator works and is evaluated properly, with a
September MAE of about 48 W. It does **not** meet the 30 W threshold. A
diagnostic indicates that no model on the weather data available on this
platform can meet it at this site (see "What limits the error" below).

### 1. Finding the data (real data first, nothing simulated)

- The ontology search found two real PV series. One is an APSystems
  micro-inverter ("Wechselrichter", `root.powerTotal`, about 60 s while
  producing, silent at night). The other is a smart plug metering a small
  string ("Leiste PV", 0 to 300 W). The developer named the inverter as the
  evaluation target.
- No simulation was needed. Both devices hold years of history.
- A day-ahead forecast needs weather that was **known a day earlier**. The
  import "Open-Meteo forecast archive (previous runs, 51.7/10)" stores forecasts
  as they were issued, with `issued_at`, `lead_days` and `forecasted_for`. Its
  export holds history from January 2024. That makes it a source with no
  look-ahead: the model never sees a forecast issued after the moment its
  prediction would have been made.

### 2. Matching the evaluation exactly

Operator Lib's own scoring code, read before any model was written, set three
rules:

- **The target series must be an operator input.** The inverter is therefore
  wired in as input `pv`, and the score resolves against it.
- **Predictions are scored by the hour bucket of their result timestamp.**
  `infer()` therefore stamps each prediction with the hour it is *about*, not
  the hour it was made.
- **The target is the mean of the samples in [h, h + 1h).** Training builds
  its target with the same rule. Because the inverter is silent at night,
  only daytime hours are scored.

### 3. The model

- **Inputs:**
  - the lead-1 forecasts (`lead_days == 1`) for hour B and for hour B + 1 h:
    global, direct, diffuse and beam irradiance, cloud cover and temperature;
  - the sun's position at the middle of the hour, computed for 51.7 / 10;
  - day of year.
- **Regressor:** scikit-learn `HistGradientBoostingRegressor` with absolute-error
  loss, so it is trained directly on the evaluation's metric.
- **At run time:** the operator buffers the archived forecast messages by hour.
  When the forecast for B + 1 h arrives (issued about 23 h before B), it
  predicts B.
- **Validation:** every training pass reports the MAE on its own last 30 days,
  which were held out from fitting. That holdout tracked the scored test month
  within a few watts in every run.

### 4. Experiments (all evaluations on the same split, 407 scored daytime hours)

| Run | Change | Holdout MAE | September MAE |
|---|---|---|---|
| 1 (`865f481`) | Gradient boosting on the lead-1 forecast; alignment of the forecast hour chosen on the holdout | 47.3 W | 52.8 W |
| 2 (`37681cf`) | Forecast for B **and** B + 1 h (the two alignments had tied within 0.5 W) | 45.9 W | 48.9 W |
| 3 (`37681cf` + env) | Stronger regularisation (`min_samples_leaf=60`, `max_leaf_nodes=15`, `lr=0.03`) | 45.0 W | 48.0 W |
| 4 (`d51c794`) | Diagnostic: the same model on the weather *as it turned out* (Open-Meteo history); model unchanged | 45.0 W | 48.0 W |
| 5 (`841ed48`) | Diagnostic fixed: `weather_time` is Europe/Berlin local time, not UTC | 45.0 W | 48.0 W |
| 6 (`a59cd00`) | Added `pv_lag24h`, the inverter's own hourly mean from 24 h earlier | 44.6 W | 48.3 W |

### 5. What limits the error

Runs 4 and 5 asked how good the forecast could be even with perfect weather
knowledge. On the same holdout hours, a model given the weather as it turned
out, correctly aligned, reaches **37.6 W**. The day-ahead forecast model
reaches 45.0 W.

- Only about 7 W of the error is forecast error.
- The remaining ~38 W sits between the weather variables and what the inverter
  reports. Possible causes are clouds smaller than the weather model's grid,
  shading, outages and frozen readings, and partial hours at dawn and dusk.
- Run 6 tested whether the inverter's own recent behaviour captures that
  remainder. It does not.
- So 30 W lies below even the perfect-weather ceiling. Reaching it would need
  data this platform does not have, such as irradiance measured at the panels.
- Caveat: the "weather as it turned out" is Open-Meteo's estimate for a grid
  cell, not a measurement at the site. The true ceiling could be somewhat
  lower, but not by the 8+ W needed.

Whether the threshold should stay at 30 W is the developer's decision.
`evaluation.yaml` was not changed.

### 6. State of this repository, and the recommended cleanup

The committed code (`a59cd00`) still carries two things that did not earn their
place:

- **`pv_lag24h`:** the gain was within noise. It also costs operator state and
  a degraded first day after every restart.
- **The diagnostic `history` input (`obs_*`, `_observed_ceiling`):** it has
  answered its question. Leaving it in makes every deployment wire in a third
  import.

The recommended production version is the run 5 model: two inputs (`pv` and the
forecast archive), September MAE 48.0 W. Remove both items above, then run one
final evaluation, so that the latest recorded result describes the code that
would be deployed.

### Pipeline inputs (current code)

| Input | Source |
|---|---|
| `pv` | Inverter `urn:infai:ses:device:89169393-…`, service `…9ddd93b2-…`, `value.root.powerTotal` |
| forecast fields, `forecasted_for`, `lead_days` | Import `urn:infai:ses:import:6fd7bbee-…` (Open-Meteo previous runs), `value.*` |
| `obs_*` (diagnostic, removable) | Import `urn:infai:ses:import:06e7cfe6-…` (Open-Meteo history), `value.*` |

Tunables are environment variables read at training time, among them
`PV_TRAINING_DAYS` (360), `PV_LEAD_DAYS` (1), `PV_HOLDOUT_DAYS` (30) and
`PV_LR`, `PV_MAX_ITER`, `PV_MAX_LEAF_NODES`, `PV_MIN_SAMPLES_LEAF`, `PV_L2`.
They are logged as MLflow params (`hp.*`).

---

An analytics operator for the SENERGY platform, scaffolded by the Operator
Development Environment. Every file here is yours to change, including this one.

## Layout

| File | What it is |
|---|---|
| "main.py" | Entry point of the deployed operator. Hands the process to Operator Lib. |
| "train.py" | Entry point of an experiment. Trains through Operator Lib, then exits. |
| "op.py" | The operator: "infer", "train", "need_retraining", and its config. |
| "training.py" | The Ray training pass and the model MLflow registers. |
| "pyproject.toml" | Dependencies, with Operator Lib pinned at "v1.8.1". |
| "uv.lock" | The resolved dependencies. Written by the scaffold; refresh it yourself. See below. |
| "Dockerfile" | The image. Built by CI; buildable by hand. |
| ".github/workflows/build.yml" | Builds and pushes "ghcr.io/senergy-platform/ode-ablation-pv-l2". Change the registry here. |
| "operator.yaml" | What the analytics stack registers: inputs, outputs, config. |
| "evaluation.yaml" | Your criteria for whether a run is good, plus what Operator Lib needs to score a test window itself. ODE never writes this. |

## The lock file

The scaffold ran "uv lock" for you and "uv.lock" is in this working copy, uncommitted
like everything else here. Commit it with the rest.

Refresh it whenever you change a dependency in "pyproject.toml", and commit the two
together:

    uv lock

An experiment runs "uv run python train.py" on the cluster, and uv builds the
environment from "pyproject.toml" and this file — on the Ray head for the driver and
on each worker node for the tasks, out of its own cache.

Without a lock file uv resolves at run time, which works and is worse in one
specific way: the run records a commit SHA as the code that produced it, and two
runs of the same commit can then resolve different dependency versions. The lock
file is what makes the recorded SHA describe the whole run rather than only its
source. That is why it is not left to be remembered — and if the scaffold reported
that it could not write one, the command above is the repair.

## Building by hand

    docker build --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t ghcr.io/senergy-platform/ode-ablation-pv-l2:dev .

## The Operator Lib pin

"pyproject.toml" pins Operator Lib at "v1.8.1", the newest at the time
this repository was scaffolded. The library tracks latest and promises no
stability, so moving the pin is a deliberate edit — change it, run "uv lock", and
commit the two together.
