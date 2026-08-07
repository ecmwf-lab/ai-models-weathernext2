# ai-models-weathernext2
[![Upload Python Package](https://github.com/ecmwf-lab/ai-models-weathernext2/actions/workflows/python-publish.yml/badge.svg)](https://github.com/ecmwf-lab/ai-models-weathernext2/actions/workflows/python-publish.yml)

`ai-models-weathernext2` is an [ai-models](https://github.com/ecmwf-lab/ai-models) plugin to run Google DeepMind's [WeatherNext 2](https://github.com/google-deepmind/weathernext) and WeatherNext Cyclones models.

WeatherNext 2: Accurate and Efficient Global Weather Forecasting with a Functional Generative Network, Nature, 2026. https://www.nature.com/articles/s41586-026-10953-2

The model weights are made available for use under the terms of the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0). You may obtain a copy of the License at: https://creativecommons.org/licenses/by-nc-sa/4.0/.

## Installation

To install the package, run:

```bash
pip install ai-models-weathernext2
```

This will install the package and most of its dependencies.

Then to install WeatherNext dependencies (and JAX on GPU):

> [!CAUTION]
> WeatherNext 2 requires significant GPU and memory resources.

### WeatherNext and JAX

WeatherNext depends on JAX, which needs special installation instructions for your specific hardware.

Please see the [installation guide](https://github.com/google/jax#installation) to follow the correct instructions.

For GPU usage:
```
pip install jax[cuda12]
```

For the slower CPU usage:
```
pip install jax
```

## Available models

| Entry point | Model | Resolution | Description |
| ----------- | ----- | ---------- | ----------- |
| `weathernext2` | WeatherNext2 2025 | 0.25 deg | Full model with 100m wind |
| `weathernext-cyclones` | WeatherNextCyclones 2025 | 0.25 deg | Cyclone-focused model |
| `weathernext-cyclones-2024` | WeatherNextCyclones 2024 | 0.25 deg | Cyclone-focused (2024 weights) |
| `weathernext-cyclones-2023` | WeatherNextCyclones 2023 | 0.25 deg | Cyclone-focused (2023 weights) |
| `weathernext-cyclones-mini` | WeatherNextCyclones Mini | 1.0 deg | Lighter cyclone model |

## Specifying ensemble numbers

There are three ways to control the ensemble members and behaviour of the `WeatherNext2` `ai-model`.

| Description | Args | Result |
| ----------- | ---- | ------ |
| `type=fc`, single member | `--num-ensemble-members 0` | Will create a `grib` file of `type=fc` |
| N members per process with ID = `range(num-ensemble-members)` | `--num-ensemble-members $N>1` | N ensemble members created all in same process, with id from the range|
| N members per process with controlled ID | `--num-ensemble-members $N>1` `--member-number 1,2...N` | N ensemble members created all in same process, with id controlled from `member-number` |

With these approaches it is possible to create either a single forecast, many ensembles in a single process, or many ensembles over many processes.

## Cyclone tracking

When using the cyclone models (`weathernext-cyclones*`), the model predicts gridded cyclone fields (existence probability, wind speed, pressure, wind radii) alongside standard meteorological variables. These gridded fields can be converted into tropical cyclone tracks using WeatherNext's built-in `DirectTracker`.

### Command line

Use `--cyclone-tracks` to run the tracker and write output in [cyclops](https://github.com/ecmwf/cyclops) TAR format:

```bash
ai-models weathernext-cyclones --lead-time 240 --cyclone-tracks tracks.tar
```

The tracker requires at least 2.5 days of forecast data to detect cyclogenesis. For deterministic runs the tracks are tagged as `da` (HRES); for ensemble runs each member is tracked separately and tagged `001`, `002`, etc.

Output files follow the cyclops naming convention:
```
{expver}{basetime}_{member}_{fclen}_{basin}
```

### Python API

```python
from ai_models_weathernext2.cyclones import run_tracker, tracks_to_cyclops, write_cyclops_tarfile

# Run tracker on gridded cyclone predictions (xarray Dataset with time, lat, lon)
tracks_df = run_tracker(predictions, init_time=datetime(2026, 8, 6))

# Convert to cyclops TrackFile objects grouped by basin
trackfiles = tracks_to_cyclops(tracks_df, init_time=datetime(2026, 8, 6), member=0, fclen=240)

# Write to TAR archive
write_cyclops_tarfile(trackfiles, "tracks.tar", hres=0)
```

### Requirements

Cyclone tracking requires the `cyclops` optional dependency:

```bash
pip install ai-models-weathernext2[cyclones]
```
