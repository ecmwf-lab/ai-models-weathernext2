# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
import datetime
import logging
import math
import os
import warnings

import numpy as np
from ai_models.model import Model

from .input import create_training_xarray
from .output import save_output_xarray

LOG = logging.getLogger(__name__)


try:
    import haiku as hk
    import jax
    import xarray_jax
    from weathernext.utils import checkpoint as wn_checkpoint
    from weathernext.utils import data_utils
    from weathernext.utils import fiddle_config_io
    from weathernext.utils import rollout
    from weathernext.weathernext2 import fgn

except ModuleNotFoundError as e:
    msg = "You need to install WeatherNext from git to use this model. See README.md for details."
    LOG.error(msg)
    raise ModuleNotFoundError(f"{msg}\n{e}")


jax.config.update("jax_default_matmul_precision", "float32")


class WeatherNext2Base(Model):
    """Base class for WeatherNext 2 and WeatherNext Cyclones models"""

    # Download from GCS
    download_url = "https://storage.googleapis.com/dm_graphcast/weathernext2/{file}"

    grib_edition = 1
    grib_extra_metadata = {"type": "pf"}

    # Input
    area = [90, 0, -90, 360]
    grid = [0.25, 0.25]

    # Config name (relative to weathernext package)
    config_name = "weathernext2/configs/WeatherNext2"

    # Shared by all WN2/Cyclones models
    param_level_pl = (
        ["t", "z", "u", "v", "w", "q"],
        [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
    )

    forcing_variables = []  # Computed internally by data_utils

    use_an = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hour_steps = 6
        self.lagged = [-6, 0]
        self.params = None
        self.ordering = self.param_sfc + [
            f"{param}{level}"
            for param in self.param_level_pl[0]
            for level in self.param_level_pl[1]
        ]

        # Handle ensemble members (same pattern as GenCast)
        if isinstance(self.member_number, str):
            self.member_number = list(set(map(int, self.member_number.split(","))))
        elif isinstance(self.member_number, int):
            self.member_number = [int(self.member_number)]
        elif self.member_number is None:
            self.member_number = list(range(1, self.num_ensemble_members + 1))
        else:
            raise TypeError(
                f"`member_number` must be a string or int, not {type(self.member_number)}"
            )

        if not len(self.member_number) == self.num_ensemble_members:
            raise ValueError(
                f"Number of ensemble members must match `member_number`,\n"
                f"Not {self.num_ensemble_members=} and {self.member_number=}"
            )

        if "stream" not in getattr(self, "metadata", {}):
            self.grib_extra_metadata["stream"] = "enfo"

    @staticmethod
    def _patch_for_gpu(config):
        """Patch config for GPU: switch attention and tune block sizes."""
        transformer_kwargs = config.predictor_kwargs["noisy_function_kwargs"][
            "mesh_model_ctor"
        ].keywords["transformer_kwargs"]
        if transformer_kwargs.get("attention_type") == "splash_mha":
            transformer_kwargs["attention_type"] = "triblockdiag_mha"
            LOG.info(
                "Patched attention_type from splash_mha to triblockdiag_mha "
                "(splash_mha is TPU-only)"
            )

    def load_model(self):
        with self.timer(f"Loading config {self.config_name}"):
            self.config = fiddle_config_io.get_fiddle_config_by_name(self.config_name)

            # Splash Attention is TPU-only; use triblockdiag_mha on GPU/CPU
            if jax.default_backend() != "tpu":
                self._patch_for_gpu(self.config)

        with self.timer(f"Loading checkpoint {self.download_files[0]}"):
            checkpoint_path = os.path.join(self.assets, self.download_files[0])
            with open(checkpoint_path, "rb") as f:
                self.ckpt = wn_checkpoint.load(f, fgn.CheckPoint)
                self.params = self.ckpt.params

            LOG.info("Model description: %s", self.ckpt.description)
            LOG.info("Model license: %s", self.ckpt.license)

        with self.timer("Building JAX model"):
            # Remove ensemble wrapper for inference -- ensemble members are
            # handled via separate forward passes with different RNG keys
            config_inference = fgn.PredictorConfig(
                task=self.config.task,
                predictor_constructor=self.config.predictor_constructor,
                predictor_kwargs=self.config.predictor_kwargs,
                predictor_wrappers=self.config.predictor_wrappers[:-1],
            )

            @hk.transform
            def run_forward(inputs, targets_template, forcings):
                predictor = fgn.construct_predictor(config_inference)
                return predictor(
                    inputs, targets_template=targets_template, forcings=forcings
                )

            run_forward_jitted = jax.jit(
                lambda rng, inputs, targets_template, forcings: run_forward.apply(
                    self.params, rng, inputs, targets_template, forcings
                )
            )

            num_devices = len(jax.local_devices())
            if num_devices > 1:
                self.model = xarray_jax.pmap(run_forward_jitted, dim="sample")
                self._pmap_devices = jax.local_devices()
            else:
                self.model = run_forward_jitted
                self._pmap_devices = None

    def run(self):
        oper_fcst = False
        if self.num_ensemble_members == 0:
            oper_fcst = True
            self.num_ensemble_members = 1
            self.member_number = [0]
            self.grib_extra_metadata = {"type": "fc"}

        if not (self.num_ensemble_members % len(jax.local_devices())) == 0:
            raise ValueError(
                f"Number of ensemble members must be divisible by number of devices, "
                f"not {self.num_ensemble_members} and {len(jax.local_devices())}"
            )

        # Write input fields to output at step 0
        self.write_input_fields(self.fields_sfc, ignore=["tp"], accumulations=["tp"])
        self.write_input_fields(self.fields_pl)

        # Load model
        with self.timer("Building model"):
            self.load_model()

        # Create xarray input
        with self.timer("Creating input data (total)"):
            with self.timer("Creating input data"):
                training_xarray, time_deltas = create_training_xarray(
                    fields_sfc=self.fields_sfc,
                    fields_pl=self.fields_pl,
                    lagged=self.lagged,
                    start_date=self.start_datetime,
                    hour_steps=self.hour_steps,
                    lead_time=self.lead_time,
                    forcing_variables=self.forcing_variables,
                    constants=None,
                    timer=self.timer,
                )

            # Add derived variables (year/day progress)
            data_utils.add_derived_vars(training_xarray)

            # Add NaN placeholders for target-only variables (e.g. cyclone
            # fields) that the model predicts but are not present in input data
            with self.timer("Adding target placeholders"):
                task_config = self.config.task
                all_needed = set(task_config.target_variables) | set(
                    task_config.forcing_variables
                )
                missing = all_needed - set(training_xarray.data_vars)
                if missing:
                    import xarray as xr

                    # Surface-like shape: (batch, time, lat, lon)
                    shape = (
                        training_xarray.sizes["batch"],
                        training_xarray.sizes["time"],
                        training_xarray.sizes["lat"],
                        training_xarray.sizes["lon"],
                    )
                    for var in missing:
                        training_xarray[var] = xr.DataArray(
                            data=np.full(shape, np.nan, dtype=np.float32),
                            dims=["batch", "time", "lat", "lon"],
                        )
                    LOG.info(
                        "Added %d target placeholder variables: %s",
                        len(missing),
                        ", ".join(sorted(missing)),
                    )

            with self.timer("Extracting inputs/targets/forcings"):
                task_config = self.config.task
                (input_xr, template, forcings) = (
                    data_utils.extract_inputs_targets_forcings(
                        training_xarray,
                        target_lead_times=[
                            f"{int(delta.days * 24 + delta.seconds / 3600):d}h"
                            for delta in time_deltas[len(self.lagged) :]
                        ],
                        **dataclasses.asdict(task_config),
                    )
                )

        # Generate random seeds for ensemble members
        rng = jax.random.PRNGKey(0)
        rngs = np.stack(
            [jax.random.fold_in(rng, i) for i in self.member_number], axis=0
        )

        # Run chunked prediction
        from contextlib import nullcontext

        samples_per_chunk = len(self._pmap_devices) if self._pmap_devices else 1
        can_simple_step = self.num_ensemble_members // samples_per_chunk == 1
        stepper = nullcontext()
        if can_simple_step:
            stepper = self.stepper(self.hour_steps)

        # Accumulate cyclone fields for tracking if requested
        cyclone_vars = [
            v for v in task_config.target_variables if v.startswith("cyclone_")
        ]
        accumulate_cyclones = (
            bool(getattr(self, "cyclone_tracks", None)) and cyclone_vars
        )
        # Per-member accumulation: {member_number: [(time_step, ds), ...]}
        cyclone_chunks_by_member = {} if accumulate_cyclones else None

        with stepper:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                for i, chunk in enumerate(
                    rollout.chunked_prediction_generator_multiple_runs(
                        self.model,
                        rngs=rngs,
                        inputs=input_xr,
                        targets_template=template * np.nan,
                        forcings=forcings,
                        num_steps_per_chunk=1,
                        num_samples=self.num_ensemble_members,
                        pmap_devices=self._pmap_devices,
                    )
                ):
                    num_steps = math.ceil(self.lead_time / self.hour_steps)
                    time_step = (i % num_steps) + 1
                    ensemble_chunk = (i // num_steps) * samples_per_chunk
                    member_number_subset = self.member_number[
                        ensemble_chunk : ensemble_chunk + samples_per_chunk
                    ]

                    # Accumulate cyclone fields per member before saving GRIB
                    if accumulate_cyclones:
                        # Keep cyclone vars plus lat/lon coords
                        cyclone_ds = chunk[cyclone_vars].compute()
                        # Ensure lat/lon are present as coordinates
                        for coord in ("lat", "lon"):
                            if coord in chunk.coords and coord not in cyclone_ds.coords:
                                cyclone_ds = cyclone_ds.assign_coords(
                                    {coord: chunk.coords[coord]}
                                )
                        for member in member_number_subset:
                            if member not in cyclone_chunks_by_member:
                                cyclone_chunks_by_member[member] = []
                            cyclone_chunks_by_member[member].append(
                                (time_step, cyclone_ds)
                            )

                    save_output_xarray(
                        output=chunk,
                        write=self.write,
                        target_variables=task_config.target_variables,
                        all_fields=self.all_fields,
                        ordering=self.ordering,
                        time=time_step,
                        hour_steps=self.hour_steps,
                        lagged=self.lagged,
                        oper_fcst=oper_fcst,
                        num_ensemble_members=samples_per_chunk,
                        member_numbers=member_number_subset,
                    )
                    if can_simple_step:
                        stepper(i, time_step * self.hour_steps)

        # Run cyclone tracker and write tracks
        if accumulate_cyclones and cyclone_chunks_by_member:
            self._run_cyclone_tracker(cyclone_chunks_by_member)

    def _run_cyclone_tracker(self, cyclone_chunks_by_member):
        """Run DirectTracker on accumulated cyclone fields and write to TAR.

        Tracks each ensemble member separately and writes all to one TAR.
        """
        import xarray as xr

        from .cyclones import run_tracker, tracks_to_cyclops, write_cyclops_tarfile

        with self.timer("Running cyclone tracker"):
            import pandas as pd

            all_trackfiles = {}  # basin -> list of TrackFile

            for member, chunks in cyclone_chunks_by_member.items():
                # Build xarray with time dimension for this member
                datasets = []
                for time_step, ds in chunks:
                    lead_td = pd.Timedelta(hours=time_step * self.hour_steps)
                    ds = ds.assign_coords(time=[lead_td])
                    datasets.append(ds)

                predictions = xr.concat(datasets, dim="time")
                if "batch" in predictions.dims:
                    predictions = predictions.isel(batch=0, drop=True)
                if "sample" in predictions.coords:
                    predictions = predictions.drop_vars("sample")

                # Run the tracker for this member
                tracks_df = run_tracker(predictions, init_time=self.start_datetime)

                if tracks_df.empty:
                    LOG.info("Member %d: no cyclone tracks found", member)
                    continue

                # Convert to cyclops format
                member_trackfiles = tracks_to_cyclops(
                    tracks_df,
                    init_time=self.start_datetime,
                    member=member,
                    fclen=self.lead_time,
                )

                # Merge into all_trackfiles
                for basin, tf in member_trackfiles.items():
                    if basin not in all_trackfiles:
                        all_trackfiles[basin] = []
                    all_trackfiles[basin].append(tf)

            if not all_trackfiles:
                LOG.info("No cyclone tracks found across any member")
                return

            # Flatten: write all TrackFiles to one TAR
            all_tfs = [tf for tfs in all_trackfiles.values() for tf in tfs]

            output_path = self.cyclone_tracks
            is_deterministic = self.grib_extra_metadata.get("type") == "fc"
            hres_member = self.member_number[0] if is_deterministic else None
            write_cyclops_tarfile(
                {f"{tf.basin.name}_{tf.number}": tf for tf in all_tfs},
                output_path,
                hres=hres_member,
            )

            total_tracks = sum(len(tf.tracks) for tf in all_tfs)
            LOG.info(
                "Wrote %d tracks (%d members, %d basins) to %s",
                total_tracks,
                len(cyclone_chunks_by_member),
                len(all_trackfiles),
                output_path,
            )

    # SCDA stream was decommissioned on 12 May 2026
    _SCDA_CUTOFF = datetime.datetime(2026, 5, 12)

    def patch_retrieve_request(self, r):
        if r.get("class", "od") != "od":
            return

        if r.get("type", "an") not in ("an", "fc"):
            return

        if r.get("stream", "oper") not in ("oper", "scda"):
            return

        if self.use_an:
            r["type"] = "an"
        else:
            r["type"] = "fc"

        time = r.get("time", 12)
        date = r.get("date")

        # Parse date to determine whether SCDA existed
        if isinstance(date, int):
            date = datetime.datetime.strptime(str(date), "%Y%m%d")
        elif isinstance(date, str):
            date = datetime.datetime.strptime(date, "%Y-%m-%d")

        if date < self._SCDA_CUTOFF:
            r["stream"] = {0: "oper", 6: "scda", 12: "oper", 18: "scda"}[time]
        else:
            r["stream"] = "oper"

    def parse_model_args(self, args):
        import argparse

        parser = argparse.ArgumentParser("ai-models weathernext2")
        parser.add_argument(
            "--num-ensemble-members",
            type=int,
            default=0,
            help="Number of ensemble members. 0 means deterministic (type=fc).",
        )
        parser.add_argument(
            "--member-number",
            default=None,
            help="Member numbers, comma-separated. Auto-generated if not given.",
        )
        parser.add_argument("--use-an", action="store_true")
        parser.add_argument(
            "--cyclone-tracks",
            default=None,
            help="Output cyclone tracks to a TAR file (cyclops format). "
            "Requires cyclops to be installed.",
        )
        return parser.parse_args(args)


# WeatherNext2 models (with 100m wind)
class WeatherNext2_2025(WeatherNext2Base):
    config_name = "weathernext2/configs/WeatherNext2"
    expver = "wn2a"

    param_sfc = [
        "lsm",
        "2t",
        "sst",
        "msl",
        "10u",
        "10v",
        "100u",
        "100v",
        "tp",
        "z",
    ]

    download_files = [
        "params/WeatherNext2_<2025_model1.npz",
    ]


# WeatherNextCyclones models (no 100m wind)
class WeatherNextCyclones_2025(WeatherNext2Base):
    config_name = "weathernext2/configs/WeatherNextCyclones"
    expver = "wnca"

    param_sfc = [
        "lsm",
        "2t",
        "sst",
        "msl",
        "10u",
        "10v",
        "tp",
        "z",
    ]

    download_files = [
        "params/WeatherNextCyclones_<2025_model1.npz",
    ]


class WeatherNextCyclones_2024(WeatherNextCyclones_2025):
    download_files = [
        "params/WeatherNextCyclones_<2024_model1.npz",
    ]


class WeatherNextCyclones_2023(WeatherNextCyclones_2025):
    download_files = [
        "params/WeatherNextCyclones_<2023_model1.npz",
    ]


# WeatherNextCyclones Mini (1.0 degree, smaller)
class WeatherNextCyclonesMini_2024(WeatherNext2Base):
    config_name = "weathernext2/configs/WeatherNextCyclones_Mini"
    grid = [1.0, 1.0]
    expver = "wncm"

    param_sfc = [
        "lsm",
        "2t",
        "sst",
        "msl",
        "10u",
        "10v",
        "tp",
        "z",
    ]

    download_files = [
        "params/WeatherNextCyclones_Mini_<2024.npz",
    ]


class WeatherNextCyclonesMini_2023(WeatherNextCyclonesMini_2024):
    download_files = [
        "params/WeatherNextCyclones_Mini_<2023.npz",
    ]
