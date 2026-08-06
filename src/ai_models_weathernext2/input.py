# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging
from collections import defaultdict

import numpy as np
import xarray as xr

from .convert import GRIB_TO_XARRAY_PL
from .convert import PARAM_TO_XARRAY_SFC

LOG = logging.getLogger(__name__)


def create_training_xarray(
    *,
    fields_sfc,
    fields_pl,
    lagged,
    start_date,
    hour_steps,
    lead_time,
    forcing_variables,
    constants,
    timer,
):
    time_deltas = [
        datetime.timedelta(hours=h)
        for h in lagged + [hour for hour in range(hour_steps, lead_time + hour_steps, hour_steps)]
    ]

    all_datetimes = [start_date + time_delta for time_delta in time_deltas]

    with timer("Converting GRIB to xarray"):
        # Create Input dataset

        lat = fields_sfc[0].metadata("distinctLatitudes")
        lon = fields_sfc[0].metadata("distinctLongitudes")

        # SURFACE FIELDS

        fields_sfc = fields_sfc.order_by("param", "valid_datetime")
        sfc = defaultdict(list)
        given_datetimes = set()
        for field in fields_sfc:
            given_datetimes.add(field.metadata("valid_datetime"))
            sfc[field.metadata("param")].append(field)

        # PRESSURE LEVEL FIELDS

        fields_pl = fields_pl.order_by("param", "valid_datetime", "level")
        pl = defaultdict(list)
        levels = set()
        given_datetimes = set()
        for field in fields_pl:
            given_datetimes.add(field.metadata("valid_datetime"))
            pl[field.metadata("param")].append(field)
            levels.add(field.metadata("level"))

        data_vars = {}

        for param, fields in sfc.items():
            if param in ("z", "lsm"):
                data_vars[PARAM_TO_XARRAY_SFC[param]] = (
                    ["lat", "lon"],
                    fields[0].to_numpy(),
                )
                continue

            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(
                1,
                len(given_datetimes),
                len(lat),
                len(lon),
            )

            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, len(all_datetimes) - len(given_datetimes)),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=(np.nan,),
            )

            data_vars[PARAM_TO_XARRAY_SFC[param]] = (
                ["batch", "time", "lat", "lon"],
                data,
            )

        for param, fields in pl.items():
            data = np.stack([field.to_numpy(dtype=np.float32) for field in fields]).reshape(
                1,
                len(given_datetimes),
                len(levels),
                len(lat),
                len(lon),
            )
            data = np.pad(
                data,
                (
                    (0, 0),
                    (0, len(all_datetimes) - len(given_datetimes)),
                    (0, 0),
                    (0, 0),
                    (0, 0),
                ),
                constant_values=(np.nan,),
            )

            data_vars[GRIB_TO_XARRAY_PL[param]] = (
                ["batch", "time", "level", "lat", "lon"],
                data,
            )

        training_xarray = xr.Dataset(
            data_vars=data_vars,
            coords=dict(
                lon=lon,
                lat=lat,
                time=time_deltas,
                datetime=(
                    ("batch", "time"),
                    [all_datetimes],
                ),
                level=sorted(levels),
            ),
        )

    with timer("Reindexing"):
        # And we want the grid south to north
        training_xarray = training_xarray.reindex(lat=sorted(training_xarray.lat.values), copy=False)

    if constants:
        # Add geopotential_at_surface and land_sea_mask back in
        x = xr.load_dataset(constants)

        for patch in ("geopotential_at_surface", "land_sea_mask"):
            LOG.info("PATCHING %s", patch)
            training_xarray[patch] = x[patch]

    return training_xarray, time_deltas
