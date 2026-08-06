# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# GRIB shortName → WN2 xarray name (surface fields)
# Keys use the eccodes shortName convention (t2m, u10, etc.)
GRIB_TO_XARRAY_SFC = {
    "t2m": "2m_temperature",
    "sst": "sea_surface_temperature",
    "msl": "mean_sea_level_pressure",
    "u10": "10m_u_component_of_wind",
    "v10": "10m_v_component_of_wind",
    "u100": "100m_u_component_of_wind",
    "v100": "100m_v_component_of_wind",
    "tp": "total_precipitation_6hr",
    "z": "geopotential_at_surface",
    "lsm": "land_sea_mask",
}

# GRIB shortName → WN2 xarray name (pressure level fields)
GRIB_TO_XARRAY_PL = {
    "t": "temperature",
    "z": "geopotential",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
    "q": "specific_humidity",
}

# MARS param shortNames that differ from eccodes shortNames
# e.g. MARS returns "2t" but eccodes uses "t2m"
GRIB_TO_CF = {
    "2t": "t2m",
    "10u": "u10",
    "10v": "v10",
    "100u": "u100",
    "100v": "v100",
}

# Reverse mapping: eccodes shortName → MARS param shortName
CF_TO_GRIB = {v: k for k, v in GRIB_TO_CF.items()}

# Combined mapping from any GRIB shortName (MARS or eccodes) to WN2 xarray name
# Used by input.py for GRIB → xarray conversion
PARAM_TO_XARRAY_SFC = {
    **GRIB_TO_XARRAY_SFC,
    **{mars: GRIB_TO_XARRAY_SFC[eccodes] for mars, eccodes in GRIB_TO_CF.items() if eccodes in GRIB_TO_XARRAY_SFC},
}
