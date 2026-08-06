# (C) Copyright 2025 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Cyclone tracking and conversion to cyclops format.

Uses WeatherNext's built-in DirectTracker to extract tropical cyclone tracks
from gridded model output, and converts them to the cyclops Track/Point format
for use in ECMWF TC workflows.

The DirectTracker operates on the gridded cyclone fields predicted by the
WeatherNextCyclones models:
  - cyclone_exists_gaussian_unit_mode (existence probability)
  - cyclone_usa_wind_disc (max sustained wind speed)
  - cyclone_usa_pres_disc (central pressure)
  - cyclone_usa_rmw_disc (radius of maximum winds)
  - cyclone_usa_r{34,50,64}_{ne,se,sw,nw}_radius_disc (wind radii)
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

LOG = logging.getLogger(__name__)

# Unit conversion constants
KM_TO_NMI = 1.0 / 1.852
KNOTS_TO_MPS = 1852.0 / 3600.0
MPS_TO_KNOTS = 1.0 / KNOTS_TO_MPS

# WeatherNext tracker DataFrame column names (from weathernext.cyclones.constants)
WN_LAT = "lat"
WN_LON = "lon"
WN_LEAD_TIME = "lead_time"
WN_VALID_TIME = "valid_time"
WN_TRACK_ID = "track_id"
WN_MAX_WIND = "maximum_sustained_wind_speed_knots"
WN_MIN_PRES = "minimum_sea_level_pressure_hpa"
WN_RMW = "radius_of_maximum_winds_km"

# Quadrant radii column names in WN tracker output (km)
WN_R34_NE = "radius_34_knot_winds_ne_km"
WN_R34_SE = "radius_34_knot_winds_se_km"
WN_R34_SW = "radius_34_knot_winds_sw_km"
WN_R34_NW = "radius_34_knot_winds_nw_km"
WN_R50_NE = "radius_50_knot_winds_ne_km"
WN_R50_SE = "radius_50_knot_winds_se_km"
WN_R50_SW = "radius_50_knot_winds_sw_km"
WN_R50_NW = "radius_50_knot_winds_nw_km"
WN_R64_NE = "radius_64_knot_winds_ne_km"
WN_R64_SE = "radius_64_knot_winds_se_km"
WN_R64_SW = "radius_64_knot_winds_sw_km"
WN_R64_NW = "radius_64_knot_winds_nw_km"

# Ordered quadrant radius columns: [r34, r50, r64] x [NE, SE, SW, NW]
WN_RADII_COLUMNS = [
    [WN_R34_NE, WN_R34_SE, WN_R34_SW, WN_R34_NW],
    [WN_R50_NE, WN_R50_SE, WN_R50_SW, WN_R50_NW],
    [WN_R64_NE, WN_R64_SE, WN_R64_SW, WN_R64_NW],
]


def _classify_storm(max_wind_knots: float) -> str:
    """Classify storm intensity using Saffir-Simpson scale.

    Parameters
    ----------
    max_wind_knots : float
        Maximum sustained wind speed in knots.

    Returns
    -------
    str
        Storm classification label: TD, TS, or HR1-HR5.
    """
    if max_wind_knots < 34:
        return "TD"
    elif max_wind_knots < 64:
        return "TS"
    elif max_wind_knots < 83:
        return "HR1"
    elif max_wind_knots < 96:
        return "HR2"
    elif max_wind_knots < 113:
        return "HR3"
    elif max_wind_knots < 137:
        return "HR4"
    else:
        return "HR5"


def run_tracker(
    predictions: xr.Dataset,
    init_time: datetime,
    initial_storms_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run the WeatherNext DirectTracker on gridded cyclone predictions.

    Parameters
    ----------
    predictions : xr.Dataset
        Gridded predictions containing cyclone fields. Must have dimensions
        (time, lat, lon) and contain at minimum the cyclone existence
        probability field.
    init_time : datetime
        Forecast initialisation time.
    initial_storms_df : pd.DataFrame, optional
        Initial storm positions for tracking. If None, cyclogenesis
        detection is used to find new storms.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: track_id, lat, lon, lead_time, valid_time,
        maximum_sustained_wind_speed_knots, minimum_sea_level_pressure_hpa,
        radius_of_maximum_winds_km, and quadrant wind radii.
    """
    from weathernext.cyclones.direct_tracker_6h_v1_config import get_config

    config = get_config()
    tracker = config.tracker_constructor(**config.tracker_kwargs)

    # Ensure init_time is set as a coordinate
    if "init_time" not in predictions.coords:
        predictions = predictions.assign_coords(init_time=pd.Timestamp(init_time))

    # The tracker requires initial_storms_df to be a DataFrame (not None)
    # with at minimum a 'lead_time' column
    if initial_storms_df is None:
        from weathernext.cyclones import constants as wn_constants

        initial_storms_df = pd.DataFrame(columns=[wn_constants.LEAD_TIME, wn_constants.TRACK_ID])

    # Preprocess gridded data into the format expected by the tracker
    processed = tracker.preprocess_gridded_ds(predictions)

    try:
        tracks_df = tracker(
            gridded_ds=processed,
            initial_storms_df=initial_storms_df,
        )
    except (KeyError, IndexError, ValueError) as e:
        LOG.warning(
            "Cyclone tracker failed (may need longer lead time for " "cyclogenesis detection, minimum 2.5 days): %s",
            e,
        )
        return pd.DataFrame()

    LOG.info(
        "Tracked %d points across %d storms",
        len(tracks_df),
        tracks_df[WN_TRACK_ID].nunique() if len(tracks_df) > 0 else 0,
    )

    return tracks_df


def tracks_to_cyclops(
    tracks_df: pd.DataFrame,
    init_time: datetime,
    expver: str = "wn2a",
    member: int = 0,
    fclen: int = 240,
):
    """Convert WeatherNext tracker DataFrame to cyclops Track/TrackFile objects.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        Output from :func:`run_tracker`. Must contain columns for lat, lon,
        lead_time, valid_time, track_id, wind speed, pressure, and radii.
    init_time : datetime
        Forecast base time.
    expver : str
        Experiment version identifier (4 chars).
    member : int
        Ensemble member number.
    fclen : int
        Forecast length in hours.

    Returns
    -------
    dict[str, cyclops.data.TrackFile]
        Dictionary mapping basin name to TrackFile objects. Each TrackFile
        contains all tracks found in that basin.
    """
    from cyclops.basins import BASINS
    from cyclops.data import Point
    from cyclops.data import Track
    from cyclops.data import TrackFile

    if len(tracks_df) == 0:
        return {}

    track_ids = tracks_df[WN_TRACK_ID].unique()
    basin_tracks = {}  # basin_name -> list of Track

    for track_id in track_ids:
        track_rows = tracks_df[tracks_df[WN_TRACK_ID] == track_id].sort_values(WN_LEAD_TIME)

        points = []
        max_wind_overall = 0.0

        for _, row in track_rows.iterrows():
            lat = float(row[WN_LAT])
            lon = float(row[WN_LON])
            valid_time = pd.Timestamp(row[WN_VALID_TIME]).to_pydatetime()

            # Wind speed (knots -> m/s for cyclops Point.vel)
            wind_knots = float(row.get(WN_MAX_WIND, 0.0))
            max_wind_overall = max(max_wind_overall, wind_knots)
            vel_mps = wind_knots * KNOTS_TO_MPS

            # Pressure in hPa
            pres = float(row.get(WN_MIN_PRES, 1013.0))

            # Wind radii: shape (3, 4) for [r34, r50, r64] x [NE, SE, SW, NW]
            # Convert from km to nautical miles
            wind_radii = np.zeros((3, 4), dtype=np.int32)
            for i, radii_row in enumerate(WN_RADII_COLUMNS):
                for j, col in enumerate(radii_row):
                    if col in row.index and not pd.isna(row[col]):
                        wind_radii[i, j] = int(round(max(0.0, float(row[col])) * KM_TO_NMI))

            points.append(
                Point(
                    time=valid_time,
                    lat1=lat,
                    lon1=lon,
                    pres=pres,
                    lat2=lat,  # max wind lat = storm centre (best available)
                    lon2=lon,  # max wind lon = storm centre (best available)
                    vel=vel_mps,
                    wind_radii=wind_radii,
                )
            )

        if not points:
            continue

        # Classify the storm by peak intensity
        label = _classify_storm(max_wind_overall)

        # Determine basin from first point's location
        first_lat = points[0].lat1
        first_lon = points[0].lon1 % 360.0  # ensure 0-360

        basin = _assign_basin(first_lat, first_lon)
        if basin is None:
            LOG.warning(
                "Could not assign basin for track %s at lat=%.1f, lon=%.1f",
                track_id,
                first_lat,
                first_lon,
            )
            continue

        basin_name = basin.name
        if basin_name not in basin_tracks:
            basin_tracks[basin_name] = []

        storm_num = len(basin_tracks[basin_name]) + 1
        tot_num = storm_num

        basin_tracks[basin_name].append(
            Track(
                storm_num=storm_num,
                tot_num=tot_num,
                label=label,
                points=points,
            )
        )

    # Build TrackFile for each basin
    result = {}
    for basin_name, tracks in basin_tracks.items():
        basin = BASINS[basin_name]
        result[basin_name] = TrackFile(
            expver=expver,
            base_time=init_time,
            number=member,
            fclen=fclen,
            basin=basin,
            tracks=tracks,
        )

    return result


def _assign_basin(lat: float, lon: float):
    """Assign a cyclops basin based on latitude and longitude.

    Parameters
    ----------
    lat : float
        Latitude in degrees.
    lon : float
        Longitude in degrees (0-360).

    Returns
    -------
    Basin or None
        The assigned cyclops Basin, or None if outside TC basins.
    """
    from cyclops.basins import BASINS

    # Simple basin assignment based on lat/lon ranges
    # Northern hemisphere
    if lat >= 0:
        if lon >= 260 or lon < 360:  # Atlantic + East Pacific
            if lon >= 260 and lon < 360:
                return BASINS["atl"]
            elif lon >= 180 and lon < 260:
                return BASINS["enp"]
        if lon >= 100 and lon < 180:
            return BASINS["wnp"]
        if lon >= 30 and lon < 100:
            return BASINS["nin"]
    # Southern hemisphere
    else:
        if lon >= 135 and lon < 240:
            return BASINS["spc"]
        if lon >= 90 and lon < 135:
            return BASINS["aus"]
        if lon >= 30 and lon < 90:
            return BASINS["sin"]

    return None


def write_cyclops_tarfile(
    trackfiles: dict,
    output_path: str,
    control: Optional[int] = None,
    hres: Optional[int] = None,
):
    """Write cyclops TrackFiles to a TAR archive.

    Parameters
    ----------
    trackfiles : dict[str, TrackFile]
        Dictionary mapping basin names to TrackFile objects, as returned
        by :func:`tracks_to_cyclops`.
    output_path : str
        Path to the output TAR file.
    control : int, optional
        Member number to label as control (000).
    hres : int, optional
        Member number to label as HRES (da).
    """
    import tarfile

    from cyclops.formats.track import iter_write_tarfile

    with tarfile.open(output_path, "w") as tar:
        iter_write_tarfile(
            tar,
            trackfiles.values(),
            control=control,
            hres=hres,
        )

    LOG.info("Wrote cyclone tracks to %s", output_path)
