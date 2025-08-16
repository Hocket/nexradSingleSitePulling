#!/usr/bin/env python3
"""
NEXRAD 3D Radar Visualization System
====================================================

A comprehensive system for downloading, processing, and visualizing NEXRAD radar data
in interactive 3D plots using PyVista with timestamp-specific targeting.

- Uses new S3 bucket (unidata-nexrad-level2)
- Added fallback to original bucket
- Better error handling and debugging
- More recent default date
- Alternative data access methods

Requirements:
    pip install numpy matplotlib pyvista boto3 arm_pyart netCDF4 xarray scipy

Usage:
    python nexrad_3d_radar_fixed.py
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import tempfile
import os
from datetime import datetime, timedelta
import pyart
import warnings

warnings.filterwarnings("ignore")


class NEXRADDownloader:
    """
    Class to handle downloading NEXRAD data from AWS S3 bucket - UPDATED VERSION
    """

    def __init__(self):
        # Configure S3 client for anonymous access to NEXRAD data
        self.s3_client = boto3.client(
            "s3", config=Config(signature_version=UNSIGNED), region_name="us-east-1"
        )

        # Try new bucket first, fallback to old one
        self.bucket_names = [
            "unidata-nexrad-level2",  # New bucket
            "noaa-nexrad-level2",  # Old bucket (deprecated Sept 2025)
        ]
        self.active_bucket = None

        # Test which bucket is accessible
        self._test_bucket_access()

    def _test_bucket_access(self):
        """Test which bucket is currently accessible"""
        for bucket in self.bucket_names:
            try:
                # Test a simple list operation
                response = self.s3_client.list_objects_v2(
                    Bucket=bucket, Prefix="2024/", MaxKeys=1
                )
                if response.get("Contents"):
                    self.active_bucket = bucket
                    print(f"Using S3 bucket: {bucket}")
                    return
            except Exception as e:
                print(f"Cannot access bucket {bucket}: {e}")
                continue

        print("Warning: No accessible buckets found. Trying with default...")
        self.active_bucket = self.bucket_names[0]  # Default to new bucket

    def list_available_radars(self, date=None):
        """
        List available radar sites for a given date

        Parameters:
        -----------
        date : datetime, optional
            Date to check for available radars (default: recent date)

        Returns:
        --------
        list : Available radar site codes
        """
        if date is None:
            date = datetime.now() - timedelta(
                days=2
            )  # Use 2 days ago for more reliable data

        prefix = date.strftime("%Y/%m/%d/")
        print(
            f"Searching for radars on {date.strftime('%Y-%m-%d')} with prefix: {prefix}"
        )

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.active_bucket, Prefix=prefix, Delimiter="/"
            )

            radars = []
            if "CommonPrefixes" in response:
                for obj in response["CommonPrefixes"]:
                    radar_code = obj["Prefix"].split("/")[-2]
                    radars.append(radar_code)
                print(
                    f"Found {len(radars)} radar sites for {date.strftime('%Y-%m-%d')}"
                )
            else:
                print("No CommonPrefixes found in response")
                print(f"Response keys: {response.keys()}")

            return sorted(radars)

        except Exception as e:
            print(f"Error listing radars: {e}")
            print(f"Bucket: {self.active_bucket}, Prefix: {prefix}")
            return []

    def list_radar_files(self, radar_site, date=None, limit=1000):
        """
        List available files for a specific radar site and date

        Parameters:
        -----------
        radar_site : str
            4-letter radar site code (e.g., 'KTLX')
        date : datetime, optional
            Date to search (default: recent date)
        limit : int
            Maximum number of files to return

        Returns:
        --------
        list : Available file keys
        """
        if date is None:
            date = datetime.now() - timedelta(days=2)

        prefix = f"{date.strftime('%Y/%m/%d')}/{radar_site}/"
        print(f"Searching for files with prefix: {prefix}")

        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.active_bucket,
                Prefix=prefix,
                MaxKeys=limit,  # Get more files to filter
            )

            files = []
            if "Contents" in response:
                for obj in response["Contents"]:
                    key = obj["Key"]
                    if key.endswith(".gz") or key.endswith("_V06"):
                        files.append(key)
                print(f"Found {len(files)} radar files")
            else:
                print("No files found in response")

            return sorted(files)[-limit:]  # Return most recent files

        except Exception as e:
            print(f"Error listing files: {e}")
            print(f"Bucket: {self.active_bucket}, Prefix: {prefix}")
            return []

    def find_radar_file_by_timestamp(
        self, radar_site, target_datetime, tolerance_minutes=30
    ):
        """
        Find radar file closest to a specific timestamp

        Parameters:
        -----------
        radar_site : str
            4-letter radar site code (e.g., 'KLWX')
        target_datetime : datetime
            Target date and time for radar data
        tolerance_minutes : int
            Maximum time difference to accept (default: 30 minutes)

        Returns:
        --------
        str or None : S3 key of closest radar file, or None if not found
        """
        print(f"Searching for {radar_site} data near {target_datetime}")

        # Search files for the target date
        files = self.list_radar_files(radar_site, target_datetime, limit=1000)

        if not files:
            print(
                f"No files found for {radar_site} on {target_datetime.strftime('%Y-%m-%d')}"
            )
            return None

        best_file = None
        min_time_diff = timedelta(hours=24)  # Initialize with large value

        for file_key in files:
            try:
                # Extract timestamp from filename
                # V06 filename format: KLWX20160621_000202_V06
                # Legacy format: RADAR_YYYYMMDD_HHMMSS_V06
                filename = file_key.split("/")[-1]

                # Handle different filename formats
                #if filename.endswith(".gz"):
                #    filename = filename[:-3]  # Remove .gz

                # Debug print
                print(f"Processing file: {filename}")

                # Parse V06 format: KLWX20160621_000202_V06
                if filename.startswith(radar_site) and filename.endswith("_V06"):
                    # Remove radar site prefix and V06 suffix
                    timestamp_part = filename[len(radar_site) :]  # Remove KLWX
                    if timestamp_part.endswith("_V06"):
                        timestamp_part = timestamp_part[:-4]  # Remove _V06

                    parts = timestamp_part.split("_")
                    print(f"  V06 format - timestamp parts: {parts}")

                    if len(parts) == 2:
                        date_str = parts[0]  # YYYYMMDD (20160621)
                        time_str = parts[1]  # HHMMSS (000202)

                        file_datetime = datetime.strptime(
                            f"{date_str}_{time_str}", "%Y%m%d_%H%M%S"
                        )
                        time_diff = abs(target_datetime - file_datetime)

                        #print(f"  File time: {file_datetime}, diff: {time_diff}")

                        if time_diff < min_time_diff and time_diff <= timedelta(
                            minutes=tolerance_minutes
                        ):
                            min_time_diff = time_diff
                            best_file = file_key

                # Also try legacy format: RADAR_YYYYMMDD_HHMMSS_V06
                else:
                    parts = filename.split("_")
                    print(f"  Legacy format - parts: {parts}")

                    if len(parts) >= 3:
                        date_str = parts[1]  # YYYYMMDD
                        time_str = parts[2]  # HHMMSS

                        file_datetime = datetime.strptime(
                            f"{date_str}_{time_str}", "%Y%m%d_%H%M%S"
                        )
                        time_diff = abs(target_datetime - file_datetime)

                        #print(f"  File time: {file_datetime}, diff: {time_diff}")

                        if time_diff < min_time_diff and time_diff <= timedelta(
                            minutes=tolerance_minutes
                        ):
                            min_time_diff = time_diff
                            best_file = file_key

            except (ValueError, IndexError) as e:
                # Skip files with unexpected naming format
                print(f"Skipping file {filename}: {e}")
                continue

        if best_file:
            print(f"Found radar file: {best_file}")
            print(f"Time difference: {min_time_diff}")
        else:
            print(
                f"No radar file found within {tolerance_minutes} minutes of {target_datetime}"
            )

        return best_file

    def get_available_timestamps(self, radar_site, date):
        """
        Get all available timestamps for a radar site on a specific date

        Parameters:
        -----------
        radar_site : str
            4-letter radar site code
        date : datetime
            Date to search

        Returns:
        --------
        list : List of datetime objects for available radar scans
        """
        files = self.list_radar_files(radar_site, date, limit=200)
        timestamps = []

        for file_key in files:
            try:
                filename = file_key.split("/")[-1]
                if filename.endswith(".gz"):
                    filename = filename[:-3]

                # Parse V06 format: KLWX20160621_000202_V06
                if filename.startswith(radar_site) and filename.endswith("_V06"):
                    # Remove radar site prefix and V06 suffix
                    timestamp_part = filename[len(radar_site) :]  # Remove KLWX
                    if timestamp_part.endswith("_V06"):
                        timestamp_part = timestamp_part[:-4]  # Remove _V06

                    parts = timestamp_part.split("_")

                    if len(parts) == 2:
                        date_str = parts[0]  # YYYYMMDD (20160621)
                        time_str = parts[1]  # HHMMSS (000202)

                        file_datetime = datetime.strptime(
                            f"{date_str}_{time_str}", "%Y%m%d_%H%M%S"
                        )
                        timestamps.append(file_datetime)

                # Also try legacy format
                else:
                    parts = filename.split("_")

                    if len(parts) >= 3:
                        date_str = parts[1]  # YYYYMMDD
                        time_str = parts[2]  # HHMMSS

                        file_datetime = datetime.strptime(
                            f"{date_str}_{time_str}", "%Y%m%d_%H%M%S"
                        )
                        timestamps.append(file_datetime)

            except (ValueError, IndexError):
                continue

        return sorted(timestamps)

    def download_radar_file(self, file_key):
        """
        Download a NEXRAD file from S3

        Parameters:
        -----------
        file_key : str
            S3 key for the radar file

        Returns:
        --------
        str : Path to downloaded temporary file
        """
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".ar2v")
        temp_file.close()
        print(f"Downloading from bucket: {self.active_bucket}")
        print(f"File key: {file_key}")

        try:
            self.s3_client.download_file(self.active_bucket, file_key, temp_file.name)
            print(f"Downloaded to: {temp_file.name}")
            return temp_file.name
        except Exception as e:
            print(f"Error downloading file: {e}")
            return None


class RadarProcessor:
    """
    Class to process NEXRAD radar data and prepare for 3D visualization
    """

    def __init__(self):
        pass

    def load_radar_data(self, file_path):
        """
        Load radar data using PyART

        Parameters:
        -----------
        file_path : str
            Path to radar file

        Returns:
        --------
        pyart.core.Radar : Radar object
        """
        try:
            print(f"Loading radar data from: {file_path}")
            radar = pyart.io.read_nexrad_archive(file_path)
            print(f"Successfully loaded radar data with {radar.nsweeps} sweeps")
            return radar
        except Exception as e:
            print(f"Error loading radar data: {e}")
            return None

    def extract_reflectivity_data(self, radar):
        """
        Extract reflectivity data from all elevation angles

        Parameters:
        -----------
        radar : pyart.core.Radar
            Radar object from PyART

        Returns:
        --------
        dict : Dictionary containing processed radar data
        """
        if radar is None:
            return None

        # Get reflectivity field (usually 'reflectivity' or 'REF')
        ref_field = None
        available_fields = list(radar.fields.keys())
        print(f"Available fields: {available_fields}")

        for field in ["reflectivity", "REF", "DBZH", "DBZ"]:
            if field in radar.fields:
                ref_field = field
                break

        if ref_field is None:
            print(f"No reflectivity field found. Available fields: {available_fields}")
            return None

        print(f"Using reflectivity field: {ref_field}")

        # Extract data for all sweeps (elevation angles)
        data = {
            "reflectivity": [],
            "elevation_angles": [],
            "azimuth": [],
            "range": [],
            "coordinates": {"x": [], "y": [], "z": []},
            "radar_info": {
                "site_name": radar.metadata.get("instrument_name", "Unknown"),
                "lat": radar.latitude["data"][0],
                "lon": radar.longitude["data"][0],
                "alt": radar.altitude["data"][0],
            },
        }

        # Process each sweep (elevation angle)
        for sweep_idx in range(radar.nsweeps):
            sweep_slice = radar.get_slice(sweep_idx)
            elevation = radar.elevation["data"][sweep_slice]
            azimuth = radar.azimuth["data"][sweep_slice]

            # Get range data
            range_data = radar.range["data"]

            # Get reflectivity data for this sweep
            ref_data = radar.fields[ref_field]["data"][sweep_slice]

            # Convert to Cartesian coordinates
            x, y, z = self._polar_to_cartesian(
                range_data,
                azimuth,
                elevation,
                radar.latitude["data"][0],
                radar.longitude["data"][0],
                radar.altitude["data"][0],
            )

            data["reflectivity"].append(ref_data)
            data["elevation_angles"].append(np.mean(elevation))
            data["azimuth"].append(azimuth)
            data["range"].append(range_data)
            data["coordinates"]["x"].append(x)
            data["coordinates"]["y"].append(y)
            data["coordinates"]["z"].append(z)

        print(f"Processed {len(data['reflectivity'])} elevation sweeps")
        return data

    def _polar_to_cartesian(self, range_data, azimuth, elevation, lat, lon, alt):
        """
        Convert polar coordinates to Cartesian coordinates

        Parameters:
        -----------
        range_data : array
            Range values in meters
        azimuth : array
            Azimuth angles in degrees
        elevation : array
            Elevation angles in degrees
        lat, lon, alt : float
            Radar location

        Returns:
        --------
        tuple : (x, y, z) coordinates
        """
        # Create meshgrid for range and azimuth
        range_mesh, az_mesh = np.meshgrid(range_data, np.deg2rad(azimuth))
        elev_mesh = np.meshgrid(range_data, np.deg2rad(elevation))[1]

        # Convert to Cartesian coordinates
        x = range_mesh * np.cos(elev_mesh) * np.sin(az_mesh)
        y = range_mesh * np.cos(elev_mesh) * np.cos(az_mesh)
        z = range_mesh * np.sin(elev_mesh) + alt

        return x, y, z


class Radar3DVisualizer:
    """
    Class for creating interactive 3D visualizations of radar data using PyVista
    """

    def __init__(self):
        # Use a dark theme for better contrast with radar colors
        pv.set_plot_theme("dark")

    def create_3d_plot(
        self,
        radar_data,
        elevation_indices=None,
        field_name="reflectivity",
        reflectivity_threshold=0,
        max_range_km=200,
    ):
        """
        Create interactive 3D plot of radar data.

        Parameters:
        -----------
        radar_data : dict
            Processed radar data from RadarProcessor.
        elevation_indices : list[int], optional
            Specific elevation sweeps to plot (default: all sweeps).
        field_name : str
            Field to visualize (default: "reflectivity").
        reflectivity_threshold : float
            Minimum field value to display.
        max_range_km : float
            Maximum range to display in kilometers.
        """
        if radar_data is None:
            print("No radar data provided.")
            return

        if field_name not in radar_data:
            print(f"Field '{field_name}' not found in radar_data.")
            print(f"Available fields: {list(radar_data.keys())}")
            return

        # Create PyVista plotter
        plotter = pv.Plotter(window_size=[1200, 800])
        plotter.add_text(
            f"NEXRAD Radar: {radar_data['radar_info'].get('site_name', 'Unknown Site')}",
            position="upper_left",
            font_size=16,
        )

        colormap = "turbo"  # works well for radar data

        if elevation_indices is None:
            elevation_indices = range(len(radar_data[field_name]))

        any_points_added = False

        # Loop over sweeps
        for sweep_idx in elevation_indices:
            if sweep_idx >= len(radar_data[field_name]):
                continue

            field_data = radar_data[field_name][sweep_idx]
            x = radar_data["coordinates"]["x"][sweep_idx]
            y = radar_data["coordinates"]["y"][sweep_idx]
            z = radar_data["coordinates"]["z"][sweep_idx]

            # Filter by range and threshold
            max_range_m = max_range_km * 1000
            range_mask = np.sqrt(x**2 + y**2) <= max_range_m
            ref_mask = field_data >= reflectivity_threshold

            # Keep valid points (ignore masked values)
            combined_mask = range_mask & ref_mask & ~np.ma.getmaskarray(field_data)

            if not np.any(combined_mask):
                continue

            any_points_added = True

            # Create 3D points array
            points = np.column_stack([
                x[combined_mask].flatten(),
                y[combined_mask].flatten(),
                z[combined_mask].flatten(),
            ])

            values = field_data[combined_mask].flatten()

            # Create PyVista mesh
            cloud = pv.PolyData(points)
            cloud[field_name] = values

            plotter.add_mesh(
                cloud,
                scalars=field_name,
                cmap=colormap,
                clim=[-20, 80],  # typical dBZ range
                point_size=4,
                render_points_as_spheres=True,
                opacity=0.85,
                scalar_bar_args={
                    "title": f"{field_name.capitalize()}",
                    "vertical": True,
                    "n_labels": 5,
                },
            )

        if not any_points_added:
            print("No points passed the filter — try lowering the threshold or increasing range.")
            return

        plotter.add_axes()
        plotter.show_grid()
        plotter.show()


if __name__ == "__main__":
    downloader = NEXRADDownloader()

    # Use specific date and time: 2016-08-21 17:50:37 UTC
    target_time = datetime(2016, 6, 21, 17, 50, 37)

    # List available radar sites on that day
    radars = downloader.list_available_radars(date=target_time)
    print(f"Available radars on {target_time.date()}: {radars}")

    # Choose a radar site; KTLX if available, else first in list
    radar_site = "KLWX" if "KLWX" in radars else (radars[0] if radars else None)

    if radar_site is None:
        print("No radar sites available for this date.")
    else:
        # Find radar file near the target timestamp
        radar_file_key = downloader.find_radar_file_by_timestamp(
            radar_site, target_time, tolerance_minutes=2
        )

        if radar_file_key:
            # Download radar file
            file_path = downloader.download_radar_file(radar_file_key)

            # Process radar data
            processor = RadarProcessor()
            radar = processor.load_radar_data(file_path)
            radar_data = processor.extract_reflectivity_data(radar)

            # Visualize
            visualizer = Radar3DVisualizer()
            visualizer.create_3d_plot(
                radar_data,
                reflectivity_threshold=10,
                max_range_km=150,
            )

            # Cleanup temp file
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        else:
            print("No suitable radar file found near the specified time.")