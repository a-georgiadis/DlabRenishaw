# 
# Based from Renishaw Wire WDF GUI information extraction package
#
# Written by: Antony Georgiadis

import os
import glob
import json
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from PIL import Image
import struct

from wdf import Wdf, WdfBlockId, WdfFlags, WdfDataUnit, WdfType, WdfScanType, WdfDataType, WdfSpectrumFlags


def extract_wdf_data(
    input_folder, 
    output_folder, 
    file_pattern="*.wdf",
    image_format="png",
    spectral_image_format="hdf5"
):
    """
    Extract all data from WDF files into structured output.
    
    Parameters:
    -----------
    input_folder : str
        Path to folder containing WDF files
    output_folder : str
        Path to folder where extracted data will be saved
    file_pattern : str
        Glob pattern for matching WDF files (default: "*.wdf")
    image_format : str
        Format for saving images (default: "png", options: "png", "tiff", "jpg")
    spectral_image_format : str
        Format for 3D spectral images (default: "hdf5", options: "hdf5", "npy", "csv_flat")
    
    Returns:
    --------
    list : Paths to created output folders
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all WDF files
    wdf_files = list(input_path.glob(file_pattern))
    
    if not wdf_files:
        print(f"No files matching '{file_pattern}' found in {input_folder}")
        return []
    
    print(f"Found {len(wdf_files)} WDF file(s) to process")
    
    output_folders = []
    
    for wdf_file in wdf_files:
        print(f"\nProcessing: {wdf_file.name}")
        try:
            output_folder_path = process_single_wdf(
                wdf_file, 
                output_path, 
                image_format, 
                spectral_image_format
            )
            output_folders.append(output_folder_path)
            print(f"  ✓ Successfully processed to {output_folder_path.name}")
        except Exception as e:
            print(f"  ✗ Error processing {wdf_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Completed: {len(output_folders)}/{len(wdf_files)} files processed successfully")
    return output_folders

def process_single_wdf(wdf_path, output_base, image_format, spectral_image_format):
    """Process a single WDF file and extract all data."""
    
    print(f"  DEBUG: Starting to process {wdf_path}")
    
    # Create output folder for this file
    file_stem = wdf_path.stem
    output_folder = output_base / file_stem
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"  DEBUG: Created output folder: {output_folder}")
    
    # Load WDF file
    print(f"  DEBUG: Attempting to load WDF file...")
    wdf = Wdf(str(wdf_path))
    print(f"  DEBUG: WDF loaded successfully")
    print(f"  DEBUG: nspectra={wdf.hdr.nspectra}, ncollected={wdf.hdr.ncollected}")
    print(f"  DEBUG: xlistcount={wdf.hdr.xlistcount}, ylistcount={wdf.hdr.ylistcount}")
    print(f"  DEBUG: npoints={wdf.hdr.npoints}")
    print(f"  DEBUG: scantype={WdfScanType(wdf.hdr.scantype).name}")
    
    # Check if map_area exists - this is the ONLY reliable way to detect a map
    has_map_area = False
    map_area_info = None
    try:
        if hasattr(wdf, 'map_area') and wdf.map_area is not None:
            has_map_area = True
            ma = wdf.map_area
            map_area_info = {
                'x_count': ma.count.x,
                'y_count': ma.count.y,
                'z_count': ma.count.z,
            }
            print(f"  DEBUG: map_area found: {ma.count.y}x{ma.count.x} (z={ma.count.z})")
    except Exception as e:
        print(f"  DEBUG: No map_area or error accessing it: {e}")
    
    # 1. Extract metadata
    print(f"  DEBUG: Extracting metadata...")
    metadata = extract_metadata(wdf)
    metadata_path = output_folder / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"  DEBUG: Metadata saved")
    
    # 2. Determine if this is a spectral image (map)
    # A file is a spectral map if and ONLY if map_area exists with 2D spatial dimensions
    is_spectral_image = False
    
    if has_map_area and map_area_info:
        # If map_area exists with x_count > 1 AND y_count > 1, it's a 2D map
        if map_area_info['x_count'] > 1 and map_area_info['y_count'] > 1:
            is_spectral_image = True
            # Verify that ncollected matches the expected grid
            expected = map_area_info['x_count'] * map_area_info['y_count']
            if wdf.hdr.ncollected != expected:
                print(f"  WARNING: ncollected ({wdf.hdr.ncollected}) != map dimensions ({expected})")
            print(f"  DEBUG: Detected as 2D spectral map")
        elif map_area_info['x_count'] > 1 or map_area_info['y_count'] > 1:
            # 1D line scan
            print(f"  DEBUG: Detected as 1D line scan (treating as regular spectra)")
    
    print(f"  DEBUG: is_spectral_image={is_spectral_image}")
    
    if is_spectral_image:
        print(f"  DEBUG: Exporting spectral image...")
        print(f"  DEBUG: Spatial grid from map_area: {map_area_info['y_count']}x{map_area_info['x_count']}, {wdf.hdr.npoints} wavenumber points per spectrum")
        export_spectral_image(wdf, output_folder, spectral_image_format, map_area_info)
    else:
        print(f"  DEBUG: Exporting spectra to CSV...")
        print(f"  DEBUG: {wdf.hdr.ncollected} spectra, {wdf.hdr.npoints} wavenumber points per spectrum")
        export_spectra_csv(wdf, output_folder)
    
    # 3. Extract whitelight image if present
    print(f"  DEBUG: Attempting to extract whitelight image...")
    try:
        extract_whitelight_image(wdf, output_folder, image_format)
    except Exception as e:
        print(f"    No whitelight image or error extracting: {e}")
    
    print(f"  DEBUG: Finished processing {wdf_path.name}")
    return output_folder


def extract_metadata(wdf):
    """Extract all metadata from WDF file into a dictionary."""
    
    metadata = {
        "header": {},
        "sections": {},
        "comment": wdf.comment() if hasattr(wdf, 'comment') else None
    }
    
    # Extract header information
    hdr = wdf.hdr
    metadata["header"] = {
        "title": hdr.title,
        "appname": hdr.appname,
        "appversion": hdr.appversion,
        "flags": [flag.name for flag in WdfFlags if flag.value & hdr.flags],
        "laser_wavenumber": hdr.laser_wavenumber,
        "naccum": hdr.naccum,
        "nspectra": hdr.nspectra,
        "npoints": hdr.npoints,
        "ncollected": hdr.ncollected,
        "ntracks": hdr.ntracks,
        "units": WdfDataUnit(hdr.units).name,
        "type": WdfType(hdr.type).name,
        "scantype": WdfScanType(hdr.scantype).name,
        "xlistcount": hdr.xlistcount,
        "ylistcount": hdr.ylistcount,
        "user": hdr.user,
        "time_start": str(hdr.time_start),
        "time_end": str(hdr.time_end),
        "uuid": str(hdr.uuid),
    }
    
    # Extract information from each section
    WdfBlockIdMap = dict([(getattr(WdfBlockId, name), name)
                          for name in dir(WdfBlockId) if not name.startswith('__')])
    
    for section in wdf.sections():
        section_name = WdfBlockIdMap.get(section.id, f"Unknown_{section.id}")
        
        if section.uid > 0:
            section_key = f"{section_name}_{section.uid}"
        else:
            section_key = section_name
        
        # Get section properties if available
        try:
            props = wdf.get_section_properties(section.id, section.uid)
            if props:
                metadata["sections"][section_key] = pset_to_dict(props)
            else:
                metadata["sections"][section_key] = {"size": section.size}
        except:
            metadata["sections"][section_key] = {"size": section.size}
    
    # Extract map area if present - wrap in try/except
    try:
        if hasattr(wdf, 'map_area') and wdf.map_area is not None:
            ma = wdf.map_area
            metadata["map_area"] = {
                "flags": ma.flags.name if hasattr(ma.flags, 'name') else str(ma.flags),
                "start": {"x": ma.start.x, "y": ma.start.y, "z": ma.start.z},
                "step": {"x": ma.step.x, "y": ma.step.y, "z": ma.step.z},
                "count": {"x": ma.count.x, "y": ma.count.y, "z": ma.count.z},
            }
    except Exception as e:
        print(f"    Note: No map area data (this is normal for non-map spectra)")
    
    # Extract origins information
    if hasattr(wdf, 'origins') and wdf.origins:
        metadata["origins"] = {}
        for key, origin in wdf.origins.items():
            origin_key = key.name if hasattr(key, 'name') else str(key)
            metadata["origins"][origin_key] = {
                "label": origin.label,
                "datatype": origin.datatype.name,
                "dataunit": origin.dataunit.name,
                "is_primary": origin.is_primary,
                "count": len(origin)
            }
    
    return metadata


def pset_to_dict(pset):
    """Convert a Pset object to a dictionary."""
    from wdf import PsetType
    
    result = {}
    for key, item in pset:
        if item.type == PsetType.Pset:
            result[key] = pset_to_dict(item.value)
        else:
            result[key] = item.value
    return result


def export_spectra_csv(wdf, output_folder):
    """Export 1D spectra to CSV in wide format."""
    
    # For regular spectra (not maps):
    # xlistcount = number of wavenumber points
    # ncollected = number of spectra
    
    # Get wavenumber/x-axis values
    # Handle multitrack data
    track = 0
    if (wdf.hdr.flags & WdfFlags.Multitrack 
        and wdf.hdr.ntracks == wdf.hdr.nspectra):
        # For multitrack, each spectrum might have its own x-axis
        # We'll use the first track's x-axis
        track = 0
    
    xlist = wdf.xlist(track)
    
    # Build dataframe
    data = {"wavenumber": xlist}
    
    for i in range(wdf.hdr.ncollected):
        spectrum = wdf[i]
        data[f"spectrum_{i}"] = spectrum
    
    df = pd.DataFrame(data)
    
    csv_path = output_folder / "spectral_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"    Saved {wdf.hdr.ncollected} spectra with {len(xlist)} points each to CSV")

def export_spectral_image(wdf, output_folder, format="hdf5", map_area_info=None):
    """Export 3D spectral image data."""
    
    # Determine spatial dimensions from map_area if available, otherwise from header
    if map_area_info:
        y_count = map_area_info['y_count']
        x_count = map_area_info['x_count']
        print(f"    Using dimensions from map_area: {y_count}x{x_count}")
    else:
        y_count = wdf.hdr.ylistcount
        x_count = wdf.hdr.xlistcount
        print(f"    Using dimensions from header: {y_count}x{x_count}")
    
    n_points = wdf.hdr.npoints
    
    print(f"    Processing spectral image: {y_count}x{x_count} spatial grid with {n_points} wavenumber points each")
    
    # Build 3D cube
    spectral_cube = np.zeros((y_count, x_count, n_points))
    
    for i in range(wdf.hdr.ncollected):
        y_idx = i // x_count
        x_idx = i % x_count
        
        # Get spectrum - convert tuple/list to numpy array
        spectrum_data = wdf[i]
        spectrum = np.array(spectrum_data)
        
        # Ensure it's 1D
        if len(spectrum.shape) > 1:
            spectrum = spectrum.flatten()
        
        # Verify length matches expected points
        if len(spectrum) != n_points:
            print(f"    WARNING: Spectrum {i} has {len(spectrum)} points, expected {n_points}")
            # Pad or trim as needed
            if len(spectrum) < n_points:
                spectrum = np.pad(spectrum, (0, n_points - len(spectrum)), mode='constant')
            else:
                spectrum = spectrum[:n_points]
        
        spectral_cube[y_idx, x_idx, :] = spectrum
    
    # Get wavenumber array
    wavenumbers = np.array(wdf.xlist(0))
    
    # Get spatial coordinates from map_area
    has_map_coords = False
    x_coords = np.arange(x_count, dtype=float)
    y_coords = np.arange(y_count, dtype=float)
    z_coord = 0.0
    
    try:
        if hasattr(wdf, 'map_area') and wdf.map_area is not None:
            ma = wdf.map_area
            
            # Calculate actual coordinates based on start, step, and count
            x_coords = ma.start.x + np.arange(x_count) * ma.step.x
            y_coords = ma.start.y + np.arange(y_count) * ma.step.y
            z_coord = ma.start.z
            
            has_map_coords = True
            print(f"    Using spatial coordinates from map_area:")
            print(f"      X: {x_coords[0]:.2f} to {x_coords[-1]:.2f} (step: {ma.step.x})")
            print(f"      Y: {y_coords[0]:.2f} to {y_coords[-1]:.2f} (step: {ma.step.y})")
            print(f"      Z: {z_coord:.2f}")
                
    except Exception as e:
        print(f"    Could not extract map_area coordinates: {e}")
        print(f"    Using index-based coordinates instead")
    
    if format == "hdf5":
        hdf5_path = output_folder / "spectral_image.h5"
        with h5py.File(hdf5_path, 'w') as f:
            # Save the spectral cube
            f.create_dataset('spectral_cube', data=spectral_cube, compression='gzip')
            f.create_dataset('wavenumbers', data=wavenumbers)
            f.create_dataset('x_coords', data=x_coords)
            f.create_dataset('y_coords', data=y_coords)
            
            # Add attributes with metadata
            f.attrs['shape_description'] = '(y, x, wavenumber)'
            f.attrs['y_count'] = y_count
            f.attrs['x_count'] = x_count
            f.attrs['n_wavenumbers'] = n_points
            f.attrs['has_spatial_coords'] = has_map_coords
            
            if has_map_coords:
                f.attrs['x_start'] = float(x_coords[0])
                f.attrs['x_end'] = float(x_coords[-1])
                f.attrs['x_step'] = float(x_coords[1] - x_coords[0] if len(x_coords) > 1 else 0)
                f.attrs['y_start'] = float(y_coords[0])
                f.attrs['y_end'] = float(y_coords[-1])
                f.attrs['y_step'] = float(y_coords[1] - y_coords[0] if len(y_coords) > 1 else 0)
                f.attrs['z_position'] = float(z_coord)
                f.attrs['spatial_units'] = 'micrometers'
            
        print(f"    Saved spectral image to HDF5: {spectral_cube.shape}")
    
    elif format == "npy":
        np.save(output_folder / "spectral_cube.npy", spectral_cube)
        np.save(output_folder / "wavenumbers.npy", wavenumbers)
        np.save(output_folder / "x_coords.npy", x_coords)
        np.save(output_folder / "y_coords.npy", y_coords)
        
        # Save metadata as JSON
        map_metadata = {
            'shape': list(spectral_cube.shape),
            'shape_description': '(y, x, wavenumber)',
            'has_spatial_coords': has_map_coords,
        }
        if has_map_coords:
            map_metadata.update({
                'x_range': [float(x_coords[0]), float(x_coords[-1])],
                'y_range': [float(y_coords[0]), float(y_coords[-1])],
                'z_position': float(z_coord),
                'spatial_units': 'micrometers'
            })
        
        with open(output_folder / "spectral_image_metadata.json", 'w') as f:
            json.dump(map_metadata, f, indent=2)
        
        print(f"    Saved spectral image to NPY files: {spectral_cube.shape}")
    
    elif format == "csv_flat":
        # Flatten to CSV with actual coordinates
        rows = []
        for y_idx in range(y_count):
            for x_idx in range(x_count):
                row = {
                    'y_index': y_idx,
                    'x_index': x_idx,
                    'x_coord': float(x_coords[x_idx]),
                    'y_coord': float(y_coords[y_idx]),
                }
                if has_map_coords:
                    row['z_coord'] = float(z_coord)
                
                for wvn_idx, wvn in enumerate(wavenumbers):
                    row[f'wvn_{wvn:.2f}'] = spectral_cube[y_idx, x_idx, wvn_idx]
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_folder / "spectral_image_flat.csv", index=False)
        print(f"    Saved flattened spectral image to CSV")

def extract_whitelight_image(wdf, output_folder, image_format="png"):
    """Extract whitelight image if present."""
    
    # Try to get the whitelight image section
    try:
        stream = wdf.get_section_stream(WdfBlockId.WHITELIGHT, 0)
        image = Image.open(stream)
        
        # Save with specified format
        image_path = output_folder / f"whitelight.{image_format}"
        image.save(image_path)
        print(f"    Saved whitelight image: {image.size}")
        
    except Exception as e:
        # No whitelight image present
        raise e

