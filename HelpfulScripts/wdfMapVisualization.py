# 
# Generated with Claude 4-5 sonnet on November 30th for Looking at WDF Spectra
# 


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import h5py
from pathlib import Path
from PIL import Image

def explore_spectral_image(h5_file_path):
    """
    Explore and visualize a spectral image HDF5 file.
    
    Parameters:
    -----------
    h5_file_path : str
        Path to the HDF5 file
    """
    
    with h5py.File(h5_file_path, 'r') as f:
        # Print file structure
        print("=" * 60)
        print(f"HDF5 File: {h5_file_path}")
        print("=" * 60)
        
        # List all datasets
        print("\nDatasets:")
        for key in f.keys():
            dataset = f[key]
            print(f"  {key}: shape={dataset.shape}, dtype={dataset.dtype}")
        
        # Print attributes
        print("\nAttributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")
        
        # Load data
        spectral_cube = f['spectral_cube'][:]
        wavenumbers = f['wavenumbers'][:]
        x_coords = f['x_coords'][:]
        y_coords = f['y_coords'][:]
        
        print("\n" + "=" * 60)
        print("Data Summary:")
        print("=" * 60)
        print(f"Spectral cube shape: {spectral_cube.shape} (y, x, wavenumber)")
        print(f"Number of spectra: {spectral_cube.shape[0] * spectral_cube.shape[1]}")
        print(f"Wavenumber range: {wavenumbers[0]:.2f} to {wavenumbers[-1]:.2f} cm⁻¹")
        print(f"Number of wavenumber points: {len(wavenumbers)}")
        
        if f.attrs.get('has_spatial_coords', False):
            print(f"\nSpatial coordinates:")
            print(f"  X: {x_coords[0]:.2f} to {x_coords[-1]:.2f} µm")
            print(f"  Y: {y_coords[0]:.2f} to {y_coords[-1]:.2f} µm")
            print(f"  Z: {f.attrs.get('z_position', 0):.2f} µm")
        else:
            print(f"\nUsing index-based coordinates")
        
        print(f"\nIntensity range: {spectral_cube.min():.2f} to {spectral_cube.max():.2f}")
        
    return spectral_cube, wavenumbers, x_coords, y_coords


def plot_spectral_image(h5_file_path, wavenumber_idx=None, save_plots=False):
    """
    Create visualizations of the spectral image.
    
    Parameters:
    -----------
    h5_file_path : str
        Path to the HDF5 file
    wavenumber_idx : int, optional
        Index of wavenumber to visualize (default: middle of range)
    save_plots : bool
        Whether to save plots to disk
    """
    
    with h5py.File(h5_file_path, 'r') as f:
        spectral_cube = f['spectral_cube'][:]
        wavenumbers = f['wavenumbers'][:]
        x_coords = f['x_coords'][:]
        y_coords = f['y_coords'][:]
        has_spatial_coords = f.attrs.get('has_spatial_coords', False)
    
    y_count, x_count, n_wvn = spectral_cube.shape
    
    # Default to middle wavenumber if not specified
    if wavenumber_idx is None:
        wavenumber_idx = n_wvn // 2
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Spatial map at specific wavenumber
    ax1 = plt.subplot(2, 3, 1)
    intensity_map = spectral_cube[:, :, wavenumber_idx]
    
    if has_spatial_coords:
        extent = [x_coords[0], x_coords[-1], y_coords[0], y_coords[-1]]
        im1 = ax1.imshow(intensity_map, extent=extent, origin='lower', cmap='viridis')
        ax1.set_xlabel('X position (µm)')
        ax1.set_ylabel('Y position (µm)')
    else:
        im1 = ax1.imshow(intensity_map, origin='lower', cmap='viridis')
        ax1.set_xlabel('X index')
        ax1.set_ylabel('Y index')
    
    ax1.set_title(f'Intensity at {wavenumbers[wavenumber_idx]:.1f} cm⁻¹')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # 2. Mean spectrum across all spatial points
    ax2 = plt.subplot(2, 3, 2)
    mean_spectrum = spectral_cube.mean(axis=(0, 1))
    ax2.plot(wavenumbers, mean_spectrum, 'b-', linewidth=1)
    ax2.axvline(wavenumbers[wavenumber_idx], color='r', linestyle='--', 
                label=f'Selected: {wavenumbers[wavenumber_idx]:.1f} cm⁻¹')
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Intensity')
    ax2.set_title('Mean Spectrum')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Single spectrum from center point
    ax3 = plt.subplot(2, 3, 3)
    center_y, center_x = y_count // 2, x_count // 2
    center_spectrum = spectral_cube[center_y, center_x, :]
    ax3.plot(wavenumbers, center_spectrum, 'g-', linewidth=1)
    ax3.axvline(wavenumbers[wavenumber_idx], color='r', linestyle='--')
    ax3.set_xlabel('Wavenumber (cm⁻¹)')
    ax3.set_ylabel('Intensity')
    ax3.set_title(f'Spectrum at center point [{center_y}, {center_x}]')
    ax3.grid(True, alpha=0.3)
    
    # 4. Integrated intensity map (sum over all wavenumbers)
    ax4 = plt.subplot(2, 3, 4)
    integrated_map = spectral_cube.sum(axis=2)
    
    if has_spatial_coords:
        im4 = ax4.imshow(integrated_map, extent=extent, origin='lower', cmap='hot')
        ax4.set_xlabel('X position (µm)')
        ax4.set_ylabel('Y position (µm)')
    else:
        im4 = ax4.imshow(integrated_map, origin='lower', cmap='hot')
        ax4.set_xlabel('X index')
        ax4.set_ylabel('Y index')
    
    ax4.set_title('Total Integrated Intensity')
    plt.colorbar(im4, ax=ax4, label='Intensity')
    
    # 5. Standard deviation map (spectral variability)
    ax5 = plt.subplot(2, 3, 5)
    std_map = spectral_cube.std(axis=2)
    
    if has_spatial_coords:
        im5 = ax5.imshow(std_map, extent=extent, origin='lower', cmap='plasma')
        ax5.set_xlabel('X position (µm)')
        ax5.set_ylabel('Y position (µm)')
    else:
        im5 = ax5.imshow(std_map, origin='lower', cmap='plasma')
        ax5.set_xlabel('X index')
        ax5.set_ylabel('Y index')
    
    ax5.set_title('Spectral Variability (Std Dev)')
    plt.colorbar(im5, ax=ax5, label='Std Dev')
    
    # 6. Spectra from multiple points overlaid
    ax6 = plt.subplot(2, 3, 6)
    
    # Sample a few spectra from different locations
    n_samples = min(5, y_count * x_count)
    indices = np.random.choice(y_count * x_count, n_samples, replace=False)
    
    for idx in indices:
        y_idx = idx // x_count
        x_idx = idx % x_count
        spectrum = spectral_cube[y_idx, x_idx, :]
        ax6.plot(wavenumbers, spectrum, alpha=0.7, linewidth=1, 
                label=f'[{y_idx},{x_idx}]')
    
    ax6.set_xlabel('Wavenumber (cm⁻¹)')
    ax6.set_ylabel('Intensity')
    ax6.set_title(f'Sample Spectra (n={n_samples})')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        output_path = h5_file_path.replace('.h5', '_visualization.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {output_path}")
    
    plt.show()


def get_spectrum_at_position(h5_file_path, x_idx, y_idx):
    """
    Extract spectrum at a specific spatial position.
    
    Parameters:
    -----------
    h5_file_path : str
        Path to the HDF5 file
    x_idx : int
        X index position
    y_idx : int
        Y index position
    
    Returns:
    --------
    wavenumbers, spectrum : numpy arrays
    """
    with h5py.File(h5_file_path, 'r') as f:
        wavenumbers = f['wavenumbers'][:]
        spectrum = f['spectral_cube'][y_idx, x_idx, :]
    
    return wavenumbers, spectrum


def get_map_at_wavenumber(h5_file_path, wavenumber_target):
    """
    Extract spatial map at a specific wavenumber (finds closest match).
    
    Parameters:
    -----------
    h5_file_path : str
        Path to the HDF5 file
    wavenumber_target : float
        Target wavenumber in cm⁻¹
    
    Returns:
    --------
    intensity_map, actual_wavenumber : 2D array, float
    """
    with h5py.File(h5_file_path, 'r') as f:
        wavenumbers = f['wavenumbers'][:]
        
        # Find closest wavenumber
        idx = np.argmin(np.abs(wavenumbers - wavenumber_target))
        actual_wavenumber = wavenumbers[idx]
        
        intensity_map = f['spectral_cube'][:, :, idx]
    
    print(f"Requested: {wavenumber_target} cm⁻¹, Found: {actual_wavenumber:.2f} cm⁻¹")
    
    return intensity_map, actual_wavenumber
