## Methods used for Exporting from Renishaw Wire Files
This folder contains a package with methods for exporting data to a standardized h5 file for Renishaw data
these files are great for working with data


### Unified for Generic h5 Data from Renishaw Experiments
filename_unified.h5
├── /                        # Root Attributes (Global Metadata)
│   ├── original_filename    # e.g., "scan_01.wdf"
│   ├── measurement_type     # "Single", "Series", or "Mapping"
│   ├── laser_wavelength_nm  # Excitation wavelength
│   ├── duration_seconds     # Total elapsed time of the scan
│   └── avg_time_per_spectrum# Calculated acquisition speed
│
├── /spectra                 # Spectral data and independent variables
│   ├── wavenumbers          # 1D array: Raman shift / Wavenumber axis
│   ├── spectral_cube        # 3D array: (Y, X, Intensities) [Present if mapped]
│   │                        # Attributes: map_start_x, map_step_x, map_span_x, etc.
│   └── spectral_matrix      # 2D array: (sample, Intensities) [Present if single/list]
│
├── /coordinates             # Hardware tracking data (Aligned with Spectra)
│   ├── Spatial_X            # 1D array: Absolute X stage coordinates
│   ├── Spatial_Y            # 1D array: Absolute Y stage coordinates
│   ├── Time                 # 1D array: Relative timestamps (seconds)
│   └── [Other]              # Attributes on all: 'unit' and 'annotation'
│
└── /optical                 # Correlative imagery (Native & External)
    ├── whitelight           # 2D/3D array: Native microscope camera snapshot
    │                        # Attributes: physical_width, physical_height, origin_x, map_crop_box_px
    └── [custom_images]      # Appended datasets (e.g., pre_experiment_fluorescence)
