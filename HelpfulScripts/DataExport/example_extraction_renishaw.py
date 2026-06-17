import matplotlib.pyplot as plt
from PIL import Image

# Some Examples of Code for Exporting
def get_experiment_metadata(h5_path):
    """Extract global experiment and kinetic metadata from the root of the file."""
    with h5py.File(h5_path, 'r') as f:
        return dict(f.attrs.items())

def get_main_spectra(h5_path):
    """
    Extracts the main spectral data, wavenumber axis, and map attributes.
    Returns: (wavenumbers, raman_data, is_map_boolean, map_metadata_dict)
    """
    map_meta = {}
    with h5py.File(h5_path, 'r') as f:
        grp = f['spectra']
        wavenumbers = grp['wavenumbers'][:]
        
        if 'spectral_cube' in grp:
            dset = grp['spectral_cube']
            raman_data = dset[:]
            is_map = True
            map_meta = dict(dset.attrs.items())
        elif 'spectral_matrix' in grp:
            raman_data = grp['spectral_matrix'][:]
            is_map = False
        else:
            raise KeyError("No standard spectral dataset found.")
            
    return wavenumbers, raman_data, is_map, map_meta

def get_coordinates(h5_path):
    """
    Extracts tracking dimensions into a dictionary of arrays and their units.
    Format: {'Spatial_X': {'data': array, 'unit': 'Micron'}, ...}
    """
    coords = {}
    with h5py.File(h5_path, 'r') as f:
        if 'coordinates' in f:
            for key, dset in f['coordinates'].items():
                coords[key] = {
                    'data': dset[:],
                    'unit': dset.attrs.get('unit', 'Unknown')
                }
    return coords

def get_optical_context(h5_path):
    """
    Extracts images and their crucial spatial alignment metadata.
    """
    images = {}
    with h5py.File(h5_path, 'r') as f:
        if 'optical' in f:
            for img_name, dset in f['optical'].items():
                images[img_name] = {
                    'data': dset[:],
                    'meta': dict(dset.attrs.items())
                }
    return images


file_path = "0_Ethanol_50x_Map_Rd2_unified.h5"

print("--- 1. GLOBAL METADATA & KINETICS ---")
meta = get_experiment_metadata(file_path)
print(f"File:              {meta.get('original_filename')}")
print(f"Measurement Type:  {meta.get('measurement_type')}")
print(f"Laser:             {meta.get('laser_wavelength_nm')} nm")
print(f"Total Duration:    {meta.get('duration_seconds'):.2f} s")
print(f"Acquisition Speed: {meta.get('avg_time_per_spectrum'):.3f} s/point")

print("\n--- 2. RAMAN SPECTRA ---")
wavs, data, is_map, map_meta = get_main_spectra(file_path)
if is_map:
    print(f"Loaded a 3D Spectral Map with shape: {data.shape} (Y, X, Shift)")
    print(f"Map Spatial Span: {map_meta.get('map_span_x')} x {map_meta.get('map_span_y')} microns")
    # Generate a dummy heatmap (e.g., peak intensity)
    heatmap = data.max(axis=2) 
else:
    print(f"Loaded Point Spectra with shape: {data.shape} (Samples, Shift)")
    mean_spectrum = data.mean(axis=0)

print("\n--- 3. HARDWARE TRACKING ---")
coords = get_coordinates(file_path)
for axis, c_dict in coords.items():
    arr = c_dict['data']
    unit = c_dict['unit']
    print(f"Found {axis}: {len(arr)} points | Range: {arr.min():.2f} to {arr.max():.2f} [{unit}]")

print("\n--- 4. OPTICAL IMAGES & ALIGNMENT ---")
images = get_optical_context(file_path)
for name, img_dict in images.items():
    img_array = img_dict['data']
    i_meta = img_dict['meta']
    print(f"Loaded '{name}': {img_array.shape}")
    print(f"  -> Physical Size: {i_meta.get('physical_width'):.2f} x {i_meta.get('physical_height'):.2f} {i_meta.get('unit')}")
    print(f"  -> Origin (X,Y):  ({i_meta.get('origin_x'):.2f}, {i_meta.get('origin_y'):.2f})")
    
    # Optional: Preview image
    # Image.fromarray(img_array).show()