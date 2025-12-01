"""
Abstract base classes for all microscope devices.

Defines common interfaces that all hardware devices must implement,
enabling polymorphic device management and state persistence.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np


class Device(ABC):
    """
    Base class for all controllable devices.
    
    All hardware components (cameras, stages, lasers, etc.) inherit from this
    class and must implement its abstract methods for initialization, state
    management, and cleanup.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize hardware connection and prepare device for use.
        
        This method should:
        - Establish communication with hardware
        - Verify device is responding
        - Set device to a known safe state
        - Read initial device parameters
        
        Raises
        ------
        RuntimeError
            If initialization fails
        """
        pass
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Return current device state as a dictionary.
        
        The returned dictionary should contain all device parameters needed
        to restore the device to its current state. This enables state
        persistence and reproducible experiments.
        
        Returns
        -------
        dict
            Dictionary containing current device state
            
        Examples
        --------
        >>> stage.get_state()
        {'x': 1000.0, 'y': 2000.0, 'z': 50.0, 'speed': 100}
        """
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """
        Restore device to a previously saved state.
        
        Parameters
        ----------
        state : dict
            State dictionary as returned by get_state()
            
        Raises
        ------
        ValueError
            If state dictionary is invalid or incomplete
        RuntimeError
            If device cannot be set to requested state
        """
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """
        Safely disconnect from hardware and release resources.
        
        This method should:
        - Return device to a safe state (e.g., disable laser)
        - Close communication connections
        - Release any system resources
        
        Should not raise exceptions - use try/except internally.
        """
        pass
    
    def __enter__(self):
        """Context manager entry - initialize device"""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - shutdown device"""
        self.shutdown()
        return False


class Stage(Device):
    """
    Abstract base class for XYZ positioning stages.
    
    Provides interface for absolute and relative positioning in three dimensions.
    Coordinates are in micrometers unless otherwise specified.
    """
    
    @abstractmethod
    def move_to(self, x: Optional[float] = None, 
                y: Optional[float] = None, 
                z: Optional[float] = None) -> None:
        """
        Move stage to absolute position.
        
        Parameters
        ----------
        x : float, optional
            Target X position in micrometers. If None, X axis not moved.
        y : float, optional
            Target Y position in micrometers. If None, Y axis not moved.
        z : float, optional
            Target Z position in micrometers. If None, Z axis not moved.
            
        Raises
        ------
        RuntimeError
            If movement fails or position is out of range
            
        Examples
        --------
        >>> stage.move_to(x=1000, y=2000)  # Move XY, leave Z unchanged
        >>> stage.move_to(z=50)  # Move only Z axis
        """
        pass
    
    @abstractmethod
    def get_position(self) -> Dict[str, float]:
        """
        Get current stage position.
        
        Returns
        -------
        dict
            Dictionary with keys 'x', 'y', 'z' containing positions in micrometers
            
        Examples
        --------
        >>> stage.get_position()
        {'x': 1000.0, 'y': 2000.0, 'z': 50.0}
        """
        pass
    
    def move_relative(self, dx: float = 0, dy: float = 0, dz: float = 0) -> None:
        """
        Move stage by relative offset.
        
        Parameters
        ----------
        dx, dy, dz : float
            Relative movement in micrometers
            
        Examples
        --------
        >>> stage.move_relative(dx=100)  # Move 100 µm in X
        """
        current = self.get_position()
        self.move_to(
            x=current['x'] + dx,
            y=current['y'] + dy,
            z=current['z'] + dz
        )
    
    def get_limits(self) -> Dict[str, Tuple[float, float]]:
        """
        Get stage travel limits.
        
        Returns
        -------
        dict
            Dictionary with keys 'x', 'y', 'z', values are (min, max) tuples
            
        Examples
        --------
        >>> stage.get_limits()
        {'x': (0, 10000), 'y': (0, 10000), 'z': (0, 1000)}
        """
        # Default implementation - subclasses should override with actual limits
        return {
            'x': (float('-inf'), float('inf')),
            'y': (float('-inf'), float('inf')),
            'z': (float('-inf'), float('inf')),
        }


class Camera(Device):
    """
    Abstract base class for image acquisition devices.
    
    Provides interface for camera control, image acquisition, and ROI management.
    """
    
    @abstractmethod
    def snap_image(self) -> np.ndarray:
        """
        Acquire a single image with current settings.
        
        Returns
        -------
        np.ndarray
            Acquired image as 2D array (grayscale) or 3D array (color)
            
        Raises
        ------
        RuntimeError
            If acquisition fails
            
        Examples
        --------
        >>> image = camera.snap_image()
        >>> print(image.shape)
        (1024, 1280)
        """
        pass
    
    @abstractmethod
    def set_exposure(self, exposure_ms: float) -> None:
        """
        Set camera exposure time.
        
        Parameters
        ----------
        exposure_ms : float
            Exposure time in milliseconds
            
        Raises
        ------
        ValueError
            If exposure is outside valid range
        """
        pass
    
    @abstractmethod
    def get_exposure(self) -> float:
        """
        Get current exposure time.
        
        Returns
        -------
        float
            Exposure time in milliseconds
        """
        pass
    
    def get_image_size(self) -> Tuple[int, int]:
        """
        Get image dimensions.
        
        Returns
        -------
        tuple
            (width, height) in pixels
        """
        # Default implementation - subclasses should override
        return (1024, 1024)
    
    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """
        Set region of interest for acquisition.
        
        Parameters
        ----------
        x, y : int
            Top-left corner of ROI in pixels
        width, height : int
            ROI dimensions in pixels
        """
        # Default implementation does nothing - subclasses override if supported
        pass
    
    def clear_roi(self) -> None:
        """Reset to full sensor area"""
        # Default implementation - subclasses override if supported
        pass


class LaserSource(Device):
    """
    Abstract base class for laser/illumination sources.
    
    Provides interface for power control and enable/disable functionality.
    Safety note: Always ensure laser is disabled during shutdown.
    """
    
    @abstractmethod
    def set_power(self, power_percent: float) -> None:
        """
        Set laser power level.
        
        Parameters
        ----------
        power_percent : float
            Power level as percentage (0-100)
            
        Raises
        ------
        ValueError
            If power is outside valid range (0-100)
        RuntimeError
            If power cannot be set
        """
        pass
    
    @abstractmethod
    def get_power(self) -> float:
        """
        Get current power setting.
        
        Returns
        -------
        float
            Power level as percentage (0-100)
        """
        pass
    
    @abstractmethod
    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable laser output.
        
        Parameters
        ----------
        enabled : bool
            True to enable laser, False to disable
            
        Raises
        ------
        RuntimeError
            If laser state cannot be changed
        """
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """
        Check if laser is currently enabled.
        
        Returns
        -------
        bool
            True if laser is enabled
        """
        pass
    
    def get_wavelength(self) -> Optional[float]:
        """
        Get laser wavelength.
        
        Returns
        -------
        float or None
            Wavelength in nanometers, or None if not applicable
        """
        return None


class FilterWheel(Device):
    """
    Abstract base class for filter wheels and filter management.
    
    Provides interface for selecting filters by position or name.
    """
    
    @abstractmethod
    def set_position(self, position: int) -> None:
        """
        Set filter wheel to specific position.
        
        Parameters
        ----------
        position : int
            Filter position (0-indexed)
            
        Raises
        ------
        ValueError
            If position is out of range
        RuntimeError
            If filter wheel movement fails
        """
        pass
    
    @abstractmethod
    def get_position(self) -> int:
        """
        Get current filter position.
        
        Returns
        -------
        int
            Current position (0-indexed)
        """
        pass
    
    @abstractmethod
    def get_num_positions(self) -> int:
        """
        Get total number of filter positions.
        
        Returns
        -------
        int
            Number of available positions
        """
        pass
    
    def set_filter_by_name(self, name: str) -> None:
        """
        Set filter by name (if filter labels are defined).
        
        Parameters
        ----------
        name : str
            Filter name/label
            
        Raises
        ------
        ValueError
            If filter name not found
        """
        # Default implementation - subclasses override if they support named filters
        raise NotImplementedError("Named filter selection not supported")
    
    def get_filter_names(self) -> Dict[int, str]:
        """
        Get mapping of positions to filter names.
        
        Returns
        -------
        dict
            Dictionary mapping position (int) to name (str)
        """
        # Default implementation - subclasses override
        return {}


class Shutter(Device):
    """
    Abstract base class for shutters.
    
    Simple open/close interface for light path control.
    """
    
    @abstractmethod
    def open(self) -> None:
        """Open the shutter"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the shutter"""
        pass
    
    @abstractmethod
    def is_open(self) -> bool:
        """
        Check if shutter is open.
        
        Returns
        -------
        bool
            True if shutter is open
        """
        pass


class Spectrometer(Device):
    """
    Abstract base class for spectrometers.
    
    Provides interface for spectral acquisition (e.g., Raman spectra).
    """
    
    @abstractmethod
    def acquire_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire a single spectrum.
        
        Returns
        -------
        wavelengths : np.ndarray
            Wavelength or wavenumber axis
        intensities : np.ndarray
            Spectral intensities
            
        Examples
        --------
        >>> wavelengths, intensities = spectrometer.acquire_spectrum()
        >>> plt.plot(wavelengths, intensities)
        """
        pass
    
    @abstractmethod
    def set_integration_time(self, time_ms: float) -> None:
        """
        Set integration time for spectral acquisition.
        
        Parameters
        ----------
        time_ms : float
            Integration time in milliseconds
        """
        pass
    
    @abstractmethod
    def get_integration_time(self) -> float:
        """
        Get current integration time.
        
        Returns
        -------
        float
            Integration time in milliseconds
        """
        pass
    
    def set_wavelength_range(self, min_wl: float, max_wl: float) -> None:
        """
        Set wavelength range for acquisition.
        
        Parameters
        ----------
        min_wl, max_wl : float
            Minimum and maximum wavelength/wavenumber
        """
        # Default implementation - subclasses override if supported
        pass