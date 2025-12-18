"""
Device adapter for Renishaw Raman microscope via WiRE ECM API.

Provides Device-compatible interface for stage, camera, and laser control
based on the WiRE External Client Interface specification.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from DLabRenishaw.core.device_base import Device, Stage, Spectrometer
from DLabRenishaw.devices.ecm import ECMConnection, ECMException
from base64 import b64decode
from PIL import Image
from io import BytesIO


class RenishawAdapter(Device):
    """
    Main adapter for Renishaw WiRE system.
    
    Provides unified access to stage, spectrometer, and laser components
    through the WiRE ECM API.
    
    Parameters
    ----------
    api_url : str
        Base URL for ECM API (e.g., "'http://localhost:9880/api'")
    timeout : float
        Default timeout for API calls in seconds
    """
    
    def __init__(self, api_url: str = "'http://localhost:9880/api'", timeout: float = 5.0):
        self.api_url = api_url
        self.ecm = ECMConnection(api_url)
        self.ecm.rpctimeout = timeout
        
        # Sub-devices
        self.stage = RenishawStage(self.ecm)
        self.spectrometer = RenishawSpectrometer(self.ecm)
        self.camera = RenishawImage(self.ecm)
        
        self._initialized = False
        self._wire_version = None
    
    def initialize(self) -> None:
        """Initialize connection to Renishaw WiRE system"""
        try:
            # Test connection and get WiRE version
            self._wire_version = self.ecm.call("version")
            print(f"Connected to WiRE version: {self._wire_version}")
            
            # Initialize sub-devices
            self.stage.initialize()
            self.spectrometer.initialize()
            self.camera.initialize()
            
            self._initialized = True
            
        except ECMException as e:
            raise RuntimeError(f"Failed to initialize Renishaw system: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of all components"""
        return {
            'wire_version': self._wire_version,
            'stage': self.stage.get_state(),
            'spectrometer': self.spectrometer.get_state(),
        }
        
    def debug_mode(self, debug:bool) -> None:
        self.ecm.debug = debug
        pass
    
    def get_debug_mode(self)-> bool:
        return self.ecm.debug
    
    def set_optical_path(self, path: int):
        """
        Can use to return or set the optical path
        0 =	Internal Silicon 		        ( Laser and silicon )
        1 =	Sample with video and laser	    ( Laser and sample+video )
        2 =	Standard data collection 	    ( Laser and sample )
        3 =	External 			            ( External )
        4 =	Internal calibration source     ( Internal calibration source )
        5 =	Sample with eyepieces only	    ( Eye piece and sample )
        6 =	Sample with eyepieces and video ( Eye piece and sample+video )
        7 =	Null podule path		        ( Null )	
        8 =	Livetrack plus raman		    ( Livetrack + raman )
        
        """
        if path > 8 or path < 0:
            print("Please enter a valid optical path")
            return 
        try:
            # Attempt to move to the correct path
            self.ecm.call("MoveMotorsToOpticalPath", opticalPath=path)
            
        except ECMException as e:
            raise RuntimeError(f"Failed to Set Optical Path: {e}")
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state of all components"""
        if 'stage' in state:
            self.stage.set_state(state['stage'])
        if 'spectrometer' in state:
            self.spectrometer.set_state(state['spectrometer'])
    
    def shutdown(self) -> None:
        """Disconnect from Renishaw system"""
        if self._initialized:
            self.spectrometer.shutdown()
            self.stage.shutdown()
            self._initialized = False


class RenishawStage(Stage):
    """
    Stage control for Renishaw system.
    
    Provides XYZ positioning control via WiRE ECM API.
    Coordinates are in micrometers.
    """
    
    def __init__(self, ecm: ECMConnection, log:bool = False):
        """
        Initialize stage

        Args:
            ecm (ECMConnection): Connection to Renishaw WireAPI
            log (bool, optional): If want to print additional text during movement for debug. Defaults to False.
        """
        self.ecm = ecm
        self._current_position = np.array([0.,0.,0.])
        self._limits = None
        self.log = False
    
    def initialize(self) -> None:
        """Initialize stage and read current position"""
        try:
            # Get current position from hardware
            self._update_position()
            
            # Get stage limits
            self._limits = self._get_stage_limits()
            
        except ECMException as e:
            print(f"Warning: Could not read initial stage position: {e}")
    
    
    def move_to(self, x: Optional[float] = None, 
                y: Optional[float] = None, 
                z: Optional[float] = None) -> None:
        """
        Move stage to absolute position.
        
        Parameters
        ----------
        x, y, z : float, optional
            Target positions in micrometers. If None, axis is not moved.
        """
        # Get Current Position for if any of the following
        
        # Use current position for any unspecified axes
        target_x = x if x is not None else self._current_position[0] # or some default value
        moveX = True if x is not None else False
        target_y = y if y is not None else self._current_position[1] # or some default value
        moveY = True if y is not None else False
        target_z = z if z is not None else self._current_position[2] # or some default value
        moveZ = True if z is not None else False
        
        
        try:
            # Wire.MoveXYZStage
            self.ecm.call("WiRE.MoveXYZStage", 
                         xTargetPos=float(target_x), 
                         yTargetPos=float(target_y), 
                         zTargetPos=float(target_z),
                         moveX = moveX,
                         moveY = moveY,
                         moveZ = moveZ)
            
            # Update cached position
            self._update_position()
            
            
        except ECMException as e:
            raise RuntimeError(f"Stage move failed: {e}")
    
    def get_position(self) -> np.ndarray:
        """Get current stage position"""
        self._update_position()
        return self._current_position.copy()
    
    def _update_position(self) -> None:
        """Read position from hardware"""
        try:
            # WiRE API: Stages.GetPosition
            pos = self.ecm.call("WiRE.GetXYZStagePosition")
            self._current_position = np.array([pos["xPosition"], pos["yPosition"], pos["zPosition"]])
        except ECMException as e:
            raise RuntimeError(f"Failed to read stage position: {e}")
    
    def _get_stage_limits(self) -> Dict[str, Tuple[float, float]]:
        """Query stage travel limits from hardware
        TODO
        Currently not documented in code - add in in future steps
        """
        return {}
    
    def get_state(self) -> Dict[str, Any]:
        """Get stage state"""
        return {
            'position': self.get_position(),
            'limits': self.get_limits(),
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore stage state"""
        if 'position' in state:
            pos = state['position']
            self.move_to(pos)
    
    def shutdown(self) -> None:
        """No cleanup needed for stage"""
        pass


class RenishawSpectrometer(Spectrometer):
    """
    Spectrometer control for Renishaw Raman system.
    
    Handles spectrum acquisition through WiRE measurement queue system.
    """
    
    def __init__(self, ecm: ECMConnection):
        self.ecm = ecm
        self._integration_time = 1.0  # seconds
        self._accumulations = 1
        self._measurement_type = "Single"
        self._last_measurement_handle = None
    
    def initialize(self) -> None:
        """Initialize spectrometer"""
        try:
            # Could query available measurement types
            measurement_types = self.ecm.call("Measurement.GetTypes")
            if self.ecm.debug:
                print(f"Available measurement types: {measurement_types}")
        except ECMException as e:
            print(f"Warning: Could not query measurement types: {e}")
    
    def acquire_spectrum(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Acquire a single Raman spectrum.
        
        Returns
        -------
        wavenumbers : np.ndarray
            Raman shift wavenumber axis (cm⁻¹)
        intensities : np.ndarray
            Spectral intensities (counts)
        """
        try:
            # Create measurement parameters
            params = {
                "MeasurementType": self._measurement_type,
                "ExposureTime": self._integration_time,
                "Accumulations": self._accumulations,
            }
            
            # Queue measurement
            # WiRE API: Queue.AddMeasurement
            handle = self.ecm.call("Queue.AddMeasurement", **params)
            self._last_measurement_handle = handle
            
            # Start measurement
            # WiRE API: Queue.Start
            self.ecm.call("Queue.Start")
            
            # Wait for completion
            status = self.ecm.wait(handle, timeout=int(self._integration_time * 1000 * 2 + 10000))
            
            if status != "COMPLETE":
                raise RuntimeError(f"Measurement failed with status: {status}")
            
            # Retrieve spectrum data
            # WiRE API: Queue.GetData
            data = self.ecm.call("Queue.GetData", handle=handle)
            
            # Extract wavenumbers and intensities
            # Note: Actual structure depends on WiRE API response format
            # Adjust based on your actual API response
            if isinstance(data, dict):
                wavenumbers = np.array(data.get('wavenumbers', []))
                intensities = np.array(data.get('intensities', []))
            else:
                # If data is returned as list/array, may need different parsing
                wavenumbers = np.array(data[0]) if len(data) > 0 else np.array([])
                intensities = np.array(data[1]) if len(data) > 1 else np.array([])
            
            return wavenumbers, intensities
            
        except ECMException as e:
            raise RuntimeError(f"Spectrum acquisition failed: {e}")
    
    def set_integration_time(self, time_s: float) -> None:
        """
        Set integration time for spectral acquisition.
        
        Parameters
        ----------
        time_s : float
            Integration time in seconds
        """
        if time_s <= 0:
            raise ValueError("Integration time must be positive")
        self._integration_time = time_s
    
    def get_integration_time(self) -> float:
        """
        Get current integration time.
        
        Returns
        -------
        float
            Integration time in seconds
        """
        return self._integration_time
    
    def set_accumulations(self, num_accumulations: int) -> None:
        """
        Set number of accumulations to average.
        
        Parameters
        ----------
        num_accumulations : int
            Number of spectra to accumulate
        """
        if num_accumulations < 1:
            raise ValueError("Accumulations must be >= 1")
        self._accumulations = num_accumulations
    
    def get_accumulations(self) -> int:
        """Get current number of accumulations"""
        return self._accumulations
    
    def set_measurement_type(self, measurement_type: str) -> None:
        """
        Set measurement type.
        
        Parameters
        ----------
        measurement_type : str
            Measurement type (e.g., "Single", "Extended", "StreamLine")
        """
        self._measurement_type = measurement_type
    
    def get_last_measurement_handle(self) -> Optional[str]:
        """Get handle of most recent measurement"""
        return self._last_measurement_handle
    
    def get_state(self) -> Dict[str, Any]:
        """Get spectrometer state"""
        return {
            'integration_time': self._integration_time,
            'accumulations': self._accumulations,
            'measurement_type': self._measurement_type,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore spectrometer state
        
        Empty Function
        
        """
        pass
        # if 'integration_time' in state:
        #     self._integration_time = state['integration_time']
        # if 'accumulations' in state:
        #     self._accumulations = state['accumulations']
        # if 'measurement_type' in state:
        #     self._measurement_type = state['measurement_type']
    
    def shutdown(self) -> None:
        """No cleanup needed for spectrometer"""
        pass
        

    
   

class RenishawMeasurementQueue:
    """
    Helper class for managing WiRE measurement queue.
    
    This class provides higher-level interface to the Queue.* API methods
    for creating, executing, and retrieving measurements.
    """
    
    def __init__(self, ecm: ECMConnection):
        self.ecm = ecm
    
    def add_measurement(self, 
                       measurement_type: str = "Single",
                       exposure: float = 1.0,
                       accumulations: int = 1,
                       paused: bool = False,
                       monitor: bool = False,
                       **kwargs) -> str:
        """
        Add a measurement to the queue.
        
        Parameters
        ----------
        measurement_type : str
            Type of measurement (e.g., "Single", "Extended", "StreamLine")
        exposure : float
            Exposure time in seconds
        accumulations : int
            Number of accumulations
        paused : bool
            If True, measurement starts paused
        monitor : bool
            If True, WiRE will display measurement progress
        **kwargs
            Additional measurement parameters
        
        Returns
        -------
        str
            Measurement handle for subsequent operations
        """
        try:
            # Build measurement parameters
            params = {
                "MeasurementType": measurement_type,
                "ExposureTime": exposure,
                "Accumulations": accumulations,
                "Paused": paused,
                "Monitor": monitor,
            }
            params.update(kwargs)
            
            # WiRE API: Queue.Add
            result = self.ecm.call("Queue.Add", 
                                  paused=paused, 
                                  monitor=monitor,
                                  **params)
            
            # Extract handle from result
            if isinstance(result, dict):
                handle = result.get('handle')
            else:
                handle = result
            
            return handle
            
        except ECMException as e:
            raise RuntimeError(f"Failed to add measurement to queue: {e}")
    
    def continue_measurement(self, handle: str) -> None:
        """
        Continue a paused measurement.
        
        Parameters
        ----------
        handle : str
            Measurement handle
        """
        try:
            # WiRE API: Queue.Continue
            self.ecm.call("Queue.Continue", handle=handle)
        except ECMException as e:
            raise RuntimeError(f"Failed to continue measurement: {e}")
    
    def get_measurement_state(self, handle: str) -> str:
        """
        Get current state of a measurement.
        
        Parameters
        ----------
        handle : str
            Measurement handle
        
        Returns
        -------
        str
            State: "IDLE", "RUNNING", "COMPLETE", "ERROR", etc.
        """
        try:
            # WiRE API: Queue.GetMeasurementState
            state = self.ecm.call("Queue.GetMeasurementState", handle=handle)
            return state
        except ECMException as e:
            raise RuntimeError(f"Failed to get measurement state: {e}")
    
    def wait_for_completion(self, handle: str, timeout: int = 30000) -> str:
        """
        Wait for measurement to complete.
        
        Parameters
        ----------
        handle : str
            Measurement handle
        timeout : int
            Maximum wait time in milliseconds
        
        Returns
        -------
        str
            Final state
        """
        return self.ecm.wait(handle, timeout=timeout)
    
    def get_data(self, handle: str) -> Dict[str, Any]:
        """
        Retrieve data from completed measurement.
        
        Parameters
        ----------
        handle : str
            Measurement handle
        
        Returns
        -------
        dict
            Measurement data including spectra, metadata, etc.
        """
        try:
            # WiRE API: Queue.GetData
            data = self.ecm.call("Queue.GetData", handle=handle)
            return data
        except ECMException as e:
            raise RuntimeError(f"Failed to retrieve measurement data: {e}")
    
    def remove_measurement(self, handle: str) -> None:
        """
        Remove measurement from queue.
        
        Parameters
        ----------
        handle : str
            Measurement handle
        """
        try:
            # WiRE API: Queue.Remove
            self.ecm.call("Queue.Remove", handle=handle)
        except ECMException as e:
            raise RuntimeError(f"Failed to remove measurement: {e}")
    
    def abort_measurement(self, handle: str) -> None:
        """
        Abort a running measurement.
        
        Parameters
        ----------
        handle : str
            Measurement handle
        """
        try:
            # WiRE API: Queue.Abort
            self.ecm.call("Queue.Abort", handle=handle)
        except ECMException as e:
            raise RuntimeError(f"Failed to abort measurement: {e}")
    
    def clear_queue(self) -> None:
        """Clear all measurements from queue"""
        try:
            # WiRE API: Queue.Clear
            self.ecm.call("Queue.Clear")
        except ECMException as e:
            raise RuntimeError(f"Failed to clear queue: {e}")


class RenishawImage:
    """_summary_
    
    """
    def __init__(self, ecm: ECMConnection):
        self.ecm = ecm
        
    def initialize(self):
        pass
        
    def acquire_image(self, ) -> np.ndarray:
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
        wireData = self.ecm.call("WiRE.GetImage") # Call Camera to take brightfield image
        imgData = wireData.pop('data') # 
        imgdata = Image.open(BytesIO(b64decode(imgData)))
        
        return imgdata, wireData
        

    
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
    
    def set_illumination(self, lamppower: float)->None:
        """_summary_
            Set lamp power from 0-100%
        Args:
            lamppower (float): 
        """
        # Set lamp power
        # WiRE.SetIlluminationLamp
            # int "illuminationPower (0-255)"
    
    def get_exposure(self) -> float:
        """
        Get current exposure time.
        
        Returns
        -------
        float
            Exposure time in milliseconds
        """
        pass
        return np.nan
    
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
