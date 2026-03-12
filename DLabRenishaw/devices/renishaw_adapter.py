"""
Device adapter for Renishaw Raman microscope via WiRE ECM API.

Provides Device-compatible interface for stage, camera, and laser control
based on the WiRE External Client Interface specification.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from DLabRenishaw.core.device_base import Device, Stage, Spectrometer, LaserSource
from DLabRenishaw.devices.ecm import ECMConnection, ECMException
from base64 import b64decode
from PIL import Image
from io import BytesIO
import time
import sys

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
    
    def __init__(self, api_url: str = "http://localhost:9880/api", timeout: float = 5.0):
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
        
    def set_debug_mode(self, debug:bool) -> None:
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


class RenishawSpectrometer():
    """
    Spectrometer control for Renishaw Raman system.
    
    Handles spectrum acquisition through WiRE measurement queue system.
    """
    
    def __init__(self, ecm: ECMConnection):
        self.ecm = ecm
        self._integration_time = 1.0  # seconds
        self._accumulations = 1
        self._last_measurement_handle = None
        self.defaultTemplate = "" # String to the default template
        
    
    def initialize(self) -> None:
        """Initialize spectrometer"""
    
    def acquire_single_spectrum(self, template=None, filename=None, timeout=5000):
        """
        Acquire a single Raman spectrum.
        
        Returns
        -------
        wavenumbers : np.ndarray
            Raman shift wavenumber axis (cm⁻¹)
        intensities : np.ndarray
            Spectral intensities (counts)
        """
        if template == None:
            template = self.defaultTemplate
        
        # Create a new measurement on the remote system in the paused state
        handle = self.ecm.call("Queue.Add", paused=True, monitor=False, remoteString=template)
        print("Measurement queued with handle = " + str(handle))
        
        try:

            # Set the data filename on the remote measurement
            if filename is not None:
                filename = self.ecm.call("Measurement.SetFilename", handle=handle, filename=filename, autoincrement=False)
                print(f"File name set to {filename}")

            # Release the measurement to run on the remote system
            _ = self.ecm.call("Queue.Continue", handle=handle)

            # Wait for the measurement to complete
            status = self.ecm.wait(handle=handle, timeout=timeout)

            # If the trigger handling loop has exited and the measurement is not "COMPLETE"
            # then a measurement has timed out or status is "IDLE" due to aborting"
            if status != "COMPLETE":
                print(f"Timeout after {timeout}ms with status {status}. Use --timeout to adjust. Aborting.", file=sys.stderr)
                self.ecm.call("Queue.Abort", handle=handle)
                time.sleep(0.500)
            else:
                filename = self.ecm.call('Measurement.GetFilename', handle=handle)
                print(f"Measurement complete using \"{filename}\"")


        finally:
            # Remove the measurement once completed.
            self.ecm.call("Queue.Remove", handle=handle)
            print("Measurement removed")
    
    def acquire_map_spectrum(self, filename, center, xy_spacing, grid_size, measurement_time, template=None, laserPower = "100", timeout_multiple=1.4, print_process = False, snake=False):
        # Helper Function
        def generate_grid_params(center, xy_spacing, grid_size, row_major=True, snake=False):
            """ This function is used to generate the inputs for rectangleMap fuction in the Renishaw Wire API
            it takes in the below arguments and returns the array of params for passing through the API call

            Args:
                center (list, tuple)(2): List or Tuple of the center of the rectangle array
                xy_spacing (int/float or list, tuple): Spacing between points either as int/float or pair in list/tuple format if different along x and y
                grid_size (int/float or list, tuple): Num of pots either as int or pair in list/tuple format if different along x and y
                row_major (bool, optional): Input for Renishaw Software on whether to scan in rows or columns. Defaults to True.
                snake (bool, optional): Param for Renishaw software to snake or raster scan. Defaults to False.

            Raises:
                ValueError: Errors to center, xy_spacing or grid_size params

            Returns:
                list: Input list for rectangleMap API call in Renishaw Wire Software
            """
            # Ensure center is a list or tuple
            if not isinstance(center, (list, tuple)) or len(center) != 2:
                raise ValueError("center must be a list or tuple of length 2")

            # Handle xy_spacing: can be a single number or a list/tuple
            if isinstance(xy_spacing, (int, float)):
                x_spacing = y_spacing = xy_spacing
            elif isinstance(xy_spacing, (list, tuple)) and len(xy_spacing) == 2:
                x_spacing, y_spacing = xy_spacing
            else:
                raise ValueError("xy_spacing must be a number or a list/tuple of length 2")

            # Handle grid_size: can be a single number or a list/tuple
            if isinstance(grid_size, int):
                nx = ny = grid_size
            elif isinstance(grid_size, (list, tuple)) and len(grid_size) == 2:
                nx, ny = grid_size
            else:
                raise ValueError("grid_size must be an integer or a list/tuple of length 2")

            center_x, center_y = center

            # Calculate the start points
            x_start = center_x - (x_spacing * (nx - 1) / 2)
            y_start = center_y - (y_spacing * (ny - 1) / 2)

            # Create the input array
            params = [
                x_start,    # double xStart
                y_start,    # double yStart
                x_spacing,  # double xStep
                y_spacing,  # double yStep
                nx,         # number nX
                ny,         # number nY
                row_major,  # boolean row_major
                snake       # boolean snake
            ]

            return params

        
        if template == None:
            template = self.defaultTemplate
        
        map_settings = generate_grid_params(center, xy_spacing, grid_size)

        # Create a new measurement on the remote system in the paused state
        handle = self.ecm.call("Queue.Add", paused=True, monitor=False, remoteString=template)
        if print_process: print("Measurement queued with handle = " + str(handle))

        try:

            # Set the data filename on the remote measurement
            if filename is not None:
                filename = self.ecm.call("Measurement.SetFilename", handle=handle, filename=filename)
                if print_process: print(f"File name set to '{filename}'")

            # if measurement_time is not None:
            #     _ = self.ecm.call("Measurement.SetExposure", handle=handle, exposure=measurement_time)
            #     if print_process: print(f"Exposure set to '{measurement_time}'")

            # if laserPower is not None:
            #     power = self.ecm.call("Measurement.SetLaserPower", handle=handle, power=laserPower)
            #     print(f"LaserPower set to '{power}'")
            #     if print_process: print(f"LaserPower set to '{power}'")

            # Configure the measurement into a series measurement
            _ = self.ecm.call("Measurement.SetMap", handle=handle, rectangleParam=map_settings)
            if print_process: print("Series measurement options set")

            # Release the measurement to run on the remote system
            _ = self.ecm.call("Queue.Continue", handle=handle)
            if print_process: print("Begin data collection")

            # Wait for the measurement to complete
            timeout_time=measurement_time*map_settings[5]*map_settings[6]*timeout_multiple
            status = self.ecm.wait(handle=handle, timeout=timeout_time)

            # If the trigger handling loop has exited and the measurement is not "COMPLETE"
            # then a measurement has timed out or status is "IDLE" due to aborting"
            if status != "COMPLETE":
                print(f"Timeout after {timeout_time}ms with status {status}. Use --timeout to adjust. Aborting.", file=sys.stderr)
                self.ecm.call("Queue.Abort", handle=handle)
                time.sleep(0.500)
            else:
                print("Measurement complete")

        finally:
            # Retrieve the currently queued measurement handles to
            # check if the measurement is still in the queue
            handlesPresent = self.ecm.call("Queue.GetHandles")
            if handlesPresent is not None:
                for queuedHandle in handlesPresent:
                    # if the measurement is present remove it from the queue
                    if queuedHandle == handle:
                        # Remove the measurement
                        self.ecm.call("Queue.Remove", handle=handle)
                        print("Measurement with handle = " + str(handle) + " removed")
        
    def acquire_series_spectrum(self, filename, xyList, measurement_time, template=None, timeout_multiple=1.4, print_process = False):

        if template == None:
            template = self.defaultTemplate
            

        # Create a new measurement on the remote system in the paused state
        handle = self.ecm.call("Queue.Add", paused=True, monitor=False, remoteString=template)
        if print_process: print("Measurement queued with handle = " + str(handle))

        try:
            # Set the data filename on the remote measurement
            if filename is not None:
                filename = self.ecm.call("Measurement.SetFilename", handle=handle, filename=filename)
                if print_process: print(f"File name set to '{filename}'")

            # Configure the measurement into a series measurement
            _ = self.ecm.call("Measurement.SetMap", handle=handle, mapXYPoints={'xy_values':xyList})
            if print_process: print("Series measurement options set")

            # Release the measurement to run on the remote system
            _ = self.ecm.call("Queue.Continue", handle=handle)
            if print_process: print("Begin data collection")

            # Wait for the measurement to complete
            timeout_time=measurement_time*len(xyList)*timeout_multiple
            status = self.ecm.wait(handle=handle, timeout=timeout_time)

            # If the trigger handling loop has exited and the measurement is not "COMPLETE"
            # then a measurement has timed out or status is "IDLE" due to aborting"
            if status != "COMPLETE":
                print(f"Timeout after {timeout_time}ms with status {status}. Use --timeout to adjust. Aborting.", file=sys.stderr)
                self.ecm.call("Queue.Abort", handle=handle)
                time.sleep(0.500)
            else:
                print("Measurement complete")

        finally:
            # Retrieve the currently queued measurement handles to
            # check if the measurement is still in the queue
            handlesPresent = self.ecm.call("Queue.GetHandles")
            if handlesPresent is not None:
                for queuedHandle in handlesPresent:
                    # if the measurement is present remove it from the queue
                    if queuedHandle == handle:
                        # Remove the measurement
                        self.ecm.call("Queue.Remove", handle=handle)
                        print("Measurement with handle = " + str(handle) + " removed")
        
        
    def get_template(self) -> str:
        return self.defaultTemplate
    
    def set_template(self, template:str):
        self.defaultTemplate = template


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
        

    
    # def set_exposure(self, exposure_ms: float) -> None:
    #     """
    #     Set camera exposure time.
        
    #     Parameters
    #     ----------
    #     exposure_ms : float
    #         Exposure time in milliseconds
            
    #     Raises
    #     ------
    #     ValueError
    #         If exposure is outside valid range
    #     """
    #     pass
    
    # def set_illumination(self, lamppower: float)->None:
    #     """_summary_
    #         Set lamp power from 0-100%
    #     Args:
    #         lamppower (float): 
    #     """
    #     # Set lamp power
    #     # WiRE.SetIlluminationLamp
    #         # int "illuminationPower (0-255)"
    
    # def get_exposure(self) -> float:
    #     """
    #     Get current exposure time.
        
    #     Returns
    #     -------
    #     float
    #         Exposure time in milliseconds
    #     """
    #     pass
    #     return np.nan
    
    # def get_image_size(self) -> Tuple[int, int]:
    #     """
    #     Get image dimensions.
        
    #     Returns
    #     -------
    #     tuple
    #         (width, height) in pixels
    #     """
    #     # Default implementation - subclasses should override
    #     return (1024, 1024)