"""
Device adapter for Renishaw Raman microscope via WiRE ECM API.

Provides Device-compatible interface for stage, camera, and laser control.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from DLabRenishaw.core.device_base import Device, Stage, Camera, LaserSource
from DLabRenishaw.devices.ecm import ECMConnection, ECMException


class RenishawAdapter(Device):
    """
    Main adapter for Renishaw Raman system.
    
    Provides unified access to stage, camera, and laser components
    through the WiRE ECM API.
    
    Parameters
    ----------
    api_url : str
        Base URL for ECM API (e.g., "http://192.168.1.100:8080")
    timeout : float
        Default timeout for API calls in seconds
    """
    
    def __init__(self, api_url: str, timeout: float = 5.0):
        self.api_url = api_url
        self.ecm = ECMConnection(api_url)
        self.ecm.rpctimeout = timeout
        
        # Sub-devices
        self.stage = RenishawStage(self.ecm)
        self.camera = RenishawCamera(self.ecm)
        self.laser = RenishawLaser(self.ecm)
        
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize connection to Renishaw system"""
        try:
            # Test connection
            version = self.ecm.call("System.GetVersion")
            print(f"Connected to WiRE version: {version}")
            
            # Initialize sub-devices
            self.stage.initialize()
            self.camera.initialize()
            self.laser.initialize()
            
            self._initialized = True
            
        except ECMException as e:
            raise RuntimeError(f"Failed to initialize Renishaw system: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current state of all components"""
        return {
            'stage': self.stage.get_state(),
            'camera': self.camera.get_state(),
            'laser': self.laser.get_state(),
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore state of all components"""
        if 'stage' in state:
            self.stage.set_state(state['stage'])
        if 'camera' in state:
            self.camera.set_state(state['camera'])
        if 'laser' in state:
            self.laser.set_state(state['laser'])
    
    def shutdown(self) -> None:
        """Disconnect from Renishaw system"""
        self.stage.shutdown()
        self.camera.shutdown()
        self.laser.shutdown()
        self._initialized = False


class RenishawStage(Stage):
    """Stage control for Renishaw system"""
    
    def __init__(self, ecm: ECMConnection):
        self.ecm = ecm
        self._current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
    
    def initialize(self) -> None:
        """Initialize stage and read current position"""
        try:
            # Get current position from hardware
            self._update_position()
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
        # Use current position for any unspecified axes
        target_x = x if x is not None else self._current_position['x']
        target_y = y if y is not None else self._current_position['y']
        target_z = z if z is not None else self._current_position['z']
        
        try:
            self.ecm.call("Stage.MoveTo", x=target_x, y=target_y, z=target_z)
            self._current_position = {'x': target_x, 'y': target_y, 'z': target_z}
        except ECMException as e:
            raise RuntimeError(f"Stage move failed: {e}")
    
    def get_position(self) -> Dict[str, float]:
        """Get current stage position"""
        self._update_position()
        return self._current_position.copy()
    
    def _update_position(self) -> None:
        """Read position from hardware"""
        try:
            pos = self.ecm.call("Stage.GetPosition")
            self._current_position = {
                'x': pos.get('x', 0.0),
                'y': pos.get('y', 0.0),
                'z': pos.get('z', 0.0),
            }
        except ECMException as e:
            raise RuntimeError(f"Failed to read stage position: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """Get stage state"""
        return {'position': self.get_position()}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore stage state"""
        if 'position' in state:
            pos = state['position']
            self.move_to(x=pos['x'], y=pos['y'], z=pos['z'])
    
    def shutdown(self) -> None:
        """No cleanup needed for stage"""
        pass


class RenishawCamera(Camera):
    """Camera control for Renishaw system"""
    
    def __init__(self, ecm: ECMConnection):
        self.ecm = ecm
        self._exposure = 1.0  # seconds
    
    def initialize(self) -> None:
        """Initialize camera"""
        pass
    
    def snap_image(self) -> np.ndarray:
        """
        Acquire single image.
        
        Returns
        -------
        np.ndarray
            Acquired image
        """
        try:
            # Start acquisition
            handle = self.ecm.call("Camera.Acquire", exposure=self._exposure)
            
            # Wait for completion
            status = self.ecm.wait(handle)
            
            if status != "COMPLETE":
                raise RuntimeError(f"Acquisition failed with status: {status}")
            
            # Retrieve image data
            image_data = self.ecm.call("Camera.GetImage", handle=handle)
            
            # Convert to numpy array (adjust based on actual API response)
            return np.array(image_data)
            
        except ECMException as e:
            raise RuntimeError(f"Image acquisition failed: {e}")
    
    def set_exposure(self, exposure_s: float) -> None:
        """Set exposure time in seconds"""
        self._exposure = exposure_s
    
    def get_exposure(self) -> float:
        """Get current exposure time in seconds"""
        return self._exposure
    
    def get_state(self) -> Dict[str, Any]:
        """Get camera state"""
        return {'exposure': self._exposure}
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore camera state"""
        if 'exposure' in state:
            self._exposure = state['exposure']
    
    def shutdown(self) -> None:
        """No cleanup needed for camera"""
        pass


class RenishawLaser(LaserSource):
    """Laser control for Renishaw system"""
    
    def __init__(self, ecm: ECMConnection):
        self.ecm = ecm
        self._power = 0.0  # percent
        self._enabled = False
    
    def initialize(self) -> None:
        """Initialize laser (ensure it's off)"""
        self.set_enabled(False)
    
    def set_power(self, power_percent: float) -> None:
        """
        Set laser power.
        
        Parameters
        ----------
        power_percent : float
            Power level as percentage (0-100)
        """
        if not 0 <= power_percent <= 100:
            raise ValueError("Power must be between 0 and 100 percent")
        
        try:
            self.ecm.call("Laser.SetPower", power=power_percent)
            self._power = power_percent
        except ECMException as e:
            raise RuntimeError(f"Failed to set laser power: {e}")
    
    def get_power(self) -> float:
        """Get current laser power setting"""
        return self._power
    
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable laser"""
        try:
            if enabled:
                self.ecm.call("Laser.Enable")
            else:
                self.ecm.call("Laser.Disable")
            self._enabled = enabled
        except ECMException as e:
            raise RuntimeError(f"Failed to set laser state: {e}")
    
    def is_enabled(self) -> bool:
        """Check if laser is enabled"""
        return self._enabled
    
    def get_state(self) -> Dict[str, Any]:
        """Get laser state"""
        return {
            'power': self._power,
            'enabled': self._enabled,
        }
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore laser state"""
        if 'power' in state:
            self.set_power(state['power'])
        if 'enabled' in state:
            self.set_enabled(state['enabled'])
    
    def shutdown(self) -> None:
        """Safely disable laser"""
        self.set_enabled(False)