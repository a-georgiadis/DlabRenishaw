import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class MicroscopeState:
    """Container for microscope state data"""
    def __init__(self):
        self.device_positions = {}
        self.surface_maps = {}
        self.focus_offsets = {}
        self.calibrations = {}
        self.imaging_parameters = {}
        self.timestamp = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            'device_positions': self.device_positions,
            'surface_maps': {k: v.to_dict() for k, v in self.surface_maps.items()},
            'focus_offsets': self.focus_offsets,
            'calibrations': self.calibrations,
            'imaging_parameters': self.imaging_parameters,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MicroscopeState':
        """Restore state from dictionary"""
        state = cls()
        state.device_positions = data.get('device_positions', {})
        # ... restore other fields
        return state

class MicroscopeController:
    """Main controller for microscope hardware and state"""
    
    def __init__(self, config_path: str = None):
        self.devices = {}
        self.state = MicroscopeState()
        
        if config_path:
            self.load_config(config_path)
    
    # Hardware control methods
    def add_device(self, name: str, device):
        self.devices[name] = device
    
    def initialize(self):
        for device in self.devices.values():
            device.initialize()
    
    def move_to(self, x, y, z=None, modality=None):
        # Movement logic
        pass
    
    # State management methods (built-in)
    def save_state(self, filepath: str):
        """Save current microscope state to file"""
        filepath = Path(filepath)
        
        # Update timestamp
        self.state.timestamp = datetime.now()
        
        # Capture current device states
        for name, device in self.devices.items():
            self.state.device_positions[name] = device.get_state()
        
        # Save to file
        data = self.state.to_dict()
        
        if filepath.suffix == '.json':
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    def load_state(self, filepath: str):
        """Load microscope state from file"""
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
        elif filepath.suffix in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        self.state = MicroscopeState.from_dict(data)
        
        # Optionally restore device states
        # self._restore_device_states()
    
    def load_config(self, filepath: str):
        """Load system configuration"""
        # Configuration loading logic
        pass
    
    
