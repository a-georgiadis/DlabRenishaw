"""
DLabRenishaw - Modular microscope control for Raman-fluorescence imaging.

Main package providing unified control interface for hybrid microscope systems.
"""

__version__ = "0.1.0"
__author__ = "antonyg"

# Import main controller for convenient access
from DLabRenishaw.core.microscope import MicroscopeController, MicroscopeState

# Make these available at package level
__all__ = [
    "MicroscopeController",
    "MicroscopeState",
]