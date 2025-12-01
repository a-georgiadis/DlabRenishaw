Modular Python control system for coordinated Raman-fluorescence microscopy requiring precision positioning and multi-modal calibration.

## Overview

DLabRenishaw provides a flexible framework for controlling a hybrid microscope system combining a commercial Raman microscope with additional imaging modalities (fluorescence, brightfield, etc.). The system is designed for measurements requiring coordination and precision across multiple devices and imaging modes.

### Key Features

- **Unified control interface** for heterogeneous hardware (commercial REST API + Micro-Manager devices)
- **Persistent state management** for calibrations and experimental configurations
- **Pluggable focus optimization** metrics for different imaging modalities
- **2D surface mapping** for extended area imaging with topography correction
- **Modular protocol system** for complex acquisition sequences
- **Multi-modal focus tracking** - different optimal focal planes per imaging mode
- Designed for scripting with PyQt GUI integration

## System Architecture
```python
┌─────────────────────────────────────────────────────────┐
│  Application Layer                                      │
│  (Custom scripts, PyQt GUI, Jupyter notebooks)          │
└───────────────────┬─────────────────────────────────────┘
│
┌───────────────────▼─────────────────────────────────────┐
│  Microscope Controller API                              │
│  - Unified device control                               │
│  - State management & persistence                       │
│  - Protocol execution engine                            │
└──┬──────────────────┬───────────────────────────────────┘
│                  │
┌──▼─────────────┐  ┌─▼────────────────┐
│ Raman System   │  │ Pycro-Manager    │
│ (REST API)     │  │ Devices          │
│ - Stage        │  │ - Cameras        │
│ - Laser        │  │ - Light sources  │
│ - Camera       │  │ - Filters        │
└────────────────┘  └──────────────────┘
```
### Module Structure
```python
dlabrenishaw/
├── core/
│   ├── device_base.py          # Abstract device interfaces
│   ├── microscope.py            # Main MicroscopeController class
│   └── state_manager.py         # State persistence & configuration
│
├── devices/
│   ├── renishaw_adapter.py      # Raman system REST API wrapper
│   ├── mm_bridge.py             # Pycro-Manager device integration
│   ├── camera_base.py           # Camera abstraction
│   └── stage_base.py            # Stage abstraction
│
├── acquisition/
│   ├── protocols.py             # Acquisition protocol base classes
│   ├── autofocus.py             # Focus optimization routines
│   ├── surface_mapping.py       # 2D surface map generation
│   └── multiposition.py         # Grid/list-based positioning
│
├── analysis/
│   ├── focus_metrics.py         # Pluggable focus quality metrics
│   ├── image_processing.py      # Downsampling, stitching utilities
│   └── signal_metrics.py        # Raman-specific signal analysis
│
├── storage/
│   ├── metadata.py              # Metadata schemas
│   └── persistence.py           # Data/state serialization
│
└── utils/
├── config.py                # Configuration management
├── coordinates.py           # Coordinate system transforms
└── logging_setup.py         # Logging configuration
```