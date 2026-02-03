import napari
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QCheckBox, QLabel, QGroupBox, QSpinBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import numpy as np
import time

# ==========================================
# 1. HARDWARE ABSTRACTION LAYER (The Switch)
# ==========================================

# OPTION A: REAL HARDWARE (Uncomment this on the microscope computer)
# from pycromanager import Core

# OPTION B: MOCK HARDWARE (Use this for dev)
class MockCore:
    """Mimics pycromanager.Core for offline development"""
    def __init__(self):
        print("[MockCore] Initialized (Offline Mode)")
        self.exposure = 100
        self.shutter_open = False
        
        # Simulate stage position
        self.x = 0.0
        self.y = 0.0

    def set_exposure(self, ms):
        self.exposure = ms
        print(f"[MockCore] Exposure set to {ms} ms")

    def set_shutter_open(self, state):
        self.shutter_open = state
        status = "OPEN" if state else "CLOSED"
        print(f"[MockCore] Shutter {status}")

    def snap_image(self):
        # Simulate the delay of taking a picture
        time.sleep(self.exposure / 1000) 

    def get_tagged_image(self):
        """Returns a fake object acting like the Pycro-manager tagged image"""
        class TaggedImage:
            def __init__(self):
                # Generate fake noise image (512x512)
                self.tags = {'Height': 512, 'Width': 512}
                # Create random noise + a moving bright spot to prove it's "live"
                base = np.random.randint(0, 100, (512, 512), dtype=np.uint16)
                
                # Add a fake "cell" that moves randomly
                cx, cy = np.random.randint(200, 300), np.random.randint(200, 300)
                base[cx-10:cx+10, cy-10:cy+10] = 2000 
                self.pix = base.flatten() # PM returns flattened arrays

        return TaggedImage()

    def get_x_position(self):
        return self.x + np.random.random() # Add jitter

    def get_y_position(self):
        return self.y + np.random.random()

# ==========================================
# 2. WORKER THREAD (Handles the Loop)
# ==========================================
class LiveWorker(QThread):
    new_frame = pyqtSignal(object)

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.running = True

    def run(self):
        # Setup for live mode
        self.core.set_exposure(50)
        self.core.set_shutter_open(True)
        
        while self.running:
            # 1. Trigger Hardware
            self.core.snap_image()
            
            # 2. Read Data
            tagged = self.core.get_tagged_image()
            
            # 3. Reshape (Crucial step for Napari)
            h = tagged.tags['Height']
            w = tagged.tags['Width']
            img = tagged.pix.reshape(h, w)
            
            # 4. Send to GUI
            self.new_frame.emit(img)
            
        # Cleanup
        self.core.set_shutter_open(False)

    def stop(self):
        self.running = False
        self.wait()

# ==========================================
# 3. THE GUI WIDGET
# ==========================================
class MicroscopeControl(QWidget):
    def __init__(self, viewer, core):
        super().__init__()
        self.viewer = viewer
        self.core = core
        self.live_thread = None
        
        # --- UI LAYOUT ---
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # GROUP 1: ACQUISITION
        self.acq_group = QGroupBox("Capture Controls")
        self.acq_layout = QVBoxLayout()
        self.acq_group.setLayout(self.acq_layout)
        
        self.btn_snap = QPushButton("Acquire High-Res Snapshot")
        self.btn_snap.setStyleSheet("background-color: #007ACC; color: white; font-weight: bold; padding: 8px;")
        self.btn_snap.clicked.connect(self.snap_image)
        self.acq_layout.addWidget(self.btn_snap)
        
        # GROUP 2: LIVE PREVIEW
        self.live_group = QGroupBox("Live Preview")
        self.live_layout = QVBoxLayout()
        self.live_group.setLayout(self.live_layout)
        
        self.chk_live = QCheckBox("Enable Live Mode")
        self.chk_live.setStyleSheet("font-size: 14px;")
        self.chk_live.stateChanged.connect(self.toggle_live)
        self.live_layout.addWidget(self.chk_live)
        
        self.lbl_info = QLabel("Status: Ready")
        self.live_layout.addWidget(self.lbl_info)

        # Add to main
        self.layout.addWidget(self.acq_group)
        self.layout.addWidget(self.live_group)
        self.layout.addStretch()

    # --- LOGIC ---
    def toggle_live(self, state):
        if state == 2: # Checked
            self.lbl_info.setText("Status: Streaming...")
            self.live_thread = LiveWorker(self.core)
            self.live_thread.new_frame.connect(self.update_live_layer)
            self.live_thread.start()
        else:
            self.lbl_info.setText("Status: Stopping...")
            if self.live_thread:
                self.live_thread.stop()
            self.lbl_info.setText("Status: Idle")

    def update_live_layer(self, img):
        # Check if layer exists, if not create it
        if 'Live_Preview' not in self.viewer.layers:
            self.viewer.add_image(img, name='Live_Preview', colormap='gray', blending='additive')
        else:
            self.viewer.layers['Live_Preview'].data = img

    def snap_image(self):
        # Pause live mode if it's running (good practice)
        was_live = self.chk_live.isChecked()
        if was_live:
            self.chk_live.setChecked(False)
            if self.live_thread: self.live_thread.wait()

        print("Taking Snapshot...")
        
        # --- REAL ACQUISITION LOGIC ---
        self.core.set_exposure(200) # Longer exposure
        self.core.set_shutter_open(True)
        self.core.snap_image()
        self.core.set_shutter_open(False)
        
        tagged = self.core.get_tagged_image()
        h, w = tagged.tags['Height'], tagged.tags['Width']
        img = tagged.pix.reshape(h, w)
        
        # Add to Napari as a permanent layer (Green)
        self.viewer.add_image(img, name="Snapshot", colormap='green', blending='additive')

        # Restore live mode if it was on
        if was_live:
            self.chk_live.setChecked(True)

# ==========================================
# 4. MAIN ENTRY POINT
# ==========================================
if __name__ == "__main__":
    
    # --- HARDWARE INIT ---
    # use_real_hardware = True  <-- Toggle this later
    use_real_hardware = False
    
    if use_real_hardware:
        from pycromanager import Core
        core = Core()
    else:
        core = MockCore()

    # --- LAUNCH GUI ---
    viewer = napari.Viewer()
    
    # Create the widget and inject the core
    controls = MicroscopeControl(viewer, core)
    
    # Dock it to the right
    viewer.window.add_dock_widget(controls, area='right', name="Microscope Controller")
    
    napari.run()