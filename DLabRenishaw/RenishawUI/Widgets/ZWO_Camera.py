import os
import sys
from datetime import datetime
import numpy as np
import cv2
import zwoasi as asi

# Force napari to agree with your direct PyQt6 imports
os.environ["QT_API"] = "pyqt6"

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QSlider, QFileDialog, QSpinBox, QHBoxLayout,
                             QTabWidget, QListWidget, QApplication, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import napari

# 1. Initialize SDK (Replace with your actual path!)
SDK_PATH = '' # or .so / .dylib
try:
    asi.init(SDK_PATH)
except asi.ZWO_Error as e:
    print(f"SDK Init Error (Already initialized?): {e}")

class CameraStreamer(QThread):
    # This signal carries the numpy array from the camera to the GUI
    new_frame = pyqtSignal(np.ndarray) 

    def __init__(self, camera):
        super().__init__()
        self.camera = camera
        self.is_streaming = False

    def run(self):
        self.is_streaming = True
        self.camera.start_video_capture()
        
        while self.is_streaming:
            try:
                # Grab the frame and emit it to the main UI thread
                frame = self.camera.capture_video_frame()
                self.new_frame.emit(frame)
            except Exception as e:
                print(f"Frame drop: {e}")

    def stop(self):
        self.is_streaming = False
        self.camera.stop_video_capture()
        self.wait() # Safely wait for the thread to close

class ZWOControlPanel(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.camera = None
        self.optical_center = None
        self.save_dir = os.path.expanduser("~")
        self.is_streaming = False
        self.worker = None

        if 'Crosshair' not in self.viewer.layers:
            self.viewer.add_shapes([], edge_color='red', face_color='transparent', edge_width=3, name='Crosshair')

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        # --- TAB 1: Camera Controls ---
        self.tab_controls = QWidget()
        control_layout = QVBoxLayout()

        self.btn_connect = QPushButton("Connect Camera")
        self.btn_connect.clicked.connect(self.connect_camera)
        control_layout.addWidget(self.btn_connect)

        self.status_label = QLabel("Status: Disconnected")
        control_layout.addWidget(self.status_label)

        control_layout.addWidget(QLabel("Exposure (µs):"))
        self.exposure_spin = QSpinBox()
        self.exposure_spin.setRange(100, 1000000)
        self.exposure_spin.setValue(10000)
        self.exposure_spin.valueChanged.connect(self.update_settings)
        control_layout.addWidget(self.exposure_spin)

        control_layout.addWidget(QLabel("Gain:"))
        self.gain_slider = QSlider(Qt.Horizontal)
        self.gain_slider.setRange(0, 300)
        self.gain_slider.valueChanged.connect(self.update_settings)
        control_layout.addWidget(self.gain_slider)

        self.btn_dir = QPushButton("Set Save Directory")
        self.btn_dir.clicked.connect(self.set_save_dir)
        control_layout.addWidget(self.btn_dir)

        self.dir_label = QLabel(f"Saving to:\n{self.save_dir}")
        self.dir_label.setWordWrap(True)
        control_layout.addWidget(self.dir_label)

        self.btn_capture = QPushButton("CAPTURE IMAGE")
        self.btn_capture.setStyleSheet("background-color: darkred; color: white; font-weight: bold;")
        self.btn_capture.clicked.connect(self.capture_image)
        control_layout.addWidget(self.btn_capture)

        # --- Crosshair Settings ---
        crosshair_layout = QHBoxLayout()
        crosshair_layout.addWidget(QLabel("Crosshair Shape:"))
        self.crosshair_combo = QComboBox()
        self.crosshair_combo.addItems(["Large Cross", "Circle", "None"])
        self.crosshair_combo.currentTextChanged.connect(self.update_crosshair)
        crosshair_layout.addWidget(self.crosshair_combo)
        control_layout.addLayout(crosshair_layout)

        # --- Crosshair Customization ---
        control_layout.addWidget(QLabel("Crosshair Size:"))
        self.crosshair_size_slider = QSlider(Qt.Horizontal)
        self.crosshair_size_slider.setRange(5, 1000)
        self.crosshair_size_slider.setValue(100)
        self.crosshair_size_slider.valueChanged.connect(lambda: self.update_crosshair())
        control_layout.addWidget(self.crosshair_size_slider)

        control_layout.addWidget(QLabel("Crosshair Thickness:"))
        self.crosshair_thickness_slider = QSlider(Qt.Horizontal)
        self.crosshair_thickness_slider.setRange(1, 20)
        self.crosshair_thickness_slider.setValue(3)
        self.crosshair_thickness_slider.valueChanged.connect(lambda: self.update_crosshair())
        control_layout.addWidget(self.crosshair_thickness_slider)
        
        # --- Crop / Masking ---
        crop_layout = QHBoxLayout()
        self.crop_checkbox = QCheckBox("Enable ROI Masking")
        crop_layout.addWidget(self.crop_checkbox)
        self.btn_add_roi = QPushButton("Add ROI Rectangle")
        self.btn_add_roi.clicked.connect(self.add_roi_rectangle)
        crop_layout.addWidget(self.btn_add_roi)
        control_layout.addLayout(crop_layout)

        self.tab_controls.setLayout(control_layout)

        # --- TAB 2: Image Gallery ---
        self.tab_gallery = QWidget()
        gallery_layout = QVBoxLayout()
        
        # Clear Button
        self.btn_clear_gallery = QPushButton("Clear Gallery")
        self.btn_clear_gallery.clicked.connect(self.clear_gallery)
        gallery_layout.addWidget(self.btn_clear_gallery)
        
        # List of Files
        self.image_list = QListWidget()
        self.image_list.itemDoubleClicked.connect(self.load_image_to_viewer) 
        gallery_layout.addWidget(self.image_list)
        
        self.tab_gallery.setLayout(gallery_layout)

        # --- Assemble Tabs ---
        self.tabs.addTab(self.tab_controls, "Camera")
        self.tabs.addTab(self.tab_gallery, "Gallery")
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def connect_camera(self):
        if asi.get_num_cameras() == 0:
            self.status_label.setText("Status: No Camera Found!")
            return
        
        self.camera = asi.Camera(0)
        
        # Force 8-bit for the live stream to save bandwidth
        self.camera.set_image_type(asi.ASI_IMG_RAW8) 
        
        self.status_label.setText(f"Status: Connected to {self.camera.get_camera_property()['Name']}")
        
        self.update_settings()
        self.start_stream()

    def update_settings(self):
        if self.camera:
            self.camera.set_control_value(asi.ASI_EXPOSURE, self.exposure_spin.value())
            self.camera.set_control_value(asi.ASI_GAIN, self.gain_slider.value())

    def set_save_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.save_dir = directory
            self.dir_label.setText(f"Saving to: {self.save_dir}")

    def capture_image(self):
        if not self.camera:
            return

        print("Pausing stream for 16-bit capture...")

        # 1. Stop the 8-bit video stream and the background worker
        self.is_streaming = False
        self.camera.stop_video_capture()

        # 2. Switch camera to 16-bit mode
        self.camera.set_image_type(asi.ASI_IMG_RAW16)

        try:
            # 3. Capture a single 16-bit frame directly from the camera
            img_data = self.camera.capture()

            # 4. Save the image with a timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.save_dir, f"ZWO_Capture_{timestamp}.tiff")
            cv2.imwrite(filepath, img_data)
            print(f"Saved 16-bit image: {filepath}")
            self.image_list.addItem(filepath)

        except Exception as e:
            print(f"Error during capture: {e}")

        finally:
            # 5. Revert back to 8-bit mode and restart the live stream
            self.camera.set_image_type(asi.ASI_IMG_RAW8)
            self.start_stream()
    
    def load_image_to_viewer(self, item):
        filepath = item.text()
        if os.path.exists(filepath):
            # napari natively knows how to open .tiff files as new layers
            self.viewer.open(filepath, colormap='gray')
        else:
            print("File not found on disk.")

    def start_stream(self):
        # 1. Create the worker thread
        self.stream_thread = CameraStreamer(self.camera)
        
        # 2. Connect the worker's signal to your update function
        self.stream_thread.new_frame.connect(self.update_layer)
        
        # 3. Start the background thread
        self.stream_thread.start()

    def add_roi_rectangle(self):
        if 'Crop ROI' not in self.viewer.layers:
            if 'Live Feed' in self.viewer.layers:
                h, w = self.viewer.layers['Live Feed'].data.shape
            else:
                h, w = 1000, 1000
            
            rect = np.array([[h//4, w//4], [h//4, w*3//4], [h*3//4, w*3//4], [h*3//4, w//4]])
            self.viewer.add_shapes([rect], shape_type='rectangle', edge_color='blue', face_color='transparent', edge_width=5, name='Crop ROI')
            self.viewer.layers['Crop ROI'].mode = 'select'

    def update_crosshair(self, text=None):
        if 'Crosshair' not in self.viewer.layers:
            return
            
        if text is None:
            text = self.crosshair_combo.currentText()
            
        layer = self.viewer.layers['Crosshair']
        layer.edge_width = self.crosshair_thickness_slider.value()

        if text == "None":
            layer.data = []
            return
            
        if self.optical_center is not None:
            cy, cx = self.optical_center
        elif 'Live Feed' in self.viewer.layers:
            h, w = self.viewer.layers['Live Feed'].data.shape
            cy, cx = h // 2, w // 2
        else:
            cy, cx = 500, 500  

        size = self.crosshair_size_slider.value()

        if text == "Large Cross":
            layer.data = [
                np.array([[cy - size, cx], [cy + size, cx]]),
                np.array([[cy, cx - size], [cy, cx + size]]) 
            ]
            layer.shape_type = ['line', 'line']
        elif text == "Circle":
            radius = size / 2
            r = radius
            box = np.array([
                [cy - r, cx - r],
                [cy + r, cx - r],
                [cy + r, cx + r],
                [cy - r, cx + r]
            ])
            layer.data = [box]
            layer.shape_type = ['ellipse']

    def update_layer(self, frame):
        # Apply ROI Masking if enabled
        if self.crop_checkbox.isChecked() and 'Crop ROI' in self.viewer.layers:
            roi_layer = self.viewer.layers['Crop ROI']
            if len(roi_layer.data) > 0:
                rect = roi_layer.data[0] 
                y_min = int(np.min(rect[:, 0]))
                y_max = int(np.max(rect[:, 0]))
                x_min = int(np.min(rect[:, 1]))
                x_max = int(np.max(rect[:, 1]))
                
                mask = np.zeros_like(frame)
                h, w = frame.shape
                y_min = max(0, min(h, y_min))
                y_max = max(0, min(h, y_max))
                x_min = max(0, min(w, x_min))
                x_max = max(0, min(w, x_max))
                
                if y_max > y_min and x_max > x_min:
                    mask[y_min:y_max, x_min:x_max] = 1
                    frame = frame * mask

        # This receives the frame safely in the main GUI thread
        if 'Live Feed' in self.viewer.layers:
            self.viewer.layers['Live Feed'].data = frame
        else:
            self.viewer.add_image(frame, name='Live Feed', colormap='gray')
            # Center the crosshair when the first frame arrives
            if self.crosshair_combo.currentText() != "None":
                self.update_crosshair()
            
    def clear_gallery(self):
        # 1. Clear the text list in your PyQt tab
        self.image_list.clear()
        
        # 2. (Optional) Remove the captured layers from the napari canvas
        # Uncomment the code below if you want the "Clear" button to also 
        # delete the loaded images from the screen, leaving only the Live Feed and Crosshair.
        
        # layers_to_remove = [layer for layer in self.viewer.layers 
        #                     if layer.name not in ['Live Feed', 'Crosshair']]
        # for layer in layers_to_remove:
        #     self.viewer.layers.remove(layer)


# --- Main Application Setup ---
def main():
    # 1. Create the napari viewer
    viewer = napari.Viewer()

    # 2. Add the Crosshair (Shapes Layer)
    viewer.add_shapes([], edge_color='red', face_color='transparent', edge_width=3, name='Crosshair')

    # 3. Inject our PyQt Widget into the napari dock
    control_panel = ZWOControlPanel(viewer)
    viewer.window.add_dock_widget(control_panel, area='right', name='ZWO Controls')

    # 4. Start the app loop
    napari.run()

if __name__ == '__main__':
    main()