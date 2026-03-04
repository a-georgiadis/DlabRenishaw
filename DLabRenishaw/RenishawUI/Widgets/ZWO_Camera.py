import napari
import zwoasi as asi
import numpy as np
import cv2
import os
from datetime import datetime
from qtpy.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                            QSlider, QFileDialog, QSpinBox, QHBoxLayout,
                            QTabWidget, QListWidget)
from qtpy.QtCore import Qt
from napari.qt.threading import thread_worker

# 1. Initialize SDK (Replace with your actual path!)
SDK_PATH = 'path/to/your/ASICamera2.dll' # or .so / .dylib
try:
    asi.init(SDK_PATH)
except asi.ZWO_Error as e:
    print(f"SDK Init Error (Already initialized?): {e}")

class ZWOControlPanel(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.camera = None
        self.save_dir = os.path.expanduser("~")
        self.is_streaming = False
        self.worker = None

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
        self.camera.start_video_capture()
        self.is_streaming = True
        
        # Setup the background worker
        self.worker = self.frame_grabber()
        self.worker.yielded.connect(self.update_layer)
        self.worker.start()

    @thread_worker
    def frame_grabber(self):
        # This runs in the background and constantly yields frames
        while self.is_streaming:
            frame = self.camera.capture_video_frame()
            yield frame

    def update_layer(self, frame):
        # This receives the frame from the worker and updates the UI
        if 'Live Feed' in self.viewer.layers:
            self.viewer.layers['Live Feed'].data = frame
        else:
            self.viewer.add_image(frame, name='Live Feed', colormap='gray')
            
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
    # This creates a simple cross at the center (adjust coordinates based on your sensor size)
    crosshair_data = np.array([
        [[500, 400], [500, 600]], # Vertical line
        [[400, 500], [600, 500]]  # Horizontal line
    ])
    viewer.add_shapes(crosshair_data, shape_type='line', edge_color='red', edge_width=3, name='Crosshair')

    # 3. Inject our PyQt Widget into the napari dock
    control_panel = ZWOControlPanel(viewer)
    viewer.window.add_dock_widget(control_panel, area='right', name='ZWO Controls')

    # 4. Start the app loop
    napari.run()

if __name__ == '__main__':
    main()