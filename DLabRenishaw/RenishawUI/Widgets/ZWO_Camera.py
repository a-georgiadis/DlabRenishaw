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
                             QTabWidget, QListWidget, QApplication, QComboBox, QCheckBox, QLineEdit)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

import pyqtgraph as pg
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

        self.initial_zoom = None
        self.initial_center = None
        self.viewer.camera.events.zoom.connect(self.clamp_zoom)
        self.viewer.mouse_move_callbacks.append(self.on_mouse_move)

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

        self.intensity_label = QLabel("Intensity: N/A")
        self.intensity_label.setStyleSheet("font-weight: bold;")
        control_layout.addWidget(self.intensity_label)

        control_layout.addWidget(QLabel("Exposure (ms):"))
        exp_layout = QHBoxLayout()
        self.exposure_slider = QSlider(Qt.Orientation.Horizontal)
        self.exposure_slider.setRange(1, 10000)
        self.exposure_slider.setValue(10)
        self.exposure_slider.valueChanged.connect(self.update_exposure)
        
        self.exposure_label = QLabel("10 ms")
        self.exposure_label.setFixedWidth(50)
        
        exp_layout.addWidget(self.exposure_slider)
        exp_layout.addWidget(self.exposure_label)
        control_layout.addLayout(exp_layout)

        # Preset 1
        preset1_layout = QHBoxLayout()
        preset1_layout.addWidget(QLabel("Preset 1:"))
        self.exp_base_combo1 = QComboBox()
        self.exp_base_combo1.addItems(["1", "2", "5"])
        preset1_layout.addWidget(self.exp_base_combo1)
        self.exp_mag_combo1 = QComboBox()
        self.exp_mag_combo1.addItems(["1ms", "10ms", "100ms", "1s"])
        preset1_layout.addWidget(self.exp_mag_combo1)
        self.btn_set_exp1 = QPushButton("Apply")
        self.btn_set_exp1.clicked.connect(lambda: self.apply_quick_exposure(1))
        preset1_layout.addWidget(self.btn_set_exp1)
        control_layout.addLayout(preset1_layout)

        # Preset 2
        preset2_layout = QHBoxLayout()
        preset2_layout.addWidget(QLabel("Preset 2:"))
        self.exp_base_combo2 = QComboBox()
        self.exp_base_combo2.addItems(["1", "2", "5"])
        preset2_layout.addWidget(self.exp_base_combo2)
        self.exp_mag_combo2 = QComboBox()
        self.exp_mag_combo2.addItems(["1ms", "10ms", "100ms", "1s"])
        preset2_layout.addWidget(self.exp_mag_combo2)
        self.btn_set_exp2 = QPushButton("Apply")
        self.btn_set_exp2.clicked.connect(lambda: self.apply_quick_exposure(2))
        preset2_layout.addWidget(self.btn_set_exp2)
        control_layout.addLayout(preset2_layout)

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

        # File Name Settings
        filename_layout = QHBoxLayout()
        filename_layout.addWidget(QLabel("File Name:"))
        self.filename_input = QLineEdit("ZWO_Capture")
        filename_layout.addWidget(self.filename_input)
        
        filename_layout.addWidget(QLabel("Index:"))
        self.file_index_spinner = QSpinBox()
        self.file_index_spinner.setRange(0, 99999)
        self.file_index_spinner.setValue(1)
        filename_layout.addWidget(self.file_index_spinner)
        
        control_layout.addLayout(filename_layout)

        # --- Capture Controls ---
        capture_layout = QHBoxLayout()
        self.btn_capture = QPushButton("CAPTURE IMAGE")
        self.btn_capture.setStyleSheet("background-color: darkred; color: white; font-weight: bold;")
        self.btn_capture.clicked.connect(self.capture_image)
        capture_layout.addWidget(self.btn_capture)

        self.bit_depth_combo = QComboBox()
        self.bit_depth_combo.addItems(["16-bit", "8-bit"])
        self.bit_depth_combo.setToolTip("Select the bit depth for saved images")
        capture_layout.addWidget(self.bit_depth_combo)
        
        control_layout.addLayout(capture_layout)

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

        # --- Slice Settings ---
        slice_layout = QVBoxLayout()
        self.slice_checkbox = QCheckBox("Enable X/Y Slices")
        self.slice_checkbox.toggled.connect(self.toggle_slices)
        slice_layout.addWidget(self.slice_checkbox)

        self.slice_plots_widget = QWidget()
        plots_vbox = QVBoxLayout()
        plots_vbox.setContentsMargins(0, 0, 0, 0)
        self.x_slice_plot = pg.PlotWidget(title="X Slice (Horizontal Distribution)")
        self.x_slice_plot.setFixedHeight(120)
        self.y_slice_plot = pg.PlotWidget(title="Y Slice (Vertical Distribution)")
        self.y_slice_plot.setFixedHeight(120)
        self.x_curve = self.x_slice_plot.plot(pen='y')
        self.y_curve = self.y_slice_plot.plot(pen='r')
        plots_vbox.addWidget(self.x_slice_plot)
        plots_vbox.addWidget(self.y_slice_plot)
        self.slice_plots_widget.setLayout(plots_vbox)
        self.slice_plots_widget.setVisible(False)
        slice_layout.addWidget(self.slice_plots_widget)

        control_layout.addLayout(slice_layout)

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

    def clamp_zoom(self, event=None):
        if self.initial_zoom is not None:
            if self.viewer.camera.zoom < self.initial_zoom:
                self.viewer.camera.zoom = self.initial_zoom
                self.viewer.camera.center = self.initial_center

    def on_mouse_move(self, viewer, event):
        if 'Live Feed' in viewer.layers:
            layer = viewer.layers['Live Feed']
            try:
                coords = layer.world_to_data(event.position)
                y, x = int(round(coords[0])), int(round(coords[1]))
                data = layer.data
                if 0 <= y < data.shape[0] and 0 <= x < data.shape[1]:
                    val = data[y, x]
                    self.intensity_label.setText(f"Intensity: {val} at ({x}, {y})")
                else:
                    self.intensity_label.setText("Intensity: N/A")
            except Exception:
                pass

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

    def apply_quick_exposure(self, preset):
        if preset == 1:
            base_val = int(self.exp_base_combo1.currentText())
            mag_idx = self.exp_mag_combo1.currentIndex()
        else:
            base_val = int(self.exp_base_combo2.currentText())
            mag_idx = self.exp_mag_combo2.currentIndex()
            
        multiplier = 10 ** mag_idx
        total_exp = base_val * multiplier
        total_exp = min(total_exp, self.exposure_slider.maximum())
        self.exposure_slider.setValue(total_exp)

    def update_exposure(self):
        val_ms = self.exposure_slider.value()
        self.exposure_label.setText(f"{val_ms} ms")
        self.update_settings()

    def update_settings(self):
        if self.camera:
            self.camera.set_control_value(asi.ASI_EXPOSURE, self.exposure_slider.value() * 1000)
            self.camera.set_control_value(asi.ASI_GAIN, self.gain_slider.value())

    def set_save_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.save_dir = directory
            self.dir_label.setText(f"Saving to: {self.save_dir}")

    def capture_image(self):
        if not self.camera:
            return

        is_16_bit = (self.bit_depth_combo.currentText() == "16-bit")
        
        print(f"Pausing stream for {self.bit_depth_combo.currentText()} capture...")

        # 1. Stop the background worker cleanly
        if hasattr(self, 'stream_thread') and self.stream_thread.isRunning():
            self.stream_thread.stop()

        # 2. Switch camera mode if needed
        if is_16_bit:
            self.camera.set_image_type(asi.ASI_IMG_RAW16)
        else:
            self.camera.set_image_type(asi.ASI_IMG_RAW8)

        try:
            # 3. Restart video capture to grab a frame properly
            self.camera.start_video_capture()
            # Toss first two frames which are often left over from previous buffer
            for _ in range(2):
                _ = self.camera.capture_video_frame(timeout=5000)
                
            img_data = self.camera.capture_video_frame(timeout=5000)
            self.camera.stop_video_capture()
            
            # 4. Verify shape (ZWO bugs sometimes return 1D or wrong sizes)
            roi = self.camera.get_roi_format()
            expected_shape = (roi[1], roi[0]) # height, width
            if img_data.shape != expected_shape:
                try:
                    img_data = img_data.reshape(expected_shape)
                except ValueError:
                    print(f"Reshape failed: img_data.size={img_data.size}, expected={expected_shape[0]*expected_shape[1]}")
                    
            if is_16_bit and img_data.dtype != np.uint16:
                img_data = img_data.astype(np.uint16)
            elif not is_16_bit and img_data.dtype != np.uint8:
                img_data = img_data.astype(np.uint8)

            # 5. Save the image with a timestamp and custom filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            depth_str = "16bit" if is_16_bit else "8bit"
            
            base_name = self.filename_input.text().strip() or "ZWO_Capture"
            idx = self.file_index_spinner.value()
            filename = f"{base_name}_{idx:03d}_{depth_str}_{timestamp}.tiff"
            filepath = os.path.join(self.save_dir, filename)
            
            cv2.imwrite(filepath, img_data)
            print(f"Saved {depth_str} image: {filepath}")
            self.image_list.addItem(filepath)
            
            # Auto-increment
            self.file_index_spinner.setValue(idx + 1)

        except Exception as e:
            print(f"Error during capture: {e}")

        finally:
            # 5. Revert back to 8-bit mode (live stream default) and restart the live stream
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
        else:
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
                
        # Move Crosshair to Top
        idx = self.viewer.layers.index(layer)
        if idx != len(self.viewer.layers) - 1:
            self.viewer.layers.move(idx, len(self.viewer.layers) - 1)

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
            
            # Setup initial zoom constraint
            self.viewer.reset_view()
            self.initial_zoom = self.viewer.camera.zoom
            self.initial_center = self.viewer.camera.center

        # Update slices if enabled
        if self.slice_checkbox.isChecked():
            self.update_slice_plots()

    def toggle_slices(self, checked):
        self.slice_plots_widget.setVisible(checked)
        if checked:
            if 'Slices' not in self.viewer.layers:
                if 'Live Feed' in self.viewer.layers:
                    h, w = self.viewer.layers['Live Feed'].data.shape
                else:
                    h, w = 1000, 1000
                cy, cx = h//2, w//2
                
                # A horizontal line across the center, a vertical line across the center
                x_line = np.array([[cy, 0], [cy, w]])
                y_line = np.array([[0, cx], [h, cx]])
                
                self.viewer.add_shapes([x_line, y_line], shape_type='line', edge_color=['yellow', 'red'], edge_width=3, name='Slices')
                self.viewer.layers['Slices'].events.data.connect(self.update_slice_plots)
                # Keep Crosshair on top
                if 'Crosshair' in self.viewer.layers:
                    idx = self.viewer.layers.index(self.viewer.layers['Crosshair'])
                    if idx != len(self.viewer.layers) - 1:
                        self.viewer.layers.move(idx, len(self.viewer.layers) - 1)
            else:
                self.viewer.layers['Slices'].visible = True
            self.update_slice_plots()
        else:
            if 'Slices' in self.viewer.layers:
                self.viewer.layers['Slices'].visible = False

    def update_slice_plots(self, event=None):
        if not self.slice_checkbox.isChecked() or 'Slices' not in self.viewer.layers or 'Live Feed' not in self.viewer.layers:
            return
            
        data = self.viewer.layers['Live Feed'].data
        shapes = self.viewer.layers['Slices'].data
        if len(shapes) < 2:
            return
            
        x_line = shapes[0] # horizontal
        y_line = shapes[1] # vertical
        
        # Determine the position of the slicing lines
        cy = int(round(np.mean(x_line[:, 0])))
        cx = int(round(np.mean(y_line[:, 1])))
        
        h, w = data.shape
        cy = max(0, min(h-1, cy))
        cx = max(0, min(w-1, cx))
        
        # Update plots
        x_profile = data[cy, :]
        self.x_curve.setData(x_profile)
        
        y_profile = data[:, cx]
        self.y_curve.setData(y_profile)
            
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