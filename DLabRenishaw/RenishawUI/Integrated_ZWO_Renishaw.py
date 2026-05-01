import os
import sys
import csv
import json
import numpy as np
from datetime import datetime

# Force napari to agree with your direct PyQt6 imports
os.environ["QT_API"] = "pyqt6"

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QHBoxLayout, QTabWidget, QComboBox, QFileDialog, 
                             QMessageBox, QApplication, QGroupBox, QRadioButton, QButtonGroup)
from PyQt6.QtCore import Qt

import napari

# Import the widgets from the other files!
# Add parent dir to path so we can import properly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Widgets.ZWO_Camera import ZWOControlPanel
from Widgets.BatchMap_Renishaw import BatchMappingWidget
from DLabRenishaw.devices import RenishawAdapter

class IntegratedAppWidget(QWidget):
    def __init__(self, viewer, renishaw_adapter):
        super().__init__()
        self.viewer = viewer
        self.renishaw_adapter = renishaw_adapter
        
        # Sub-widgets
        self.zwo_panel = ZWOControlPanel(self.viewer)
        self.batch_map_panel = BatchMappingWidget(self.renishaw_adapter)
        
        # Calibration state
        self.calibration_active = False
        self.calibration_step = 0
        self.calib_origin_stage = None # (x, y, z)
        self.calib_pts_pixel = [] # List of (py, px) from image clicks
        
        self.base_calib_matrix = None # 2x2 matrix relating pixels to um
        self.base_calib_mag = 1.0
        self.calib_type = None
        
        self.objectives = {} # Name -> dict of metadata
        self.session_calibrated = {} # Name -> bool
        self.preload_objectives()
        
        self.init_ui()
        self.setup_mouse_callbacks()

    def preload_objectives(self):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objectives.csv")
        if os.path.exists(csv_path):
            self.load_csv_from_path(csv_path)
        
    def init_ui(self):
        main_layout = QVBoxLayout()
        self.tabs = QTabWidget()
        
        # TAB 1: ZWO Camera native controls
        self.tabs.addTab(self.zwo_panel, "Camera Settings")
        
        # TAB 2: Batch Mapping
        self.tabs.addTab(self.batch_map_panel, "Batch Mapping")
        
        # TAB 3: Calibration & Movement
        self.tab_calib = QWidget()
        calib_layout = QVBoxLayout()
        
        # Objective Config
        obj_group = QGroupBox("Objectives Configuration")
        obj_layout = QVBoxLayout()
        
        row1 = QHBoxLayout()
        self.btn_load_csv = QPushButton("Load Objectives CSV")
        self.btn_load_csv.clicked.connect(self.load_objectives_csv)
        row1.addWidget(self.btn_load_csv)
        
        row1.addWidget(QLabel("Current Objective:"))
        self.combo_objective = QComboBox()
        self.combo_objective.addItems(list(self.objectives.keys()))
        self.combo_objective.currentTextChanged.connect(self.update_session_status)
        row1.addWidget(self.combo_objective)
        
        self.lbl_calib_status = QLabel("❌")
        row1.addWidget(self.lbl_calib_status)
        
        obj_layout.addLayout(row1)
        obj_group.setLayout(obj_layout)
        calib_layout.addWidget(obj_group)
        
        # Wizard
        wiz_group = QGroupBox("Calibration Wizard")
        wiz_layout = QVBoxLayout()
        
        type_layout = QHBoxLayout()
        self.radio_laser = QRadioButton("Laser Center")
        self.radio_laser.setChecked(True)
        self.radio_umpx = QRadioButton("um/px")
        self.radio_dist = QRadioButton("Distortion Target")
        self.calib_group = QButtonGroup()
        self.calib_group.addButton(self.radio_laser)
        self.calib_group.addButton(self.radio_umpx)
        self.calib_group.addButton(self.radio_dist)
        
        type_layout.addWidget(self.radio_laser)
        type_layout.addWidget(self.radio_umpx)
        type_layout.addWidget(self.radio_dist)
        wiz_layout.addLayout(type_layout)
        
        self.lbl_wizard_status = QLabel("Status: Idle")
        self.lbl_wizard_status.setWordWrap(True)
        self.lbl_wizard_status.setMinimumHeight(50)
        self.lbl_wizard_status.setStyleSheet("font-weight: bold;")
        wiz_layout.addWidget(self.lbl_wizard_status)
        
        self.btn_start_calib = QPushButton("Start Calibration Workflow")
        self.btn_start_calib.clicked.connect(self.start_calibration)
        wiz_layout.addWidget(self.btn_start_calib)
        
        self.btn_recenter = QPushButton("Recenter Camera View")
        self.btn_recenter.clicked.connect(self.recenter_view)
        wiz_layout.addWidget(self.btn_recenter)
        
        wiz_group.setLayout(wiz_layout)
        calib_layout.addWidget(wiz_group)
        
        # Save/Load Calib
        io_layout = QHBoxLayout()
        self.btn_save_calib = QPushButton("Save Calibration")
        self.btn_save_calib.clicked.connect(self.save_calibration)
        self.btn_load_calib = QPushButton("Load Calibration")
        self.btn_load_calib.clicked.connect(self.load_calibration)
        io_layout.addWidget(self.btn_save_calib)
        io_layout.addWidget(self.btn_load_calib)
        calib_layout.addLayout(io_layout)
        
        calib_layout.addStretch()
        self.tab_calib.setLayout(calib_layout)
        self.tabs.addTab(self.tab_calib, "Stage Calibration")
        
        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def load_objectives_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Objectives CSV", "", "CSV Files (*.csv)")
        if path:
            self.load_csv_from_path(path)

    def load_csv_from_path(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                self.objectives.clear()
                self.session_calibrated.clear()
                for row in reader:
                    if len(row) >= 2:
                        name = row[0].strip()
                        if name.lower() == "name": continue # header
                        try:
                            mag = float(row[1].strip())
                            um_per_px = float(row[2].strip()) if len(row) > 2 and row[2].strip() else None
                            cx = float(row[3].strip()) if len(row) > 3 and row[3].strip() else None
                            cy = float(row[4].strip()) if len(row) > 4 and row[4].strip() else None
                            last_calib = row[5].strip() if len(row) > 5 else None
                            
                            self.objectives[name] = {
                                'mag': mag,
                                'um_per_px': um_per_px,
                                'laser_center_x': cx,
                                'laser_center_y': cy,
                                'last_calibrated': last_calib
                            }
                            self.session_calibrated[name] = False
                        except ValueError:
                            pass # skip bad rows header
            
            if hasattr(self, 'combo_objective'):
                self.combo_objective.clear()
                self.combo_objective.addItems(list(self.objectives.keys()))
                self.lbl_wizard_status.setText(f"Loaded {len(self.objectives)} objectives from {os.path.basename(path)}.")
                self.update_session_status()
        except Exception as e:
            if hasattr(self, 'lbl_wizard_status'):
                self.lbl_wizard_status.setText(f"Error loading CSV: {e}")

    def save_objectives_csv(self):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "objectives.csv")
        try:
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Name", "Magnification", "um_per_px", "laser_center_x", "laser_center_y", "last_calibrated"])
                for name, data in self.objectives.items():
                    writer.writerow([
                        name, 
                        data.get('mag', 1.0), 
                        data.get('um_per_px', ''), 
                        data.get('laser_center_x', ''), 
                        data.get('laser_center_y', ''), 
                        data.get('last_calibrated', '')
                    ])
        except Exception as e:
            print(f"Failed to save objectives: {e}")

    def append_calibration_history(self, calib_type, obj_name, cx=None, cy=None, um_per_px=None, stage_pos=None, matrix=None):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibrationHistory.csv")
        file_exists = os.path.exists(csv_path)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        sx, sy, sz = ("", "", "")
        if stage_pos is not None:
            sx, sy, sz = stage_pos
            
        m00, m01, m10, m11 = ("", "", "", "")
        if matrix is not None:
            m00, m01 = matrix[0, 0], matrix[0, 1]
            m10, m11 = matrix[1, 0], matrix[1, 1]
            
        try:
            with open(csv_path, 'a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "CalibrationType", "Objective", "Magnification", "CenterX", "CenterY", "um_per_px", "StageX", "StageY", "StageZ", "Matrix_M00", "Matrix_M01", "Matrix_M10", "Matrix_M11"])
                
                mag = self.objectives.get(obj_name, {}).get('mag', 1.0)
                writer.writerow([timestamp, calib_type, obj_name, mag, cx, cy, um_per_px, sx, sy, sz, m00, m01, m10, m11])
        except Exception as e:
            print(f"Failed to append history: {e}")

    def update_session_status(self, text=None):
        if not hasattr(self, 'combo_objective'): return
        obj = self.combo_objective.currentText()
        if self.session_calibrated.get(obj, False):
            self.lbl_calib_status.setText("✅")
            self.lbl_calib_status.setStyleSheet("color: green; font-weight: bold; font-size: 16px;")
        else:
            self.lbl_calib_status.setText("❌")
            self.lbl_calib_status.setStyleSheet("color: red; font-weight: bold; font-size: 16px;")

    def recenter_view(self):
        # Snap the napari camera back to the center of the image
        if 'Live Feed' in self.viewer.layers:
            h, w = self.viewer.layers['Live Feed'].data.shape
            self.viewer.camera.center = (h/2, w/2)
            self.viewer.camera.zoom = 1.0

    def start_calibration(self):
        self.calibration_active = True
        self.calibration_step = 1
        self.calib_pts_pixel = []
        self.btn_start_calib.setEnabled(False)
        
        if self.radio_laser.isChecked():
            self.calib_type = "laser"
            self.lbl_wizard_status.setText("Laser Center Calib: Fire the laser, then DOUBLE-CLICK the location in the Napari viewer.")
        elif self.radio_umpx.isChecked():
            self.calib_type = "umpx"
            self.lbl_wizard_status.setText("um/px Calib Step 1: Move target into view, then DOUBLE-CLICK the target in Napari.")
        elif self.radio_dist.isChecked():
            self.calib_type = "dist"
            self.lbl_wizard_status.setText("Distortion Calib: Feature not yet implemented. Please select another mode.")
            self.abort_calibration()

    def abort_calibration(self):
        self.calibration_active = False
        self.calibration_step = 0
        self.btn_start_calib.setEnabled(True)

    def save_calibration(self):
        if self.base_calib_matrix is None:
             QMessageBox.warning(self, "No Calibration", "Perform or load a calibration first.")
             return
        path, _ = QFileDialog.getSaveFileName(self, "Save Calibration", "", "JSON Files (*.json)")
        if path:
             data = {
                 "matrix": self.base_calib_matrix.tolist(),
                 "base_mag": self.base_calib_mag,
                 "optical_center": self.zwo_panel.optical_center
             }
             with open(path, 'w') as f:
                 json.dump(data, f)
             self.lbl_wizard_status.setText("Calibration saved.")

    def load_calibration(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Calibration", "", "JSON Files (*.json)")
        if path:
             try:
                 with open(path, 'r') as f:
                     data = json.load(f)
                 self.base_calib_matrix = np.array(data["matrix"])
                 self.base_calib_mag = data["base_mag"]
                 center = data.get("optical_center")
                 if center is not None:
                     self.zwo_panel.optical_center = tuple(center)
                     self.zwo_panel.update_crosshair()
                     
                 self.lbl_wizard_status.setText(f"Loaded calibration. Base mag: {self.base_calib_mag}x")
             except Exception as e:
                 self.lbl_wizard_status.setText(f"Failed to load: {e}")

    def setup_mouse_callbacks(self):
        @self.viewer.mouse_double_click_callbacks.append
        def on_double_click(viewer, event):
            if self.calibration_active:
                data_coords = event.position
                if len(data_coords) < 2: return
                click_y, click_x = data_coords[-2], data_coords[-1]
                
                try:
                    pos = self.renishaw_adapter.stage.get_position()
                    if isinstance(pos, dict):
                         curr_stage = (pos['x'], pos['y'], pos['z'])
                    else:
                         curr_stage = pos
                except Exception as e:
                    self.lbl_wizard_status.setText(f"Error reading stage: {e}")
                    self.abort_calibration()
                    return
                
                obj_name = self.combo_objective.currentText()
                obj_data = self.objectives.get(obj_name, {})
                
                if self.calib_type == "laser":
                    self.calib_pts_pixel.append((click_y, click_x))
                    
                    self.zwo_panel.optical_center = (click_y, click_x)
                    self.zwo_panel.update_crosshair()
                    
                    obj_data['laser_center_x'] = click_x
                    obj_data['laser_center_y'] = click_y
                    obj_data['last_calibrated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.objectives[obj_name] = obj_data
                    
                    self.session_calibrated[obj_name] = True
                    self.update_session_status()
                    self.save_objectives_csv()
                    
                    self.append_calibration_history("Laser Center", obj_name, cx=click_x, cy=click_y, stage_pos=curr_stage)
                    
                    self.lbl_wizard_status.setText(f"Laser Center Calibrated at px({click_x:.1f}, {click_y:.1f})")
                    self.abort_calibration()
                    
                elif self.calib_type == "umpx":
                    self.calib_pts_pixel.append((click_y, click_x))
                    
                    if self.calibration_step == 1:
                        self.calib_origin_stage = curr_stage
                        
                        mag = obj_data.get('mag', 1.0)
                        if mag == 0: mag = 1.0
                        self.step_size = (20.0 * 5.0) / mag
                        
                        try:
                            self.renishaw_adapter.stage.move_to(curr_stage[0] + self.step_size, curr_stage[1], curr_stage[2])
                        except Exception as e:
                            self.lbl_wizard_status.setText(f"Error moving stage: {e}")
                            self.abort_calibration()
                            return
                            
                        self.calibration_step = 2
                        self.lbl_wizard_status.setText(f"Step 2: Stage moved +{self.step_size:.1f}um in X. DOUBLE-CLICK the target.")
                        
                    elif self.calibration_step == 2:
                        try:
                            self.renishaw_adapter.stage.move_to(self.calib_origin_stage[0], self.calib_origin_stage[1] + self.step_size, self.calib_origin_stage[2])
                        except Exception as e:
                            self.lbl_wizard_status.setText(f"Error moving stage: {e}")
                            self.abort_calibration()
                            return

                        self.calibration_step = 3
                        self.lbl_wizard_status.setText(f"Step 3: Stage moved +{self.step_size:.1f}um in Y. DOUBLE-CLICK the target.")
                        
                    elif self.calibration_step == 3:
                        cy, cx = self.calib_pts_pixel[0]
                        y1, x1 = self.calib_pts_pixel[1]
                        y2, x2 = self.calib_pts_pixel[2]
                        
                        dp_x = np.array([x1 - cx, y1 - cy])
                        dp_y = np.array([x2 - cx, y2 - cy])
                        
                        dU_x = np.array([self.step_size, 0.0])
                        dU_y = np.array([0.0, self.step_size])
                        
                        dU_mat = np.column_stack((dU_x, dU_y))
                        dP_mat = np.column_stack((dp_x, dp_y))
                        
                        try:
                            matrix = dU_mat @ np.linalg.inv(dP_mat)
                            self.base_calib_matrix = matrix
                            self.base_calib_mag = obj_data.get('mag', 1.0)
                            
                            um_px_x = np.linalg.norm(dU_x) / np.linalg.norm(dp_x)
                            um_px_y = np.linalg.norm(dU_y) / np.linalg.norm(dp_y)
                            avg_um_px = (um_px_x + um_px_y) / 2.0
                            
                            obj_data['um_per_px'] = avg_um_px
                            obj_data['last_calibrated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            self.objectives[obj_name] = obj_data
                            
                            self.session_calibrated[obj_name] = True
                            self.update_session_status()
                            self.save_objectives_csv()
                            
                            self.append_calibration_history("um/px", obj_name, cx=cx, cy=cy, um_per_px=avg_um_px, stage_pos=self.calib_origin_stage, matrix=matrix)
                            
                            self.lbl_wizard_status.setText(f"um/px Calib Complete! um/px={avg_um_px:.4f}\nM=\n{matrix}")
                            
                            self.renishaw_adapter.stage.move_to(*self.calib_origin_stage)
                            
                        except np.linalg.LinAlgError:
                            self.lbl_wizard_status.setText("Error: Collinear points. Calibration Failed.")
                            
                        self.abort_calibration()
                
                return # Don't do standard click-to-move during calibration
                
            # If not calibrating, normal click-to-move logic
            obj_name = self.combo_objective.currentText()
            obj_data = self.objectives.get(obj_name, {})
            
            data_coords = event.position
            if len(data_coords) < 2: return
            click_y, click_x = data_coords[-2], data_coords[-1]
            
            cx = obj_data.get('laser_center_x')
            cy = obj_data.get('laser_center_y')
            if cx is None or cy is None or cx == '' or cy == '':
                if self.zwo_panel.optical_center is not None:
                    cy, cx = self.zwo_panel.optical_center
                elif 'Live Feed' in self.viewer.layers:
                    h, w = self.viewer.layers['Live Feed'].data.shape
                    cy, cx = h // 2, w // 2
                else:
                    return
            else:
                cx, cy = float(cx), float(cy)

            dp = np.array([click_x - cx, click_y - cy])
            
            # Use base_calib_matrix if available, otherwise reject
            if self.base_calib_matrix is None:
                print("No transfer matrix loaded. Please perform um/px calibration.")
                return
                
            dU_base = self.base_calib_matrix @ dp
            
            current_mag = obj_data.get('mag', 1.0)
            if current_mag != 0:
                scale_factor = self.base_calib_mag / current_mag
            else:
                scale_factor = 1.0
                
            dU_scaled = dU_base * scale_factor
            dx, dy = -dU_scaled[0], -dU_scaled[1]
            
            try:
                pos = self.renishaw_adapter.stage.get_position()
                if isinstance(pos, dict):
                     curr_x, curr_y, curr_z = pos['x'], pos['y'], pos['z']
                else:
                     curr_x, curr_y, curr_z = pos
                
                self.renishaw_adapter.stage.move_to(curr_x + dx, curr_y + dy, curr_z)
            except Exception as e:
                print(f"Move error: {e}")

        # Removed the mouse_drag_callbacks entirely since calibration is now double-click

def main():
    viewer = napari.Viewer()
    
    # Needs a real API URL. Using dummy test URL here
    renAdapt = RenishawAdapter(api_url='http://localhost:9880/api')
    renAdapt.initialize()
    
    app_widget = IntegratedAppWidget(viewer, renAdapt)
    viewer.window.add_dock_widget(app_widget, area='right', name='Integrated Control')

    napari.run()

if __name__ == '__main__':
    main()
