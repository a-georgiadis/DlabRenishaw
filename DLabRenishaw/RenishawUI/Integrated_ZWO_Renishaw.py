import os
import sys
import csv
import json
import numpy as np

# Force napari to agree with your direct PyQt6 imports
os.environ["QT_API"] = "pyqt6"

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QLabel, 
                             QHBoxLayout, QTabWidget, QComboBox, QFileDialog, 
                             QMessageBox, QApplication, QGroupBox)
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
        
        self.objectives = {"Base": 1.0} # Name -> Magnification
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
        row1.addWidget(self.combo_objective)
        obj_layout.addLayout(row1)
        obj_group.setLayout(obj_layout)
        calib_layout.addWidget(obj_group)
        
        # Wizard
        wiz_group = QGroupBox("Calibration Wizard")
        wiz_layout = QVBoxLayout()
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
                for row in reader:
                    if len(row) >= 2:
                        name = row[0].strip()
                        try:
                            mag = float(row[1].strip())
                            self.objectives[name] = mag
                        except ValueError:
                            pass # skip bad rows header
            
            if hasattr(self, 'combo_objective'):
                self.combo_objective.clear()
                self.combo_objective.addItems(list(self.objectives.keys()))
                self.lbl_wizard_status.setText(f"Loaded {len(self.objectives)} objectives from {os.path.basename(path)}.")
        except Exception as e:
            if hasattr(self, 'lbl_wizard_status'):
                self.lbl_wizard_status.setText(f"Error loading CSV: {e}")

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
        self.lbl_wizard_status.setText("Step 1: Move the Renishaw stage using external controls until your target is in view. Then DOUBLE-CLICK the target in the Napari viewer.")

    def abort_calibration(self):
        self.calibration_active = False
        self.calibration_step = 0
        self.btn_start_calib.setEnabled(True)

    def complete_calibration(self):
        print("Calib pts:", self.calib_pts_pixel)
        if len(self.calib_pts_pixel) < 3:
             self.lbl_wizard_status.setText("Calibration failed: Missing points.")
             self.abort_calibration()
             return

        # Point 0: initial center
        cy, cx = self.calib_pts_pixel[0]
        # Point 1: clicked after X moved +20
        y1, x1 = self.calib_pts_pixel[1]
        # Point 2: clicked after Y moved +20
        y2, x2 = self.calib_pts_pixel[2]
        
        # Pixel displacement vectors (dp = p_shifted - p_center)
        dp_x = np.array([x1 - cx, y1 - cy])
        dp_y = np.array([x2 - cx, y2 - cy])
        
        # Physical displacement vectors
        dU_x = np.array([20.0, 0.0])
        dU_y = np.array([0.0, 20.0])
        
        # We want to find Matrix M such that M * dP = dU
        # dU_x = M * dp_x
        # dU_y = M * dp_y
        # [dU_x, dU_y] = M * [dp_x, dp_y]
        # M = [dU_x, dU_y] * inverse([dp_x, dp_y])
        
        dU_mat = np.column_stack((dU_x, dU_y))
        dP_mat = np.column_stack((dp_x, dp_y))
        
        try:
            self.base_calib_matrix = dU_mat @ np.linalg.inv(dP_mat)
            
            obj_name = self.combo_objective.currentText()
            self.base_calib_mag = self.objectives.get(obj_name, 1.0)
            
            self.lbl_wizard_status.setText(f"Calibration Complete! M=\n{self.base_calib_matrix}\nSaved on Objective: {obj_name} ({self.base_calib_mag}x). Double-click image to move!")
            
            # Recenter stage
            self.renishaw_adapter.stage.move_to(*self.calib_origin_stage)
            
        except np.linalg.LinAlgError:
            self.lbl_wizard_status.setText("Error: Collinear points. Calibration Failed.")
        
        self.abort_calibration()

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
            # If we are calibrating, intercept double click for the wizard!
            if self.calibration_active:
                data_coords = event.position
                if len(data_coords) >= 2:
                    click_y, click_x = data_coords[-2], data_coords[-1]
                    self.calib_pts_pixel.append((click_y, click_x))
                    
                    if self.calibration_step == 1:
                        # 1. Record Center
                        self.zwo_panel.optical_center = (click_y, click_x)
                        self.zwo_panel.update_crosshair()
                        
                        try:
                            pos = self.renishaw_adapter.stage.get_position()
                            if isinstance(pos, dict):
                                 self.calib_origin_stage = (pos['x'], pos['y'], pos['z'])
                            else:
                                 self.calib_origin_stage = pos
                        except Exception as e:
                            self.lbl_wizard_status.setText(f"Error reading stage: {e}")
                            self.abort_calibration()
                            return
                        
                        # Move +20 in X
                        try:
                            new_x = self.calib_origin_stage[0] + 20.0
                            self.renishaw_adapter.stage.move_to(new_x, self.calib_origin_stage[1], self.calib_origin_stage[2])
                        except Exception as e:
                            self.lbl_wizard_status.setText(f"Error moving stage: {e}")
                            self.abort_calibration()
                            return
                            
                        self.calibration_step = 2
                        self.lbl_wizard_status.setText("Step 2: Stage moved +20um in X. DOUBLE-CLICK the target in the Napari viewer.")
                        
                    elif self.calibration_step == 2:
                        try:
                            new_y = self.calib_origin_stage[1] + 20.0
                            # Move to Center X, +20 Y
                            self.renishaw_adapter.stage.move_to(self.calib_origin_stage[0], new_y, self.calib_origin_stage[2])
                        except Exception as e:
                            self.lbl_wizard_status.setText(f"Error moving stage: {e}")
                            self.abort_calibration()
                            return

                        self.calibration_step = 3
                        self.lbl_wizard_status.setText("Step 3: Stage moved +20um in Y. DOUBLE-CLICK the target in the Napari viewer.")
                    
                    elif self.calibration_step == 3:
                        self.complete_calibration()
                
                return # Don't do standard click-to-move during calibration
                
            # If not calibrating, normal click-to-move logic
            if self.base_calib_matrix is None:
                return # Not calibrated
                
            # Get click position
            data_coords = event.position
            # Assuming 2D view, data_coords is (y, x)
            if len(data_coords) >= 2:
                click_y, click_x = data_coords[-2], data_coords[-1]
                
                # Center coordinates
                if self.zwo_panel.optical_center is not None:
                    cy, cx = self.zwo_panel.optical_center
                elif 'Live Feed' in self.viewer.layers:
                    h, w = self.viewer.layers['Live Feed'].data.shape
                    cy, cx = h // 2, w // 2
                else:
                    cy, cx = 500, 500
                
                dp = np.array([click_x - cx, click_y - cy])
                
                # Base real-world offset
                dU_base = self.base_calib_matrix @ dp
                
                # Scale by magnification
                current_obj = self.combo_objective.currentText()
                current_mag = self.objectives.get(current_obj, 1.0)
                
                if current_mag != 0:
                    scale_factor = self.base_calib_mag / current_mag
                else:
                    scale_factor = 1.0
                    
                dU_scaled = dU_base * scale_factor
                
                # We need to MOVE the stage to reverse the offset
                dx, dy = -dU_scaled[0], -dU_scaled[1]
                
                print(f"Double click at px ({click_x:.1f}, {click_y:.1f}). Offset um: dx={dx:.2f}, dy={dy:.2f}")
                
                # Send move command relative to current pos
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
