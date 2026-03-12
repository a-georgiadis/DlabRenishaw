import sys
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QTableWidget, QTableWidgetItem, QLabel, QLineEdit, 
                             QDoubleSpinBox, QSpinBox, QGroupBox, QCheckBox, 
                             QFileDialog, QHeaderView, QAbstractItemView, QPlainTextEdit)
from PyQt6.QtCore import Qt


class BatchMappingWidget(QWidget):
    def __init__(self):
        super().__init__()
        
        # --- UI SETUP ---
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        # 1. THE COORDINATE LIST
        self.create_list_section()
        
        # 2. EDIT & MOVE CONTROLS
        self.create_edit_section()
        
        # 3. ADD/REMOVE CONTROLS
        self.create_management_section()
        
        # 4. MEASUREMENT SETTINGS
        self.create_settings_section()
        
        # 5. GRID SETTINGS (Collapsible)
        self.create_grid_section()
        
        # 6. RUN SECTION (Time Estimate + Run Button)
        run_layout = QHBoxLayout()
        
        self.lbl_estimate = QLabel("Estimated Time: 0.00 s")
        self.lbl_estimate.setStyleSheet("font-weight: bold; font-size: 14px; color: #007ACC;")
        
        self.btn_run_batch = QPushButton("Run Batch Map")
        self.btn_run_batch.setStyleSheet("background-color: #D32F2F; color: white; font-weight: bold; padding: 10px;")
        self.btn_run_batch.clicked.connect(self.run_batch_map)
        
        run_layout.addWidget(self.lbl_estimate)
        run_layout.addStretch() # Pushes the button to the right
        run_layout.addWidget(self.btn_run_batch)
        
        self.layout.addLayout(run_layout)
        
        # 7. LOG OUTPUT
        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumHeight(100) # Keep it small
        self.log_output.setPlaceholderText("System messages will appear here...")
        self.log_message("Welcome! Ready to acquire.") # Initial message
        
        self.layout.addWidget(self.log_output)
        
        # Add stretch to keep things compact at the top
        self.layout.addStretch()

        # Initial calculation
        self.update_time_estimate()

    def create_list_section(self):
        """Creates the scrollable table of points."""
        lbl = QLabel("Coordinate Register:")
        self.layout.addWidget(lbl)
        
        self.table = QTableWidget(0, 4) # 0 Rows, 4 Columns
        self.table.setHorizontalHeaderLabels(["Name", "X", "Y", "Z"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.itemClicked.connect(self.on_row_selected)
        
        # Style
        self.table.setMinimumHeight(150)
        self.layout.addWidget(self.table)

    def create_edit_section(self):
        """Inputs to edit the selected row and Move button."""
        group = QGroupBox("Edit Selected Point")
        layout = QVBoxLayout()
        
        # Row 1: Data inputs
        row_layout = QHBoxLayout()
        
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("Name")
        
        self.edit_x = QDoubleSpinBox()
        self.edit_x.setRange(-10000, 10000)
        self.edit_x.setPrefix("X: ")
        
        self.edit_y = QDoubleSpinBox()
        self.edit_y.setRange(-10000, 10000)
        self.edit_y.setPrefix("Y: ")
        
        self.edit_z = QDoubleSpinBox()
        self.edit_z.setRange(-10000, 10000)
        self.edit_z.setPrefix("Z: ")

        # Connect edits to update table immediately (optional) or via button
        self.edit_name.textChanged.connect(self.update_table_from_inputs)
        self.edit_x.valueChanged.connect(self.update_table_from_inputs)
        self.edit_y.valueChanged.connect(self.update_table_from_inputs)
        self.edit_z.valueChanged.connect(self.update_table_from_inputs)

        row_layout.addWidget(self.edit_name)
        row_layout.addWidget(self.edit_x)
        row_layout.addWidget(self.edit_y)
        row_layout.addWidget(self.edit_z)
        layout.addLayout(row_layout)
        
        # Row 2: Move Button
        self.btn_move = QPushButton("Move Stage to This Point")
        self.btn_move.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_move.clicked.connect(self.move_to_selected)
        layout.addWidget(self.btn_move)
        
        group.setLayout(layout)
        self.layout.addWidget(group)

    def create_management_section(self):
        """Add/Remove buttons and Naming prefix."""
        layout = QHBoxLayout()
        
        self.input_prefix = QLineEdit("Point")
        self.input_prefix.setPlaceholderText("Prefix (e.g. 'Cell')")
        self.input_prefix.setFixedWidth(120)
        
        self.btn_add = QPushButton("Add Current Position")
        self.btn_add.clicked.connect(self.add_point)
        
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self.remove_point)
        
        layout.addWidget(self.input_prefix)
        layout.addWidget(self.btn_add)
        layout.addWidget(self.btn_remove)
        self.layout.addLayout(layout)

    def create_settings_section(self):
        """Measurement parameters and File I/O."""
        group = QGroupBox("Measurement Settings")
        layout = QVBoxLayout()
        
        # 1. Measurement Params (Intensity / Time)
        params_layout = QHBoxLayout()
        self.spin_intensity = QDoubleSpinBox()
        self.spin_intensity.setPrefix("Laser %: ")
        self.spin_intensity.setRange(0, 100)
        self.spin_intensity.setValue(10.0)
        
        self.spin_time = QDoubleSpinBox()
        self.spin_time.setPrefix("Acq Time (s): ")
        self.spin_time.setRange(0.1, 600)
        self.spin_time.setValue(1.0)
        self.spin_time.valueChanged.connect(self.update_time_estimate)
        
        params_layout.addWidget(self.spin_intensity)
        params_layout.addWidget(self.spin_time)
        layout.addLayout(params_layout)
        
        # 2. Load Settings File
        load_layout = QHBoxLayout()
        self.line_filepath = QLineEdit("C:/Default/Settings/config.json")
        self.btn_load = QPushButton("Load Config...")
        self.btn_load.clicked.connect(self.load_settings_file)
        
        load_layout.addWidget(self.line_filepath)
        load_layout.addWidget(self.btn_load)
        layout.addLayout(load_layout)

        # 3. Save Directory & Prefix (NEW)
        save_layout = QHBoxLayout()
        
        self.line_savepath = QLineEdit("C:/Data")
        self.line_savepath.setPlaceholderText("Select Save Folder")
        self.btn_savepath = QPushButton("Save Folder...")
        self.btn_savepath.clicked.connect(self.select_save_folder)
        
        self.line_sample_prefix = QLineEdit("Experiment_01")
        self.line_sample_prefix.setPlaceholderText("File Prefix")
        
        save_layout.addWidget(self.line_savepath)
        save_layout.addWidget(self.btn_savepath)
        save_layout.addWidget(self.line_sample_prefix)
        
        layout.addLayout(save_layout)
        
        group.setLayout(layout)
        self.layout.addWidget(group)

    def create_grid_section(self):
        """Collapsible grid settings."""
        self.chk_grid = QCheckBox("Enable Grid at Each Point")
        self.chk_grid.toggled.connect(self.toggle_grid_options)
        self.layout.addWidget(self.chk_grid)
        
        self.grid_group = QGroupBox("Grid Parameters")
        grid_layout = QHBoxLayout()
        
        self.spin_grid_x = QSpinBox()
        self.spin_grid_x.setPrefix("X Points: ")
        self.spin_grid_x.setRange(1, 100)
        self.spin_grid_x.setValue(1)
        self.spin_grid_x.valueChanged.connect(self.update_time_estimate)
        
        self.spin_grid_y = QSpinBox()
        self.spin_grid_y.setPrefix("Y Points: ")
        self.spin_grid_y.setRange(1, 100)
        self.spin_grid_y.setValue(1)
        self.spin_grid_y.valueChanged.connect(self.update_time_estimate)

        self.spin_spacing = QDoubleSpinBox()
        self.spin_spacing.setPrefix("Spacing (um): ")
        self.spin_spacing.setValue(10.0)
        
        grid_layout.addWidget(self.spin_grid_x)
        grid_layout.addWidget(self.spin_grid_y)
        grid_layout.addWidget(self.spin_spacing)
        
        self.grid_group.setLayout(grid_layout)
        self.grid_group.setVisible(False) # Hidden by default
        self.layout.addWidget(self.grid_group)

    # --- UI LOGIC ---

    def toggle_grid_options(self, checked):
        self.grid_group.setVisible(checked)
        self.update_time_estimate()

    def update_time_estimate(self):
        """
        Calculates: Total Time = Num_Points * (GridX * GridY) * AcqTime
        """
        num_points = self.table.rowCount()
        acq_time = self.spin_time.value()
        
        if self.chk_grid.isChecked():
            grid_points = self.spin_grid_x.value() * self.spin_grid_y.value()
        else:
            grid_points = 1
            
        total_time = num_points * grid_points * acq_time
        self.lbl_estimate.setText(f"Estimated Time: {total_time:.2f} s")

    def on_row_selected(self):
        """When a row is clicked, populate the edit boxes."""
        row = self.table.currentRow()
        if row < 0: return
        
        name = self.table.item(row, 0).text()
        x = float(self.table.item(row, 1).text())
        y = float(self.table.item(row, 2).text())
        z = float(self.table.item(row, 3).text())
        
        # Block signals to prevent infinite loops during population
        self.edit_name.blockSignals(True)
        self.edit_x.blockSignals(True)
        self.edit_y.blockSignals(True)
        self.edit_z.blockSignals(True)
        
        self.edit_name.setText(name)
        self.edit_x.setValue(x)
        self.edit_y.setValue(y)
        self.edit_z.setValue(z)
        
        self.edit_name.blockSignals(False)
        self.edit_x.blockSignals(False)
        self.edit_y.blockSignals(False)
        self.edit_z.blockSignals(False)

    def update_table_from_inputs(self):
        """Updates the table when edit boxes are changed."""
        row = self.table.currentRow()
        if row < 0: return
        
        self.table.setItem(row, 0, QTableWidgetItem(self.edit_name.text()))
        self.table.setItem(row, 1, QTableWidgetItem(str(self.edit_x.value())))
        self.table.setItem(row, 2, QTableWidgetItem(str(self.edit_y.value())))
        self.table.setItem(row, 3, QTableWidgetItem(str(self.edit_z.value())))

    def load_settings_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Settings", self.line_filepath.text(), "JSON Files (*.json);;All Files (*)")
        if path:
            self.line_filepath.setText(path)

    def select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Directory", self.line_savepath.text())
        if folder:
            self.line_savepath.setText(folder)
            
    # --- BACKEND HOOKS ---

    def add_point(self):
        # 1. Get backend Coordinates
        x, y, z = self._get_current_stage_position()
        
        # 2. Get Current Prefix
        prefix = self.input_prefix.text()
        
        # 3. Find the first available number for THIS prefix
        counter = 0
        while True:
            candidate_name = f"{prefix}_{counter}"
            if not self._name_exists(candidate_name):
                name = candidate_name
                break
            counter += 1
        
        # 4. Add to Table
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(str(x)))
        self.table.setItem(row, 2, QTableWidgetItem(str(y)))
        self.table.setItem(row, 3, QTableWidgetItem(str(z)))
        
        self.update_time_estimate()

    def _name_exists(self, name_to_check):
        """Helper to scan table for duplicates"""
        rows = self.table.rowCount()
        for i in range(rows):
            if self.table.item(i, 0).text() == name_to_check:
                return True
        return False

    def remove_point(self):
        row = self.table.currentRow()
        if row >= 0:
            self.table.removeRow(row)
            self.update_time_estimate()
            # Clear inputs if table is now empty
            if self.table.rowCount() == 0:
                self.edit_name.clear()

    def move_to_selected(self):
        row = self.table.currentRow()
        if row < 0: return
        
        x = float(self.table.item(row, 1).text())
        y = float(self.table.item(row, 2).text())
        z = float(self.table.item(row, 3).text())
        
        print(f"Moving to {x}, {y}, {z}")
        self._send_move_command(x, y, z)

    def log_message(self, message):
        """Helper to print to the GUI log window"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_output.appendPlainText(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        self.log_output.verticalScrollBar().setValue(self.log_output.verticalScrollBar().maximum())

    def run_batch_map(self):
        """Main execution logic"""
        point_count = self.table.rowCount()
        
        if point_count == 0:
            self.log_message("Error: No points in register!")
            return

        self.log_message(f"Starting batch on {point_count} points...")
        
        # TODO: Insert your batch loop here
        # Example pseudo-code:
        # for i in range(point_count):
        #     x = self.table.item(i, 1).text()
        #     y = self.table.item(i, 2).text()
        #     self._send_move_command(x, y)
        #     self._trigger_acquisition()
        
        self.log_message("Batch started (Dry Run).")

    # --- PLACEHOLDERS FOR YOUR BACKEND ---

    def _get_current_stage_position(self):
        """
        # TODO: Hook into Pycro-manager or Raman API here.
        Example:
        return core.get_x(), core.get_y(), core.get_z()
        """
        # Returning dummy data for GUI testing
        return 100.5, 200.0, 50.2

    def _send_move_command(self, x, y, z):
        """
        # TODO: Hook into Pycro-manager or Raman API here.
        Example:
        core.set_xy_position(x, y)
        core.set_z_position(z)
        """
        pass

# For testing independently
if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    widget = BatchMappingWidget()
    widget.show()
    sys.exit(app.exec())