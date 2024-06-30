from PySide6.QtWidgets import QWidget
from PySide6.QtWidgets import QPushButton
from PySide6.QtWidgets import QVBoxLayout
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QFileDialog
from PySide6.QtWidgets import QHBoxLayout
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent

import pyqtgraph as pg
import numpy as np
from anotator.annotator import Annotator

from anotator.diffannotator import DiffAnnotator, DiffArtefact
from anotator.config import (DIFFANNOTATOR_ZOOMED_WINDOW_S,
                             DIFFANNOTATOR_ZOOMED_TICK_INTERVAL_S,
                             DIFFANNOTATOR_UNZOOMED_WINDOW_S,
                             DIFFANNOTATOR_UNZOOMED_TICK_INTERVAL_S)
from anotator.helpers import ts_to_tick, ts_to_full_date


class DiffAnnotateTab(QWidget):

    def __init__(self, annotator, window, *args, **kwargs) -> None:
        QWidget.__init__(self, *args, **kwargs)

        self.mainwindow = window
        self.annotator: Annotator = annotator
        self.diffannotator: DiffAnnotator = DiffAnnotator(annotator)
        self.diffartf: DiffArtefact = None

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Info

        self.info_container = QWidget()
        self.info_layout = QHBoxLayout()
        self.info_container.setLayout(self.info_layout)

        # ARTF1 file label
        self.artf1_label = QLabel()
        self.artf1_label.setText("ARTF1: N/A")
        self.info_layout.addWidget(self.artf1_label)

        # ARTF2 file label
        self.artf2_label = QLabel()
        self.artf2_label.setText("ARTF2: N/A")
        self.info_layout.addWidget(self.artf2_label)

        # Diff file label
        self.diff_label = QLabel()
        self.diff_label.setText("Artefakt v souboru: N/A")
        self.info_layout.addWidget(self.diff_label)

        # Diff count label
        self.diff_count_label = QLabel()
        self.diff_count_label.setText("Rozdílů: N/A")
        self.info_layout.addWidget(self.diff_count_label)

        self.layout.addWidget(self.info_container)

        # Plots

        # ABP charts
        self.abp_container = QWidget()
        self.abp_layout = QHBoxLayout()
        self.abp_container.setLayout(self.abp_layout)

        # ABP Zoomed chart
        self.abp_zoomed_container = QWidget()
        self.abp_zoomed_layout = QVBoxLayout()
        self.abp_zoomed_container.setLayout(self.abp_zoomed_layout)

        self.abp_zoomed_label = QLabel()
        self.abp_zoomed_label.setText("ABP 10s")
        self.abp_zoomed_layout.addWidget(self.abp_zoomed_label)

        self.abp_zoomed = pg.PlotWidget()
        self.abp_zoomed.getAxis("bottom").setHeight(30)
        self.abp_zoomed.setMouseEnabled(x=False, y=False)
        self.abp_zoomed_layout.addWidget(self.abp_zoomed)

        self.abp_layout.addWidget(self.abp_zoomed_container)

        # ABP Unzoomed chart
        self.abp_unzoomed_container = QWidget()
        self.abp_unzoomed_layout = QVBoxLayout()
        self.abp_unzoomed_container.setLayout(self.abp_unzoomed_layout)

        self.abp_unzoomed_label = QLabel()
        self.abp_unzoomed_label.setText("ABP -+1min")
        self.abp_unzoomed_layout.addWidget(self.abp_unzoomed_label)

        self.abp_unzoomed = pg.PlotWidget()
        self.abp_unzoomed.getAxis("bottom").setHeight(30)
        self.abp_unzoomed.setMouseEnabled(x=False, y=False)
        self.abp_unzoomed_layout.addWidget(self.abp_unzoomed)

        self.abp_layout.addWidget(self.abp_unzoomed_container)

        self.layout.addWidget(self.abp_container)

        # ICP charts
        self.icp_container = QWidget()
        self.icp_layout = QHBoxLayout()
        self.icp_container.setLayout(self.icp_layout)

        # ICP Zoomed chart
        self.icp_zoomed_container = QWidget()
        self.icp_zoomed_layout = QVBoxLayout()
        self.icp_zoomed_container.setLayout(self.icp_zoomed_layout)

        self.icp_zoomed_label = QLabel()
        self.icp_zoomed_label.setText("ICP 10s")
        self.icp_zoomed_layout.addWidget(self.icp_zoomed_label)

        self.icp_zoomed = pg.PlotWidget()
        self.icp_zoomed.getAxis("bottom").setHeight(30)
        self.icp_zoomed.setMouseEnabled(x=False, y=False)
        self.icp_zoomed_layout.addWidget(self.icp_zoomed)

        self.icp_layout.addWidget(self.icp_zoomed_container)

        # ICP Unzoomed chart
        self.icp_unzoomed_container = QWidget()
        self.icp_unzoomed_layout = QVBoxLayout()
        self.icp_unzoomed_container.setLayout(self.icp_unzoomed_layout)

        self.icp_unzoomed_label = QLabel()
        self.icp_unzoomed_label.setText("ICP -+1min")
        self.icp_unzoomed_layout.addWidget(self.icp_unzoomed_label)

        self.icp_unzoomed = pg.PlotWidget()
        self.icp_unzoomed.getAxis("bottom").setHeight(30)
        self.icp_unzoomed.setMouseEnabled(x=False, y=False)
        self.icp_unzoomed_layout.addWidget(self.icp_unzoomed)

        self.icp_layout.addWidget(self.icp_unzoomed_container)

        self.layout.addWidget(self.icp_container)

        # Artefact buttons

        self.artf_button_layout = QHBoxLayout()

        # Mark real artefact button
        self.btn_mark_real_artf = QPushButton("Artefakt")
        self.btn_mark_real_artf.setStyleSheet("QPushButton { background-color: rgba(255, 0, 0, 80); }")
        self.btn_mark_real_artf.clicked.connect(self.handle_mark_real_artf)
        self.artf_button_layout.addWidget(self.btn_mark_real_artf)

        # Unmark real artefact button
        self.btn_unmark_real_artf = QPushButton("Není artefakt")
        self.btn_unmark_real_artf.setStyleSheet("QPushButton { background-color: rgba(0, 0, 255, 80); }")
        self.btn_unmark_real_artf.clicked.connect(self.handle_unmark_real_artf)
        self.artf_button_layout.addWidget(self.btn_unmark_real_artf)

        self.layout.addLayout(self.artf_button_layout)

        # Control buttons

        self.control_button_layout = QHBoxLayout()

        # Previous button
        self.btn_previous = QPushButton("Předchozí")
        self.btn_previous.clicked.connect(self.handle_previous)
        self.control_button_layout.addWidget(self.btn_previous)

        # Next button
        self.btn_next = QPushButton("Další")
        self.btn_next.clicked.connect(self.handle_next)
        self.control_button_layout.addWidget(self.btn_next)

        self.layout.addLayout(self.control_button_layout)

        # File Buttons

        self.file_button_layout = QHBoxLayout()
        # Load ARTF1 button
        self.btn_load_artf1 = QPushButton("Načíst ARTF1")
        self.btn_load_artf1.clicked.connect(self.handle_load_artf1)
        self.file_button_layout.addWidget(self.btn_load_artf1)

        # Load ARTF2 button
        self.btn_load_artf2 = QPushButton("Načíst ARTF2")
        self.btn_load_artf2.clicked.connect(self.handle_load_artf2)
        self.file_button_layout.addWidget(self.btn_load_artf2)

        # Find diffs button
        self.btn_find_diffs = QPushButton("Najít rozdíly")
        self.btn_find_diffs.clicked.connect(self.handle_find_diffs)
        self.file_button_layout.addWidget(self.btn_find_diffs)

        # Export ARTF button
        self.btn_export_artf = QPushButton("Exportovat ARTF")
        self.btn_export_artf.clicked.connect(self.handle_export_artf)
        self.file_button_layout.addWidget(self.btn_export_artf)

        self.layout.addLayout(self.file_button_layout)


    def handle_load_artf1(self):
        if not self.annotator.is_data():
            self.show_msg("Nejprve je třeba načíst HDF5 data.", "Chyba")
            return

        # File dialog
        filepath = QFileDialog.getOpenFileName(self, "Načíst ARTF1", "", "ARTF (*.artf)")[0]
        if filepath == "":
            return

        try:
            self.diffannotator.load_artf1(filepath)
        except Exception as e:
            self.show_msg(f"ARTF1 se nepodařilo načíst: {e}", "Chyba")
            return

        filename = self.diffannotator.get_artf1_filename()
        self.artf1_label.setText(f"ARTF1: {filename}")

        self.clear_all_plots()
        self.reset_diff_labels()

    def handle_load_artf2(self):
        if not self.annotator.is_data():
            self.show_msg("Nejprve je třeba načíst HDF5 data.", "Chyba")
            return

        # File dialog
        filepath = QFileDialog.getOpenFileName(self, "Načíst ARTF2", "", "ARTF (*.artf)")[0]
        if filepath == "":
            return

        try:
            self.diffannotator.load_artf2(filepath)
        except Exception as e:
            self.show_msg(f"ARTF2 se nepodařilo načíst: {e}", "Chyba")
            return

        filename = self.diffannotator.get_artf2_filename()
        self.artf2_label.setText(f"ARTF2: {filename}")

        self.clear_all_plots()
        self.reset_diff_labels()

    def handle_find_diffs(self):
        if not self.annotator.is_data():
            self.show_msg("Nejprve je třeba načíst HDF5 data.", "Chyba")
            return

        if not self.diffannotator.are_artfs_loaded():
            self.show_msg("Nejprve je třeba načíst oba ARTF soubory.", "Chyba")
            return

        count = self.diffannotator.find_diffs()
        #if count > 0:
        self.render_diff()
        self.show_msg("Nalezeno {} rozdílů.".format(count))

    def clear_all_plots(self) -> None:
        """Clears all plots"""
        self.icp_zoomed.clear()
        self.icp_unzoomed.clear()
        self.abp_zoomed.clear()
        self.abp_unzoomed.clear()

    def show_msg(self, msg: str, msgtype: str="Info") -> None:
        """Shows popup message"""
        msgbox = QMessageBox()
        msgbox.setWindowTitle(msgtype)
        msgbox.setText(msg)
        msgbox.exec_()

    def handle_previous(self):
        if not self.annotator.is_data():
            self.show_msg("Nejprve je třeba načíst HDF5 data.", "Chyba")
            return

        if not self.diffannotator.are_artfs_loaded():
            self.show_msg("Nejprve je třeba načíst oba ARTF soubory.", "Chyba")
            return

        if not self.diffannotator.get_found_diffs():
            self.show_msg("Nejprve je třeba najít rozdíly.", "Chyba")
            return

        try:
            self.diffannotator.prev_diff()
        except Exception as e:
            self.show_msg(str(e), "Chyba")
            return
        self.render_diff()

    def handle_next(self):
        if not self.annotator.is_data():
            self.show_msg("Nejprve je třeba načíst HDF5 data.", "Chyba")
            return

        if not self.diffannotator.are_artfs_loaded():
            self.show_msg("Nejprve je třeba načíst oba ARTF soubory.", "Chyba")
            return

        if not self.diffannotator.get_found_diffs():
            self.show_msg("Nejprve je třeba najít rozdíly.", "Chyba")
            return

        try:
            self.diffannotator.next_diff()
        except Exception as e:
            self.show_msg(str(e), "Chyba")
            return
        self.render_diff()

    def handle_mark_real_artf(self):
        if not self.annotator.is_data():
            self.show_msg("Nejprve je třeba načíst HDF5 data.", "Chyba")
            return
        if not self.diffannotator.are_artfs_loaded():
            self.show_msg("Nejprve je třeba načíst oba ARTF soubory.", "Chyba")
            return
        if not self.diffannotator.get_found_diffs():
            self.show_msg("Nejprve je třeba najít rozdíly.", "Chyba")
            return
        self.diffannotator.mark_real()
        self.render_diff()

    def handle_unmark_real_artf(self):
        if not self.annotator.is_data():
            self.show_msg("Nejprve je třeba načíst HDF5 data.", "Chyba")
            return
        if not self.diffannotator.are_artfs_loaded():
            self.show_msg("Nejprve je třeba načíst oba ARTF soubory.", "Chyba")
            return
        if not self.diffannotator.get_found_diffs():
            self.show_msg("Nejprve je třeba najít rozdíly.", "Chyba")
            return
        self.diffannotator.unmark_real()
        self.render_diff()

    def handle_export_artf(self):
        if not self.annotator.is_data():
            self.show_msg("Nejprve je třeba načíst HDF5 data.", "Chyba")
            return

        if not self.diffannotator.are_artfs_loaded():
            self.show_msg("Nejprve je třeba načíst oba ARTF soubory.", "Chyba")
            return

        if not self.diffannotator.get_found_diffs():
            self.show_msg("Nejprve je třeba najít rozdíly.", "Chyba")
            return

        # File dialog
        filepath = QFileDialog.getSaveFileName(self, "Exportovat ARTF", "", "ARTF (*.artf)")[0]
        if filepath == "":
            return

        if not filepath.endswith(".artf"):
            filepath += ".artf"

        try:
            self.diffannotator.export(filepath)
        except Exception as e:
            self.show_msg(str(e), "Chyba")
            return
        self.show_msg(f"ARTF soubor byl exportován do {filepath}.")

    def render_diff(self) -> None:
        """Renders diff plots and related info"""
        self.render_plots()
        self.show_artefact_file()
        self.show_diff_count()

    def reset_diff_labels(self) -> None:
        """Resets diff labels"""
        self.diff_label.setText("Artefakt v souboru: N/A")
        self.diff_count_label.setText("Rozdílů: N/A")

    def show_diff_count(self) -> None:
        """Shows current diff count"""
        diff_count = self.diffannotator.get_diff_count()
        if diff_count == 0:
            self.diff_count_label.setText("Rozdílů: 0")
            return
        current_diff = self.diffannotator.get_current_diff_index()
        self.diff_count_label.setText(f"Rozdílů: {current_diff + 1}/{diff_count}")

    def show_artefact_file(self) -> None:
        """Shows current artefact file"""
        if self.diffannotator.get_diff_count() == 0:
            return
        diff = self.diffannotator.get_diff()
        self.diff_label.setText(f"Artefakt v souboru: {diff.artf_file}")

    def render_plots(self) -> None:
        """Renders plots"""
        if self.diffannotator.get_diff_count() == 0:
            return
        diffart = self.diffannotator.get_diff()
        self.render_unzoomed_plots(diffart)
        self.render_zoomed_plots(diffart)

    def render_unzoomed_plots(self, diffart):
        """Renders unzoomed plots"""
        art_type = diffart.art_type
        base = diffart.base

        if art_type == "icp":
            data = self.annotator.data_icp
            pws = [self.icp_unzoomed]
        elif art_type == "abp":
            data = self.annotator.data_abp
            pws = [self.abp_unzoomed]
        else:
            raise ValueError("Unknown ARTF type")

        art_window_s = self.annotator.window_s
        sr = self.annotator.sr
        tick_interval_s = DIFFANNOTATOR_UNZOOMED_TICK_INTERVAL_S
        window_s = DIFFANNOTATOR_UNZOOMED_WINDOW_S

        # Artefact window
        rbase = base
        rend = rbase + art_window_s * sr

        # Main window
        delta = (window_s - art_window_s) * sr
        xmin = rbase - delta // 2
        if xmin < 0:
            xmin = 0
        xmax = rend + delta // 2
        if xmax > len(data):
            xmax = len(data)

        # Data to be plotted
        x = np.arange(xmin, xmax)
        y = data[xmin:xmax]

        # Clear all plots
        self.icp_unzoomed.clear()
        self.abp_unzoomed.clear()

        for pw in pws:
            # Plot data
            pw.clear()
            pw.plot(x, y)

            # Set ticks
            ax = pw.getAxis("bottom")
            ticksi = range(xmin, xmax + 1, sr * tick_interval_s)
            start_ts = self.annotator.get_start_time_s()
            ticks = [(i, f"{ts_to_tick(start_ts + i/sr, date=False)}") for i in ticksi]
            ax.setTicks([ticks])

            # Add span region
            brush = pg.mkBrush("#BFBFBFA0")
            region = pg.LinearRegionItem(values=(rbase, rend),
                                              orientation="vertical",
                                              movable=False,
                                              brush=brush)
            pw.addItem(region)

            # Padding
            pw.setXRange(xmin, xmax, padding=0.05)


    def render_zoomed_plots(self, diffart):
        """Renders zoomed plots"""
        art_type = diffart.art_type
        base = diffart.base

        if art_type == "icp":
            data = self.annotator.data_icp
            pws = [self.icp_zoomed]
        elif art_type == "abp":
            data = self.annotator.data_abp
            pws = [self.abp_zoomed]
        else:
            raise ValueError("Unknown ARTF type")

        art_window_s = self.annotator.window_s
        sr = self.annotator.sr
        tick_interval_s = DIFFANNOTATOR_ZOOMED_TICK_INTERVAL_S
        window_s = DIFFANNOTATOR_ZOOMED_WINDOW_S

        # Artefact window
        rbase = base
        rend = rbase + art_window_s * sr

        # Main window
        delta = (window_s - art_window_s) * sr
        xmin = rbase - delta // 2
        if xmin < 0:
            xmin = 0
        xmax = rend + delta // 2
        if xmax > len(data):
            xmax = len(data)

        # Data to be plotted
        x = np.arange(xmin, xmax)
        y = data[xmin:xmax]

        # Clear all plots
        self.icp_zoomed.clear()
        self.abp_zoomed.clear()

        for pw in pws:
            # Plot data
            pw.clear()
            pw.plot(x, y)

            # Set ticks
            ax = pw.getAxis("bottom")
            ticksi = range(xmin, xmax + 1, sr * tick_interval_s)
            start_ts = self.annotator.get_start_time_s()
            ticks = [(i, f"{ts_to_tick(start_ts + i/sr, date=False)}") for i in ticksi]
            ax.setTicks([ticks])

            # Add span region
            if diffart.is_real:
                brush = pg.mkBrush("#FF000032")
            else:
                brush = None
            region = pg.LinearRegionItem(values=(rbase, rend),
                                              orientation="vertical",
                                              movable=False,
                                              brush=brush)
            pw.addItem(region)

            # Padding
            pw.setXRange(xmin, xmax, padding=0.05)

    def keyPressEvent(self, event : QKeyEvent) -> None:
        """KeyPress event listener to enable keyboard shortcuts"""
        # <-, H, A
        if event.key() in (Qt.Key_Left, Qt.Key_H, Qt.Key_A):
            self.handle_previous()
        # ->, L, D
        elif event.key() in (Qt.Key_Right, Qt.Key_L, Qt.Key_D):
            self.handle_next()
        # Space
        elif event.key() == Qt.Key_Space:
            if not self.diffannotator.get_diff().is_real:
                self.handle_mark_real_artf()
            else:
                self.handle_unmark_real_artf()
