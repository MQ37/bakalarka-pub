from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from anotator import MainWindow

import numpy as np
import pyqtgraph as pg

from anotator.errors import EmptyDataError, InvalidIndexError
from anotator import Annotator
from anotator.helpers import ts_to_tick, ts_to_full_date
from anotator.config import (ANNOTATOR_UNZOOMED_WINDOW_S,
                             ANNOTATOR_ZOOMED_WINDOW_S,
                             ANNOTATOR_UNZOOMED_TICK_INTERVAL_S,
                             ANNOTATOR_ZOOMED_TICK_INTERVAL_S)

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QMessageBox,
    QVBoxLayout,
    QLabel,
    QFileDialog
)


class AnnotateTab(QWidget):
    """Qt widget tab for artefact annotation"""

    def __init__(self, annotator: Annotator, window: MainWindow, *args, **kwargs) -> None:
        QWidget.__init__(self, *args, **kwargs)

        self.mainwindow: MainWindow = window
        self.annotator: Annotator = annotator

        self.window_size_unzoomed_s = ANNOTATOR_UNZOOMED_WINDOW_S
        self.tick_interval_unzoomed_s = ANNOTATOR_UNZOOMED_TICK_INTERVAL_S
        self.window_size_zoomed_s = ANNOTATOR_ZOOMED_WINDOW_S
        self.tick_interval_zoomed_s = ANNOTATOR_ZOOMED_TICK_INTERVAL_S

        self.art_type = 0

        # Base layout
        layout = QGridLayout()
        self.setLayout(layout)

        info_container = QWidget()
        info_layout = QHBoxLayout()
        info_container.setLayout(info_layout)

        time_range_label = QLabel()
        time_range_label.setText("Aktuální anotace: NA - NA")
        info_layout.addWidget(time_range_label)
        self.time_range_label = time_range_label

        art_type_label = QLabel()
        art_type_label.setText(f"Typ artefaktu: {self.art_type}")
        info_layout.addWidget(art_type_label)
        self.art_type_label = art_type_label

        layout.addWidget(info_container, 0, 0)

        # ICP charts
        icp_container = QWidget()
        icp_layout = QGridLayout()
        icp_container.setLayout(icp_layout)

        # ICP Zoomed chart
        icp_zoomed_container = QWidget()
        icp_zoomed_layout = QVBoxLayout()
        icp_zoomed_container.setLayout(icp_zoomed_layout)

        icp_zoomed_label = QLabel()
        icp_zoomed_label.setText("ICP 10s")
        icp_zoomed_layout.addWidget(icp_zoomed_label)

        icp_zoomed = pg.PlotWidget()
        icp_zoomed.getAxis("bottom").setHeight(30)
        icp_zoomed.setMouseEnabled(x=False, y=False)
        icp_zoomed_layout.addWidget(icp_zoomed)
        self.icp_zoomed = icp_zoomed

        icp_layout.addWidget(icp_zoomed_container, 0, 0)

        # ICP Unzoomed chart
        icp_unzoomed_container = QWidget()
        icp_unzoomed_layout = QVBoxLayout()
        icp_unzoomed_container.setLayout(icp_unzoomed_layout)

        icp_unzoomed_label = QLabel()
        icp_unzoomed_label.setText("ICP -+1m")
        icp_unzoomed_layout.addWidget(icp_unzoomed_label)

        icp_unzoomed = pg.PlotWidget()
        icp_unzoomed.getAxis("bottom").setHeight(30)
        icp_unzoomed.setMouseEnabled(x=False, y=False)
        icp_unzoomed_layout.addWidget(icp_unzoomed)
        self.icp_unzoomed = icp_unzoomed

        icp_layout.addWidget(icp_unzoomed_container, 0, 1)

        # ABP charts
        abp_container = QWidget()
        abp_layout = QGridLayout()
        abp_container.setLayout(abp_layout)

        # ABP Zoomed chart
        abp_zoomed_container = QWidget()
        abp_zoomed_layout = QVBoxLayout()
        abp_zoomed_container.setLayout(abp_zoomed_layout)

        abp_zoomed_label = QLabel()
        abp_zoomed_label.setText("ABP 10s")
        abp_zoomed_layout.addWidget(abp_zoomed_label)

        abp_zoomed = pg.PlotWidget()
        abp_zoomed.getAxis("bottom").setHeight(30)
        abp_zoomed.setMouseEnabled(x=False, y=False)
        abp_zoomed_layout.addWidget(abp_zoomed)
        self.abp_zoomed = abp_zoomed

        abp_layout.addWidget(abp_zoomed_container, 0, 0)

        # ABP Unzoomed chart
        abp_unzoomed_container = QWidget()
        abp_unzoomed_layout = QVBoxLayout()
        abp_unzoomed_container.setLayout(abp_unzoomed_layout)

        abp_unzoomed_label = QLabel()
        abp_unzoomed_label.setText("ABP -+1min")
        abp_unzoomed_layout.addWidget(abp_unzoomed_label)

        abp_unzoomed = pg.PlotWidget()
        abp_unzoomed.getAxis("bottom").setHeight(30)
        abp_unzoomed.setMouseEnabled(x=False, y=False)
        abp_unzoomed_layout.addWidget(abp_unzoomed)
        self.abp_unzoomed = abp_unzoomed

        abp_layout.addWidget(abp_unzoomed_container, 0, 1)

        # Add chart containers to the main layout
        layout.addWidget(icp_container, 2, 0)
        layout.addWidget(abp_container, 1, 0)

        ## Next, Prev, Artefact Button container ##
        button_container = QWidget()
        button_layout = QHBoxLayout()
        button_container.setLayout(button_layout)

        # Prev button
        button_prev = QPushButton()
        button_prev.setText("Předchozí")
        button_prev.clicked.connect(self.button_prev_clicked)
        button_layout.addWidget(button_prev)

        # Next button
        button_next = QPushButton()
        button_next.setText("Další")
        button_next.clicked.connect(self.button_next_clicked)
        button_layout.addWidget(button_next)

        # Artefact button
        button_artefact = QPushButton()
        button_artefact.setText("Artefakt")
        button_artefact.setStyleSheet("QPushButton { background-color: rgba(255, 0, 0, 80); }")
        button_artefact.clicked.connect(self.button_artefact_clicked)
        button_layout.addWidget(button_artefact)

        # Not artefact button
        button_not_artefact = QPushButton()
        button_not_artefact.setText("Není artefakt")
        button_not_artefact.setStyleSheet("QPushButton { background-color: rgba(0, 0, 255, 80); }")
        button_not_artefact.clicked.connect(self.button_not_artefact_clicked)
        button_layout.addWidget(button_not_artefact)

        layout.addWidget(button_container, 3, 0, 1, 2)
        ## End container ##

        ## ICP ABP Artefact Button container ##
        button_container = QWidget()
        button_layout = QHBoxLayout()
        button_container.setLayout(button_layout)

        # ICP Artefact button
        button_icp_artefact = QPushButton()
        button_icp_artefact.setText("ICP Artefakt")
        button_icp_artefact.setStyleSheet("QPushButton { background-color: rgba(255, 0, 0, 80); }")
        button_icp_artefact.clicked.connect(self.button_icp_artefact_clicked)
        button_layout.addWidget(button_icp_artefact)

        # ICP Not artefact button
        button_icp_not_artefact = QPushButton()
        button_icp_not_artefact.setText("ICP Není artefakt")
        button_icp_not_artefact.setStyleSheet("QPushButton { background-color: rgba(0, 0, 255, 80); }")
        button_icp_not_artefact.clicked.connect(self.button_icp_not_artefact_clicked)
        button_layout.addWidget(button_icp_not_artefact)

        # ABP Artefact button
        button_abp_artefact = QPushButton()
        button_abp_artefact.setText("ABP Artefakt")
        button_abp_artefact.setStyleSheet("QPushButton { background-color: rgba(255, 0, 0, 80); }")
        button_abp_artefact.clicked.connect(self.button_abp_artefact_clicked)
        button_layout.addWidget(button_abp_artefact)

        # ABP Not artefact button
        button_abp_not_artefact = QPushButton()
        button_abp_not_artefact.setText("ABP Není artefakt")
        button_abp_not_artefact.setStyleSheet("QPushButton { background-color: rgba(0, 0, 255, 80); }")
        button_abp_not_artefact.clicked.connect(self.button_abp_not_artefact_clicked)
        button_layout.addWidget(button_abp_not_artefact)

        layout.addWidget(button_container, 4, 0, 1, 2)
        ## End container ##

        # Export button
        button_container = QWidget()
        button_container_layout = QHBoxLayout()
        button_container.setLayout(button_container_layout)

        button_export = QPushButton()
        button_export.setText("Export")
        button_export.clicked.connect(self.button_export_clicked)
        button_container_layout.addWidget(button_export)

        for i in range(6):
            button = QPushButton()
            button.setText(f"{i}")
            button.clicked.connect(getattr(self, f"set_art_type_{i}"))
            button_container_layout.addWidget(button)

        layout.addWidget(button_container, 5, 0)

    def set_art_type(self, int):
        self.art_type = int
        self.art_type_label.setText(f"Typ artefaktu: {self.art_type}")

    def set_art_type_0(self):
        self.set_art_type(0)

    def set_art_type_1(self):
        self.set_art_type(1)

    def set_art_type_2(self):
        self.set_art_type(2)
    
    def set_art_type_3(self):
        self.set_art_type(3)
    
    def set_art_type_4(self):
        self.set_art_type(4)

    def set_art_type_5(self):
        self.set_art_type(5)

    def keyPressEvent(self, event : QKeyEvent) -> None:
        """KeyPress event listener to enable keyboard shortcuts"""
        if not self.annotator.is_data():
            return
        # <-, H, A
        if event.key() in (Qt.Key_Left, Qt.Key_H, Qt.Key_A): # type: ignore
            self.switch_prev()
        # ->, L, D
        elif event.key() in (Qt.Key_Right, Qt.Key_L, Qt.Key_D): # type: ignore
            self.switch_next()
        # Space
        elif event.key() == Qt.Key_Space: # type: ignore
            if self.annotator.is_artefact(icp=True, abp=True):
                self.unmark_artefact()
            else:
                self.mark_artefact()
        # C
        elif event.key() == Qt.Key_C: # type: ignore
            if self.annotator.is_artefact(icp=True):
                self.unmark_artefact(icp=True)
            else:
                self.mark_artefact(icp=True)
        # B
        elif event.key() == Qt.Key_B: # type: ignore
            if self.annotator.is_artefact(abp=True):
                self.unmark_artefact(abp=True)
            else:
                self.mark_artefact(abp=True)
        elif event.key() == Qt.Key_0: # type: ignore
            self.art_type = 0
            self.art_type_label.setText(f"Typ artefaktu: {self.art_type}")
        elif event.key() == Qt.Key_1: # type: ignore
            self.art_type = 1
            self.art_type_label.setText(f"Typ artefaktu: {self.art_type}")
        elif event.key() == Qt.Key_2: # type: ignore
            self.art_type = 2
            self.art_type_label.setText(f"Typ artefaktu: {self.art_type}")
        elif event.key() == Qt.Key_3: # type: ignore
            self.art_type = 3
            self.art_type_label.setText(f"Typ artefaktu: {self.art_type}")
        elif event.key() == Qt.Key_4: # type: ignore
            self.art_type = 4
            self.art_type_label.setText(f"Typ artefaktu: {self.art_type}")
        elif event.key() == Qt.Key_5: # type: ignore
            self.art_type = 5
            self.art_type_label.setText(f"Typ artefaktu: {self.art_type}")

    def render_zoomed_plots(self) -> None:
        """Renders the main plots"""
        sr = self.annotator.get_sr()
        base, end, (data_icp, data_abp) = self.annotator.get_window()
        x = np.arange(base, end)


        pws = [self.icp_zoomed, self.abp_zoomed]
        data = [data_icp, data_abp]

        for i, (pw, _data) in enumerate(zip(pws, data)):
            pw.clear()
            pw.plot(x, _data)
            ax = pw.getAxis("bottom")
            ticksi = range(base, end + 1, sr * self.tick_interval_zoomed_s)
            start_ts = self.annotator.get_start_time_s()
            ticks = [(i, f"{ts_to_tick(start_ts + i/sr, date=False)}") for i in ticksi]
            ax.setTicks([ticks])

            is_artefact = self.annotator.is_artefact()
            # ICP
            if i == 0:
                is_artefact = is_artefact or self.annotator.is_artefact(icp=True)
            # ABP
            elif i == 1:
                is_artefact = is_artefact or self.annotator.is_artefact(abp=True)

            # Add span region
            if is_artefact:
                brush = pg.mkBrush("#FF000032")
            else:
                brush = None
            rbase, rend = self.annotator.get_roi()
            region = pg.LinearRegionItem(values=(rbase, rend),
                                              orientation="vertical",
                                              movable=False,
                                              brush=brush)
            pw.addItem(region)

            if is_artefact:
                try:
                    art_type = self.annotator.get_artefact_type(rbase)
                except Exception:
                    pass
                else:
                    label = pg.TextItem(f"{art_type}", color="white")
                    label.setPos(rbase, 0.5)
                    pw.addItem(label)

            # Padding
            pw.setXRange(base, end, padding=0.05) # type: ignore

    def render_plot_unzoomed(self, pw : pg.PlotWidget,
                     window_size_s : int, tick_interval_s : int,
                     mxmin : int, mxmax : int,
                     rbase : int, rend : int,
                     data: np.ndarray,
                     is_icp: bool) -> tuple[int, int]:
        """Renders secondary PlotWidget"""
        sr = self.annotator.get_sr()
        #data = self.annotator.get_data()
        mwindow = sr * (self.annotator.get_window_s() * (1 + self.annotator.window_around * 2))

        # Compute delta window between main or previous
        # and split the delta window on each side
        window = sr * window_size_s
        delta_window = window - mwindow
        xmin = mxmin - (delta_window // 2)
        if xmin < 0:
            xmin = 0
        xmax = mxmax + (delta_window // 2)
        if xmax > len(data):
            xmax = len(data)

        # Data to be plotted
        x = np.arange(xmin, xmax)
        y = data[xmin:xmax]

        # Plot data
        pw.clear()
        pw.plot(x, y)
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


        brush = pg.mkBrush("#FF000032")
        if is_icp:
            artefacts_in_region = self.annotator.get_artefacts_in_region(xmin, xmax, icp=True)
        else:
            artefacts_in_region = self.annotator.get_artefacts_in_region(xmin, xmax, abp=True)

        for base in artefacts_in_region:
            end = base + self.annotator.window_s * self.annotator.sr # type: ignore
            region = pg.LinearRegionItem(values=(base, end),
                                                orientation="vertical",
                                                brush=brush,
                                                movable=False)
            pw.addItem(region)

            try:
                art_type = self.annotator.get_artefact_type(base)
            except Exception:
                pass
            else:
                label = pg.TextItem(f"{art_type}", color="white")
                label.setPos(base, 0.5)
                pw.addItem(label)

        # Padding
        pw.setXRange(xmin, xmax, padding=0.05) # type: ignore

        return xmin, xmax

    def render_plots(self) -> None:
        """Renders all PlotWidgets on the tab"""
        self.render_zoomed_plots()

        (data_icp, data_abp) = self.annotator.get_data()

        xmin, xmax, _ = self.annotator.get_window()
        rbase = xmin
        rend = xmax
        window_size_s = self.window_size_unzoomed_s
        tick_interval_s = self.tick_interval_unzoomed_s

        self.render_plot_unzoomed(self.icp_unzoomed, window_size_s, tick_interval_s,
                                        xmin, xmax, rbase, rend, data_icp, is_icp=True)
        self.render_plot_unzoomed(self.abp_unzoomed, window_size_s, tick_interval_s,
                                        xmin, xmax, rbase, rend, data_abp, is_icp=False)

        # Time range labels
        sr = self.annotator.get_sr()
        start_dt = ts_to_full_date(self.annotator.get_start_time_s() + xmin / sr)
        end_dt = ts_to_full_date(self.annotator.get_start_time_s() + xmax / sr)
        self.time_range_label.setText(f"Aktuální anotace: {start_dt} - {end_dt}")

    def mark_artefact(self, *args, **kwargs) -> None:
        """Marks current section as artefact"""
        if not self.annotator.is_data():
            return

        kwargs["art_type"] = self.art_type
        self.annotator.mark_artefact(*args, **kwargs)
        # Switch to next section
        #if not (kwargs.get("abp") or kwargs.get("icp")):
        self.switch_next()
        self.render_plots()

    def unmark_artefact(self, *args, **kwargs) -> None:
        """Unmarks current section as artefact"""
        if not self.annotator.is_data():
            return

        self.annotator.unmark_artefact(*args, **kwargs)
        self.switch_next()
        self.render_plots()

    def switch_next(self) -> None:
        """Switches to next section"""
        try:
            self.annotator.next()
            self.render_plots()
        except Exception:
            pass

    def switch_prev(self) -> None:
        """Switches to previous section"""
        try:
            self.annotator.prev()
            self.render_plots()
        except Exception:
            pass

    def button_next_clicked(self) -> None:
        """Next button callback, switches to next section"""
        try:
            self.switch_next()
        except EmptyDataError:
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")

    def button_prev_clicked(self) -> None:
        """Previous button callback, switches to previous section"""
        try:
            self.switch_prev()
        except EmptyDataError:
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")

    def button_artefact_clicked(self) -> None:
        """Artefact button callback, marks section as artefact"""
        try:
            self.mark_artefact()
        except EmptyDataError:
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")

    def button_not_artefact_clicked(self) -> None:
        """Not artefact button callback, marks section as not artefact"""
        try:
            self.unmark_artefact()
        except EmptyDataError:
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")

    def show_msg(self, msg: str, title : str = "Info") -> None:
        """Shows popup message"""
        msgbox = QMessageBox()
        msgbox.setWindowTitle(title)
        msgbox.setText(msg)
        msgbox.exec_()

    def button_export_clicked(self) -> None:
        if not self.annotator.is_data():
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")
            return

        options = QFileDialog.Options() # type: ignore
        options |= QFileDialog.DontUseNativeDialog # type: ignore
        filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "ARTF Files (*.artf)", options=options)
        if not filename:
            return

        if not filename.endswith(".artf"):
            filename += ".artf"

        try:
            self.annotator.export(filename)
        except Exception:
            pass
        else:
            self.show_msg(f"Exportováno do {filename}")

    def button_icp_artefact_clicked(self) -> None:
        try:
            self.mark_artefact(icp=True)
        except EmptyDataError:
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")

    def button_icp_not_artefact_clicked(self) -> None:
        try:
            self.unmark_artefact(icp=True)
        except EmptyDataError:
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")

    def button_abp_artefact_clicked(self) -> None:
        try:
            self.mark_artefact(abp=True)
        except EmptyDataError:
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")

    def button_abp_not_artefact_clicked(self) -> None:
        try:
            self.unmark_artefact(abp=True)
        except EmptyDataError:
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty")
