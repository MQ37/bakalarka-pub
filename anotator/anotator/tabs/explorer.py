from __future__ import annotations

from typing import Union
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from anotator import MainWindow

import numpy as np
import datetime
import pyqtgraph as pg

from anotator import Annotator
from anotator.errors import WrongSignalGroupError, ArtefactOutOfRange
from anotator.helpers import format_timedelta, ts_to_tick, ts_to_full_date
from anotator.config import (EXPLORER_UNZOOMED_WINDOW_S,
                             EXPLORER_UNZOOMED_TICK_INTERVAL_S,
                             EXPLORER_ZOOMED_WINDOW_S,
                             EXPLORER_ZOOMED_TICK_INTERVAL_S,
                             EXPLORER_RESOLUTION)

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGridLayout,
    QPushButton,
    QWidget,
    QLabel,
    QSlider,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QDialog,
    QHBoxLayout
)

class ExplorerTab(QWidget):

    def __init__(self, annotator : Annotator, window : MainWindow, *args, **kwargs) -> None:
        QWidget.__init__(self, *args, **kwargs)

        self.mainwindow : MainWindow = window
        self.annotator : Annotator = annotator
        self.window_size_unzoomed_s = EXPLORER_UNZOOMED_WINDOW_S
        self.tick_interval_unzoomed_s = EXPLORER_UNZOOMED_TICK_INTERVAL_S
        self.window_size_zoomed_s = EXPLORER_ZOOMED_WINDOW_S
        self.tick_interval_zoomed_s = EXPLORER_ZOOMED_TICK_INTERVAL_S
        self.res : int = EXPLORER_RESOLUTION

        # Base layout
        layout = QGridLayout()
        self.setLayout(layout)

        # Time range label
        time_range_label = QLabel()
        time_range_label.setText("Časové období: NA - NA")
        layout.addWidget(time_range_label, 0, 0)
        self.time_range_label = time_range_label

        # ABP unzoomed chart
        abp_unzoomed_container = QWidget()
        abp_unzoomed_layout = QVBoxLayout()
        abp_unzoomed_container.setLayout(abp_unzoomed_layout)

        abp_unzoomed_label = QLabel()
        abp_unzoomed_label.setText("ABP 5m")
        abp_unzoomed_layout.addWidget(abp_unzoomed_label)

        abp_unzoomed = pg.PlotWidget()
        abp_unzoomed.getAxis("bottom").setHeight(30)
        abp_unzoomed.setMouseEnabled(x=False, y=False)
        abp_unzoomed_layout.addWidget(abp_unzoomed)
        self.abp_unzoomed = abp_unzoomed

        layout.addWidget(abp_unzoomed_container, 1, 0)

        # ICP unzoomed chart
        icp_unzoomed_container = QWidget()
        icp_unzoomed_layout = QVBoxLayout()
        icp_unzoomed_container.setLayout(icp_unzoomed_layout)

        icp_unzoomed_label = QLabel()
        icp_unzoomed_label.setText("ICP 5m")
        icp_unzoomed_layout.addWidget(icp_unzoomed_label)

        icp_unzoomed = pg.PlotWidget()
        icp_unzoomed.getAxis("bottom").setHeight(30)
        icp_unzoomed.setMouseEnabled(x=False, y=False)
        icp_unzoomed_layout.addWidget(icp_unzoomed)
        self.icp_unzoomed = icp_unzoomed

        layout.addWidget(icp_unzoomed_container, 2, 0)

        # ABP zoomed chart
        abp_zoomed_container = QWidget()
        abp_zoomed_layout = QVBoxLayout()
        abp_zoomed_container.setLayout(abp_zoomed_layout)

        abp_zoomed_label = QLabel()
        abp_zoomed_label.setText("ABP 1m")
        abp_zoomed_layout.addWidget(abp_zoomed_label)

        abp_zoomed = pg.PlotWidget()
        abp_zoomed.getAxis("bottom").setHeight(30)
        abp_zoomed.setMouseEnabled(x=False, y=False)
        abp_zoomed_layout.addWidget(abp_zoomed)
        self.abp_zoomed = abp_zoomed

        # ICP zoomed chart
        icp_zoomed_container = QWidget()
        icp_zoomed_layout = QVBoxLayout()
        icp_zoomed_container.setLayout(icp_zoomed_layout)

        icp_zoomed_label = QLabel()
        icp_zoomed_label.setText("ICP 1m")
        icp_zoomed_layout.addWidget(icp_zoomed_label)

        icp_zoomed = pg.PlotWidget()
        icp_zoomed.getAxis("bottom").setHeight(30)
        icp_zoomed.setMouseEnabled(x=False, y=False)
        icp_zoomed_layout.addWidget(icp_zoomed)
        self.icp_zoomed = icp_zoomed

        zoomed_container = QWidget()
        zoomed_layout = QHBoxLayout()
        zoomed_container.setLayout(zoomed_layout)

        zoomed_layout.addWidget(abp_zoomed_container)
        zoomed_layout.addWidget(icp_zoomed_container)

        # Remove padding on sides of zoomed charts
        zoomed_container.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(zoomed_container, 3, 0)

        # Annotate button
        button_annotate = QPushButton()
        button_annotate.setText("Začít Anotovat")
        button_annotate.clicked.connect(self.button_annotate_clicked)
        layout.addWidget(button_annotate, 3, 1)

        # Slider label
        label = QLabel()
        label.setText("Časová osa:")
        layout.addWidget(label, 4, 0)

        # Slider
        self.slider = QSlider(Qt.Horizontal) # type: ignore
        self.slider.setMinimum(0)
        #self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.valueChanged.connect(self.slider_changed)
        layout.addWidget(self.slider, 5, 0)
        # Disable slider
        self.slider.setEnabled(False)

    def button_annotate_clicked(self) -> None:
        """Annotate button callback, switches to annotate tab and renders
        selected section"""
        if not self.annotator.is_data():
            msgbox = QMessageBox()
            msgbox.setWindowTitle("Info")
            msgbox.setText("Nejdříve vyberte HDF5 soubor s daty")
            msgbox.exec_()
            return

        value = self.slider.value()
        sr = self.annotator.get_sr()
        window_size_s = self.window_size_zoomed_s
        lowi = value / self.res
        xmin = int(lowi * sr * window_size_s)

        # TODO: catch exception
        self.annotator.index = xmin
        # Switch to annotate tab
        #self.mainwindow.tabwidget.setCurrentIndex(2)
        self.mainwindow.switch_to_annotate_tab()
        self.mainwindow.tab_annotate.render_plots()

    def init_plots(self) -> None:
        """Initializes slides range and renders plots"""
        (data, _) = self.annotator.get_data()
        sr = self.annotator.get_sr()

        # Update slider
        window_size_s = self.window_size_zoomed_s
        maxi = (len(data) // (sr * window_size_s))
        slidermax = maxi * self.res
        self.slider.setMaximum(slidermax)
        self.slider.setValue(0)

        # Time range labels
        start_dt = ts_to_full_date(self.annotator.get_start_time_s())
        end_dt = ts_to_full_date(self.annotator.get_end_time_s())
        self.time_range_label.setText(f"Časové období: {start_dt} - {end_dt}")

        self.render_plots()

    def slider_changed(self) -> None:
        """Slider callback, moves PlotWidgets to currently selected section"""
        if not self.annotator.is_data():
            return

        #value = self.slider.value()
        #self.render_plots(value)
        self.render_plots()


    #def render_plots(self, index=0) -> None:
    def render_plots(self) -> None:
        if not self.annotator.is_data():
            return

        index = self.slider.value()
        sr = self.annotator.get_sr()
        (data_icp, data_abp) = self.annotator.get_data()

        # Populate main plotwidget
        window_size_s = self.window_size_zoomed_s
        tick_interval_s = self.tick_interval_zoomed_s

        # Get base index from slider
        lowi = index / self.res
        mwindow = sr * window_size_s
        mxmin = int(lowi * mwindow)
        mxmax = int((lowi + 1) * mwindow)

        # ROI
        annotator_window_s = self.annotator.get_window_s()
        rbase = mxmin
        rend = rbase + annotator_window_s * sr

        # Zoomed charts
        self.render_plot(self.icp_zoomed, window_size_s, tick_interval_s, mwindow,
                         mxmin, mxmax, rbase, rend, data_icp, is_icp=True)
        self.render_plot(self.abp_zoomed, window_size_s, tick_interval_s, mwindow,
                         mxmin, mxmax, rbase, rend, data_abp, is_icp=False)


        # Unzoomed charts
        rbase = mxmin
        rend = mxmax
        window_size_s = self.window_size_unzoomed_s
        tick_interval_s = self.tick_interval_unzoomed_s
        self.render_plot(self.icp_unzoomed, window_size_s, tick_interval_s, mwindow,
                         mxmin, mxmax, rbase, rend, data_icp, is_icp=True)
        self.render_plot(self.abp_unzoomed, window_size_s, tick_interval_s, mwindow,
                         mxmin, mxmax, rbase, rend, data_abp, is_icp=False)

    def render_plot(self, pw : pg.PlotWidget,
                    window_size_s : int, tick_interval_s : int,  mwindow : int,
                    mxmin : int, mxmax : int, rbase : int, rend : int,
                    data : np.ndarray, is_icp: bool) -> tuple[int, int]:
            """Moves the PlotWidget to selected section"""

            # Compute delta window between main or previous
            # and split the delta window on each side
            sr = self.annotator.get_sr()
            datalen = len(data)
            #(data_icp, data_abp) = self.annotator.get_data()

            window = sr * window_size_s
            delta_window = window - mwindow
            xmin = mxmin - (delta_window // 2)
            if xmin < 0:
                xmin = 0
            xmax = mxmax + (delta_window // 2)
            if xmax > datalen:
                xmax = datalen

            # Data to be plotted
            x = np.arange(xmin, xmax)
            y = data[xmin:xmax]

            # Plot data
            pw.clear()
            pw.plot(x, y)
            ax = pw.getAxis("bottom")
            ticksi = range(xmin, xmax + 1, sr * tick_interval_s)
            start_ts = self.annotator.get_start_time_s()
            ticks = [(i, f"{ts_to_tick(start_ts + i/sr)}") for i in ticksi]
            ax.setTicks([ticks])

            # Add span region
            region = pg.LinearRegionItem(values=(rbase, rend),
                                              orientation="vertical",
                                              movable=False)
            pw.addItem(region)

            # Padding
            pw.setXRange(xmin, xmax, padding=0.05) # type: ignore

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

            return xmin, xmax

    def show_msg(self, msg: str, msgtype: str="Info") -> None:
        """Shows popup message"""
        msgbox = QMessageBox()
        msgbox.setWindowTitle(msgtype)
        msgbox.setText(msg)
        msgbox.exec_()
