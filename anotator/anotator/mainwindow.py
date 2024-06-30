from anotator import Annotator
from anotator.tabs import ExplorerTab, AnnotateTab, DiffAnnotateTab, IntroTab

from PySide6.QtWidgets import (
    QMainWindow,
    QGridLayout,
    QTabWidget,
    QWidget,
)


class MainWindow(QMainWindow):

    def __init__(self) -> None:
        QMainWindow.__init__(self)

        annotator = Annotator()
        self.annotator = annotator

        self.setWindowTitle("Anotator")
        # Initial window size
        self.resize(800, 600)

        widget = QWidget(self)
        self.setCentralWidget(widget)

        layout = QGridLayout()
        widget.setLayout(layout)

        self.tab_intro = IntroTab(annotator, self)
        self.tab_explorer = ExplorerTab(annotator, self)
        self.tab_annotate = AnnotateTab(annotator, self)
        self.tab_diffannotate = DiffAnnotateTab(annotator, self)

        self.tabwidget = QTabWidget()
        #self.tabwidget.tabBar().setStyleSheet("QTabBar { font-size: 24px; }")
        self.tabwidget.addTab(self.tab_intro, "Výběr")
        self.tabwidget.addTab(self.tab_explorer, "Explorace")
        self.tabwidget.addTab(self.tab_annotate, "Anotace")
        self.tabwidget.addTab(self.tab_diffannotate, "Porovnání")
        self.tabwidget.currentChanged.connect(self.handle_tab_change)
        layout.addWidget(self.tabwidget, 0, 0)

    def handle_tab_change(self, index):
        if index == 1:
            self.tab_explorer.render_plots()
            self.tab_explorer.slider.setEnabled(self.annotator.is_data())

    def switch_to_annotate_tab(self):
        self.tabwidget.setCurrentIndex(2)

    def switch_to_explorer_tab(self):
        self.tabwidget.setCurrentIndex(1)

    def tab_explorer_init_plots(self):
        self.tab_explorer.init_plots()

