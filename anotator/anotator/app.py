import sys
from PySide6 import QtWidgets
from anotator import MainWindow


def run() -> None:
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
