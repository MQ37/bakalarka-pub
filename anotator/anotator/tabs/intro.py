
from anotator import Annotator
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from anotator import MainWindow
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QMessageBox,
    QDialog,
)
from PySide6.QtCore import Qt
from anotator.errors import (WrongSignalGroupError, ArtefactOutOfRange,
                             WrongARTFInfoUserID, WrongARTFInfoHDF5File)




class IntroTab(QWidget):

    def __init__(self, annotator : Annotator, mainwindow : "MainWindow", *args, **kwargs) -> None:
        QWidget.__init__(self, *args, **kwargs)

        self.annotator = annotator
        self.mainwindow = mainwindow

        layout = QVBoxLayout()
        self.setLayout(layout)

        user_id_container = QWidget()
        user_id_layout = QHBoxLayout()
        user_id_container.setLayout(user_id_layout)

        self.user_id_label = QLabel("ID uživatele:")

        self.user_id_input = QLineEdit()
        self.user_id_input.textEdited.connect(self.user_id_input_changed)

        user_id_layout.addWidget(self.user_id_label)
        user_id_layout.addWidget(self.user_id_input)

        layout.addWidget(user_id_container)

        file_info_container = QWidget()
        file_info_layout = QHBoxLayout()
        file_info_container.setLayout(file_info_layout)

        self.hdf_file_label = QLabel("HDF5 soubor: N/A")
        file_info_layout.addWidget(self.hdf_file_label)

        self.artf_file_label = QLabel("ARTF soubor: N/A")
        file_info_layout.addWidget(self.artf_file_label)

        layout.addWidget(file_info_container)

        # Load file HDF5 button
        self.button_file_hdf = QPushButton()
        self.button_file_hdf.setText("Vyber HDF5")
        self.button_file_hdf.clicked.connect(self.button_file_hdf_clicked)
        layout.addWidget(self.button_file_hdf)

        # Load file ARTF button
        self.button_file_artf = QPushButton()
        self.button_file_artf.setText("Vyber ARTF")
        self.button_file_artf.clicked.connect(self.button_file_artf_clicked)
        layout.addWidget(self.button_file_artf)

        # Unload ARTF button
        self.button_unload_artf = QPushButton()
        self.button_unload_artf.setText("Odebrat ARTF (a všechny anotace)")
        self.button_unload_artf.clicked.connect(self.button_unload_artf_clicked)
        layout.addWidget(self.button_unload_artf)

        # Begin button
        self.button_begin = QPushButton()
        self.button_begin.setText("Začít")
        self.button_begin.clicked.connect(self.button_begin_clicked)
        layout.addWidget(self.button_begin)

    def button_file_artf_clicked(self) -> None:
        """Load file button callback, opens file dialog to load file,
        and loads artefacts from ARTF file"""
        if not self.annotator.is_data():
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty", msgtype="Info")
            return
        (filename, _) = QFileDialog.getOpenFileName(
            self,
            caption="Otevřít soubor",
            filter="ARTF (*.artf)")

        # No file selected
        if filename == "":
            return

        try:
            self.annotator.load_artf(filename, show_msg=self.show_msg)
        except WrongSignalGroupError:
            self.show_msg("ARTF soubor obsahuje neočekáváné hodnoty, nejspíše načítáte ARTF obsahující ART pro HDF5 soubor očekávající ABP nebo naopak", msgtype="Chyba")
        except ArtefactOutOfRange:
            self.show_msg("ARTF soubor obsahuje artefakty mimo časové období tohoto HDF5 souboru", msgtype="Chyba")
        except ValueError as e:
            self.show_msg(f"ARTF soubor obsahuje neočekávané hodnoty: {e}", msgtype="Chyba")
        except WrongARTFInfoHDF5File:
            self.show_msg("ARTF soubor neodpovídá vybranému HDF5 souboru", msgtype="Chyba")
        else:
            # ARTF file label
            artf_filename = self.annotator.get_artf_filename()
            self.artf_file_label.setText(f"ARTF soubor: {artf_filename}")
            self.mainwindow.tab_annotate.render_plots()

    def show_msg(self, msg: str, msgtype: str="Info") -> None:
        """Shows popup message"""
        msgbox = QMessageBox()
        msgbox.setWindowTitle(msgtype)
        msgbox.setText(msg)
        msgbox.exec_()

    def button_file_hdf_clicked(self) -> None:
        """Load file button callback, opens file dialog to load file,
        initialize PlotWidgets and render data"""
        (filename, _) = QFileDialog.getOpenFileName(
            self,
            caption="Otevřít soubor",
            filter="HDF5 (*.hdf5)")
            #filter="NumPy (*.npy)")
        # No file selected
        if filename == "":
            return

        # Read data in separate thread
        self.annotator.read_data(filename, self.load_finished)

        # Disable window and show popup
        self.setEnabled(False)

        self.popup = QDialog(self)
        self.popup.setWindowTitle("Čtení souboru")
        self.popup.setFixedSize(300, 100)
        layout = QVBoxLayout()
        self.popup.setLayout(layout)
        self.popup.setModal(True)
        progressLabel = QLabel("Čtení souboru...")
        progressLabel.setAlignment(Qt.AlignCenter) # type: ignore
        layout.addWidget(progressLabel)
        self.popup.exec_()

    def load_finished(self, error=False) -> None:
        self.setEnabled(True)
        self.popup.close()

        if error:
            self.show_msg("Chyba čtení dat")
        else:
            # Enable slider
            #self.slider.setEnabled(True)
            self.mainwindow.tab_explorer_init_plots()
            self.mainwindow.tab_annotate.render_plots()
            self.hdf_file_label.setText(f"HDF5 soubor: {self.annotator.get_hdf5_filename()}")
            self.reset_artf_file()

    def reset_artf_file(self) -> None:
        self.artf_file_label.setText("ARTF soubor: N/A")

    def user_id_input_changed(self, text: str) -> None:
        """Callback for user ID input"""
        self.annotator.set_user_id(text)

    def button_begin_clicked(self) -> None:
        if not self.annotator.is_data():
            self.show_msg("Nejdříve vyberte HDF5 soubor s daty", msgtype="Info")
            return
        self.mainwindow.switch_to_explorer_tab()

    def button_unload_artf_clicked(self) -> None:
        self.annotator.reset_artf_file()
        self.reset_artf_file()

