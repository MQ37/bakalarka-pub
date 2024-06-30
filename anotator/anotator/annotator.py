import numpy as np
import math
import os


from typing import Union, Callable
from anotator.errors import (InvalidIndexError, EmptyDataError, 
                             ArtefactOutOfRange, WrongARTFInfoHDF5File,
                               WrongARTFInfoUserID)
from anotator.exporters import ARTFExporter, ARTFMetadata
from anotator.artefact import Artefact
from anotator.config import WINDOW_S, WINDOWS_AROUND
from anotator.readers import HDFReader, ARTFReader

from PySide6.QtCore import QThread, Signal

class FileReaderThread(QThread):
    signal = Signal()

    def __init__(self, pipe):
        super().__init__()
        self.pipe = pipe

    def set_filename(self, filename):
        self.filename = filename

    def run(self):
        reader = HDFReader(self.filename)
        icp_data, abp_data, sr, times, abp_name = reader.read()
        self.pipe[0] = icp_data
        self.pipe[1] = abp_data
        self.pipe[2] = sr
        self.pipe[3] = times
        self.pipe[4] = abp_name

        # Send signal
        self.signal.emit()

class Annotator:

    def __init__(self) -> None:
        self.hdf5_filename: Union[str, None] = None
        self.artf_filename: Union[str, None] = None
        self.user_id: Union[str, None] = None
        # Pipe for reader
        self.pipe = [None, None, None, (None, None), None]
        # Reader callback
        self.read_callback = None
        self.reader = FileReaderThread(self.pipe)
        # ICP data
        self.data_icp : Union[np.ndarray, None] = None
        # ABP data
        self.data_abp : Union[np.ndarray, None] = None
        # Data sampling rate
        self.sr : Union[int, None] = None
        # Data start time microsec
        self.start_time_microsec : Union[int, None] = None
        # Data end time microsec
        self.end_time_microsec : Union[int, None] = None
        # ABP dataset name in HDF5 file
        self.abp_name: Union[str, None] = None

        # Annotator current position
        self._index : int = 0
        # Window size in seconds
        self.window_s : int = WINDOW_S
        # Windows around current section
        self.window_around : int = WINDOWS_AROUND
        # Marked sections as artefact
        # Global
        self.artefacts : set[int] = set()
        # ICP
        self.icp_artefacts: set[int] = set()
        # ABP
        self.abp_artefacts: set[int] = set()

        self.artefact_types: dict[int, int] = {}

    def next(self) -> None:
        if not self.is_data() or not self.sr:
            raise EmptyDataError("No data loaded")
        self.index += self.sr * self.window_s

    def prev(self) -> None:
        if not self.is_data() or not self.sr:
            raise EmptyDataError("No data loaded")
        self.index -= self.sr * self.window_s

    def reset_artefacts(self) -> None:
        """Removes all artefacts."""
        self.artefacts = set()
        self.icp_artefacts = set()
        self.abp_artefacts = set()

    def reset_artf_file(self) -> None:
        """Resets artefact file."""
        self.artf_filename = None
        self.reset_artefacts()

    def is_data(self) -> bool:
        return self.data_icp is not None and self.data_abp is not None

    def mark_artefact(self, index : Union[int, None] = None,
                      icp: bool = False, abp: bool = False, art_type: int = 0) -> None:
        """Marks section as artefact, defaults to global artefact."""
        if not self.is_data():
            raise EmptyDataError("No data loaded")
        if index is None:
            index = self.index

        # TODO: if marked both ICP and ABP individually move to global?
        # ICP
        if icp:
            self.icp_artefacts.add(index)
        # ABP
        elif abp:
            self.abp_artefacts.add(index)
        # Global
        else:
            # Remove from ICP and ABP if present
            if index in self.icp_artefacts:
                self.icp_artefacts.remove(index)
            if index in self.abp_artefacts:
                self.abp_artefacts.remove(index)
            self.artefacts.add(index)

        self.artefact_types[index] = art_type

    def unmark_artefact(self, index : Union[int, None] = None,
                      icp: bool = False, abp: bool = False) -> None:
        """Unmarks section as artefact, defaults to global artefact."""
        if not self.is_data():
            raise EmptyDataError("No data loaded")
        if index is None:
            index = self.index

        # ICP
        if icp:
            if index in self.icp_artefacts:
                self.icp_artefacts.remove(index)
            # Inverse unmark for global mark
            if index in self.artefacts:
                self.artefacts.remove(index)
                self.abp_artefacts.add(index)
        # ABP
        elif abp:
            if index in self.abp_artefacts:
                self.abp_artefacts.remove(index)
            # Inverse unmark for global mark
            if index in self.artefacts:
                self.artefacts.remove(index)
                self.icp_artefacts.add(index)
        # Global
        else:
            if index in self.icp_artefacts:
                self.icp_artefacts.remove(index)
            if index in self.abp_artefacts:
                self.abp_artefacts.remove(index)
            if index in self.artefacts:
                self.artefacts.remove(index)

        # Remove artefact type
        if index in self.artefact_types:
            del self.artefact_types[index]

    def is_artefact(self, index : Union[int, None] = None,
                      icp: bool = False, abp: bool = False) -> bool:
        """Returns if the selected region based on index is marked as
        artefact.
        icp and abp -> icp, abp and global artefacts (all artefacts)
        icp -> icp artefacts only
        abp -> abp artefacts only
        none -> only global artefacts
        """
        if not self.is_data():
            raise EmptyDataError("No data loaded")
        if index is None:
            index = self.index

        if icp and abp:
            return index in self.artefacts.union(self.icp_artefacts)\
                                            .union(self.abp_artefacts)
        elif icp:
            return index in self.icp_artefacts.union(self.artefacts)
        elif abp:
            return index in self.abp_artefacts.union(self.artefacts)
        else:
            return index in self.artefacts

    ### Getters ###

    @property
    def index(self) -> int:
        return self._index

    def get_artefacts_in_region(self, xmin: int, xmax: int,
                                icp: bool = False, abp: bool = False) -> list[int]:
        """Returns artefact bases in region"""
        if not self.is_data() or not self.window_s or not self.sr:
            raise EmptyDataError("No data loaded")

        if icp:
            self_artefacts = self.icp_artefacts.union(self.artefacts)
        elif abp:
            self_artefacts = self.abp_artefacts.union(self.artefacts)
        else:
            self_artefacts = self.artefacts

        artefacts = []
        for idx in range(xmin, xmax+1, self.window_s * self.sr):
            if idx in self_artefacts:
                artefacts.append(idx)

        return artefacts

    def get_sr(self) -> int:
        """Returns data sampling rate."""
        if not self.is_data() or not self.sr:
            raise EmptyDataError("No data loaded")
        return self.sr

    def get_window(self) -> tuple[int, int, tuple[np.ndarray, np.ndarray]]:
        """Returns base and end index of ROI window and tuple (ICP, ABP)"""
        if not self.is_data() or not self.window_s or not self.sr\
                or self.data_icp is None or self.data_abp is None:
            raise EmptyDataError("No data loaded")

        # Base index
        if self.index >= self.sr * self.window_s:
            base = self.index - self.sr * self.window_s * self.window_around
        else:
            base = self.index

        # End index
        if self.index <= len(self.data_icp) - (self.window_around+1) * self.sr * self.window_s:
            end = self.index + ((self.window_around+1) * self.sr * self.window_s)
        else:
            end = self.index + (self.sr * self.window_s)

        return base, end, (self.data_icp[base:end], self.data_abp[base:end])

    def get_roi(self) -> tuple[int, int]:
        """Returns current section that is being annotated."""
        if not self.is_data() or not self.window_s or not self.sr:
            raise EmptyDataError("No data loaded")
        return self.index, self.index + self.sr * self.window_s

    def get_window_s(self) -> int:
        return self.window_s

    def get_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns tuple with (ICP, ABP) data"""
        if not self.is_data() or self.data_icp is None or self.data_abp is None:
            raise EmptyDataError("No data loaded")
        return self.data_icp, self.data_abp

    ### Setters ###

    @index.setter
    def index(self, value : int) -> None:
        if not self.is_data() or not self.sr or self.data_icp is None:
            raise EmptyDataError("No data loaded")
        if value > len(self.data_icp) - self.sr * self.window_s or value < 0:
            raise InvalidIndexError(
                f"Invalid index {value} when data length is {len(self.data_icp)} and window size {self.window_s * self.sr}"
            )
        self._index = value

    def set_user_id(self, user_id : str) -> None:
        """Sets user id"""
        self.user_id = user_id

    ### File ops ###

    def export(self, filename) -> None:
        """Exports artefacts into .artf file format"""
        if not self.is_data() or not self.sr or not self.start_time_microsec:
            raise EmptyDataError("No data loaded")
        if self.abp_name is None:
            raise Exception("ABP name not set")

        exporter = ARTFExporter(filename)
        start_time_s = int(self.start_time_microsec / 1000) / 1000

        global_artefacts = []
        for idx in sorted(list(self.artefacts)):
            artefact = Artefact.from_index(idx, self.sr, self.window_s, start_time_s)
            global_artefacts.append(artefact)

        icp_artefacts = []
        for idx in sorted(list(self.icp_artefacts)):
            artefact = Artefact.from_index(idx, self.sr, self.window_s, start_time_s)
            icp_artefacts.append(artefact)

        abp_artefacts = []
        for idx in sorted(list(self.abp_artefacts)):
            artefact = Artefact.from_index(idx, self.sr, self.window_s, start_time_s)
            abp_artefacts.append(artefact)

        # Create metadata
        metadata = ARTFMetadata(
            hdf5_filename=self.get_hdf5_filename(),
            user_id=self.user_id,
            )

        exporter.export(global_artefacts, icp_artefacts, abp_artefacts,
                        abp_name=self.abp_name,
                        metadata=metadata)


    def load_artf(self, filepath, show_msg: Callable[[str, str], None]) -> None:
        """Loads artefacts from ARTF file"""
        if not self.is_data() or not self.start_time_microsec:
            raise EmptyDataError("No data loaded")
        if self.abp_name is None:
            raise Exception("ABP name not set")

        # to make linter happy
        if not self.start_time_microsec or not self.end_time_microsec:
            raise Exception("No start or end time set")

        reader = ARTFReader(filepath)
        global_artefacts, icp_artefacts, abp_artefacts, metadata = reader.read(abp_name=self.abp_name)

        if metadata:
            if not metadata.hdf5_filename:
                show_msg("Nahrál jste ARTF soubor bez záznamu o zdrojovém HDF5 souboru", "Info")
            elif metadata.hdf5_filename != self.get_hdf5_filename():
                raise WrongARTFInfoHDF5File

            if not metadata.user_id:
                show_msg("Nahrál jste ARTF soubor neznámého uživatele", "Info")
            elif metadata.user_id != self.user_id:
                show_msg(f"Nahrál jste ARTF soubor jiného uživatele - {metadata.user_id}", "Info")
        else:
            show_msg("Nahrál jste ARTF soubor bez metadat", "Info")

        filename = os.path.basename(filepath)
        self.artf_filename = filename

        start_time_s = int(self.start_time_microsec / 1000) / 1000
        end_time_s = int(self.end_time_microsec / 1000) / 1000

        # Remove old artefacts
        self.reset_artefacts()

        for artefact in global_artefacts:
            if artefact.start_time.timestamp() < start_time_s:
                raise ArtefactOutOfRange(f"Data starts at {start_time_s} but artefacts starts at {artefact.start_time.timestamp()}")
            if artefact.end_time.timestamp() > end_time_s:
                raise ArtefactOutOfRange(f"Data ends at {end_time_s} but artefacts ends at {artefact.end_time.timestamp()}")
            delta_s = artefact.end_time.timestamp() - artefact.start_time.timestamp()

            rel_start_time_s = artefact.start_time.timestamp() - start_time_s
            # Align start to window size
            rel_start_time_s -= rel_start_time_s % self.window_s

            #rel_end_time_s = rel_start_time_s + delta_s

            for i in range(math.ceil(delta_s / self.window_s)):
                base = int( (rel_start_time_s + (i * self.window_s)) * self.sr )
                self.artefacts.add(base)

        for artefact in icp_artefacts:
            if artefact.start_time.timestamp() < start_time_s:
                raise ArtefactOutOfRange(f"Data starts at {start_time_s} but artefacts starts at {artefact.start_time.timestamp()}")
            if artefact.end_time.timestamp() > end_time_s:
                raise ArtefactOutOfRange(f"Data ends at {end_time_s} but artefacts ends at {artefact.end_time.timestamp()}")
            delta_s = artefact.end_time.timestamp() - artefact.start_time.timestamp()

            rel_start_time_s = artefact.start_time.timestamp() - start_time_s
            # Align start to window size
            rel_start_time_s -= rel_start_time_s % self.window_s

            #rel_end_time_s = rel_start_time_s + delta_s

            for i in range(math.ceil(delta_s / self.window_s)):
                base = int( (rel_start_time_s + (i * self.window_s)) * self.sr )
                self.icp_artefacts.add(base)

        for artefact in abp_artefacts:
            if artefact.start_time.timestamp() < start_time_s:
                raise ArtefactOutOfRange(f"Data starts at {start_time_s} but artefacts starts at {artefact.start_time.timestamp()}")
            if artefact.end_time.timestamp() > end_time_s:
                raise ArtefactOutOfRange(f"Data ends at {end_time_s} but artefacts ends at {artefact.end_time.timestamp()}")
            delta_s = artefact.end_time.timestamp() - artefact.start_time.timestamp()

            rel_start_time_s = artefact.start_time.timestamp() - start_time_s
            # Align start to window size
            rel_start_time_s -= rel_start_time_s % self.window_s

            #rel_end_time_s = rel_start_time_s + delta_s

            for i in range(math.ceil(delta_s / self.window_s)):
                base = int( (rel_start_time_s + (i * self.window_s)) * self.sr )
                self.abp_artefacts.add(base)

    def reading_done(self) -> None:
        """Called when reading thread finished reading with data in pipe.
        Reads data from pipe and calls read_callback."""
        self.data_icp = self.pipe[0]
        self.data_abp = self.pipe[1]
        self.sr = self.pipe[2]
        (start_time, end_time) = self.pipe[3]
        self.abp_name = self.pipe[4]

        if self.data_icp is None or self.data_abp is None:
            if self.read_callback is not None:
                self.read_callback(error=True)
            raise Exception("Error reading data")

        # Remove old artefacts
        self.reset_artf_file()

        self.start_time_microsec = start_time
        self.end_time_microsec = end_time

        # Remove nans
        #self.data_icp = self.data_icp[~np.isnan(self.data_icp)]
        #self.data_abp = self.data_abp[~np.isnan(self.data_abp)]

        # Cut data to same length
        minlen = min([len(self.data_icp), len(self.data_abp)])
        self.data_icp = self.data_icp[:minlen]
        self.data_abp = self.data_abp[:minlen]

        assert len(self.data_icp) == len(self.data_abp)

        if self.read_callback is not None:
            self.read_callback()

    def read_data(self, filepath : str, callback : Union[Callable, None] = None) -> None:
        """Reads data in thread and calls callback when finished"""
        filename = os.path.basename(filepath)
        self.hdf5_filename = filename
        if callback is not None:
            self.read_callback = callback
        self.reader.set_filename(filepath)
        self.reader.signal.connect(self.reading_done)
        self.reader.start()

    def get_start_time(self) -> int:
        """Returns data start time in microseconds"""
        if not self.is_data() or not self.start_time_microsec:
            raise EmptyDataError("No data loaded")
        return self.start_time_microsec

    def get_end_time(self) -> int:
        """Returns data end time in microseconds"""
        if not self.is_data() or not self.end_time_microsec:
            raise EmptyDataError("No data loaded")
        return self.end_time_microsec

    def get_data_len(self) -> int:
        """Returns length of data in samples"""
        if not self.is_data():
            raise EmptyDataError("No data loaded")
        # Make LSP shut up
        if self.data_icp is None:
            raise EmptyDataError("No data loaded")
        return len(self.data_icp)

    def get_hdf5_filename(self) -> str:
        """Returns hdf5 filename"""
        if not self.is_data():
            raise EmptyDataError("No data loaded")
        if self.hdf5_filename is None:
            raise EmptyDataError("No hdf5 filename")
        return self.hdf5_filename

    def get_start_time_s(self) -> float:
        """Returns data start time in seconds"""
        if not self.is_data():
            raise EmptyDataError("No data loaded")
        if self.start_time_microsec is None:
            raise EmptyDataError("No start time")

        return self.start_time_microsec / 1e6

    def get_end_time_s(self) -> float:
        """Returns data end time in seconds"""
        if not self.is_data():
            raise EmptyDataError("No data loaded")
        if self.end_time_microsec is None:
            raise EmptyDataError("No end time")

        return self.end_time_microsec / 1e6

    def get_artf_filename(self) -> str:
        """Returns artefact filename"""
        if not self.is_data():
            raise EmptyDataError("No data loaded")
        if self.artf_filename is None:
            raise Exception("No artefact filename")
        return self.artf_filename

    def get_artefact_type(self, index=None) -> int:
        """Returns artefact type at index"""
        if not self.is_data():
            raise EmptyDataError("No data loaded")

        if index is None:
            index = self.index

        if index not in self.artefact_types:
            raise Exception("No artefact type at index")

        return self.artefact_types[index]
