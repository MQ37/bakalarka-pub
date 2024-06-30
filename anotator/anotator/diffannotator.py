from typing import Dict, Union
from anotator.exporters import ARTFExporter, ARTFMetadata
from anotator.readers import ARTFReader
from anotator.annotator import Annotator
from anotator.artefact import Artefact
import os
import copy


class DiffArtefact:
    # art_type: str (icp or abp)
    # base: int
    # artf_file: str (artf1 or artf2) where the artefact was found (other file has this artefact missing)
    # is_real: bool - was marked as real
    def __init__(self, art_type: str, base: int, artf_file: str, is_real: bool=False):
        self.art_type = art_type
        self.base = base
        self.artf_file = artf_file
        self.is_real = is_real

    def __repr__(self) -> str:
        return f"DiffArtefact({self.art_type}, {self.base}, {self.artf_file}, {self.is_real})"

    def __str__(self) -> str:
        return repr(self)

class DiffAnnotator:

    def __init__(self, annotator) -> None:
        self.annotator: Annotator = annotator
        self.loaded_artf1: bool = False
        self.loaded_artf2: bool = False
        self.artf1_filename: Union[str, None] = None
        self.artf2_filename: Union[str, None] = None
        self.found_diffs: bool = False
        self.artf1: Dict[str, list[Artefact]] = {
                        "icp": [],
                        "abp": []
                    }
        self.artf2: Dict[str, list[Artefact]] = {
                        "icp": [],
                        "abp": []
                    }
        self.out: Dict[str, list[Artefact]] = {
                        "icp": [],
                        "abp": []
                    }
        self.diffs: list[DiffArtefact] = []
        self.current_diff: int = 0

    def reset(self) -> None:
        """Resets the diff annotator to its initial state"""
        self.found_diffs: bool = False
        self.out: Dict[str, list[Artefact]] = {
                        "icp": [],
                        "abp": []
                    }
        self.diffs: list[DiffArtefact] = []
        self.current_diff: int = 0

    def reset_artf1(self) -> None:
        self.artf1: Dict[str, list[Artefact]] = {
                        "icp": [],
                        "abp": []
                    }

    def reset_artf2(self) -> None:
        self.artf2: Dict[str, list[Artefact]] = {
                        "icp": [],
                        "abp": []
                    }

    def load_artf(self, filename: str, artf_file: str) -> None:
        # Reset
        self.reset()

        abp_name = self.annotator.abp_name
        if abp_name is None:
            raise Exception("ABP name not set")
        (_global, icp, abp, metadata) = ARTFReader(filename).read(abp_name=abp_name)
        sr = self.annotator.sr
        window_s = self.annotator.window_s
        # To make LSP shut up
        start_time_s = self.annotator.get_start_time_s()

        artf = self.artf1 if artf_file == "artf1" else self.artf2

        # Convert global artefacts to icp and abp
        for art in _global:
            base = art.to_index(sr, window_s, start_time_s)[0]
            if base < 0 or base >= self.annotator.get_data_len():
                raise Exception("Artefact out of bounds")

            artf["icp"].append(art)
            artf["abp"].append(art)

        for art in icp:
            base = art.to_index(sr, window_s, start_time_s)[0]
            if base < 0 or base >= self.annotator.get_data_len():
                raise Exception("Artefact out of bounds")

            artf["icp"].append(art)

        for art in abp:
            base = art.to_index(sr, window_s, start_time_s)[0]
            if base < 0 or base >= self.annotator.get_data_len():
                raise Exception("Artefact out of bounds")

            artf["abp"].append(art)

    def load_artf1(self, filepath: str) -> None:
        self.reset_artf1()
        artf_file = "artf1"
        self.load_artf(filepath, artf_file)
        self.loaded_artf1 = True
        filename = os.path.basename(filepath)
        self.artf1_filename = filename

    def load_artf2(self, filepath: str) -> None:
        self.reset_artf2()
        artf_file = "artf2"
        self.load_artf(filepath, artf_file)
        self.loaded_artf2 = True
        filename = os.path.basename(filepath)
        self.artf2_filename = filename

    def are_artfs_loaded(self) -> bool:
        return self.loaded_artf1 and self.loaded_artf2

    def find_diffs(self) -> int:
        if not self.loaded_artf1 or not self.loaded_artf2:
            raise Exception("Artefacts not loaded")

        if self.found_diffs:
            return len(self.diffs)

        for artf_type in self.artf1.keys():
            # Find missing artefacts in artf2
            for artf1 in self.artf1[artf_type]:
                found = False
                for artf2 in self.artf2[artf_type]:
                    if artf1== artf2:
                        found = True
                        break
                if not found:
                    base = artf1.to_index(self.annotator.sr, self.annotator.window_s, self.annotator.get_start_time_s())[0]
                    diffart = DiffArtefact(artf_type, base, "artf1", False)
                    self.diffs.append(diffart)
                # Add to output if same
                else:
                    self.out[artf_type].append(artf1)
            # Find missing artefacts in artf1
            for artf2 in self.artf2[artf_type]:
                found = False
                for artf1 in self.artf1[artf_type]:
                    if artf1== artf2:
                        found = True
                        break
                if not found:
                    base = artf2.to_index(self.annotator.sr, self.annotator.window_s, self.annotator.get_start_time_s())[0]
                    diffart = DiffArtefact(artf_type, base, "artf2", False)
                    self.diffs.append(diffart)

        # Sort diffs
        self.diffs.sort(key=lambda x: x.base)

        self.found_diffs = True

        return len(self.diffs)

    def is_diff(self) -> bool:
        return len(self.diffs) > 0

    def get_found_diffs(self) -> bool:
        """Returns whether diffs have been found for current ARTF files"""
        return self.found_diffs

    def next_diff(self) -> DiffArtefact:
        """Returns the next diff"""
        if self.current_diff >= len(self.diffs) - 1:
            raise Exception("No more diffs")
        self.current_diff += 1
        return self.diffs[self.current_diff]

    def prev_diff(self) -> DiffArtefact:
        """Returns the previous diff"""
        if self.current_diff <= 0:
            raise Exception("No previous diffs")
        self.current_diff -= 1
        return self.diffs[self.current_diff]

    def get_diff(self) -> DiffArtefact:
        """Returns the current diff"""
        if self.current_diff < 0 or self.current_diff >= len(self.diffs):
            raise Exception("No diffs")
        return self.diffs[self.current_diff]

    def mark_real(self) -> None:
        """Marks the current diff as real"""
        if not self.loaded_artf1 or not self.loaded_artf2:
            raise Exception("Artefacts not loaded")
        if not self.diffs:
            raise Exception("No diffs")
        self.diffs[self.current_diff].is_real = True

    def unmark_real(self) -> None:
        """Marks the current diff as fake"""
        if not self.loaded_artf1 or not self.loaded_artf2:
            raise Exception("Artefacts not loaded")
        if not self.diffs:
            raise Exception("No diffs")
        self.diffs[self.current_diff].is_real = False

    def get_current_diff_index(self) -> int:
        """Returns the current diff index"""
        return self.current_diff

    def get_diff_count(self) -> int:
        """Returns the number of diffs"""
        return len(self.diffs)

    def get_artf1_filename(self) -> str:
        """Returns the filename of the first artefact file"""
        if self.artf1_filename is None:
            raise Exception("Artefact 1 not loaded")
        return self.artf1_filename

    def get_artf2_filename(self) -> str:
        """Returns the filename of the second artefact file"""
        if self.artf2_filename is None:
            raise Exception("Artefact 2 not loaded")
        return self.artf2_filename

    def export(self, filename: str) -> None:
        """Exports the artefacts to ARTF file"""
        if self.annotator.abp_name is None:
            raise Exception("ABP name not set")

        exporter = ARTFExporter(filename)

        global_artefacts: list[Artefact] = []

        icp_artefacts: list[Artefact] = copy.deepcopy(self.out["icp"])
        abp_artefacts: list[Artefact] = copy.deepcopy(self.out["abp"])

        # Convert diffs to Artefact obj
        real_diffs: list[DiffArtefact] = [diff for diff in self.diffs if diff.is_real]
        for diff in real_diffs:
            art = Artefact.from_index(diff.base, self.annotator.sr, self.annotator.window_s, self.annotator.get_start_time_s())
            if diff.art_type == "icp":
                icp_artefacts.append(art)
            elif diff.art_type == "abp":
                abp_artefacts.append(art)

        # Sort Artefacts
        icp_artefacts.sort(key=lambda x: x.start_time)
        abp_artefacts.sort(key=lambda x: x.start_time)

        # Create metadata
        hdf5_filename = self.annotator.get_hdf5_filename()
        metadata = ARTFMetadata(hdf5_filename=hdf5_filename)

        exporter.export(global_artefacts, icp_artefacts, abp_artefacts,
                        abp_name=self.annotator.abp_name,
                        metadata=metadata)


