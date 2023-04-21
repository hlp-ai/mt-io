"""Base Processor for all processor."""

import abc
import json
import os
from typing import Dict, List, Union

from dataclasses import dataclass, field


@dataclass
class BaseProcessor(abc.ABC):
    data_dir: str

    symbols: List[str] = field(default_factory=list)
    speakers_map: Dict[str, int] = field(default_factory=dict)

    saved_mapper_path: str = None
    loaded_mapper_path: str = None

    items: List[List[str]] = field(default_factory=list)  # text, wav_path, speaker_name

    symbol_to_id: Dict[str, int] = field(default_factory=dict)
    id_to_symbol: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):

        if self.loaded_mapper_path is not None:
            self._load_mapper(loaded_path=self.loaded_mapper_path)
            if self.setup_eos_token():
                self.add_symbol(self.setup_eos_token())  # if this eos token not yet present in symbols list.
                self.eos_id = self.symbol_to_id[self.setup_eos_token()]
            return

        if self.symbols.__len__() < 1:
            raise ValueError("Symbols list is empty but mapper isn't loaded")

        self.create_items()
        self.create_speaker_map()
        self.reverse_speaker = {v: k for k, v in self.speakers_map.items()}
        self.create_symbols()
        if self.saved_mapper_path is not None:
            self._save_mapper(saved_path=self.saved_mapper_path)

        # processor name. usefull to use it for AutoProcessor
        self._processor_name = type(self).__name__

        if self.setup_eos_token():
            self.add_symbol(self.setup_eos_token())  # if this eos token not yet present in symbols list.
            self.eos_id = self.symbol_to_id[self.setup_eos_token()]

    def __getattr__(self, name: str) -> Union[str, int]:
        if "_id" in name:  # map symbol to id
            return self.symbol_to_id[name.replace("_id", "")]
        return self.symbol_to_id[name]  # map symbol to value

    def create_speaker_map(self):
        sp_id = 0
        for i in self.items:
            speaker_name = i[-1]
            if speaker_name not in self.speakers_map:
                self.speakers_map[speaker_name] = sp_id
                sp_id += 1

    def get_speaker_id(self, name: str) -> int:
        return self.speakers_map[name]

    def get_speaker_name(self, speaker_id: int) -> str:
        return self.speakers_map[speaker_id]

    def create_symbols(self):
        self.symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self.id_to_symbol = {i: s for i, s in enumerate(self.symbols)}

    def create_items(self):
        raise NotImplementedError()

    def add_symbol(self, symbol: Union[str, list]):
        if isinstance(symbol, str):
            if symbol in self.symbol_to_id:
                return
            self.symbols.append(symbol)
            symbol_id = len(self.symbol_to_id)
            self.symbol_to_id[symbol] = symbol_id
            self.id_to_symbol[symbol_id] = symbol

        elif isinstance(symbol, list):
            for i in symbol:
                self.add_symbol(i)
        else:
            raise ValueError("A new_symbols must be a string or list of string.")

    @abc.abstractmethod
    def get_one_sample(self, item):
        """Get one sample from dataset items.
        Args:
            item: one item in Dataset items.
                Dataset items may include (raw_text, speaker_id, wav_path, ...)

        Returns:
            sample (dict): sample dictionary return all feature used for preprocessing later.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def text_to_sequence(self, text: str, inference=True):
        raise NotImplementedError()

    @abc.abstractmethod
    def setup_eos_token(self):
        """Return eos symbol of type string."""
        return "eos"

    def _load_mapper(self, loaded_path: str = None):
        loaded_path = (
            os.path.join(self.data_dir, "mapper.json")
            if loaded_path is None
            else loaded_path
        )
        with open(loaded_path, "r") as f:
            data = json.load(f)
        self.speakers_map = data["speakers_map"]
        self.symbol_to_id = data["symbol_to_id"]
        self.id_to_symbol = {int(k): v for k, v in data["id_to_symbol"].items()}
        self._processor_name = data["processor_name"]

        # other keys
        all_data_keys = data.keys()
        for key in all_data_keys:
            if key not in ["speakers_map", "symbol_to_id", "id_to_symbol"]:
                setattr(self, key, data[key])

    def _save_mapper(self, saved_path: str = None, extra_attrs_to_save: dict = None):
        """
        Save all needed mappers to file
        """
        saved_path = (
            os.path.join(self.data_dir, "mapper.json")
            if saved_path is None
            else saved_path
        )
        with open(saved_path, "w") as f:
            full_mapper = {
                "symbol_to_id": self.symbol_to_id,
                "id_to_symbol": self.id_to_symbol,
                "speakers_map": self.speakers_map,
                "processor_name": self._processor_name,
            }
            if extra_attrs_to_save:
                full_mapper = {**full_mapper, **extra_attrs_to_save}
            json.dump(full_mapper, f)

    @abc.abstractmethod
    def save_pretrained(self, saved_path):
        """Save mappers to file"""
        pass
