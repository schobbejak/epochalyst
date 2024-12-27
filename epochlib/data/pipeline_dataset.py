"""Module that contains a dataset that can take a training pipeline."""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Tuple, TypeVar

import numpy as np
import numpy.typing as npt

from epochlib.data.enum_data_format import Data, DataRetrieval
from epochlib.training.training import TrainingPipeline
from epochlib.training.training_block import TrainingBlock

try:
    from torch.utils.data import Dataset
except ImportError:
    """User doesn't require torch"""

T = TypeVar("T")
DataTuple = Tuple[T, T]


@dataclass
class PipelineDataset(Dataset[Tuple[T, T]]):
    """Pipeline dataset takes in a pipeline to be able to process the original data.

    Useful for lazy loading data for training, where computing would take a long time or the device would run out of memory.

    :param retrieval: Data retrieval object.
    :param retrieval_type: Data retrieval enum
    :param steps: Steps to apply to the dataset
    :param result_formatter:

    :param x: Input data
    :param y: Labels
    :param indices: Indices to use

    """

    x: Data | None = None
    y: Data | None = None
    indices: npt.NDArray[np.int_] | None = None

    retrieval: list[str] | None = None
    retrieval_type: DataRetrieval | None = None
    steps: list[TrainingBlock] | None = None
    result_formatter: Callable[[Any], Any] = lambda a: a

    def __post_init__(self) -> None:
        """Set up the dataset."""
        if self.retrieval is None:
            raise ValueError("Retrieval object must be set.")
        if self.retrieval_type is None:
            raise ValueError("Retrieval type must be set.")

        # Setup data retrieval
        self._retrieval_enum = getattr(self.retrieval_type, self.retrieval[0])
        for retrieval in self.retrieval[1:]:
            self._retrieval_enum = self._retrieval_enum | getattr(self.retrieval_type, retrieval)

        # Setup pipeline
        self.setup_pipeline(use_augmentations=False)

    def initialize(self, x: Data, y: Data, indices: list[int] | npt.NDArray[np.int_] | None = None) -> None:
        """Set up the dataset for training.

        :param x: X data to initialize with
        :param y: Y data to initialize with
        :param indices: Indices to filter on
        """
        self.x = x
        self.y = y
        self.indices = np.array(indices, dtype=np.int32) if isinstance(indices, list) else indices

    def setup_pipeline(self, *, use_augmentations: bool) -> None:
        """Set whether to use the augmentations.

        :param use_augmentations: Whether to use augmentations while passing data through pipeline
        """
        self._enabled_steps: list[TrainingBlock] = []

        if self.steps is not None:
            for step in self.steps:
                if (step.is_augmentation and use_augmentations) or not step.is_augmentation:
                    self._enabled_steps.append(step)

        self._pipeline = TrainingPipeline(steps=self._enabled_steps)
        logging.getLogger("TrainingPipeline").setLevel(logging.WARNING)

    def __len__(self) -> int:
        """Get the length of the dataset."""
        if self.x is None:
            raise ValueError("Dataset is not initialized.")
        if self.indices is None:
            return len(self.x)
        return len(self.indices)

    def __getitem__(self, idx: int | list[int] | npt.NDArray[np.int_]) -> tuple[Any, Any]:
        """Get an item from the dataset.

        :param idx: Index to retrieve
        :return: Data and labels at the Index
        """
        if not isinstance(idx, (int | np.integer)):
            return self.__getitems__(idx)

        if self.x is None:
            raise ValueError("Dataset not initialized or has no x data.")
        if self.indices is not None:
            idx = self.indices[idx]

        self.x.retrieval = self._retrieval_enum
        x = np.expand_dims(self.x[idx], axis=0)
        y = np.expand_dims(self.y[idx], axis=0) if self.y is not None else None

        x, y = self._pipeline.train(x, y)
        return self.result_formatter(x)[0], self.result_formatter(y)[0] if y is not None else None

    def __getitems__(self, indices: list[int] | npt.NDArray[np.int_]) -> tuple[Any, Any]:
        """Get items from the dataset.

        :param indices: The indices to retrieve
        :return: Data and labels at the indices.
        """
        if self.x is None:
            raise ValueError("Dataset not initialized or has no x data.")
        if self.indices is not None:
            indices = self.indices[indices]

        self.x.retrieval = self._retrieval_enum
        x = self.x[indices]
        y = self.y[indices] if self.y is not None else None

        x, y = self._pipeline.train(x, y)
        return self.result_formatter(x), self.result_formatter(y) if y is not None else None
