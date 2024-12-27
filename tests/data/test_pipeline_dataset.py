from unittest import TestCase
from dataclasses import dataclass

from epochlib.data import Data, DataRetrieval, PipelineDataset
from epochlib.training import TrainingBlock
import numpy as np
import numpy.typing as npt
from typing import Any


class TestDataRetrieval(DataRetrieval):
    BASE = 2**0
    FIRST = 2**1


@dataclass
class CustomData(Data):
    data1: npt.NDArray[np.int_] | None = None
    data2: npt.NDArray[np.int_] | None = None

    def __post_init__(self) -> None:
        self.retrieval = TestDataRetrieval.BASE

    def __getitem__(self, idx: int | npt.NDArray[np.int_] | list[int] | slice) -> npt.NDArray[Any] | list[Any]:
        """Get item from the data.

        :param idx: Index to retrieve
        :return: Relevant data
        """
        if not isinstance(idx, (int | np.integer)):
            return self.__getitems__(idx)  # type: ignore[arg-type]

        result = []
        if self.retrieval & TestDataRetrieval.BASE:
            result.append(self.data1[idx])
        if self.retrieval & TestDataRetrieval.FIRST:
            result.append(self.data2[idx])

        if len(result) == 1:
            return result[0]

        return result

    def __getitems__(self, indices: npt.NDArray[np.int_] | list[int] | slice) -> npt.NDArray[Any]:
        """Retrieve items for all indices based on specified retrieval flags.

        :param indices: List of indices to retrieve
        :return: Relevant data
        """
        result = []
        if self.retrieval & TestDataRetrieval.BASE:
            result.append(self.data1[indices])
        if self.retrieval & TestDataRetrieval.FIRST:
            result.append(self.data2[indices])

        if len(result) == 1:
            return result[0]

        return result

    def __len__(self) -> int:
        if self.data1 is not None:
            return len(self.data1)
        if self.data2 is not None:
            return len(self.data2)
        return 0


class TestTrainingBlockNoAug(TrainingBlock):

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.str_],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.uint8]]:
        """Randomize the SMILES string."""
        return x, y

    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return False


class TestTrainingBlockWithAug(TrainingBlock):

    def train(
        self,
        x: npt.NDArray[np.str_],
        y: npt.NDArray[np.str_],
        **train_args: Any,
    ) -> tuple[npt.NDArray[np.str_], npt.NDArray[np.uint8]]:
        """Randomize the SMILES string."""
        return x, y

    @property
    def is_augmentation(self) -> bool:
        """Check if augmentation is enabled."""
        return True


class TestPipelineDataset(TestCase):

    def test_initialization_errors(self) -> None:
        with self.assertRaises(ValueError):
            PipelineDataset()
        with self.assertRaises(ValueError):
            PipelineDataset(retrieval=['BASE'])

    def test_initialization_steps(self) -> None:
        pd = PipelineDataset(retrieval=['BASE'], retrieval_type=TestDataRetrieval)
        step1 = TestTrainingBlockNoAug()
        pd_with_step = PipelineDataset(retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step1])
        self.assertEqual(pd_with_step._enabled_steps, [step1])

        step_with_aug = TestTrainingBlockWithAug()
        pd_aug = PipelineDataset(retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step_with_aug])
        self.assertEqual(pd_aug._enabled_steps, [])

        pd_aug.setup_pipeline(use_augmentations=True)
        self.assertEqual(pd_aug._enabled_steps, [step_with_aug])

    def test_get_item(self) -> None:
        test_data = CustomData()
        test_data.data1 = [0, 1]
        step = TestTrainingBlockNoAug()

        pd_with_data = PipelineDataset(
            retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step], x=test_data
        )
        self.assertEqual(pd_with_data[0][0], 0)

        pd_with_indices = PipelineDataset(
            retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step], x=test_data, indices=[1]
        )
        self.assertEqual(pd_with_indices[0][0], 1)

        pd_no_data = PipelineDataset(retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step])
        with self.assertRaises(ValueError):
            pd_no_data[0]

    def test_get_items(self) -> None:
        test_data = CustomData()
        test_data.data1 = np.array([0, 1, 2])
        step = TestTrainingBlockNoAug()

        pd_with_data = PipelineDataset(
            retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step], x=test_data
        )
        self.assertTrue((pd_with_data[[0, 1]][0] == [0, 1]).all())

        pd_with_indices = PipelineDataset(
            retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step], x=test_data, indices=np.array([0, 2])
        )
        self.assertTrue((pd_with_indices[[0, 1]][0] == [0, 2]).all())

        pd_no_data = PipelineDataset(retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step])
        with self.assertRaises(ValueError):
            pd_no_data[[0, 1]]

    def test_len(self) -> None:
        test_data = CustomData()
        test_data.data1 = np.array([0, 1, 2])
        step = TestTrainingBlockNoAug()

        pd_with_data = PipelineDataset(
            retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step], x=test_data
        )
        self.assertTrue(len(pd_with_data) == 3)

        pd_with_data.initialize(x=test_data, y=test_data, indices=np.array([0, 2]))
        # pd_with_indices = PipelineDataset(
        #     retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step], x=test_data, indices=np.array([0,2])
        # )
        self.assertTrue(len(pd_with_data) == 2)

        pd_no_data = PipelineDataset(retrieval=['BASE'], retrieval_type=TestDataRetrieval, steps=[step])
        with self.assertRaises(ValueError):
            self.assertTrue(len(pd_no_data))
