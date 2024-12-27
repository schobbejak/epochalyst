"""PretrainBlock to implement modules such as scalers."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from joblib import hash

from .training_block import TrainingBlock


@dataclass
class PretrainBlock(TrainingBlock):
    """Pretrain block class.

    Parameters
    ----------
    - test_size : float

    Methods
    -------
    .. code-block:: python
        @abstractmethod
        def pretrain_train(self, x: Any, y: Any, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> tuple[Any, Any]:

        @abstractmethod
        def custom_predict(self, x: Any, **pred_args: Any) -> Any: # Predict pretrain block method.

        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]: # Train pretrain block method.

        def predict(self, x: Any, **pred_args: Any) -> Any: # Predict pretrain block method.

        def train_split_hash(self, train_indices: list[int]) -> str: # Split the hash on train split

    Usage:
    .. code-block:: python
        from epochlib.pipeline.model.training.pretrain_block import PretrainBlock


        class CustomPretrainBlock(PretrainBlock):
            def pretrain_train(self, x: Any, y: Any, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> tuple[Any, Any]:
                return x, y

            def custom_predict(self, x: Any, **pred_args: Any) -> Any:
                return x


        custom_pretrain_block = CustomPretrainBlock()

        x, y = custom_pretrain_block.train(x, y)
        x = custom_pretrain_block.predict(x)
    """

    test_size: float = 0.2

    @abstractmethod
    def pretrain_train(self, x: Any, y: Any, train_indices: list[int], *, save_pretrain: bool = True, save_pretrain_with_split: bool = False) -> tuple[Any, Any]:
        """Train pretrain block method.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_indices: The indices to train on.
        :param save_pretrain: Whether to save the pretrain.
        :param save_pretrain_with_split: Whether to save the pretrain with a cross validation split.
        """
        raise NotImplementedError(
            f"Train method not implemented for {self.__class__.__name__}",
        )

    def custom_train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Call the pretrain train method.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The train arguments.
        :return: The input and output of the system.
        """
        train_indices = train_args.get("train_indices", None)
        save_pretrain = train_args.get("save_pretrain", True)
        save_pretrain_with_split = train_args.get("save_pretrain_with_split", False)

        return self.pretrain_train(
            x,
            y,
            train_indices,
            save_pretrain=save_pretrain,
            save_pretrain_with_split=save_pretrain_with_split,
        )

    def train_split_hash(self, train_indices: list[int]) -> str:
        """Split the hash on train split.

        :param train_indices: Train indices
        :return: Split hash
        """
        self._hash = hash(self.get_hash() + str(train_indices))
        return self._hash
