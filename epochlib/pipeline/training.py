"""This module contains classes for training and predicting on data."""

import copy
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from joblib import hash

from .core import Base, Block, ParallelSystem, SequentialSystem
from .transforming import TransformingSystem


class TrainType(Base):
    """Abstract train type describing a class that implements two functions train and predict."""

    @abstractmethod
    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the block.

        :param x: The input data.
        :param y: The target variable.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement train method.")

    @abstractmethod
    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the target variable.

        :param x: The input data.
        :return: The predictions.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict method.")


class Trainer(TrainType, Block):
    """The trainer block is for blocks that need to train on two inputs and predict on one.

    Methods:
    .. code-block:: python
        @abstractmethod
        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
            # Train the block.

        @abstractmethod
        def predict(self, x: Any, **pred_args: Any) -> Any:
            # Predict the target variable.

        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path

    Usage:
    .. code-block:: python
        from epochlib.pipeline import Trainer


        class MyTrainer(Trainer):
            def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
                # Train the block.
                return x, y

            def predict(self, x: Any, **pred_args: Any) -> Any:
                # Predict the target variable.
                return x


        my_trainer = MyTrainer()
        predictions, labels = my_trainer.train(x, y)
        predictions = my_trainer.predict(x)
    """


class TrainingSystem(TrainType, SequentialSystem):
    """A system that trains on the input data and labels.

    Parameters:
    - steps (list[TrainType]): The steps in the system.

    Methods:
    .. code-block:: python
        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]: # Train the system.

        def predict(self, x: Any, **pred_args: Any) -> Any: # Predict the output of the system.

        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path

    Usage:
    .. code-block:: python
        from epochlib.pipeline import TrainingSystem

        trainer_1 = CustomTrainer()
        trainer_2 = CustomTrainer()

        training_system = TrainingSystem(steps=[trainer_1, trainer_2])
        trained_x, trained_y = training_system.train(x, y)
        predictions = training_system.predict(x)
    """

    def __post_init__(self) -> None:
        """Post init method for the TrainingSystem class."""
        # Assert all steps are a subclass of Trainer
        for step in self.steps:
            if not isinstance(
                step,
                (TrainType),
            ):
                raise TypeError(f"step: {step} is not an instance of TrainType")

        super().__post_init__()

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The output of the system.
        :return: The input and output of the system.
        """
        set_of_steps = set()
        for step in self.steps:
            step_name = step.__class__.__name__
            set_of_steps.add(step_name)

        if set_of_steps != set(train_args.keys()):
            # Raise a warning and print all the keys that do not match
            warnings.warn(f"The following steps do not exist but were given in the kwargs: {set(train_args.keys()) - set_of_steps}", UserWarning, stacklevel=2)

        # Loop through each step and call the train method
        for step in self.steps:
            step_name = step.__class__.__name__

            step_args = train_args.get(step_name, {})
            if isinstance(step, (TrainType)):
                x, y = step.train(x, y, **step_args)
            else:
                raise TypeError(f"{step} is not an instance of TrainType")

        return x, y

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        set_of_steps = set()
        for step in self.steps:
            step_name = step.__class__.__name__
            set_of_steps.add(step_name)

        if set_of_steps != set(pred_args.keys()):
            # Raise a warning and print all the keys that do not match
            warnings.warn(f"The following steps do not exist but were given in the kwargs: {set(pred_args.keys()) - set_of_steps}", UserWarning, stacklevel=2)

        # Loop through each step and call the predict method
        for step in self.steps:
            step_name = step.__class__.__name__

            step_args = pred_args.get(step_name, {})

            if isinstance(step, (TrainType)):
                x = step.predict(x, **step_args)
            else:
                raise TypeError(f"{step} is not an instance of TrainType")

        return x


class ParallelTrainingSystem(TrainType, ParallelSystem):
    """A system that trains the input data in parallel.

    Parameters:
    - steps (list[Trainer | TrainingSystem | ParallelTrainingSystem]): The steps in the system.
    - weights (list[float]): The weights of steps in the system, if not specified they are all equal.

    Methods:
    .. code-block:: python
        @abstractmethod
        def concat(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed data.

        def train(self, x: Any, y: Any) -> tuple[Any, Any]: # Train the system.

        def predict(self, x: Any, pred_args: dict[str, Any] = {}) -> Any: # Predict the output of the system.

        def concat_labels(self, data1: Any, data2: Any) -> Any: # Concatenate the transformed labels.

        def get_hash(self) -> str: # Get the hash of the system.

    Usage:
    .. code-block:: python
        from epochlib.pipeline import ParallelTrainingSystem

        trainer_1 = CustomTrainer()
        trainer_2 = CustomTrainer()


        class CustomParallelTrainingSystem(ParallelTrainingSystem):
            def concat(self, data1: Any, data2: Any) -> Any:
                # Concatenate the transformed data.
                return data1 + data2


        training_system = CustomParallelTrainingSystem(steps=[trainer_1, trainer_2])
        trained_x, trained_y = training_system.train(x, y)
        predictions = training_system.predict(x)
    """

    def __post_init__(self) -> None:
        """Post init method for the ParallelTrainingSystem class."""
        # Assert all steps correct instances
        for step in self.steps:
            if not isinstance(step, (TrainType)):
                raise TypeError(f"{step} is not an instance of TrainType")

        super().__post_init__()

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :return: The input and output of the system.
        """
        # Loop through each step and call the train method
        out_x, out_y = None, None
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__

            step_args = train_args.get(step_name, {})

            if isinstance(step, (TrainType)):
                step_x, step_y = step.train(copy.deepcopy(x), copy.deepcopy(y), **step_args)
                out_x, out_y = (
                    self.concat(out_x, step_x, self.get_weights()[i]),
                    self.concat_labels(out_y, step_y, self.get_weights()[i]),
                )
            else:
                raise TypeError(f"{step} is not an instance of TrainType")

        return out_x, out_y

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :return: The output of the system.
        """
        # Loop through each trainer and call the predict method
        out_x = None
        for i, step in enumerate(self.steps):
            step_name = step.__class__.__name__

            step_args = pred_args.get(step_name, {})

            if isinstance(step, (TrainType)):
                step_x = step.predict(copy.deepcopy(x), **step_args)
                out_x = self.concat(out_x, step_x, self.get_weights()[i])
            else:
                raise TypeError(f"{step} is not an instance of TrainType")

        return out_x

    def concat_labels(self, original_data: Any, data_to_concat: Any, weight: float = 1.0) -> Any:
        """Concatenate the transformed labels. Will use concat method if not overridden.

        :param original_data: The first input data.
        :param data_to_concat: The second input data.
        :param weight: Weight of data to concat
        :return: The concatenated data.
        """
        return self.concat(original_data, data_to_concat, weight)


@dataclass
class Pipeline(TrainType):
    """A pipeline of systems that can be trained and predicted.

    Parameters:
    - x_sys (TransformingSystem | None): The system to transform the input data.
    - y_sys (TransformingSystem | None): The system to transform the labelled data.
    - train_sys (TrainingSystem | None): The system to train the data.
    - pred_sys (TransformingSystem | None): The system to transform the predictions.
    - label_sys (TransformingSystem | None): The system to transform the labels.

    Methods:
    .. code-block:: python
        def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
            # Train the system.

        def predict(self, x: Any, **pred_args) -> Any:
            # Predict the output of the system.

        def get_hash(self) -> str:
            # Get the hash of the pipeline

        def get_parent(self) -> Any:
            # Get the parent of the pipeline

        def get_children(self) -> list[Any]:
            # Get the children of the pipeline

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path

    Usage:
    .. code-block:: python
        from epochlib.pipeline import Pipeline

        x_sys = CustomTransformingSystem()
        y_sys = CustomTransformingSystem()
        train_sys = CustomTrainingSystem()
        pred_sys = CustomTransformingSystem()
        label_sys = CustomTransformingSystem()

        pipeline = Pipeline(x_sys=x_sys, y_sys=y_sys, train_sys=train_sys, pred_sys=pred_sys, label_sys=label_sys)
        trained_x, trained_y = pipeline.train(x, y)
        predictions = pipeline.predict(x)
    """

    x_sys: TransformingSystem | None = None
    y_sys: TransformingSystem | None = None
    train_sys: Trainer | TrainingSystem | ParallelTrainingSystem | None = None
    pred_sys: TransformingSystem | None = None
    label_sys: TransformingSystem | None = None

    def __post_init__(self) -> None:
        """Post initialization function of the Pipeline."""
        super().__post_init__()

        # Set children and parents
        children = []
        systems = [
            self.x_sys,
            self.y_sys,
            self.train_sys,
            self.pred_sys,
            self.label_sys,
        ]

        for sys in systems:
            if sys is not None:
                sys.set_parent(self)
                children.append(sys)

        self.set_children(children)

    def train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
        """Train the system.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The arguments to pass to the training system. (Default is {})
        :return: The input and output of the system.
        """
        if self.x_sys is not None:
            x = self.x_sys.transform(x, **train_args.get("x_sys", {}))
        if self.y_sys is not None:
            y = self.y_sys.transform(y, **train_args.get("y_sys", {}))
        if self.train_sys is not None:
            x, y = self.train_sys.train(x, y, **train_args.get("train_sys", {}))
        if self.pred_sys is not None:
            x = self.pred_sys.transform(x, **train_args.get("pred_sys", {}))
        if self.label_sys is not None:
            y = self.label_sys.transform(y, **train_args.get("label_sys", {}))

        return x, y

    def predict(self, x: Any, **pred_args: Any) -> Any:
        """Predict the output of the system.

        :param x: The input to the system.
        :param pred_args: The arguments to pass to the prediction system. (Default is {})
        :return: The output of the system.
        """
        if self.x_sys is not None:
            x = self.x_sys.transform(x, **pred_args.get("x_sys", {}))
        if self.train_sys is not None:
            x = self.train_sys.predict(x, **pred_args.get("train_sys", {}))
        if self.pred_sys is not None:
            x = self.pred_sys.transform(x, **pred_args.get("pred_sys", {}))

        return x

    def _set_hash(self, prev_hash: str) -> None:
        """Set the hash of the pipeline.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        xy_hash = ""
        if self.x_sys is not None:
            self.x_sys.set_hash(self.get_hash())
            xy_hash += self.x_sys.get_hash()
        if self.y_sys is not None:
            self.y_sys.set_hash(self.get_hash())
            xy_hash += self.y_sys.get_hash()[::-1]  # Reversed for edge case where you have two pipelines with the same system but one in x the other in y

        if xy_hash != "":
            self._hash = hash(xy_hash)

        if self.train_sys is not None:
            self.train_sys.set_hash(self.get_hash())
            training_hash = self.train_sys.get_hash()
            if training_hash != "":
                self._hash = hash(self._hash + training_hash)

        predlabel_hash = ""
        if self.pred_sys is not None:
            self.pred_sys.set_hash(self.get_hash())
            predlabel_hash += self.pred_sys.get_hash()
        if self.label_sys is not None:
            self.label_sys.set_hash(self.get_hash())
            predlabel_hash += self.label_sys.get_hash()

        if predlabel_hash != "":
            self._hash = hash(predlabel_hash)
