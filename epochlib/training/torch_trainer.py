"""TorchTrainer is a module that allows for the training of Torch models."""

import contextlib
import copy
import functools
import gc
from collections.abc import Callable
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Annotated, Any, Literal, Tuple, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from annotated_types import Ge, Gt, Interval
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

from epochlib.data import Data

from ._custom_data_parallel import _CustomDataParallel
from .training_block import TrainingBlock
from .utils import _get_onnxrt, _get_openvino, batch_to_device

T = TypeVar("T", bound=Dataset)  # type: ignore[type-arg]
T_co = TypeVar("T_co", covariant=True)


def custom_collate(batch: list[Tensor]) -> tuple[Tensor, Tensor]:
    """Collate function for the dataloader.

    :param batch: The batch to collate.
    :return: Collated batch.
    """
    X, y = batch[0], batch[1]
    return X, y


@dataclass
class TorchTrainer(TrainingBlock):
    """Abstract class for torch trainers, override necessary functions for custom implementation.

    Parameters Modules
    ----------
    - `model` (nn.Module): The model to train.
    - `optimizer` (functools.partial[Optimizer]): Optimizer to use for training.
    - `criterion` (nn.Module): Criterion to use for training.
    - `scheduler` (Callable[[Optimizer], LRScheduler] | None): Scheduler to use for training.
    - `dataloader_args (dict): Arguments for the dataloader`

    Parameters Training
    ----------
    - `epochs` (int): Number of epochs
    - `patience` (int): Stopping training after x epochs of no improvement (early stopping)
    - `batch_size` (int): Batch size

    Parameters Checkpointing
    ----------
    - `checkpointing_enabled` (bool): Whether to save checkpoints after each epoch
    - `checkpointing_keep_every` (int): Keep every i'th checkpoint (1 to keep all, 0 to keep only the last one)
    - `checkpointing_resume_if_exists` (bool): Resume training if a checkpoint exists

    Parameters Precision
    ----------
    - `use_mixed_precision` (bool): Whether to use mixed precision for the model training

    Parameters Misc
    ----------
    - `to_predict` (str): Whether to predict on the 'validation' set, 'all' data or 'none'
    - `model_name` (str): Name of the model
    - `n_folds` (float): Number of folds for cross validation (0 for train full,
    - `_fold` (int): Fold number
    - `validation_size` (float): Relative size of the validation set
    - `x_tensor_type` (str): Type of x tensor for data
    - `y_tensor_type` (str): Type of y tensor for labels

    Methods
    -------
    .. code-block:: python
        @abstractmethod
        def log_to_terminal(self, message: str) -> None:
            # Logs to terminal if implemented

        @abstractmethod
        def log_to_debug(self, message: str) -> None:
            # Logs to debugger if implemented

        @abstractmethod
        def log_to_warning(self, message: str) -> None:
            # Logs to warning if implemented

        @abstractmethod
        def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None:
            # Logs to external site

        @abstractmethod
        def external_define_metric(self, metric: str, metric_type: str) -> None:
            # Defines an external metric

        def train(self, x: Any, y: Any, cache_args: dict[str, Any] = {}, **train_args: Any) -> tuple[Any, Any]:
            # Applies caching and calls custom_train, overridding removes caching

        def predict(self, x: Any, cache_args: dict[str, Any] = {}, **pred_args: Any) -> Any:
            # Applies caching and calls custom_predict, overridding removes caching

        def custom_train(self, x: Any, y: Any, **train_args: Any) -> tuple[Any, Any]:
            # Implements torch training. If you are going to override this method and not use any other functionality, inherit from TrainingBlock.

        def custom_predict(self, x: Any, **pred_args: Any) -> Any:
            # Implements torch prediction. If you are going to override this method and not use any other functionality, inherit from TrainingBlock.

        def predict_on_loader(loader: DataLoader[tuple[Tensor, ...]]) -> npt.NDArray[np.float32]:
            # Predict using a dataloader.

        def create_datasets(
            x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], train_indices: list[int], validation_indices: list[int], cache_size: int = -1
        ) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
            # Create the datasets for training and validation.

        def create_prediction_dataset(x: npt.NDArray[np.float32]) -> Dataset[tuple[Tensor, ...]]:
            # Create the prediction dataset.

        def create_dataloaders(
            train_dataset: Dataset[tuple[Tensor, ...]], validation_dataset: Dataset[tuple[Tensor, ...]]
        ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
            # Create the dataloaders for training and validation.

        def update_model_directory(model_directory: str) -> None:
            # Update the model directory for caching (default: tm).

    Usage:
    .. code-block:: python
        from epochlib.pipeline.model.training.torch_trainer import TorchTrainer
        from torch import nn
        from torch.optim import Adam
        from torch.optim.lr_scheduler import StepLR
        from torch.nn import MSELoss

        class MyTorchTrainer(TorchTrainer):

            def log_to_terminal(self, message: str) -> None:

            ....

        model = nn.Sequential(nn.Linear(1, 1))
        optimizer = functools.partial(Adam, lr=0.01)
        criterion = MSELoss()
        scheduler = functools.partial(StepLR, step_size=1, gamma=0.1)
        epochs = 10
        batch_size = 32
        patience = 5
        validation_size = 0.2

        trainer = MyTorchTrainer(model=model, optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                 epochs=epochs, batch_size=batch_size, patience=patience, validation_size=validation_size)

        x, y = trainer.train(x, y)
        x = trainer.predict(x)
    """

    # Modules
    model: nn.Module
    optimizer: functools.partial[Optimizer]
    criterion: nn.Module
    scheduler: Callable[[Optimizer], LRScheduler] | None = None
    dataloader_args: dict[str, Any] = field(default_factory=dict, repr=False)
    dataset: functools.partial[Dataset[Tuple[Tensor, Tensor]]] | None = None

    # Training parameters
    epochs: Annotated[int, Gt(0)] = 10
    patience: Annotated[int, Gt(0)] = -1  # Early stopping
    batch_size: Annotated[int, Gt(0)] = 32
    collate_fn: Callable[[list[Tensor]], tuple[Tensor, Tensor]] = field(default=custom_collate, init=True, repr=False, compare=False)

    # Checkpointing
    checkpointing_enabled: bool = field(default=True, init=True, repr=False, compare=False)
    checkpointing_keep_every: Annotated[int, Gt(0)] = field(default=0, init=True, repr=False, compare=False)
    checkpointing_resume_if_exists: bool = field(default=True, init=True, repr=False, compare=False)

    # Precision
    use_mixed_precision: bool = field(default=False)

    # Misc
    model_name: str | None = None  # No spaces allowed
    trained_models_directory: PathLike[str] = field(default=Path("tm/"), repr=False, compare=False)
    to_predict: Literal["validation", "all", "none"] = field(default="validation", repr=False, compare=False)

    # Parameters relevant for Hashing
    n_folds: Annotated[int, Ge(0)] = field(default=-1, init=True, repr=False, compare=False)
    _fold: int = field(default=-1, init=False, repr=False, compare=False)
    validation_size: Annotated[float, Interval(ge=0, le=1)] = 0.2

    # Types for tensors
    x_tensor_type: str = "float"
    y_tensor_type: str = "float"

    # Prefix and postfix for logging to external
    logging_prefix: str = field(default="", init=True, repr=False, compare=False)
    logging_postfix: str = field(default="", init=True, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Post init method for the TorchTrainer class."""
        # Make sure to_predict is either "validation" or "all" or "none"
        if self.to_predict not in ["validation", "all", "none"]:
            raise ValueError("to_predict should be either 'validation', 'all' or 'none'")

        if self.n_folds == -1:
            raise ValueError(
                "Please specify the number of folds for cross validation or set n_folds to 0 for train full.",
            )

        if self.model_name is None:
            raise ValueError("self.model_name is None, please specify a model_name")

        self.save_model_to_disk = True
        self.best_model_state_dict: dict[Any, Any] = {}

        # Set optimizer
        self.initialized_optimizer = self.optimizer(self.model.parameters())

        # Set scheduler
        self.initialized_scheduler: LRScheduler | None
        if self.scheduler is not None:
            self.initialized_scheduler = self.scheduler(self.initialized_optimizer)
        else:
            self.initialized_scheduler = None

        # Set the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_to_terminal(f"Setting device: {self.device}")

        # If multiple GPUs are available, distribute batch size over the GPUs
        if torch.cuda.device_count() > 1:
            self.log_to_terminal(f"Using {torch.cuda.device_count()} GPUs")
            self.model = _CustomDataParallel(self.model)

        self.model.to(self.device)

        # Early stopping
        self.last_val_loss = np.inf
        self.lowest_val_loss = np.inf

        # Mixed precision
        if self.use_mixed_precision:
            self.log_to_terminal("Using mixed precision training.")
            self.scaler = torch.GradScaler(device=self.device.type)
            torch.set_float32_matmul_precision("high")

        # Check validity of model_name
        if " " in self.model_name:
            raise ValueError("Spaces in model_name not allowed")

        super().__post_init__()

    def custom_train(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.float32], **train_args: Any) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Train the model.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_args: The keyword arguments.
            - train_indices: The indices to train on.
            - validation_indices: The indices to validate on.
            - save_model: Whether to save the model.
            - fold: Fold number if running cv.
        :return: The input and output of the system.
        """
        train_indices = train_args.get("train_indices")
        if train_indices is None:
            raise ValueError("train_indices not provided")
        validation_indices = train_args.get("validation_indices")
        if validation_indices is None:
            raise ValueError("validation_indices not provided")
        save_model = train_args.get("save_model", True)
        self._fold = train_args.get("fold", -1)

        self.save_model_to_disk = save_model

        # Create datasets
        train_dataset, validation_dataset = self.create_datasets(
            x,
            y,
            train_indices,
            validation_indices,
        )

        # Create dataloaders
        train_loader, validation_loader = self.create_dataloaders(train_dataset, validation_dataset)

        # Check if a trained model exists
        if self._model_exists():
            self.log_to_terminal(
                f"Model exists in {self.get_model_path()}. Loading model...",
            )
            self._load_model()

            # Return the predictions
            return self.predict_after_train(
                x,
                y,
                train_dataset,
                validation_dataset,
                train_indices,
                validation_indices,
            )

        # Log the model being trained
        self.log_to_terminal(f"Training model: {self.model.__class__.__name__}")

        # Resume from checkpoint if enabled and checkpoint exists
        start_epoch = 0
        if self.checkpointing_resume_if_exists:
            saved_checkpoints = list(Path(self.trained_models_directory).glob(f"{self.get_hash()}_checkpoint_*.pt"))
            if len(saved_checkpoints) > 0:
                self.log_to_terminal("Resuming training from checkpoint")
                epochs = [int(checkpoint.stem.split("_")[-1]) for checkpoint in saved_checkpoints]
                self._load_model(saved_checkpoints[np.argmax(epochs)])
                start_epoch = max(epochs) + 1

        # Train the model
        self.log_to_terminal(f"Training model for {self.epochs} epochs{', starting at epoch ' + str(start_epoch) if start_epoch > 0 else ''}")

        train_losses: list[float] = []
        val_losses: list[float] = []

        self.lowest_val_loss = np.inf
        if len(validation_loader) == 0:
            self.log_to_warning(
                f"Doing train full, model will be trained for {self.epochs} epochs",
            )

        self._training_loop(
            train_loader,
            validation_loader,
            train_losses,
            val_losses,
            self._fold,
            start_epoch,
        )
        self.log_to_terminal(
            f"Done training the model: {self.model.__class__.__name__}",
        )

        # Revert to the best model
        if self.best_model_state_dict:
            self.log_to_terminal(
                f"Reverting to model with best validation loss {self.lowest_val_loss}",
            )
            self.model.load_state_dict(self.best_model_state_dict)

        if save_model:
            self._save_model()

        return self.predict_after_train(
            x,
            y,
            train_dataset,
            validation_dataset,
            train_indices,
            validation_indices,
        )

    def predict_after_train(
        self,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        train_dataset: Dataset[Any],
        validation_dataset: Dataset[Any],
        train_indices: list[int],
        validation_indices: list[int],
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Predict after training the model.

        :param x: The input to the system.
        :param y: The expected output of the system.
        :param train_dataset: The training dataset.
        :param validation_dataset: The validation dataset.
        :param train_indices: The indices to train on.
        :param validation_indices: The indices to validate on.

        :return: The predictions and the expected output.
        """
        match self.to_predict:
            case "all":
                concat_dataset: Dataset[Any] = self._concat_datasets(
                    train_dataset,
                    validation_dataset,
                    train_indices,
                    validation_indices,
                )
                pred_dataloader = DataLoader(
                    concat_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    collate_fn=(self.collate_fn if hasattr(concat_dataset, "__getitems__") else None),
                )
                return self.predict_on_loader(pred_dataloader), y
            case "validation":
                train_loader, validation_loader = self.create_dataloaders(
                    train_dataset,
                    validation_dataset,
                )
                return self.predict_on_loader(validation_loader), y[validation_indices]
            case "none":
                return x, y
            case _:
                raise ValueError("to_predict should be either 'validation', 'all' or 'none")

    def custom_predict(self, x: Any, **pred_args: Any) -> npt.NDArray[np.float32]:
        """Predict on the validation data.

        :param x: The input to the system.
        :return: The output of the system.
        """
        self.log_section_separator(f"Predicting model: {self.model.__class__.__name__}")
        self.log_to_debug(f"Predicting model: {self.model.__class__.__name__}")

        # Parse pred_args
        curr_batch_size = pred_args.get("batch_size", self.batch_size)

        # Create dataset
        pred_dataset = self.create_prediction_dataset(x)
        pred_dataloader = DataLoader(
            pred_dataset,
            batch_size=curr_batch_size,
            shuffle=False,
            collate_fn=(self.collate_fn if hasattr(pred_dataset, "__getitems__") else None),
        )

        # Predict with a single model
        if self.n_folds < 1 or pred_args.get("use_single_model", False):
            self._load_model()
            return self.predict_on_loader(pred_dataloader, pred_args.get("compile_method", None))

        predictions = []
        # Predict with multiple models
        for i in range(int(self.n_folds)):
            self._fold = i  # set the fold, which updates the hash
            # Try to load the next fold if it exists
            try:
                self._load_model()
            except FileNotFoundError as e:
                if i == 0:
                    raise FileNotFoundError(f"First model of {self.n_folds} folds not found...") from e
                self.log_to_warning(f"Model for fold {self._fold} not found, skipping the rest of the folds...")
                break
            self.log_to_terminal(f"Predicting with model fold {i + 1}/{self.n_folds}")
            predictions.append(self.predict_on_loader(pred_dataloader, pred_args.get("compile_method", None)))

        # Average the predictions using numpy
        validation_predictions = np.array(predictions)

        return np.mean(validation_predictions, axis=0)

    def predict_on_loader(
        self,
        loader: DataLoader[tuple[Tensor, ...]],
        compile_method: str | None = None,
    ) -> npt.NDArray[np.float32]:
        """Predict on the loader.

        :param loader: The loader to predict on.
        :return: The predictions.
        """
        self.log_to_terminal("Running inference on the given dataloader")
        self.model.eval()
        predictions = []
        # Create a new dataloader from the dataset of the input dataloader with collate_fn
        loader = DataLoader(
            loader.dataset,
            batch_size=loader.batch_size,
            shuffle=False,
            collate_fn=(self.collate_fn if hasattr(loader.dataset, "__getitems__") else None),
            **self.dataloader_args,
        )
        if compile_method is None:
            with torch.no_grad(), tqdm(loader, unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)

                    y_pred = self.model(X_batch).squeeze(1).cpu().numpy()
                    predictions.extend(y_pred)

        elif compile_method == "ONNX":
            if self.device != torch.device("cpu"):
                raise ValueError("ONNX compilation only works on CPU. To disable CUDA use the environment variable CUDA_VISIBLE_DEVICES=-1")
            input_tensor = next(iter(loader))[0].to(self.device).float()
            input_names = ["actual_input"]
            output_names = ["output"]
            self.log_to_terminal("Compiling model to ONNX")
            torch.onnx.export(self.model, input_tensor, f"{self.get_hash()}.onnx", verbose=False, input_names=input_names, output_names=output_names)
            onnx_model = _get_onnxrt().InferenceSession(f"{self.get_hash()}.onnx")
            self.log_to_terminal("Done compiling model to ONNX")
            with torch.no_grad(), tqdm(loader, desc="Predicting", unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)
                    y_pred = onnx_model.run(output_names, {input_names[0]: X_batch.numpy()})[0]
                    predictions.extend(y_pred)

            # Remove the onnx file
            Path(f"{self.get_hash()}.onnx").unlink()

        elif compile_method == "Openvino":
            if self.device != torch.device("cpu"):
                raise ValueError("Openvino compilation only works on CPU. To disable CUDA use the environment variable CUDA_VISIBLE_DEVICES=-1")
            input_tensor = next(iter(loader))[0].to(self.device).float()
            self.log_to_terminal("Compiling model to Openvino")
            ov = _get_openvino()
            openvino_model = ov.compile_model(ov.convert_model(self.model, example_input=input_tensor))
            self.log_to_terminal("Done compiling model to Openvino")
            with torch.no_grad(), tqdm(loader, desc="Predicting", unit="batch", disable=False) as tepoch:
                for data in tepoch:
                    X_batch = batch_to_device(data[0], self.x_tensor_type, self.device)
                    y_pred = openvino_model(X_batch)[0]
                    predictions.extend(y_pred)

        self.log_to_terminal("Done predicting!")
        return np.array(predictions)

    def get_hash(self) -> str:
        """Get the hash of the block.

        Override the get_hash method to include the fold number in the hash.

        :return: The hash of the block.
        """
        result = f"{self._hash}_{self.n_folds}"
        if self._fold != -1:
            result += f"_f{self._fold}"
        return result

    def create_datasets(
        self,
        x: npt.NDArray[np.float32] | Data,
        y: npt.NDArray[np.float32] | Data,
        train_indices: list[int],
        validation_indices: list[int],
    ) -> tuple[Dataset[tuple[Tensor, ...]], Dataset[tuple[Tensor, ...]]]:
        """Create the datasets for training and validation.

        :param x: The input data.
        :param y: The target variable.
        :param train_indices: The indices to train on.
        :param validation_indices: The indices to validate on.
        :return: The training and validation datasets.
        """
        if self.dataset is None:
            train_dataset = TensorDataset(
                torch.from_numpy(x[train_indices]),
                torch.from_numpy(y[train_indices]),
            )
            validation_dataset = TensorDataset(
                torch.from_numpy(x[validation_indices]),
                torch.from_numpy(y[validation_indices]),
            )
            return train_dataset, validation_dataset

        return self.dataset(x[train_indices], y[train_indices]), self.dataset(x[validation_indices], y[validation_indices])

    def create_prediction_dataset(
        self,
        x: npt.NDArray[np.float32] | Data,
    ) -> Dataset[tuple[Tensor, ...]]:
        """Create the prediction dataset.

        :param x: The input data.
        :return: The prediction dataset.
        """
        if self.dataset is None:
            return TensorDataset(torch.from_numpy(x))
        return self.dataset(x)

    def create_dataloaders(
        self,
        train_dataset: Dataset[tuple[Tensor, ...]],
        validation_dataset: Dataset[tuple[Tensor, ...]],
    ) -> tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]]:
        """Create the dataloaders for training and validation.

        :param train_dataset: The training dataset.
        :param validation_dataset: The validation dataset.
        :return: The training and validation dataloaders.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=(self.collate_fn if hasattr(train_dataset, "__getitems__") else None),
            **self.dataloader_args,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=(self.collate_fn if hasattr(validation_dataset, "__getitems__") else None),
            **self.dataloader_args,
        )
        return train_loader, validation_loader

    def save_model_to_external(self) -> None:
        """Save model to external database."""
        self.log_to_warning(
            "Saving model to external is not implemented for TorchTrainer, if you want uploaded models. Please overwrite",
        )

    def _training_loop(
        self,
        train_loader: DataLoader[tuple[Tensor, ...]],
        validation_loader: DataLoader[tuple[Tensor, ...]],
        train_losses: list[float],
        val_losses: list[float],
        fold: int = -1,
        start_epoch: int = 0,
    ) -> None:
        """Training loop for the model.

        :param train_loader: Dataloader for the validation data.
        :param validation_loader: Dataloader for the training data. (can be empty)
        :param train_losses: List of train losses.
        :param val_losses: List of validation losses.
        """
        fold_no = ""

        if fold > -1:
            fold_no = f"_{fold}"

        self.external_define_metric(self.wrap_log(f"Training/Train Loss{fold_no}"), self.wrap_log("epoch"))
        self.external_define_metric(self.wrap_log(f"Validation/Validation Loss{fold_no}"), self.wrap_log("epoch"))

        # Set the scheduler to the correct epoch
        if self.initialized_scheduler is not None:
            self.initialized_scheduler.step(epoch=start_epoch)

        for epoch in range(start_epoch, self.epochs):
            # Train using train_loader
            train_loss = self.train_one_epoch(train_loader, epoch)
            self.log_to_debug(f"Epoch {epoch} Train Loss: {train_loss}")
            train_losses.append(train_loss)

            # Log train loss
            self.log_to_external(
                message={
                    self.wrap_log(f"Training/Train Loss{fold_no}"): train_losses[-1],
                    self.wrap_log("epoch"): epoch,
                },
            )

            # Step the scheduler
            if self.initialized_scheduler is not None:
                self.initialized_scheduler.step(epoch=epoch + 1)

            # Checkpointing
            if self.checkpointing_enabled:
                # Save checkpoint
                self._save_model(self.get_model_checkpoint_path(epoch), save_to_external=False, quiet=True)

                # Remove old checkpoints
                if (self.checkpointing_keep_every == 0 or epoch % self.checkpointing_keep_every != 0) and self.get_model_checkpoint_path(epoch - 1).exists():
                    self.get_model_checkpoint_path(epoch - 1).unlink()

            # Compute validation loss
            if len(validation_loader) > 0:
                self.last_val_loss = self.val_one_epoch(
                    validation_loader,
                    desc=f"Epoch {epoch} Valid",
                )
                self.log_to_debug(f"Epoch {epoch} Valid Loss: {self.last_val_loss}")
                val_losses.append(self.last_val_loss)

                # Log validation loss and plot train/val loss against each other
                self.log_to_external(
                    message={
                        self.wrap_log(f"Validation/Validation Loss{fold_no}"): val_losses[-1],
                        self.wrap_log("epoch"): epoch,
                    },
                )

                self.log_to_external(
                    message={
                        "type": "wandb_plot",
                        "plot_type": "line_series",
                        "data": {
                            "xs": list(
                                range(epoch + 1),
                            ),  # Ensure it's a list, not a range object
                            "ys": [train_losses, val_losses],
                            "keys": [f"Train{fold_no}", f"Validation{fold_no}"],
                            "title": self.wrap_log(f"Training/Loss{fold_no}"),
                            "xname": "Epoch",
                        },
                    },
                )

                # Early stopping
                if self._early_stopping():
                    self.log_to_external(message={self.wrap_log(f"Epochs{fold_no}"): (epoch + 1) - self.patience})
                    break

            # Log the trained epochs to wandb if we finished training
            self.log_to_external(message={self.wrap_log(f"Epochs{fold_no}"): epoch + 1})

    def train_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        epoch: int,
    ) -> float:
        """Train the model for one epoch.

        :param dataloader: Dataloader for the training data.
        :param epoch: Epoch number.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.train()
        pbar = tqdm(
            dataloader,
            unit="batch",
            desc=f"Epoch {epoch} Train ({self.initialized_optimizer.param_groups[0]['lr']:0.8f})",
        )
        for batch in pbar:
            X_batch, y_batch = batch

            X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
            y_batch = batch_to_device(y_batch, self.y_tensor_type, self.device)

            # Forward pass
            with torch.autocast(self.device.type) if self.use_mixed_precision else contextlib.nullcontext():  # type: ignore[attr-defined]
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

            # Backward pass
            self.initialized_optimizer.zero_grad()
            if self.use_mixed_precision:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.initialized_optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.initialized_optimizer.step()

            # Print tqdm
            losses.append(loss.item())
            pbar.set_postfix(loss=sum(losses) / len(losses))

        # Remove the cuda cache
        torch.cuda.empty_cache()
        gc.collect()

        return sum(losses) / len(losses)

    def val_one_epoch(
        self,
        dataloader: DataLoader[tuple[Tensor, ...]],
        desc: str,
    ) -> float:
        """Compute validation loss of the model for one epoch.

        :param dataloader: Dataloader for the validation data.
        :param desc: Description for the tqdm progress bar.
        :return: Average loss for the epoch.
        """
        losses = []
        self.model.eval()
        pbar = tqdm(dataloader, unit="batch")
        with torch.no_grad():
            for batch in pbar:
                X_batch, y_batch = batch

                X_batch = batch_to_device(X_batch, self.x_tensor_type, self.device)
                y_batch = batch_to_device(y_batch, self.y_tensor_type, self.device)

                # Forward pass
                y_pred = self.model(X_batch).squeeze(1)
                loss = self.criterion(y_pred, y_batch)

                # Print losses
                losses.append(loss.item())
                pbar.set_description(desc=desc)
                pbar.set_postfix(loss=sum(losses) / len(losses))
        return sum(losses) / len(losses)

    def _save_model(self, model_path: Path | None = None, *, save_to_external: bool = True, quiet: bool = False) -> None:
        """Save the model in the model_directory folder."""
        model_path = model_path if model_path is not None else self.get_model_path()

        if not quiet:
            self.log_to_terminal(
                f"Saving model to {model_path}",
            )

        model_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.model, model_path)

        if save_to_external:
            self.save_model_to_external()

    def _load_model(self, path: Path | None = None) -> None:
        """Load the model from the model_directory folder."""
        model_path = path if path is not None else self.get_model_path()

        # Check if the model exists
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found in {model_path}",
            )

        # Load model
        self.log_to_terminal(
            f"Loading model from {model_path}",
        )
        checkpoint = torch.load(model_path)

        # Load the weights from the checkpoint
        if isinstance(checkpoint, nn.DataParallel):
            model = checkpoint.module
        else:
            model = checkpoint

        # Set the current model to the loaded model
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(model.state_dict())
        else:
            self.model.load_state_dict(model.state_dict())

    def _model_exists(self) -> bool:
        """Check if the model exists in the model_directory folder."""
        return self.get_model_path().exists() and self.save_model_to_disk

    def _early_stopping(self) -> bool:
        """Check if early stopping should be performed.

        :return: Whether to perform early stopping.
        """
        # Store the best model so far based on validation loss
        if self.patience != -1:
            if self.last_val_loss < self.lowest_val_loss:
                self.lowest_val_loss = self.last_val_loss
                self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
                if self.early_stopping_counter >= self.patience:
                    self.log_to_terminal(
                        f"Early stopping after {self.early_stopping_counter} epochs",
                    )
                    return True
        return False

    def _concat_datasets(
        self,
        train_dataset: T,
        validation_dataset: T,
        train_indices: list[int] | npt.NDArray[np.int32],
        validation_indices: list[int] | npt.NDArray[np.int32],
    ) -> Dataset[T_co]:
        """Concatenate the training and validation datasets according to original order specified by train_indices and validation_indices.

        :param train_dataset: The training dataset.
        :param validation_dataset: The validation dataset.
        :param train_indices: The indices for the training data.
        :param validation_indices: The indices for the validation data.
        :return: A new dataset containing the concatenated data in the original order.
        """
        return TrainValidationDataset(
            train_dataset,
            validation_dataset,
            list(train_indices),
            list(validation_indices),
        )

    def get_model_path(self) -> Path:
        """Get the model path.

        :return: The model path.
        """
        return Path(self.trained_models_directory) / f"{self.get_hash()}.pt"

    def get_model_checkpoint_path(self, epoch: int) -> Path:
        """Get the checkpoint path.

        :param epoch: The epoch number.
        :return: The checkpoint path.
        """
        return Path(self.trained_models_directory) / f"{self.get_hash()}_checkpoint_{epoch}.pt"

    def wrap_log(self, text: str) -> str:
        """Add logging prefix and postfix to the message."""
        return f"{self.logging_prefix}{text}{self.logging_postfix}"


class TrainValidationDataset(Dataset[T_co]):
    """Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    :param train_dataset: The train dataset
    :param validation_dataset: The validation dataset
    :param train_indices: The train indices
    :param validation_indices: The validation indices
    """

    train_dataset: Dataset[T_co]
    validation_dataset: Dataset[T_co]
    train_indices: list[int]
    validation_indices: list[int]

    def __init__(
        self,
        train_dataset: Dataset[T_co],
        validation_dataset: Dataset[T_co],
        train_indices: list[int],
        validation_indices: list[int],
    ) -> None:
        """Initialize TrainValidationDataset.

        :param train_dataset: The train dataset.
        :param validation_dataset: The validation dataset.
        :param train_indices: The train indices.
        :param validation_indices: The validation indices.
        """
        super().__init__()
        if len(train_dataset) != len(train_indices):  # type: ignore[arg-type]
            raise ValueError("Train_dataset should be the same length as train_indices")
        if len(validation_dataset) != len(validation_indices):  # type: ignore[arg-type]
            raise ValueError("Validation_dataset should be the same length as validation_indices")
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.train_indices = train_indices
        self.validation_indices = validation_indices

    def __len__(self) -> int:
        """Get the length of the dataset.

        :return: The length of the dataset.
        """
        return len(self.train_dataset) + len(self.validation_dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> T_co:
        """Get the item at an idx.

        :param idx: Index to retrieve.
        :return: Value to return.
        """
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length",
                )
            idx = len(self) + idx

        item = self.train_dataset[0]

        if idx in self.train_indices:
            train_index = self.train_indices.index(idx)
            item = self.train_dataset[train_index]
        else:
            validation_index = self.validation_indices.index(idx)
            item = self.validation_dataset[validation_index]

        return item
