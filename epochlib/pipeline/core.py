"""This module contains the core classes for all classes in the epochlib package."""

from abc import abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from joblib import hash


@dataclass
class Base:
    """The Base class is the base class for all classes in the epochlib package.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str:
            # Get the hash of base

        def get_parent(self) -> Any:
            # Get the parent of base.

        def get_children(self) -> list[Any]:
            # Get the children of base

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """

    def __post_init__(self) -> None:
        """Initialize the block."""
        self.set_hash("")
        self.set_parent(None)
        self.set_children([])

    def set_hash(self, prev_hash: str) -> None:
        """Set the hash of the block.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = hash(prev_hash + str(self))

    def get_hash(self) -> str:
        """Get the hash of the block.

        :return: The hash of the block.
        """
        return self._hash

    def get_parent(self) -> Any:
        """Get the parent of the block.

        :return: Parent of the block
        """
        return self._parent

    def get_children(self) -> list[Any]:
        """Get the children of the block.

        :return: Children of the block
        """
        return self._children

    def save_to_html(self, file_path: Path) -> None:
        """Write html representation of class to file.

        :param file_path: File path to write to
        """
        html = self._repr_html_()
        with open(file_path, "w") as file:
            file.write(html)

    def set_parent(self, parent: Any) -> None:
        """Set the parent of the block.

        :param parent: Parent of the block
        """
        self._parent = parent

    def set_children(self, children: list[Any]) -> None:
        """Set the children of the block.

        :param children: Children of the block
        """
        self._children = children

    def _repr_html_(self) -> str:
        """Return representation of class in html format.

        :return: String representation of html
        """
        html = "<div style='border: 1px solid black; padding: 10px;'>"
        html += f"<p><strong>Class:</strong> {self.__class__.__name__}</p>"
        html += "<ul>"
        html += f"<li><strong>Hash:</strong> {self.get_hash()}</li>"
        html += f"<li><strong>Parent:</strong> {self.get_parent()}</li>"
        html += "<li><strong>Children:</strong> "
        if self.get_children():
            html += "<ul>"
            for child in self.get_children():
                html += f"<li>{child._repr_html_()}</li>"
            html += "</ul>"
        else:
            html += "None"
        html += "</li>"
        html += "</ul>"
        html += "</div>"
        return html


class Block(Base):
    """The Block class is the base class for all blocks.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """


@dataclass
class ParallelSystem(Base):
    """The System class is the base class for all systems.

    Parameters:
    - steps (list[_Base]): The steps in the system.
    - weights (list[float]): Weights of steps in the system, if not specified they are equal.

    Methods:
    .. code-block:: python
        @abstractmethod
        def concat(self, original_data: Any), data_to_concat: Any, weight: float = 1.0) -> Any:
            # Specifies how to concat data after parallel computations

        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """

    steps: list[Base] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function of _System class."""
        # Sort the steps by name, to ensure consistent ordering of parallel computations
        self.steps = sorted(self.steps, key=lambda x: x.__class__.__name__)

        super().__post_init__()

        # Set parent and children
        for step in self.steps:
            step.set_parent(self)

        # Set weights if they exist
        if len(self.weights) == len(self.get_steps()):
            [w / sum(self.weights) for w in self.weights]
        else:
            num_steps = len(self.get_steps())
            self.weights = [1 / num_steps for x in self.steps]

        self.set_children(self.steps)

    def get_steps(self) -> list[Base]:
        """Return list of steps of ParallelSystem.

        :return: List of steps
        """
        return self.steps

    def get_weights(self) -> list[float]:
        """Return list of weights of ParallelSystem.

        :return: List of weights
        """
        if len(self.get_steps()) != len(self.weights):
            raise TypeError("Mismatch between weights and steps")
        return self.weights

    def set_hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        # System has no steps and as such hash should not be affected
        if len(self.steps) == 0:
            return

        # System is one step and should act as such
        if len(self.steps) == 1:
            step = self.steps[0]
            step.set_hash(prev_hash)
            self._hash = step.get_hash()
            return

        # System has at least two steps so hash should become a combination
        total = self.get_hash()
        for step in self.steps:
            step.set_hash(prev_hash)
            total = total + step.get_hash()

        self._hash = hash(total)

    @abstractmethod
    def concat(self, original_data: Any, data_to_concat: Any, weight: float = 1.0) -> Any:
        """Concatenate the transformed data.

        :param original_data: The first input data.
        :param data_to_concat: The second input data.
        :param weight: Weight of data to concat
        :return: The concatenated data.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not implement concat method.")


@dataclass
class SequentialSystem(Base):
    """The SequentialSystem class is the base class for all systems.

    Parameters:
    - steps (list[_Base]): The steps in the system.

    Methods:
    .. code-block:: python
        def get_hash(self) -> str:
            # Get the hash of the block.

        def get_parent(self) -> Any:
            # Get the parent of the block.

        def get_children(self) -> list[Any]:
            # Get the children of the block

        def save_to_html(self, file_path: Path) -> None:
            # Save html format to file_path
    """

    steps: list[Base] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Post init function of _System class."""
        super().__post_init__()

        # Set parent and children
        for step in self.steps:
            step.set_parent(self)

        self.set_children(self.steps)

    def get_steps(self) -> list[Base]:
        """Return list of steps of _ParallelSystem.

        :return: List of steps
        """
        return self.steps

    def set_hash(self, prev_hash: str) -> None:
        """Set the hash of the system.

        :param prev_hash: The hash of the previous block.
        """
        self._hash = prev_hash

        # Set hash of each step using previous hash and then update hash with last step
        for step in self.steps:
            step.set_hash(self.get_hash())
            self._hash = step.get_hash()
