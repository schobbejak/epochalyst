"""TransformationPipeline that extends from TransformingSystem, Cacher and _Logger."""

from dataclasses import dataclass
from typing import Any

from epochlib.caching.cacher import CacheArgs, Cacher
from epochlib.pipeline import TransformingSystem, TransformType


@dataclass
class TransformationPipeline(TransformingSystem, Cacher):
    """TransformationPipeline is the class used to create the pipeline for the transformation of the data.

    ### Parameters:
    - `steps` (List[Union[Transformer, TransformationPipeline]]): The steps to transform the data. Can be a list of any Transformer type.
    - `title` (str): The title of the pipeline. (Default: "Transformation Pipeline")

    Methods
    -------
    .. code-block:: python
        @abstractmethod
        def log_to_terminal(self, message: str) -> None: # Log the message to the terminal.

        @abstractmethod
        def log_to_debug(self, message: str) -> None: # Log the message to the debug file.

        @abstractmethod
        def log_to_warning(self, message: str) -> None: # Log the message to the warning file.

        @abstractmethod
        def log_to_external(self, message: str) -> None: # Log the message to an external file.

        @abstractmethod
        def log_section_separator(self, title: str) -> None: # Log a section separator to the terminal.

        @abstractmethod
        def external_define_metric(self, metric: str, metric_type: str) -> None: # Define a metric to be logged to an external file.

        def transform(self, data: Any, cache_args: dict[str, Any] = {}, **transform_args: Any) -> Any: # Transform the input data.

        def get_hash(self) -> str: # Get the hash of the pipeline.

    Usage:
    .. code-block:: python
        from epochlib.pipeline.model.transformation import TransformationPipeline

        class MyTransformationPipeline(TransformationPipeline):
            def log_to_terminal(self, message: str) -> None:
                print(message)

            ....

        step1 = MyTransformer1()
        step2 = MyTransformer2()
        pipeline = MyTransformationPipeline(steps=[step1, step2])

        data = pipeline.transform(data)
    """

    title: str = "Transformation Pipeline"  # The title of the pipeline since transformation pipeline can be used for multiple purposes. (Feature, Label, etc.)

    def transform(self, data: Any, cache_args: CacheArgs | None = None, **transform_args: Any) -> Any:
        """Transform the input data.

        :param data: The input data.
        :param cache_args: The cache arguments.
        :return: The transformed data.
        """
        if cache_args and self.cache_exists(self.get_hash(), cache_args):
            self.log_to_terminal(
                f"Cache exists for {self.title} with hash: {self.get_hash()}. Using the cache.",
            )
            return self._get_cache(self.get_hash(), cache_args)

        if self.get_steps():
            self.log_section_separator(self.title)

        self.all_steps = self.get_steps()

        # Furthest step
        for i, step in enumerate(self.get_steps()):
            # Check if step is instance of Cacher and if cache_args exists
            if not isinstance(step, Cacher) or not isinstance(step, TransformType):
                self.log_to_debug(f"{step} is not instance of Cacher or TransformType")
                continue

            step_args = transform_args.get(step.__class__.__name__, None)
            if step_args is None:
                self.log_to_debug(f"{step} is not given transform_args")
                continue

            step_cache_args = step_args.get("cache_args", None)
            if step_cache_args is None:
                self.log_to_debug(f"{step} is not given cache_args")
                continue

            step_cache_exists = step.cache_exists(step.get_hash(), step_cache_args)
            if step_cache_exists:
                self.log_to_debug(
                    f"Cache exists for {step}, moving index of steps to {i}",
                )
                self.steps = self.all_steps[i:]

        data = super().transform(data, **transform_args)

        if cache_args:
            self.log_to_terminal(f"Storing cache for pipeline to {cache_args['storage_path']}")
            self._store_cache(self.get_hash(), data, cache_args)

        # Set steps to original in case class is called again
        self.steps = self.all_steps

        return data
