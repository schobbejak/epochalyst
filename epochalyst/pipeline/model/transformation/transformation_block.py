from typing import Any
from agogos.transforming import Transformer
from epochalyst._core._logging._logger import _Logger
from epochalyst._core._caching._cacher import _Cacher
from abc import abstractmethod


class TransformationBlock(Transformer, _Cacher, _Logger):
    """The transformation block is a flexible block that allows for transformation of any data.

    cache_args can be passed to the transform method to cache the output of the transformation block. The cache_args are as follows:
    - 'output_data_type': The type of the output data. (options: dask_array, numpy_array, pandas_dataframe, dask_dataframe)
    - 'storage_type': The type of the storage. (options: .npy, .parquet, .csv, .npy_stack)
    - 'storage_path': The path to the storage.
    - example: cache_args = {
        "output_data_type": "numpy_array",
        "storage_type": ".npy",
        "storage_path": "data/processed"
    }

    ### Methods:
    ```python
    @abstractmethod
    def custom_transform(self, data: Any, **transform_args) -> Any: # Custom transformation implementation

    @abstractmethod
    def log_to_terminal(self, message: str) -> None: # Logs to terminal if implemented

    @abstractmethod
    def log_to_debug(self, message: str) -> None: # Logs to debugger if implemented

    @abstractmethod
    def log_to_warning(self, message: str) -> None: # Logs to warning if implemented

    @abstractmethod
    def log_to_external(self, message: dict[str, Any], **kwargs: Any) -> None: # Logs to external site

    @abstractmethod
    def external_define_metric(self, metric: str, metric_type: str) -> None: # Defines an external metric

    def transform(self, data: Any, cache_args: dict[str, Any] = {}, **transform_args: Any) -> Any: # Applies caching and calls custom_transform
    ```

    ### Usage:
    ```python
        from epochalyst.pipeline.model.transformation.transformation_block import TransformationBlock

        class CustomTransformationBlock(TransformationBlock):
            def custom_transform(self, data: Any) -> Any:
                return data

            ....

        custom_transformation_block = CustomTransformationBlock()

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": "data/processed",
        }

        data = custom_transformation_block.transform(data, cache=cache_args)
    ```
    """

    def transform(
        self, data: Any, cache_args: dict[str, Any] = {}, **transform_args: Any
    ) -> Any:
        """Transform the input data using a custom method.

        :param data: The input data.
        :param cache_args: The cache arguments.
        - 'output_data_type': The type of the output data. (options: dask_array, numpy_array, pandas_dataframe, dask_dataframe)
        - 'storage_type': The type of the storage. (options: .npy, .parquet, .csv, .npy_stack)
        - 'storage_path': The path to the storage.
        :return: The transformed data.
        """

        if cache_args and self._cache_exists(
            name=self.get_hash(), cache_args=cache_args
        ):
            return self._get_cache(name=self.get_hash(), cache_args=cache_args)

        data = self.custom_transform(data, **transform_args)

        self._store_cache(
            name=self.get_hash(), data=data, cache_args=cache_args
        ) if cache_args else None

        return data

    @abstractmethod
    def custom_transform(self, data: Any, **transform_args: Any) -> Any:
        """Transform the input data using a custom method.

        :param data: The input data.
        :return: The transformed data.
        """
        raise NotImplementedError(
            f"Custom transform method not implemented for {self.__class__}"
        )
