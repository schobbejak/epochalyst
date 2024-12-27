"""EnsemblePipeline for ensebling multiple ModelPipelines."""

from typing import Any

from epochlib.caching import CacheArgs
from epochlib.pipeline import ParallelTrainingSystem


class EnsemblePipeline(ParallelTrainingSystem):
    """EnsemblePipeline is the class used to create the pipeline for the model.

    :param steps: Trainers to ensemble
    """

    def get_x_cache_exists(self, cache_args: CacheArgs) -> bool:
        """Get status of x.

        :param cache_args: Cache arguments
        :return: Whether cache exists
        """
        if len(self.steps) == 0:
            return False

        return all(step.get_x_cache_exists(cache_args) for step in self.steps)

    def get_y_cache_exists(self, cache_args: CacheArgs) -> bool:
        """Get status of y cache.

        :param cache_args: Cache arguments
        :return: Whether cache exists
        """
        if len(self.steps) == 0:
            return False

        return all(step.get_y_cache_exists(cache_args) for step in self.steps)

    def concat(self, original_data: Any, data_to_concat: Any, weight: float = 1.0) -> Any:
        """Concatenate the trained data.

        :param original_data: First input data
        :param data_to_concat: Second input data
        :param weight: Weight of data to concat
        :return: Concatenated data
        """
        if original_data is None:
            if data_to_concat is None:
                return None
            return data_to_concat * weight

        return original_data + data_to_concat * weight
