from pathlib import Path

import numpy as np
import pytest
from epochlib.pipeline import Transformer

from epochlib.transformation import TransformationPipeline
from epochlib.transformation import TransformationBlock
from tests.constants import TEMP_DIR


class ExampleTransformationBlock(TransformationBlock):
    def log_to_debug(self, message):
        pass

    def log_to_terminal(self, message):
        pass

    def custom_transform(self, x, **transform_args):
        if x is None:
            return None
        return x * 2


class CustomTransformationPipeline(TransformationPipeline):
    def log_to_debug(self, message: str) -> None:
        return None

    def log_to_terminal(self, message: str) -> None:
        return None


class TestTransformationPipeline:
    cache_path = TEMP_DIR

    @pytest.fixture(autouse=True)
    def run_always(self, setup_temp_dir):
        pass

    def test_transformation_pipeline_init(self):
        tp = CustomTransformationPipeline()
        assert tp.steps is not None

    def test_transformation_pipeline_transform(self):
        tp = CustomTransformationPipeline()
        x = 1
        assert tp.transform(x) == x
        assert tp.transform(x, transform_args={"a": 1}) == x
        assert tp.transform(x, transform_args={"a": 1, "b": 2}) == x

    def test_transformation_pipeline_with_steps(self):
        t1 = ExampleTransformationBlock()
        t2 = ExampleTransformationBlock()
        tp = CustomTransformationPipeline(steps=[t1, t2])

        assert tp.transform(None) is None

    def test_transformation_pipeline_with_cache(self):
        t1 = ExampleTransformationBlock()
        t2 = ExampleTransformationBlock()

        tp = CustomTransformationPipeline(steps=[t1, t2])

        cache_args = {
            "output_data_type": "numpy_array",
            "storage_type": ".npy",
            "storage_path": f"{self.cache_path}",
        }

        assert tp.transform(np.array([1]), cache_args=cache_args) == np.array([4])
        assert tp.transform(np.array([1]), cache_args=cache_args) == np.array([4])

    def test_transformation_pipeline_with_halfway_cache(self):
        t1 = ExampleTransformationBlock()
        t2 = ExampleTransformationBlock()

        tp1 = CustomTransformationPipeline(steps=[t1, t2])
        tp2 = CustomTransformationPipeline(steps=[t1, t2])

        transform_args = {
            "ExampleTransformationBlock": {
                "cache_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": f"{self.cache_path}",
                }
            }
        }

        assert tp1.transform(np.array([1]), **transform_args) == np.array([4])
        assert tp2.transform(np.array([1]), **transform_args) == np.array([4])

    def test_transformation_pipeline_with_halfway_cache_no_step_cache_args(self):
        t1 = ExampleTransformationBlock()
        t2 = ExampleTransformationBlock()

        tp1 = CustomTransformationPipeline(steps=[t1, t2])
        tp2 = CustomTransformationPipeline(steps=[t1, t2])

        transform_args = {
            "ExampleTransformationBlock": {
                "wrong_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": f"{self.cache_path}",
                }
            }
        }

        assert tp1.transform(np.array([1]), **transform_args) == np.array([4])
        assert tp2.transform(np.array([1]), **transform_args) == np.array([4])

    def test_transformation_pipeline_with_halfway_cache_not_instance_cacher(self):
        class ImplementedTransformer(Transformer):
            def transform(self, x, **transform_args):
                return x * 2

        t1 = ExampleTransformationBlock()
        t2 = ImplementedTransformer()

        tp1 = CustomTransformationPipeline(steps=[t1, t2])
        tp2 = CustomTransformationPipeline(steps=[t1, t2])

        transform_args = {
            "ExampleTransformationBlock": {
                "cache_args": {
                    "output_data_type": "numpy_array",
                    "storage_type": ".npy",
                    "storage_path": f"{self.cache_path}",
                }
            }
        }

        assert tp1.transform(np.array([1]), **transform_args) == np.array([4])
        assert tp2.transform(np.array([2]), **transform_args) == np.array([4])
