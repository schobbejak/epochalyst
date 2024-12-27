"""Defines function for importing onnxruntime."""

from typing import Any


def _get_onnxrt() -> Any:
    """Return onnxruntime."""
    try:
        import onnxruntime as onnxrt

    except ImportError:
        raise ImportError(
            "If you want to compile models to ONNX format install onnxruntime",
        ) from None

    else:
        return onnxrt


def _get_openvino() -> Any:
    """Return openvino."""
    try:
        import openvino

    except ImportError:
        raise ImportError(
            "If you want to compile models to OpenVINO format install openvino",
        ) from None

    else:
        return openvino
