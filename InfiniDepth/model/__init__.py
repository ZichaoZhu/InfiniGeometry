from .registry import MODEL_REGISTRY, register_model
from .model import (
    DisparityAlignment,
    InfiniDepth,
    InfiniDepth_DepthSensor,
    InfiniDepthEncoding,
    align_reference_disparity,
)
from .ssr import InfiniDepthSSR, InfiniDepthSSROutput
from .ssr_geometry import InfiniDepthSSRInputs, build_ssr_inputs

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "InfiniDepth",
    "InfiniDepth_DepthSensor",
    "InfiniDepthEncoding",
    "DisparityAlignment",
    "align_reference_disparity",
    "InfiniDepthSSR",
    "InfiniDepthSSROutput",
    "InfiniDepthSSRInputs",
    "build_ssr_inputs",
]


def __getattr__(name):
    if name in {"InfiniDepth", "InfiniDepth_DepthSensor"}:
        from .model import InfiniDepth, InfiniDepth_DepthSensor

        return {
            "InfiniDepth": InfiniDepth,
            "InfiniDepth_DepthSensor": InfiniDepth_DepthSensor,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
