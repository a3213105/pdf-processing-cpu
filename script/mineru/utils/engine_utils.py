#  Copyright (c) Opendatalab. All rights reserved.
from loguru import logger

from mineru.utils.check_sys_env import is_mac_os_version_supported, is_windows_environment, is_mac_environment, \
    is_linux_environment


def get_vlm_engine(inference_engine: str, is_async: bool = False) -> str:
    """
    Automatically select or verify VLM inference engine

    Args:
        inference_engine: the specified engine name or 'auto' Make automatic selection
        is_async: Whether to use an asynchronous engine (only for vllm efficient)

    Returns:
        Final chosen engine name
    """
    if inference_engine == 'auto':
        # Automatic engine selection based on operating system
        if is_windows_environment():
            inference_engine = _select_windows_engine()
        elif is_linux_environment():
            inference_engine = _select_linux_engine(is_async)
        elif is_mac_environment():
            inference_engine = _select_mac_engine()
        else:
            logger.warning("Unknown operating system, falling back to transformers")
            inference_engine = 'transformers'

    formatted_engine = _format_engine_name(inference_engine)
    logger.info(f"Using {formatted_engine} as the inference engine for VLM.")
    return formatted_engine


def _select_windows_engine() -> str:
    """Windows Platform engine selection"""
    try:
        import lmdeploy
        return 'lmdeploy'
    except ImportError:
        return 'transformers'


def _select_linux_engine(is_async: bool) -> str:
    """Linux Platform engine selection"""
    try:
        import vllm
        return 'vllm-async' if is_async else 'vllm'
    except ImportError:
        try:
            import lmdeploy
            return 'lmdeploy'
        except ImportError:
            return 'transformers'


def _select_mac_engine() -> str:
    """macOS Platform engine selection"""
    try:
        from mlx_vlm import load as mlx_load
        if is_mac_os_version_supported():
            return 'mlx'
        else:
            return 'transformers'
    except ImportError:
        return 'transformers'


def _format_engine_name(engine: str) -> str:
    """Unified formatting engine name"""
    if engine != 'transformers':
        return f"{engine}-engine"
    return engine
