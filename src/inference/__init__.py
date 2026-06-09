def get_inference_manager():
    from src.inference.service import get_inference_manager as _get_inference_manager

    return _get_inference_manager()

__all__ = ["get_inference_manager"]
