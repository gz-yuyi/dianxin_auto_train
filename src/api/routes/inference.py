from fastapi import APIRouter, HTTPException

from src.inference.service import get_inference_manager
from src.schemas import (
    LoraModelLoadRequest,
    LoraModelLoadResponse,
    LoraModelUnloadRequest,
    LoraModelUnloadResponse,
    LoraPredictRequest,
    LoraPredictResponse,
)


router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/models/load", response_model=LoraModelLoadResponse)
def load_lora_model(payload: LoraModelLoadRequest) -> LoraModelLoadResponse:
    manager = get_inference_manager()
    try:
        model_id = manager.load_model(payload.model_dir, payload.max_length)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LoraModelLoadResponse(model_id=model_id, status="loaded", message="model loaded")


@router.post("/models/unload", response_model=LoraModelUnloadResponse)
def unload_lora_model(payload: LoraModelUnloadRequest) -> LoraModelUnloadResponse:
    manager = get_inference_manager()
    try:
        manager.unload_model(payload.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    return LoraModelUnloadResponse(model_id=payload.model_id, status="unloaded", message="model unloaded")


@router.post("/predict", response_model=LoraPredictResponse)
def predict_lora(payload: LoraPredictRequest) -> LoraPredictResponse:
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    manager = get_inference_manager()
    try:
        future = manager.enqueue(payload.model_id, payload.texts, payload.top_n)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    result = future.result()
    return LoraPredictResponse(**result)
