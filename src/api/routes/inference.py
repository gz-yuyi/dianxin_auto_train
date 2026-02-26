from fastapi import APIRouter, HTTPException

from src.inference.service import get_inference_manager
from src.schemas import (
    InferenceServiceStatusResponse,
    LoraModelLoadRequest,
    LoraModelLoadResponse,
    LoraModelUnloadRequest,
    LoraModelUnloadResponse,
    LoraPredictRequest,
    LoraPredictResponse,
    ModelInfo,
    ModelListResponse,
    ModelQueryRequest,
    WorkerStatus,
)


router = APIRouter(prefix="/inference", tags=["推理服务"])


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


@router.get(
    "/models",
    response_model=ModelListResponse,
    summary="获取模型列表",
    description="获取所有可用模型列表及其加载状态。",
    response_description="模型列表",
)
def list_models() -> ModelListResponse:
    """获取所有可用模型列表及其状态"""
    manager = get_inference_manager()
    models_data = manager.list_models()
    models = [ModelInfo(**m) for m in models_data]
    loaded_count = sum(1 for m in models if m.status == "loaded")
    return ModelListResponse(models=models, total=len(models), loaded_count=loaded_count)


@router.post(
    "/models/query",
    response_model=ModelListResponse,
    summary="查询模型",
    description="根据模型 ID 列表批量查询模型信息。",
    response_description="查询结果",
)
def query_models(payload: ModelQueryRequest) -> ModelListResponse:
    """根据模型ID列表查询模型"""
    if not payload.model_ids:
        raise HTTPException(status_code=400, detail="model_ids must not be empty")
    manager = get_inference_manager()
    models_data = manager.query_models(payload.model_ids)
    models = [ModelInfo(**m) for m in models_data]
    loaded_count = sum(1 for m in models if m.status == "loaded")
    return ModelListResponse(models=models, total=len(models), loaded_count=loaded_count)


@router.get(
    "/status",
    response_model=InferenceServiceStatusResponse,
    summary="获取服务状态",
    description="获取推理服务状态，包括 Worker 状态、显存信息和待处理请求数。",
    response_description="服务状态详情",
)
def get_service_status() -> InferenceServiceStatusResponse:
    """获取推理服务状态，包括Worker和显存信息"""
    manager = get_inference_manager()
    status_data = manager.get_service_status()
    workers = [WorkerStatus(**w) for w in status_data["workers"]]
    return InferenceServiceStatusResponse(
        service_status=status_data["service_status"],
        workers=workers,
        total_workers=status_data["total_workers"],
        loaded_models_count=status_data["loaded_models_count"],
        pending_requests=status_data["pending_requests"],
    )
