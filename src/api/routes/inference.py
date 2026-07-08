from fastapi import APIRouter, HTTPException

from src.config import is_inference_gateway_enabled
from src.inference.gateway import InferenceGatewayError, get_inference_gateway
from src.schemas import (
    InferenceServiceStatusResponse,
    LoraModelLoadRequest,
    LoraModelLoadResponse,
    LoraModelPublishRequest,
    LoraModelPublishResponse,
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


def _payload_dict(payload) -> dict:
    if hasattr(payload, "model_dump"):
        return payload.model_dump()
    return payload.dict()


def _raise_gateway_error(exc: InferenceGatewayError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


def _get_inference_manager():
    from src.inference.service import get_inference_manager

    return get_inference_manager()


def _is_model_state_error(exc: Exception) -> bool:
    from src.inference.service import InferenceModelStateError

    return isinstance(exc, InferenceModelStateError)


@router.post(
    "/models/load",
    response_model=LoraModelLoadResponse,
    summary="加载模型",
    description="将指定目录下的模型加载到推理服务中，并返回分配后的模型 ID。",
    response_description="模型加载结果",
)
def load_lora_model(payload: LoraModelLoadRequest) -> LoraModelLoadResponse:
    if is_inference_gateway_enabled():
        gateway = get_inference_gateway()
        try:
            result = gateway.load_model(_payload_dict(payload))
        except InferenceGatewayError as exc:
            _raise_gateway_error(exc)
        return LoraModelLoadResponse(**result)

    manager = _get_inference_manager()
    try:
        model_id = manager.load_model(payload.model_dir, payload.max_length)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LoraModelLoadResponse(model_id=model_id, status="loaded", message="model loaded")


@router.post(
    "/models/publish",
    response_model=LoraModelPublishResponse,
    summary="发布模型",
    description="将 artifacts 下指定模型目录发布为可预测状态；目录名通常为训练时的模型英文名；已加载模型可重新读取磁盘产物并刷新显存缓存。",
    response_description="模型发布结果",
)
def publish_lora_model(payload: LoraModelPublishRequest) -> LoraModelPublishResponse:
    load_payload = {"model_dir": payload.model_id, "max_length": payload.max_length}

    if is_inference_gateway_enabled():
        gateway = get_inference_gateway()
        if not payload.reload:
            try:
                query_result = gateway.query_models({"model_ids": [payload.model_id]})
            except InferenceGatewayError as exc:
                _raise_gateway_error(exc)
            if any(model.get("model_id") == payload.model_id and model.get("status") == "loaded" for model in query_result.get("models", [])):
                return LoraModelPublishResponse(
                    model_id=payload.model_id,
                    status="loaded",
                    message="model already published",
                )
        try:
            result = gateway.load_model(load_payload)
        except InferenceGatewayError as exc:
            _raise_gateway_error(exc)
        return LoraModelPublishResponse(
            model_id=result.get("model_id", payload.model_id),
            status="loaded",
            message="model published",
        )

    manager = _get_inference_manager()
    if not payload.reload:
        models = manager.query_models([payload.model_id])
        if any(model.get("model_id") == payload.model_id and model.get("status") == "loaded" for model in models):
            return LoraModelPublishResponse(
                model_id=payload.model_id,
                status="loaded",
                message="model already published",
            )
    try:
        model_id = manager.load_model(payload.model_id, payload.max_length)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return LoraModelPublishResponse(model_id=model_id, status="loaded", message="model published")


@router.post(
    "/models/unload",
    response_model=LoraModelUnloadResponse,
    summary="卸载模型",
    description="根据模型 ID 从推理服务中卸载已加载的模型。",
    response_description="模型卸载结果",
)
def unload_lora_model(payload: LoraModelUnloadRequest) -> LoraModelUnloadResponse:
    if is_inference_gateway_enabled():
        gateway = get_inference_gateway()
        try:
            result = gateway.unload_model(_payload_dict(payload))
        except InferenceGatewayError as exc:
            _raise_gateway_error(exc)
        return LoraModelUnloadResponse(**result)

    manager = _get_inference_manager()
    try:
        manager.unload_model(payload.model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    return LoraModelUnloadResponse(model_id=payload.model_id, status="unloaded", message="model unloaded")


@router.post(
    "/predict",
    response_model=LoraPredictResponse,
    summary="执行预测",
    description="使用指定模型对输入文本列表执行批量预测，并返回标签及概率信息。",
    response_description="预测结果",
)
def predict_lora(payload: LoraPredictRequest) -> LoraPredictResponse:
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must not be empty")
    if is_inference_gateway_enabled():
        gateway = get_inference_gateway()
        try:
            result = gateway.predict(_payload_dict(payload))
        except InferenceGatewayError as exc:
            _raise_gateway_error(exc)
        return LoraPredictResponse(**result)

    manager = _get_inference_manager()
    try:
        future = manager.enqueue(payload.model_id, payload.texts, payload.top_n)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        result = future.result()
    except Exception as exc:
        if _is_model_state_error(exc):
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        raise HTTPException(status_code=500, detail=str(exc)) from exc
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
    if is_inference_gateway_enabled():
        gateway = get_inference_gateway()
        try:
            result = gateway.list_models()
        except InferenceGatewayError as exc:
            _raise_gateway_error(exc)
        models = [ModelInfo(**m) for m in result["models"]]
        return ModelListResponse(
            models=models,
            total=result["total"],
            loaded_count=result["loaded_count"],
        )

    manager = _get_inference_manager()
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
    if is_inference_gateway_enabled():
        gateway = get_inference_gateway()
        try:
            result = gateway.query_models(_payload_dict(payload))
        except InferenceGatewayError as exc:
            _raise_gateway_error(exc)
        models = [ModelInfo(**m) for m in result["models"]]
        return ModelListResponse(
            models=models,
            total=result["total"],
            loaded_count=result["loaded_count"],
        )

    manager = _get_inference_manager()
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
    if is_inference_gateway_enabled():
        gateway = get_inference_gateway()
        status_data = gateway.get_service_status()
        workers = [WorkerStatus(**w) for w in status_data["workers"]]
        return InferenceServiceStatusResponse(
            service_status=status_data["service_status"],
            workers=workers,
            total_workers=status_data["total_workers"],
            loaded_models_count=status_data["loaded_models_count"],
            pending_requests=status_data["pending_requests"],
        )

    manager = _get_inference_manager()
    status_data = manager.get_service_status()
    workers = [WorkerStatus(**w) for w in status_data["workers"]]
    return InferenceServiceStatusResponse(
        service_status=status_data["service_status"],
        workers=workers,
        total_workers=status_data["total_workers"],
        loaded_models_count=status_data["loaded_models_count"],
        pending_requests=status_data["pending_requests"],
    )
