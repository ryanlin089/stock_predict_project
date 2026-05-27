import logging
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from services.constants import SUPPORTED_STOCKS
from services.errors import ErrorType, StockPredictionError
from services.prediction_service import predict_stock
from services.schemas import PredictRequest
from services.stock_service import (
    check_model_files_exist,
    get_trained_stocks,
    get_stock_name,
    is_valid_code_format,
    normalize_code,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stock Prediction API", version="1.0.0")


def get_allowed_origins() -> list[str]:
    origins = os.getenv("ALLOWED_ORIGINS")
    if origins:
        return [origin.strip().rstrip("/") for origin in origins.split(",") if origin.strip()]

    return ["http://localhost:3000", "http://127.0.0.1:3000"]


allowed_origins = get_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials="*" not in allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"name": "Stock Prediction API", "status": "ok"}


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.get("/api/trained-stocks")
def trained_stocks():
    return {"stocks": get_trained_stocks()}


@app.post("/api/predict")
def predict(request: PredictRequest):
    code = normalize_code(request.code)

    if not is_valid_code_format(code):
        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error_type": ErrorType.INVALID_CODE_FORMAT.value,
                "message": "股票代號格式錯誤，請輸入 4 位數字",
            },
        )

    name = get_stock_name(code)
    if name is None:
        return {
            "success": False,
            "error_type": ErrorType.UNSUPPORTED_STOCK.value,
            "code": code,
            "message": "目前尚未支援此股票代號",
        }

    if not check_model_files_exist(code):
        return {
            "success": True,
            "code": code,
            "name": name,
            "is_trained": False,
            "message": f"尚未訓練 {code} {name} 的模型，請先完成模型訓練後再進行預測",
        }

    try:
        return predict_stock(code)
    except StockPredictionError as exc:
        logger.exception("Stock prediction error: %s", exc.message)
        status_code = 500
        if exc.error_type in {ErrorType.DATA_FILE_NOT_FOUND, ErrorType.MODEL_FILE_NOT_FOUND}:
            status_code = 500
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "error_type": exc.error_type.value,
                "code": code,
                "message": exc.message,
            },
        )
    except Exception as exc:
        logger.exception("Unexpected internal server error")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error_type": ErrorType.INTERNAL_SERVER_ERROR.value,
                "code": code,
                "message": f"後端內部錯誤：{exc}",
            },
        )


@app.exception_handler(HTTPException)
async def http_exception_handler(_, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": ErrorType.INTERNAL_SERVER_ERROR.value,
            "message": str(exc.detail),
        },
    )
