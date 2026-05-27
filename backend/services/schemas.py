from pydantic import BaseModel, Field
from typing import Literal

class PredictRequest(BaseModel):
    code: str = Field(..., description="4 位數台股代號，例如 2303")

class StockStatus(BaseModel):
    code: str
    name: str
    is_trained: bool

class TrainedStocksResponse(BaseModel):
    stocks: list[StockStatus]

class ErrorResponse(BaseModel):
    success: Literal[False]
    error_type: str
    message: str
    code: str | None = None

class UntrainedResponse(BaseModel):
    success: Literal[True]
    code: str
    name: str
    is_trained: Literal[False]
    message: str

class PredictSuccessResponse(BaseModel):
    success: Literal[True]
    code: str
    name: str
    is_trained: Literal[True]
    buy_count: int
    total_models: int
    recommendation: str
    summary: str
    models: dict[str, str]
