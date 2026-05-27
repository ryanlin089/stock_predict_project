from enum import Enum

class ErrorType(str, Enum):
    INVALID_CODE_FORMAT = "INVALID_CODE_FORMAT"
    UNSUPPORTED_STOCK = "UNSUPPORTED_STOCK"
    UNTRAINED_MODEL = "UNTRAINED_MODEL"
    DATA_FILE_NOT_FOUND = "DATA_FILE_NOT_FOUND"
    MODEL_FILE_NOT_FOUND = "MODEL_FILE_NOT_FOUND"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    PREDICTION_FAILED = "PREDICTION_FAILED"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"

class StockPredictionError(Exception):
    def __init__(self, error_type: ErrorType, message: str):
        self.error_type = error_type
        self.message = message
        super().__init__(message)
