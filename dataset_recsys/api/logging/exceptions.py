# exceptions.py

from typing import Any, List, Optional

from fastapi import HTTPException, status
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    code: int
    error: str


class ValidationErrorDetail(BaseModel):
    Key: str
    Value: List[str]


class ValidationErrorResponse(BaseModel):
    code: int
    error: str
    message: List[ValidationErrorDetail]


class FailedDependencyMessage(BaseModel):
    statusCode: int
    source: str
    correlationId: Optional[str] = None
    payload: Optional[Any] = None


class FailedDependencyResponse(BaseModel):
    code: int
    error: str
    message: FailedDependencyMessage


class FailedDependencyException(HTTPException):
    def __init__(
        self,
        source: str,
        status_code: int,
        detail: str,
        correlation_id: Optional[str] = None,
        payload: Optional[Any] = None,
    ):
        self.source = source
        self.downstream_status_code = status_code
        self.correlation_id = correlation_id
        self.downstream_payload = payload
        super().__init__(status_code=status.HTTP_424_FAILED_DEPENDENCY, detail=detail)