"""
api/errors.py
--------------
The API's error contract (task A2). Every failure response is:

    {"code": ..., "message": ..., "retryable": bool, "request_id": ...}

`message` is always a plain sentence written here or by our own code — never
an exception string, stack trace or provider text. Those go to the logs only,
tagged with the same request_id so a user report can be traced.
"""

import contextvars
import uuid
from enum import Enum

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agents.langgraph_agent.utils.utils import logger

REQUEST_ID_HEADER = "X-Request-ID"

# Set per request by the middleware, readable anywhere (logs, controllers).
request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class ErrorCode(str, Enum):
    provider_unavailable = "provider_unavailable"
    rate_limited = "rate_limited"
    no_context = "no_context"
    timeout = "timeout"
    invalid_file = "invalid_file"
    invalid_request = "invalid_request"
    unauthorized = "unauthorized"
    forbidden = "forbidden"
    not_found = "not_found"
    internal = "internal"


# code → (HTTP status, default message, retryable)
_DEFAULTS: dict[ErrorCode, tuple[int, str, bool]] = {
    ErrorCode.provider_unavailable: (503, "The assistant is unavailable right now. Try again in a moment.", True),
    ErrorCode.rate_limited: (429, "Too many requests. Wait a moment and try again.", True),
    ErrorCode.no_context: (404, "I couldn't find that in your documents.", False),
    ErrorCode.timeout: (504, "That took too long to answer. Try again.", True),
    ErrorCode.invalid_file: (400, "This file can't be processed.", False),
    ErrorCode.invalid_request: (400, "The request was not valid.", False),
    ErrorCode.unauthorized: (401, "Your session has ended. Sign in again.", False),
    ErrorCode.forbidden: (403, "You don't have permission to do that.", False),
    ErrorCode.not_found: (404, "That item doesn't exist or was removed.", False),
    ErrorCode.internal: (500, "Something went wrong on our side. Try again.", True),
}

_STATUS_TO_CODE = {
    400: ErrorCode.invalid_request,
    401: ErrorCode.unauthorized,
    403: ErrorCode.forbidden,
    404: ErrorCode.not_found,
    408: ErrorCode.timeout,
    413: ErrorCode.invalid_file,
    415: ErrorCode.invalid_file,
    422: ErrorCode.invalid_request,
    429: ErrorCode.rate_limited,
    503: ErrorCode.provider_unavailable,
    504: ErrorCode.timeout,
}


class ApiError(Exception):
    """Raise anywhere in the API to return a contract error.

    `message` must be safe to show a user. Pass `log_detail` for anything
    internal you want in the logs instead.
    """

    def __init__(self, code: ErrorCode, message: str | None = None, *, log_detail: str | None = None):
        status, default_message, retryable = _DEFAULTS[code]
        super().__init__(message or default_message)
        self.code = code
        self.status = status
        self.message = message or default_message
        self.retryable = retryable
        self.log_detail = log_detail


def error_response(code: ErrorCode, message: str | None = None) -> JSONResponse:
    status, default_message, retryable = _DEFAULTS[code]
    request_id = request_id_var.get()
    return JSONResponse(
        status_code=status,
        content={
            "code": code.value,
            "message": message or default_message,
            "retryable": retryable,
            "request_id": request_id,
        },
        headers={REQUEST_ID_HEADER: request_id},
    )


def install_error_handling(app: FastAPI) -> None:
    """Request-id middleware plus handlers that turn every failure into the contract."""

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        # Reuse the browser's id when it sends one (F3), otherwise mint one.
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if 8 <= len(incoming) <= 64 and incoming.isascii() else uuid.uuid4().hex
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        except Exception:
            # Exceptions raised past the routing layer (e.g. in other middleware).
            logger.exception("Unhandled error | request_id=%s", request_id)
            response = error_response(ErrorCode.internal)
        finally:
            request_id_var.reset(token)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        level = logger.error if exc.status >= 500 else logger.info
        level("ApiError %s | %s | request_id=%s | %s",
              exc.code.value, request.url.path, request_id_var.get(), exc.log_detail or exc.message)
        return error_response(exc.code, exc.message)

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException):
        code = _STATUS_TO_CODE.get(exc.status_code, ErrorCode.internal)
        # 4xx details are sentences our own code wrote, so they may be shown.
        # 5xx details have historically carried str(e) — never show those.
        message = exc.detail if exc.status_code < 500 and isinstance(exc.detail, str) else None
        if exc.status_code >= 500:
            logger.error("HTTP %d | %s | request_id=%s | %s",
                         exc.status_code, request.url.path, request_id_var.get(), exc.detail)
        return error_response(code, message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError):
        logger.info("Validation error | %s | request_id=%s | %s",
                    request.url.path, request_id_var.get(), exc.errors())
        return error_response(ErrorCode.invalid_request, "Some required fields are missing or invalid.")

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        logger.exception("Unhandled error | %s | request_id=%s", request.url.path, request_id_var.get())
        return error_response(ErrorCode.internal)
