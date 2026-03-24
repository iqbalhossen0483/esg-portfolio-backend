"""Standardized API response helpers."""

from fastapi.responses import JSONResponse


def success_response(data=None, message: str = "Success", status_code: int = 200):
    """Standard success response: {success: true, message, data}"""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": True,
            "message": message,
            "data": data,
        },
    )


def error_response(
    message: str = "An error occurred",
    details=None,
    status_code: int = 500,
):
    """Standard error response: {success: false, message, details}"""
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "details": details,
        },
    )
