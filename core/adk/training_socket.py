"""Socket.IO handlers for training pipeline real-time progress.

Handlers are registered under the /training namespace so that the
chat namespace can have its own auth/connect handler without conflict.
"""

from socketio.exceptions import ConnectionRefusedError

from core.auth.security import decode_token
from core.logging import get_logger

log = get_logger(__name__)

NAMESPACE = "/training"


def _extract_token(auth, environ) -> str | None:
    if auth and isinstance(auth, dict):
        token = auth.get("token")
        if token:
            return token
    query = environ.get("QUERY_STRING", "")
    for param in query.split("&"):
        if param.startswith("token="):
            return param.split("=", 1)[1]
    return None


def register_training_handlers(sio):
    """Register Socket.IO event handlers for training progress."""

    @sio.on("connect", namespace=NAMESPACE)
    async def on_connect(sid, environ, auth=None):
        token = _extract_token(auth, environ)
        if not token:
            raise ConnectionRefusedError("authentication required")

        try:
            payload = decode_token(token)
        except Exception:
            raise ConnectionRefusedError("invalid token")

        if payload.get("role") != "admin":
            raise ConnectionRefusedError("admin role required")

        await sio.save_session(sid, {
            "user_id": payload.get("user_id"),
            "role": payload.get("role"),
        }, namespace=NAMESPACE)
        log.info("training socket connected sid=%s user_id=%s",
                 sid, payload.get("user_id"))

    @sio.on("training:subscribe", namespace=NAMESPACE)
    async def on_subscribe(sid, data):
        session = await sio.get_session(sid, namespace=NAMESPACE)
        if session.get("role") != "admin":
            await sio.emit("error", {"message": "Admin access required"},
                           room=sid, namespace=NAMESPACE)
            return

        job_id = data.get("job_id")
        if job_id:
            sio.enter_room(sid, f"training:{job_id}", namespace=NAMESPACE)
            await sio.emit("training:subscribed", {"job_id": job_id},
                           room=sid, namespace=NAMESPACE)
