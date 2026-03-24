"""Socket.IO handlers for training pipeline real-time progress."""

from core.auth.security import decode_token


def register_training_handlers(sio):
    """Register Socket.IO event handlers for training progress."""

    @sio.on("connect")
    async def on_connect(sid, environ, auth=None):
        """Authenticate Socket.IO connection via JWT token."""
        # Token can be sent as query param or auth header
        token = None
        if auth and isinstance(auth, dict):
            token = auth.get("token")

        if not token:
            # Try query string
            query = environ.get("QUERY_STRING", "")
            for param in query.split("&"):
                if param.startswith("token="):
                    token = param.split("=", 1)[1]
                    break

        if token:
            try:
                payload = decode_token(token)
                await sio.save_session(sid, {
                    "user_id": payload.get("user_id"),
                    "role": payload.get("role"),
                })
            except Exception:
                pass  # Allow connection but without auth context

    @sio.on("training:subscribe")
    async def on_subscribe(sid, data):
        """Admin subscribes to training updates for a specific job."""
        session = await sio.get_session(sid)
        if session.get("role") != "admin":
            await sio.emit("error", {"message": "Admin access required"}, room=sid)
            return

        job_id = data.get("job_id")
        if job_id:
            sio.enter_room(sid, f"training:{job_id}")
            await sio.emit("training:subscribed", {"job_id": job_id}, room=sid)
