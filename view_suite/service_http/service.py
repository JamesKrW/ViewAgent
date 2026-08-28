# service.py
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from .handler import BaseHandler, MyHandler
from .multipart import encode_multipart, parse_meta_field, read_images

# ----------------------------
# Config (env-driven)
# ----------------------------
API_KEY = os.getenv("UNIFIED_API_KEY", "")  # Empty => no auth required
MAX_INFLIGHT = int(os.getenv("UNIFIED_MAX_INFLIGHT", "0"))  # 0 => unlimited
ADMIT_TIMEOUT = float(os.getenv("UNIFIED_ADMIT_TIMEOUT", "2.0"))  # seconds

# Response image encoding (keep it simple and consistent)
IMAGE_FORMAT = os.getenv("UNIFIED_IMAGE_FORMAT", "PNG")
IMAGE_MIME = os.getenv("UNIFIED_IMAGE_MIME", "image/png")


def _auth(request: Request) -> None:
    """
    Optional API key auth.

    Accepts:
      - Query param: ?token=...
      - Header: X-API-Key: ...

    If UNIFIED_API_KEY is empty, auth is disabled.
    """
    if not API_KEY:
        return
    token = request.query_params.get("token") or request.headers.get("x-api-key")
    if token != API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")


def build_app(handler: BaseHandler) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # asyncio synchronization primitives must be created from the event
        # loop that will use them.  build_app() runs before uvicorn creates its
        # serving loop; a module-global Semaphore can eventually remain bound
        # to a stale loop and turn every request into HTTP 500.
        app.state.request_semaphore = (
            asyncio.Semaphore(MAX_INFLIGHT) if MAX_INFLIGHT > 0 else None
        )
        try:
            yield
        finally:
            # Allow handler to clean up resources (GPU contexts, threads, etc.)
            await handler.aclose()

    app = FastAPI(lifespan=lifespan)
    app.state.handler = handler

    @app.get("/health")
    async def health():
        return {
            "ok": True,
            "max_inflight": MAX_INFLIGHT if MAX_INFLIGHT > 0 else "unlimited",
            "image_format": IMAGE_FORMAT,
            "image_mime": IMAGE_MIME,
        }

    @app.post("/render")
    async def render(
        request: Request,
        meta: str | None = Form(default=None),
        images: list[UploadFile] | None = File(default=None),  # noqa: B008
    ):
        """
        Request (recommended):
            Content-Type: multipart/form-data
            - meta: JSON string (optional)
            - images: repeated file field (optional, 0..n)

        Response:
            Content-Type: multipart/mixed; boundary="..."
            - meta part: application/json; charset=utf-8
            - images parts: image/* (repeated)
        """
        _auth(request)

        # Optional global concurrency control (limits in-flight requests)
        semaphore = app.state.request_semaphore
        acquired = False
        if semaphore is not None:
            try:
                # ADMIT_TIMEOUT bounds how long a request waits to be admitted.
                await asyncio.wait_for(semaphore.acquire(), timeout=ADMIT_TIMEOUT)
                acquired = True
            except asyncio.TimeoutError:
                raise HTTPException(status_code=503, detail="server busy")

        try:
            meta_dict = parse_meta_field(meta)
            img_list = await read_images(images)

            # User-defined business logic
            result = await app.state.handler.handle(meta_dict, img_list)

            encode_format = result.image_format or IMAGE_FORMAT
            encode_mime = result.image_mime or IMAGE_MIME
            boundary, body = encode_multipart(
                result.meta or {},
                result.images or [],
                encoded_images=result.encoded_images or None,
                image_format=encode_format,
                image_mime=encode_mime,
            )

            return Response(
                content=body,
                media_type=f'multipart/mixed; boundary="{boundary}"',
            )
        finally:
            if acquired and semaphore is not None:
                semaphore.release()

    return app


# Default app instance (uvicorn can import service:app)
app = build_app(MyHandler())
