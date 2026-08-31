"""Generic FastAPI service for stateful VAGEN environments."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from vagen_agent.envs.remote.handler import BaseGymHandler, SessionNotFoundError
from vagen_agent.envs.remote.multipart_codec import decode_multipart, encode_multipart

LOGGER = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import Response
except ImportError as exc:  # client-only installs do not need FastAPI
    FastAPI = HTTPException = Request = Response = None  # type: ignore[assignment]
    _FASTAPI_IMPORT_ERROR: ImportError | None = exc
else:
    _FASTAPI_IMPORT_ERROR = None


class GymService:
    """Wrap a :class:`BaseGymHandler` with the VAGEN remote HTTP protocol."""

    def __init__(
        self,
        handler: BaseGymHandler,
        max_inflight: int = 0,
        admit_timeout: float = 5.0,
        api_key: str = "",
        image_format: str = "JPEG",
        image_mime: str = "image/jpeg",
        image_options: dict[str, Any] | None = None,
    ) -> None:
        if _FASTAPI_IMPORT_ERROR is not None:
            raise ImportError(
                "serving remote environments requires FastAPI; install fastapi and uvicorn"
            ) from _FASTAPI_IMPORT_ERROR
        if max_inflight < 0 or admit_timeout <= 0:
            raise ValueError(
                "max_inflight must be non-negative and admit_timeout positive"
            )
        self.handler = handler
        self.max_inflight = int(max_inflight)
        self.admit_timeout = float(admit_timeout)
        self.api_key = api_key or os.getenv("GYM_API_KEY", "")
        self.image_format = image_format
        self.image_mime = image_mime
        self.image_options = dict(image_options or {})
        if image_options is None and image_format.upper() in {"JPEG", "JPG"}:
            # 4:4:4 avoids chroma smearing around small coloured GUI text. Quality 90
            # remains much faster and smaller than PNG for normal screenshots.
            self.image_options = {"quality": 90, "subsampling": 0}
        self._semaphore = (
            asyncio.Semaphore(self.max_inflight) if self.max_inflight > 0 else None
        )

    def authenticate(self, request: Request) -> None:
        if not self.api_key:
            return
        token = request.query_params.get("token") or request.headers.get("x-api-key")
        if token != self.api_key:
            raise HTTPException(status_code=401, detail="unauthorized")

    async def acquire(self) -> bool:
        if self._semaphore is None:
            return False
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.admit_timeout
            )
            return True
        except TimeoutError as exc:
            raise HTTPException(status_code=503, detail="server busy") from exc

    def release(self) -> None:
        if self._semaphore is not None:
            self._semaphore.release()

    def handle_connect_error(self, error: Exception) -> None:
        if isinstance(error, RuntimeError) and "maximum active sessions" in str(error):
            raise HTTPException(status_code=503, detail=str(error)) from error
        LOGGER.exception("remote environment connect failed", exc_info=error)
        raise HTTPException(status_code=500, detail=str(error)) from error

    def handle_call_error(self, error: Exception) -> None:
        if isinstance(error, SessionNotFoundError):
            raise HTTPException(status_code=404, detail="session not found") from error
        if isinstance(error, TimeoutError):
            raise HTTPException(status_code=504, detail=str(error)) from error
        LOGGER.exception("remote environment call failed", exc_info=error)
        raise HTTPException(status_code=500, detail=str(error)) from error

    def _response(self, result: Any) -> Any:
        boundary, body = encode_multipart(
            result.data,
            result.images,
            image_format=self.image_format,
            image_mime=self.image_mime,
            image_options=self.image_options,
        )
        return Response(
            content=body, media_type=f'multipart/mixed; boundary="{boundary}"'
        )

    async def health(self) -> dict[str, Any]:
        return {
            "ok": True,
            "service": "gym-env-service",
            "protocol_version": 3,
            "max_inflight": self.max_inflight or "unlimited",
        }

    async def sessions(self, request: Request) -> dict[str, Any]:
        self.authenticate(request)
        return self.handler.get_session_stats()

    async def connect(self, request: Request) -> Response:
        self.authenticate(request)
        acquired = await self.acquire()
        try:
            data, _ = decode_multipart(
                request.headers.get("content-type", ""), await request.body()
            )
            result = await self.handler.connect(
                dict(data.get("env_config") or {}), seed=data.get("seed")
            )
            return self._response(result)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - service hook maps handler failures to HTTP
            self.handle_connect_error(exc)
        finally:
            if acquired:
                self.release()

    async def call(self, request: Request) -> Response:
        self.authenticate(request)
        acquired = await self.acquire()
        try:
            data, images = decode_multipart(
                request.headers.get("content-type", ""), await request.body()
            )
            session_id = str(data.get("session_id", ""))
            method = str(data.get("method", ""))
            if not session_id or not method:
                raise HTTPException(
                    status_code=400, detail="session_id and method are required"
                )
            result = await self.handler.call(
                session_id, method, dict(data.get("params") or {}), images
            )
            return self._response(result)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001 - service hook maps handler failures to HTTP
            self.handle_call_error(exc)
        finally:
            if acquired:
                self.release()

    def register_routes(self, app: Any) -> None:
        app.add_api_route("/health", self.health, methods=["GET"])
        app.add_api_route("/sessions", self.sessions, methods=["GET"])
        app.add_api_route("/connect", self.connect, methods=["POST"])
        app.add_api_route("/call", self.call, methods=["POST"])

    def build(self, startup_callback=None, shutdown_callback=None) -> Any:
        handler = self.handler

        @asynccontextmanager
        async def lifespan(_app):
            if startup_callback is not None:
                startup_callback()
            try:
                yield
            finally:
                if shutdown_callback is not None:
                    shutdown_callback()
                await handler.aclose()

        app = FastAPI(
            title="Gym Environment Service",
            description="Generic HTTP service for remote VAGEN environments",
            lifespan=lifespan,
        )
        app.state.handler = handler
        self.register_routes(app)
        return app


__all__ = ["GymService"]
