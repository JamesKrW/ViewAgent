from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from vagen_agent.envs import (
    BaseCompactEnv,
    EnvAction,
    get_env_cls,
)
from vagen_agent.envs.remote import BaseGymHandler, GymService, RemoteEnv
from vagen_agent.envs.remote.multipart_codec import decode_multipart, encode_multipart


class _FakeEnv(BaseCompactEnv):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.actions = []
        self.closed = False

    async def _reset(self, seed=None):
        return {
            "obs_str": "<image>ready",
            "multi_modal_input": {"<image>": [Image.new("RGB", (7, 5), "blue")]},
            "screen_size": (1280, 720),
            "dom": {"buttons": [{"id": "start", "enabled": True}]},
            "proprio": [0.25, 0.5],
        }, {"seed": seed}

    async def _system_prompt(self):
        return {"obs_str": "act carefully"}

    async def _step(self, action):
        self.actions.append(action)
        return {
            "obs_str": "done",
            "screen_size": [1280, 720],
            "dom": {"buttons": []},
        }, [0.0, 1.0], True, False, {"success": True, "score": 1}

    async def close(self):
        self.closed = True


class _FakeHandler(BaseGymHandler):
    def __init__(self):
        super().__init__(max_sessions=1)
        self.created = []

    async def create_env(self, env_config):
        env = _FakeEnv(env_config)
        self.created.append(env)
        return env


def test_remote_env_is_registered() -> None:
    assert get_env_cls("RemoteEnv") is RemoteEnv


def test_multipart_round_trip_preserves_json_and_images() -> None:
    image = Image.new("RGB", (7, 5), (10, 20, 30))
    boundary, body = encode_multipart({"value": [1, 2]}, [image])
    data, images = decode_multipart(f'multipart/mixed; boundary="{boundary}"', body)
    assert data == {"value": [1, 2]}
    assert len(images) == 1
    assert images[0].size == (7, 5)
    assert images[0].getpixel((0, 0)) == (10, 20, 30)


def test_multipart_jpeg_accepts_rgba_frames() -> None:
    image = Image.new("RGBA", (8, 4), (10, 20, 30, 128))
    boundary, body = encode_multipart(
        {"codec": "jpeg"},
        [image],
        image_format="JPEG",
        image_mime="image/jpeg",
        image_options={"quality": 90, "subsampling": 0},
    )
    data, images = decode_multipart(f'multipart/mixed; boundary="{boundary}"', body)
    assert data == {"codec": "jpeg"}
    assert images[0].mode == "RGB"
    assert images[0].size == (8, 4)


def test_remote_service_client_lifecycle_and_response_unwrapping() -> None:
    async def scenario() -> None:
        handler = _FakeHandler()
        app = GymService(handler).build()
        client = RemoteEnv(
            {
                "base_urls": "http://remote.test",
                "timeout": 5,
                "retries": 0,
                "task": "demo",
            }
        )
        client._client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app))

        observation, info = await client.reset(seed=7)
        assert observation["obs_str"] == "<image>ready"
        assert observation["multi_modal_input"]["<image>"][0].size == (7, 5)
        assert observation["screen_size"] == [1280, 720]
        assert observation["dom"] == {
            "buttons": [{"id": "start", "enabled": True}]
        }
        assert observation["proprio"] == [0.25, 0.5]
        assert info == {"seed": 7}
        assert handler.created[0].config == {"task": "demo"}

        assert await client.system_prompt() == {"obs_str": "act carefully"}
        next_observation, reward, terminated, truncated, step_info = await client.step(
            SimpleNamespace(text="move left")
        )
        assert next_observation == {
            "obs_str": "done",
            "screen_size": [1280, 720],
            "dom": {"buttons": []},
        }
        assert reward == [0.0, 1.0]
        assert terminated is True
        assert truncated is False
        assert step_info == {"success": True, "score": 1}
        assert handler.created[0].actions == ["move left"]

        source = SimpleNamespace(
            text="raw model output", call_id=9, token_ids=[4, 5]
        )
        await client.reset(seed=8)
        await client.step(
            EnvAction(
                value={"schema": "example.action.v1", "key": "w"},
                response=source,
            )
        )
        assert handler.created[0].actions[-1] == {
            "schema": "example.action.v1",
            "key": "w",
        }

        await client.close()
        assert handler.created[0].closed is True
        assert handler.get_session_stats()["num_sessions"] == 0
        await handler.aclose()

    asyncio.run(scenario())


def test_connect_failover_pins_stateful_calls_to_the_selected_server() -> None:
    async def scenario() -> None:
        calls = []

        async def transport(request: httpx.Request) -> httpx.Response:
            calls.append((request.url.host, request.url.path))
            if request.url.host == "first.test":
                return httpx.Response(503, request=request)
            if request.url.path == "/connect":
                data = {"session_id": "session-1", "obs": "ready", "info": {}}
            else:
                request_data, _ = decode_multipart(
                    request.headers["content-type"], request.content
                )
                assert request_data["session_id"] == "session-1"
                if request_data["method"] == "close":
                    data = {"closed": True}
                else:
                    data = {"obs": "next", "reward": 0.0, "done": False, "info": {}}
            boundary, body = encode_multipart(data)
            return httpx.Response(
                200,
                headers={"Content-Type": f'multipart/mixed; boundary="{boundary}"'},
                content=body,
                request=request,
            )

        client = RemoteEnv(
            {
                "base_urls": ["http://first.test", "http://second.test"],
                "retries": 1,
                "failover_after_failures": 0,
                "backoff": 0,
            }
        )
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        await client.reset(seed=0)
        await client.step("wait")
        await client.close()

        assert calls == [
            ("first.test", "/connect"),
            ("second.test", "/connect"),
            ("second.test", "/call"),
            ("second.test", "/call"),
        ]

    asyncio.run(scenario())


def test_remote_service_authentication() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    handler = _FakeHandler()
    app = GymService(handler, api_key="secret").build()
    boundary, body = encode_multipart({"env_config": {}, "seed": 1})
    headers = {"Content-Type": f'multipart/form-data; boundary="{boundary}"'}
    with TestClient(app) as client:
        assert client.post("/connect", content=body, headers=headers).status_code == 401
        headers["X-API-Key"] = "secret"
        response = client.post("/connect", content=body, headers=headers)
        assert response.status_code == 200
        assert b"Content-Type: image/jpeg" in response.content


def test_remote_service_can_opt_back_into_lossless_png() -> None:
    service = GymService(
        _FakeHandler(), image_format="PNG", image_mime="image/png", image_options={}
    )
    assert service.image_format == "PNG"
    assert service.image_mime == "image/png"
    assert service.image_options == {}
