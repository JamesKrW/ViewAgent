import asyncio
import time

from view_suite.ai2thor.service_http import handler as handler_module


class _FakeController:
    def step(self, **kwargs):
        return None

    def stop(self):
        return None

    @property
    def last_event(self):
        return type("Event", (), {"third_party_camera_frames": [object()]})()


def _task(size: int, fov: float):
    return {
        "pose": {
            "position": {"x": 0.0, "y": 1.5, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
        },
        "width": size,
        "height": size,
        "fov": fov,
    }


def _pool(monkeypatch, created):
    def create_controller(*args, **kwargs):
        time.sleep(0.02)
        created.append((args, kwargs))
        return _FakeController()

    monkeypatch.setattr(handler_module, "_create_controller", create_controller)
    monkeypatch.setattr(handler_module, "_encode_rgb_to_png_bytes", lambda frame: b"png")
    return handler_module._ThreadControllerPool(
        max_slots=8,
        max_threads=16,
        controller_config={},
        default_width=512,
        default_height=512,
        default_fov=90.0,
    )


def test_cold_same_scene_requests_create_one_controller(monkeypatch):
    created = []
    pool = _pool(monkeypatch, created)

    async def run():
        await asyncio.gather(
            *(pool.render("FloorPlan1", [_task(512, 90.0)]) for _ in range(16))
        )

    try:
        asyncio.run(run())
        assert len(created) == 1
    finally:
        pool.shutdown()


def test_same_scene_different_render_params_use_distinct_slots(monkeypatch):
    created = []
    pool = _pool(monkeypatch, created)

    async def run():
        await pool.render("FloorPlan1", [_task(512, 90.0)])
        await pool.render("FloorPlan1", [_task(256, 53.1)])

    try:
        asyncio.run(run())
        assert len(created) == 2
        assert len(set(pool.request_to_slot.values())) == 2
        assert pool.metrics().get("controller_reconstruct", 0) == 0
    finally:
        pool.shutdown()
